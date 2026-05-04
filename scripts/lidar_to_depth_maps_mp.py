"""CPU multiprocessing variant of scripts/lidar_to_depth_maps.py.

Identical behaviour and on-disk output to the reference script; only the
per-frame projection loop is parallelised across workers via fork + COW
shared memory for the LiDAR point cloud.

CLI: same as the reference script, plus
  --workers N          (default: os.cpu_count())
  --chunksize N        (default: 8; Pool.imap_unordered chunk size)

Example:
    python scripts/lidar_to_depth_maps_mp.py --config path/to/v0_1_release.yaml \\
        --workers 16 --depth_maps_dir depth_maps_lidar_mp

See lidar_to_depth_maps_common.py for shared helpers.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import laspy
except ImportError:
    print("ERROR: laspy not installed. `pip install laspy` inside the container.")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d not installed. `pip install open3d` inside the container.")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lidar_to_depth_maps_common import (  # noqa: E402
    _parse_cameras_txt,
    _parse_images_txt,
    _quat_to_rotmat,
    _project_one_frame,
    _load_cfg,
    _resolve_source_path,
    _depth_map_key,
    _assert_unique_keys,
)


# ---------------------------------------------------------------------------
# Worker state (populated in the parent BEFORE fork so children see the same
# memory pages via COW; no pickling of the big arrays).
# ---------------------------------------------------------------------------

_W_XYZ = None         # (N, 3) float64
_W_NORMALS = None     # (N, 3) float64
_W_CAMS = None        # {cam_id: (fx, fy, cx, cy, W, H)}
_W_DM_PATH = None     # pathlib.Path


def _worker_project(frame):
    """Project one frame and write its .npz. Returns (wrote, coverage, skip_reason)."""
    image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name = frame
    if cam_id not in _W_CAMS:
        return (False, 0.0, f"image_id={image_id} references unknown cam_id={cam_id}")
    fx, fy, cx, cy, W, H = _W_CAMS[cam_id]
    R = _quat_to_rotmat(qw, qx, qy, qz)
    t_vec = np.array([tx, ty, tz], dtype=np.float64)
    depth, normal = _project_one_frame(_W_XYZ, _W_NORMALS, R, t_vec,
                                       fx, fy, cx, cy, W, H)
    out_path = _W_DM_PATH / (_depth_map_key(cam_id, name) + ".npz")
    np.savez_compressed(out_path, depth=depth, normal=normal)
    cov = float((depth > 0).mean())
    return (True, cov, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path,
                    help="Path to v0_1_release.yaml (or similar).")
    ap.add_argument("--xgrids_path", type=str, default=None,
                    help="Override custom_train_2dgs.xgrids_path for this run.")
    ap.add_argument("--las_path", type=str, default=None,
                    help="Override custom_train_2dgs.las_path for this run.")
    ap.add_argument("--depth_maps_dir", type=str, default=None,
                    help="Override custom_train_2dgs.depth_maps_dir for this run.")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help=f"Number of worker processes (default: os.cpu_count() = {os.cpu_count()}).")
    ap.add_argument("--chunksize", type=int, default=8,
                    help="Pool.imap_unordered chunk size (default: 8).")
    args = ap.parse_args()

    full_cfg, dataset_dir, ct = _load_cfg(args.config)

    ct = dict(ct)
    if args.xgrids_path:    ct["xgrids_path"] = args.xgrids_path
    if args.las_path:       ct["las_path"] = args.las_path
    if args.depth_maps_dir: ct["depth_maps_dir"] = args.depth_maps_dir

    las_path = ct.get("las_path")
    if not las_path:
        print("ERROR: las_path is not set (in YAML custom_train_2dgs.las_path or --las_path).")
        sys.exit(1)
    las_path = Path(las_path).expanduser().resolve()
    if not las_path.is_file():
        print(f"ERROR: LAS file not found: {las_path}")
        sys.exit(1)

    source_path = _resolve_source_path(dataset_dir, ct)
    sparse0 = source_path / "sparse" / "0"
    cameras_txt = sparse0 / "cameras.txt"
    images_txt = sparse0 / "images.txt"

    depth_maps_dir = ct.get("depth_maps_dir") or "depth_maps_lidar"
    dm_path = Path(depth_maps_dir)
    if not dm_path.is_absolute():
        dm_path = source_path / dm_path
    dm_path.mkdir(parents=True, exist_ok=True)

    gen = ct.get("depth_map_generation", {}) or {}
    voxel_size = float(gen.get("lidar_voxel_size", 0.002))
    normal_radius = float(gen.get("normal_radius", 0.05))

    print(f"[depth-maps-mp] source_path     = {source_path}")
    print(f"[depth-maps-mp] las_path        = {las_path}")
    print(f"[depth-maps-mp] output dir      = {dm_path}")
    print(f"[depth-maps-mp] voxel_size      = {voxel_size}")
    print(f"[depth-maps-mp] normal_radius   = {normal_radius}")
    print(f"[depth-maps-mp] workers         = {args.workers}")

    # --- Load + downsample + estimate normals --------------------------------
    t0 = time.time()
    print("[depth-maps-mp] reading LAS ...")
    las = laspy.read(str(las_path))
    xyz = np.stack([np.asarray(las.x, dtype=np.float64),
                    np.asarray(las.y, dtype=np.float64),
                    np.asarray(las.z, dtype=np.float64)], axis=1)
    print(f"[depth-maps-mp]   loaded {xyz.shape[0]:,} points in {time.time()-t0:.1f}s")

    t0 = time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    xyz = np.asarray(pcd.points, dtype=np.float64)
    print(f"[depth-maps-mp]   voxel-down ({voxel_size} m) -> {xyz.shape[0]:,} points "
          f"in {time.time()-t0:.1f}s")

    t0 = time.time()
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=normal_radius))
    normals = np.asarray(pcd.normals, dtype=np.float64)
    print(f"[depth-maps-mp]   normals in {time.time()-t0:.1f}s")

    # --- Parse COLMAP --------------------------------------------------------
    cams = _parse_cameras_txt(cameras_txt)
    images = _parse_images_txt(images_txt)
    print(f"[depth-maps-mp] {len(cams)} cameras, {len(images)} images parsed")
    _assert_unique_keys(images)

    # Orient normals toward scene centroid = mean camera position.
    cam_centres = []
    for (_id, qw, qx, qy, qz, tx, ty, tz, _cam_id, _name) in images:
        R = _quat_to_rotmat(qw, qx, qy, qz)
        c = -R.T @ np.array([tx, ty, tz], dtype=np.float64)
        cam_centres.append(c)
    centroid = np.mean(np.stack(cam_centres, axis=0), axis=0)
    to_centroid = centroid[None, :] - xyz
    flip = np.einsum("ij,ij->i", normals, to_centroid) < 0.0
    normals[flip] *= -1.0
    print(f"[depth-maps-mp]   oriented normals (centroid = {centroid}); flipped {flip.sum():,}")

    # --- Stash in module globals BEFORE forking so workers inherit via COW ---
    # Using an initializer would pickle these arrays per worker (~hundreds of
    # MB each); plain globals + fork keeps them in one set of pages.
    global _W_XYZ, _W_NORMALS, _W_CAMS, _W_DM_PATH
    _W_XYZ = xyz
    _W_NORMALS = normals
    _W_CAMS = cams
    _W_DM_PATH = dm_path

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        print("[depth-maps-mp] WARNING: 'fork' start method unavailable; falling back "
              "to 'spawn'. Each worker will re-load the point cloud "
              "(expect higher RAM use and slower startup).")
        ctx = mp.get_context("spawn")

    # --- Per-frame projection (parallel) -------------------------------------
    t_all = time.time()
    n_written = 0
    n_skip = 0
    coverage_sum = 0.0
    with ctx.Pool(processes=args.workers) as pool:
        pbar = tqdm(total=len(images), desc="[depth-maps-mp] projecting", unit="frame")
        for wrote, cov, skip_reason in pool.imap_unordered(
                _worker_project, images, chunksize=args.chunksize):
            if wrote:
                n_written += 1
                coverage_sum += cov
                if n_written % 10 == 0:
                    mean_cov = coverage_sum / n_written
                    pbar.set_postfix({"cov": f"{cov*100:.1f}%",
                                      "mean_cov": f"{mean_cov*100:.1f}%"})
            else:
                n_skip += 1
                pbar.write(f"  [skip] {skip_reason}")
            pbar.update(1)
        pbar.close()

    mean_cov = coverage_sum / max(n_written, 1)
    print(f"[depth-maps-mp] wrote {n_written} frames (skip={n_skip}), "
          f"mean coverage = {mean_cov*100:.1f}%, elapsed {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
