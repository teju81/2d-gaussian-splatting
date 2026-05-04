"""Project a LAS LiDAR point cloud into each COLMAP camera frame, producing
per-image depth + normal maps used by custom_train.py for LiDAR supervision.

CLI:
    python scripts/lidar_to_depth_maps.py --config <path/to/v0_1_release.yaml>

Reads from YAML:
  dataset_dir                                           (global)
  train_and_eval.train.custom_train_2dgs.source_path    (optional override)
  train_and_eval.train.custom_train_2dgs.las_path       (required)
  train_and_eval.train.custom_train_2dgs.depth_maps_dir (output location)
  train_and_eval.train.custom_train_2dgs.depth_map_generation.lidar_voxel_size
  train_and_eval.train.custom_train_2dgs.depth_map_generation.normal_radius

Writes <depth_maps_dir>/cam<cam_id>_<image_stem>.npz per image with:
  depth:  (H, W)    float32, 0 where no LiDAR hit
  normal: (H, W, 3) float32 world-frame, zeros where no hit

Key format: 'cam{cam_id}_{stem}' where cam_id is the COLMAP camera
intrinsic id (matches 2DGS Camera.colmap_id) and stem is the image
filename stem (matches 2DGS Camera.image_name). Stems alone are NOT
unique in this dataset (441 collisions between camera_0/ and camera_1/
subdirs) but (stem, cam_id) is.

Normals stored in WORLD frame (same as 2DGS rendered normals) so the loss
can compare them directly with no per-iteration rotation.

For faster variants on the same output format see:
  scripts/lidar_to_depth_maps_mp.py    -- CPU multiprocessing
  scripts/lidar_to_depth_maps_cuda.py  -- PyTorch CUDA
"""

import argparse
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

# Ensure scripts/ is on sys.path when this file is executed directly.
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
    args = ap.parse_args()

    full_cfg, dataset_dir, ct = _load_cfg(args.config)

    # CLI overrides (apply to a copy so the YAML dict stays untouched).
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

    print(f"[depth-maps] source_path     = {source_path}")
    print(f"[depth-maps] las_path        = {las_path}")
    print(f"[depth-maps] output dir      = {dm_path}")
    print(f"[depth-maps] voxel_size      = {voxel_size}")
    print(f"[depth-maps] normal_radius   = {normal_radius}")

    # --- Load + downsample + estimate normals --------------------------------
    t0 = time.time()
    print("[depth-maps] reading LAS ...")
    las = laspy.read(str(las_path))
    xyz = np.stack([np.asarray(las.x, dtype=np.float64),
                    np.asarray(las.y, dtype=np.float64),
                    np.asarray(las.z, dtype=np.float64)], axis=1)
    print(f"[depth-maps]   loaded {xyz.shape[0]:,} points in {time.time()-t0:.1f}s")

    t0 = time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    xyz = np.asarray(pcd.points, dtype=np.float64)
    print(f"[depth-maps]   voxel-down ({voxel_size} m) -> {xyz.shape[0]:,} points "
          f"in {time.time()-t0:.1f}s")

    t0 = time.time()
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=normal_radius))
    normals = np.asarray(pcd.normals, dtype=np.float64)
    print(f"[depth-maps]   normals in {time.time()-t0:.1f}s")

    # --- Parse COLMAP --------------------------------------------------------
    cams = _parse_cameras_txt(cameras_txt)
    images = _parse_images_txt(images_txt)
    print(f"[depth-maps] {len(cams)} cameras, {len(images)} images parsed")
    _assert_unique_keys(images)

    # Orient normals toward scene centroid = mean camera position.
    # Camera position in world is -R^T @ t where R,t are world->camera.
    cam_centres = []
    for (_id, qw, qx, qy, qz, tx, ty, tz, _cam_id, _name) in images:
        R = _quat_to_rotmat(qw, qx, qy, qz)
        c = -R.T @ np.array([tx, ty, tz], dtype=np.float64)
        cam_centres.append(c)
    centroid = np.mean(np.stack(cam_centres, axis=0), axis=0)
    to_centroid = centroid[None, :] - xyz
    flip = np.einsum("ij,ij->i", normals, to_centroid) < 0.0
    normals[flip] *= -1.0
    print(f"[depth-maps]   oriented normals (centroid = {centroid}); flipped {flip.sum():,}")

    # --- Per-frame projection ------------------------------------------------
    t_all = time.time()
    n_written = 0
    n_skip = 0
    coverage_sum = 0.0
    pbar = tqdm(images, desc="[depth-maps] projecting", unit="frame")
    for image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name in pbar:
        if cam_id not in cams:
            pbar.write(f"  [skip] image_id={image_id} references unknown cam_id={cam_id}")
            n_skip += 1
            continue
        fx, fy, cx, cy, W, H = cams[cam_id]
        R = _quat_to_rotmat(qw, qx, qy, qz)
        t_vec = np.array([tx, ty, tz], dtype=np.float64)

        depth, normal = _project_one_frame(xyz, normals, R, t_vec, fx, fy, cx, cy, W, H)

        out_path = dm_path / (_depth_map_key(cam_id, name) + ".npz")
        np.savez_compressed(out_path, depth=depth, normal=normal)
        n_written += 1
        this_cov = float((depth > 0).mean())
        coverage_sum += this_cov

        if n_written % 10 == 0:
            mean_cov = coverage_sum / n_written
            pbar.set_postfix({"cov": f"{this_cov*100:.1f}%",
                              "mean_cov": f"{mean_cov*100:.1f}%"})
    pbar.close()

    mean_cov = coverage_sum / max(n_written, 1)
    print(f"[depth-maps] wrote {n_written} frames (skip={n_skip}), "
          f"mean coverage = {mean_cov*100:.1f}%, elapsed {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
