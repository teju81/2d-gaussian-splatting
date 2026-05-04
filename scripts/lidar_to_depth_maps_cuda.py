"""PyTorch CUDA variant of scripts/lidar_to_depth_maps.py.

Same on-disk output format as the reference script, but the per-frame
projection runs entirely on GPU. Key tricks:

1. xyz + normals are uploaded to CUDA once (as float32; float precision is
   ample for 1024^2 depth maps at room-to-garage scene scale).
2. Per-frame z-buffer is computed in a single scatter_reduce(reduce='amin')
   over packed int64 values, where the high 32 bits hold the float32 bits of
   z (which compare monotonically as int32 for z > 0) and the low 32 bits
   hold the kept-point index. The packed amin returns the smallest z AND the
   winning point's index in one pass -- no argsort needed.
3. np.savez_compressed runs in a bounded ThreadPoolExecutor so CPU zlib
   overlaps with the next frame's GPU kernels. Threads (not processes)
   because zlib releases the GIL.

CLI:
  --config <yaml>       (required)
  --dataset_root <path> (overrides custom_train_2dgs.dataset_root in YAML)
  --device cuda[:N]     (default: cuda)
  --save_workers N      (default: 4; set 0 to disable save-overlap)

Example:
    python scripts/lidar_to_depth_maps_cuda.py --config path/to/v0_1_release.yaml \\
        --dataset_root /path/to/mm1_data

See lidar_to_depth_maps_common.py for shared helpers.
"""

import argparse
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

try:
    import torch
except ImportError:
    print("ERROR: torch not installed. Use lidar_to_depth_maps.py or "
          "lidar_to_depth_maps_mp.py if you don't have a CUDA-capable torch.")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lidar_to_depth_maps_common import (  # noqa: E402
    _parse_cameras_txt,
    _parse_images_txt,
    _quat_to_rotmat,
    _load_cfg,
    _depth_map_key,
    _assert_unique_keys,
)
from utils.dataset_utils import (  # noqa: E402
    find_las_file,
    prepare_xgrids_dataset,
    resolve_dataset_root,
)


_INT32_MASK = 0xFFFFFFFF


def _project_one_frame_cuda(xyz_g, normals_g, R_np, t_np,
                            fx, fy, cx, cy, W, H, device):
    """GPU analogue of _project_one_frame. Returns numpy arrays for savez.

    Args:
        xyz_g:     (N, 3) float32 CUDA tensor, world-frame LiDAR points.
        normals_g: (N, 3) float32 CUDA tensor, world-frame normals.
        R_np, t_np: world->camera rotation and translation (numpy, float64).
        fx, fy, cx, cy, W, H: PINHOLE intrinsics.
    """
    R = torch.from_numpy(R_np).to(device=device, dtype=torch.float32)
    t = torch.from_numpy(t_np).to(device=device, dtype=torch.float32)

    x_c = xyz_g @ R.T + t                      # (N, 3)
    z_c = x_c[:, 2]
    front = z_c > 1e-6
    x_c = x_c[front]
    z_c = z_c[front]
    n_keep = normals_g[front]

    u = fx * x_c[:, 0] / z_c + cx
    v = fy * x_c[:, 1] / z_c + cy
    u_i = u.floor().to(torch.int64)
    v_i = v.floor().to(torch.int64)
    in_frame = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    u_i = u_i[in_frame]
    v_i = v_i[in_frame]
    z_c = z_c[in_frame]
    n_keep = n_keep[in_frame]

    if u_i.numel() == 0:
        return (np.zeros((H, W), dtype=np.float32),
                np.zeros((H, W, 3), dtype=np.float32))

    flat_idx = v_i * W + u_i                   # (K,) int64
    K = z_c.numel()

    # Pack (z_bits, local_idx) into int64. For z > 0 the IEEE-754 bit layout
    # of float32 is monotonically increasing as int32, so amin over packed
    # values recovers the smallest z AND the local index that produced it.
    z_bits = z_c.contiguous().view(torch.int32).to(torch.int64) & _INT32_MASK
    local_idx = torch.arange(K, device=device, dtype=torch.int64)
    packed = (z_bits << 32) | local_idx

    sentinel = torch.iinfo(torch.int64).max
    buf = torch.full((H * W,), sentinel, dtype=torch.int64, device=device)
    buf.scatter_reduce_(0, flat_idx, packed, reduce="amin", include_self=True)

    filled = buf != sentinel

    # Depth: high 32 bits, reinterpreted as float32. Zero where unfilled.
    z_bits_out = ((buf >> 32) & _INT32_MASK).to(torch.int32).view(torch.float32)
    depth_flat = torch.where(filled, z_bits_out, torch.zeros_like(z_bits_out))

    # Normal: gather from n_keep using the winning local index. Zero where unfilled.
    winner_local = (buf & _INT32_MASK).to(torch.int64)
    normal_flat = torch.zeros((H * W, 3), dtype=torch.float32, device=device)
    filled_pixel_idx = torch.nonzero(filled, as_tuple=False).flatten()
    normal_flat[filled_pixel_idx] = n_keep[winner_local[filled_pixel_idx]]

    depth_np = depth_flat.view(H, W).cpu().numpy()
    normal_np = normal_flat.view(H, W, 3).cpu().numpy()
    return depth_np, normal_np


def run(config_path, dataset_root, device="cuda", save_workers=4):
    """Generate per-frame LiDAR depth+normal .npz files. Library entry point.

    Returns the Path to the output depth-maps directory.
    """
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False. Run the CPU variants "
              "(lidar_to_depth_maps.py or lidar_to_depth_maps_mp.py) instead.")
        sys.exit(1)
    device = torch.device(device)

    full_cfg, _, ct = _load_cfg(Path(config_path))

    dataset_root_path = Path(dataset_root).expanduser().resolve()
    source_path_str, _ = prepare_xgrids_dataset(dataset_root_path)
    source_path = Path(source_path_str)
    las_path = find_las_file(dataset_root_path)

    sparse0 = source_path / "sparse" / "0"
    cameras_txt = sparse0 / "cameras.txt"
    images_txt = sparse0 / "images.txt"

    dm_path = source_path / "depth_maps_lidar"
    dm_path.mkdir(parents=True, exist_ok=True)

    gen = ct.get("depth_map_generation", {}) or {}
    voxel_size = float(gen.get("lidar_voxel_size", 0.002))
    normal_radius = float(gen.get("normal_radius", 0.05))

    print(f"[depth-maps-cuda] dataset_root    = {dataset_root_path}")
    print(f"[depth-maps-cuda] source_path     = {source_path}")
    print(f"[depth-maps-cuda] las_path        = {las_path}")
    print(f"[depth-maps-cuda] output dir      = {dm_path}")
    print(f"[depth-maps-cuda] voxel_size      = {voxel_size}")
    print(f"[depth-maps-cuda] normal_radius   = {normal_radius}")
    print(f"[depth-maps-cuda] device          = {device}")
    print(f"[depth-maps-cuda] save_workers    = {save_workers}")

    # --- Load + downsample + estimate normals (CPU/Open3D, same as reference)
    t0 = time.time()
    print("[depth-maps-cuda] reading LAS ...")
    las = laspy.read(str(las_path))
    xyz = np.stack([np.asarray(las.x, dtype=np.float64),
                    np.asarray(las.y, dtype=np.float64),
                    np.asarray(las.z, dtype=np.float64)], axis=1)
    print(f"[depth-maps-cuda]   loaded {xyz.shape[0]:,} points in {time.time()-t0:.1f}s")

    t0 = time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    xyz = np.asarray(pcd.points, dtype=np.float64)
    print(f"[depth-maps-cuda]   voxel-down ({voxel_size} m) -> {xyz.shape[0]:,} points "
          f"in {time.time()-t0:.1f}s")

    t0 = time.time()
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=normal_radius))
    normals = np.asarray(pcd.normals, dtype=np.float64)
    print(f"[depth-maps-cuda]   normals in {time.time()-t0:.1f}s")

    # --- Parse COLMAP --------------------------------------------------------
    cams = _parse_cameras_txt(cameras_txt)
    images = _parse_images_txt(images_txt)
    print(f"[depth-maps-cuda] {len(cams)} cameras, {len(images)} images parsed")
    _assert_unique_keys(images)

    cam_centres = []
    for (_id, qw, qx, qy, qz, tx, ty, tz, _cam_id, _name) in images:
        R = _quat_to_rotmat(qw, qx, qy, qz)
        c = -R.T @ np.array([tx, ty, tz], dtype=np.float64)
        cam_centres.append(c)
    centroid = np.mean(np.stack(cam_centres, axis=0), axis=0)
    to_centroid = centroid[None, :] - xyz
    flip = np.einsum("ij,ij->i", normals, to_centroid) < 0.0
    normals[flip] *= -1.0
    print(f"[depth-maps-cuda]   oriented normals (centroid = {centroid}); flipped {flip.sum():,}")

    # --- Upload to GPU once --------------------------------------------------
    t0 = time.time()
    xyz_g = torch.from_numpy(xyz.astype(np.float32, copy=False)).to(device)
    normals_g = torch.from_numpy(normals.astype(np.float32, copy=False)).to(device)
    print(f"[depth-maps-cuda]   uploaded {xyz.shape[0]:,} points to {device} in {time.time()-t0:.1f}s "
          f"({xyz_g.element_size() * xyz_g.numel() * 2 / 1e6:.0f} MB)")

    # --- Per-frame projection ------------------------------------------------
    t_all = time.time()
    n_written = 0
    n_skip = 0
    coverage_sum = 0.0

    save_pool = ThreadPoolExecutor(max_workers=save_workers) if save_workers > 0 else None
    MAX_OUTSTANDING = max(save_workers * 2, 4) if save_pool else 0
    pending = deque()

    def _submit_or_save(out_path, depth_np, normal_np):
        """Either enqueue to the thread pool or save inline, returning the cov."""
        if save_pool is None:
            np.savez_compressed(out_path, depth=depth_np, normal=normal_np)
        else:
            fut = save_pool.submit(np.savez_compressed, out_path,
                                   depth=depth_np, normal=normal_np)
            pending.append(fut)
            if len(pending) >= MAX_OUTSTANDING:
                pending.popleft().result()  # back-pressure

    pbar = tqdm(images, desc="[depth-maps-cuda] projecting", unit="frame")
    for image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name in pbar:
        if cam_id not in cams:
            pbar.write(f"  [skip] image_id={image_id} references unknown cam_id={cam_id}")
            n_skip += 1
            continue
        fx, fy, cx, cy, W, H = cams[cam_id]
        R = _quat_to_rotmat(qw, qx, qy, qz)
        t_vec = np.array([tx, ty, tz], dtype=np.float64)

        depth, normal = _project_one_frame_cuda(
            xyz_g, normals_g, R, t_vec, fx, fy, cx, cy, W, H, device,
        )

        out_path = dm_path / (_depth_map_key(cam_id, name) + ".npz")
        _submit_or_save(out_path, depth, normal)

        n_written += 1
        this_cov = float((depth > 0).mean())
        coverage_sum += this_cov

        if n_written % 10 == 0:
            mean_cov = coverage_sum / n_written
            pbar.set_postfix({"cov": f"{this_cov*100:.1f}%",
                              "mean_cov": f"{mean_cov*100:.1f}%"})
    pbar.close()

    # Drain remaining saves.
    if save_pool is not None:
        while pending:
            pending.popleft().result()
        save_pool.shutdown()

    mean_cov = coverage_sum / max(n_written, 1)
    print(f"[depth-maps-cuda] wrote {n_written} frames (skip={n_skip}), "
          f"mean coverage = {mean_cov*100:.1f}%, elapsed {time.time()-t_all:.1f}s")

    return dm_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path,
                    help="Path to v0_1_release.yaml (or similar).")
    ap.add_argument("--dataset_root", type=str, default=None,
                    help="Root folder containing the las file and perspective/ folder. "
                         "Overrides custom_train_2dgs.dataset_root in YAML.")
    ap.add_argument("--device", type=str, default="cuda",
                    help="CUDA device (e.g. 'cuda' or 'cuda:0'). Default: cuda.")
    ap.add_argument("--save_workers", type=int, default=4,
                    help="Threads for parallel np.savez_compressed. 0 = serial save. Default: 4.")
    args = ap.parse_args()

    dataset_root = resolve_dataset_root(args.config, args.dataset_root)
    run(config_path=args.config, dataset_root=dataset_root,
        device=args.device, save_workers=args.save_workers)


if __name__ == "__main__":
    main()
