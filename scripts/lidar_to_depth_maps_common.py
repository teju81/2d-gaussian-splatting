"""Shared helpers for the lidar_to_depth_maps* scripts.

Used by:
  scripts/lidar_to_depth_maps.py        (reference, single-threaded CPU)
  scripts/lidar_to_depth_maps_mp.py     (multiprocessing CPU)
  scripts/lidar_to_depth_maps_cuda.py   (PyTorch CUDA)

Contains COLMAP parsing, quaternion math, config resolution, depth-map
file naming, and the CPU per-frame projection kernel. The CUDA variant
implements its own projection and does not use _project_one_frame.
"""

import sys
from pathlib import Path

import numpy as np
import yaml

# Make utils.dataset_utils importable from any of the scripts that import this.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent  # 2dgs_inria/
sys.path.insert(0, str(_REPO_ROOT))

from utils.dataset_utils import prepare_xgrids_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# COLMAP parsing
# ---------------------------------------------------------------------------

def _parse_cameras_txt(path: Path):
    """Return {camera_id: (fx, fy, cx, cy, W, H)} for PINHOLE cameras."""
    cams = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            cam_id = int(toks[0])
            model = toks[1]
            W, H = int(toks[2]), int(toks[3])
            params = [float(x) for x in toks[4:]]
            if model == "PINHOLE":
                fx, fy, cx, cy = params
            elif model == "SIMPLE_PINHOLE":
                f_, cx, cy = params
                fx = fy = f_
            else:
                raise ValueError(f"Unsupported camera model '{model}' (need PINHOLE)")
            cams[cam_id] = (fx, fy, cx, cy, W, H)
    return cams


def _parse_images_txt(path: Path):
    """Return list of (image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name).

    COLMAP images.txt stores TWO lines per image: a pose line and a POINTS2D
    line. POINTS2D may be EMPTY (blank line). We drop comment lines entirely,
    then consume the remaining lines in pairs (pose, points2D) regardless of
    whether the POINTS2D line is blank. Quaternion is (qw, qx, qy, qz); the
    rotation R + translation t transform world -> camera (x_c = R @ x_w + t).
    """
    raw = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            raw.append(line.rstrip("\n"))

    i = 0
    while i < len(raw) and raw[i].strip() == "":
        i += 1

    images = []
    while i < len(raw):
        pose_line = raw[i].strip()
        if not pose_line:
            i += 1
            continue
        toks = pose_line.split()
        if len(toks) < 10:
            print(f"WARNING: malformed pose line at index {i}: {pose_line[:80]!r}")
            i += 1
            continue
        image_id = int(toks[0])
        qw, qx, qy, qz = map(float, toks[1:5])
        tx, ty, tz = map(float, toks[5:8])
        cam_id = int(toks[8])
        name = " ".join(toks[9:])
        images.append((image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name))
        i += 2
    return images


def _quat_to_rotmat(qw, qx, qy, qz):
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix. World -> camera."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    return R


# ---------------------------------------------------------------------------
# Per-frame projection (CPU, vectorised)
# ---------------------------------------------------------------------------

def _project_one_frame(points_w, normals_w, R, t, fx, fy, cx, cy, W, H):
    """Project world-frame LiDAR points + normals into one camera frame.

    Returns:
        depth:  (H, W) float32, 0 where no hit.
        normal: (H, W, 3) float32 world-frame, zeros where no hit.
    """
    x_c = points_w @ R.T + t
    z_c = x_c[:, 2]
    front = z_c > 1e-6
    x_c = x_c[front]
    z_c = z_c[front]
    n_keep = normals_w[front]

    u = fx * x_c[:, 0] / z_c + cx
    v = fy * x_c[:, 1] / z_c + cy
    u_i = np.floor(u).astype(np.int64)
    v_i = np.floor(v).astype(np.int64)
    in_frame = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    u_i = u_i[in_frame]
    v_i = v_i[in_frame]
    z_c = z_c[in_frame]
    n_keep = n_keep[in_frame]

    if u_i.size == 0:
        return (np.zeros((H, W), dtype=np.float32),
                np.zeros((H, W, 3), dtype=np.float32))

    # Z-buffer: for each pixel keep the smallest z. Sort by -z_c (descending)
    # so that *last* write at a given pixel is the nearest, then overwrite.
    order = np.argsort(-z_c, kind="stable")
    u_i = u_i[order]
    v_i = v_i[order]
    z_c = z_c[order]
    n_keep = n_keep[order]

    flat_idx = v_i * W + u_i
    depth_flat = np.zeros(H * W, dtype=np.float32)
    normal_flat = np.zeros((H * W, 3), dtype=np.float32)
    depth_flat[flat_idx] = z_c.astype(np.float32)
    normal_flat[flat_idx] = n_keep.astype(np.float32)

    return depth_flat.reshape(H, W), normal_flat.reshape(H, W, 3)


# ---------------------------------------------------------------------------
# Config + source-path resolution
# ---------------------------------------------------------------------------

def _load_cfg(cfg_path: Path):
    with open(cfg_path, "r") as f:
        full = yaml.safe_load(f)
    dataset_dir = full.get("dataset_dir")
    ct = full["train_and_eval"]["train"]["custom_train_2dgs"]
    return full, dataset_dir, ct


def _resolve_source_path(dataset_dir, ct_cfg):
    """Return a Path to the COLMAP root (containing sparse/0 and images/).

    Priority: custom_train_2dgs.xgrids_path > .source_path > global dataset_dir.
    """
    xgrids_path = ct_cfg.get("xgrids_path")
    if xgrids_path:
        sp_str, _ = prepare_xgrids_dataset(Path(xgrids_path).expanduser().resolve())
        return Path(sp_str)
    ct_source_path = ct_cfg.get("source_path")
    if ct_source_path:
        sp = Path(ct_source_path).expanduser().resolve()
        if not (sp / "sparse" / "0" / "cameras.txt").exists():
            print(f"ERROR: source_path {sp} is not a prepared COLMAP dataset "
                  f"(no sparse/0/cameras.txt).")
            sys.exit(1)
        return sp
    if not dataset_dir:
        print("ERROR: set custom_train_2dgs.xgrids_path OR "
              "custom_train_2dgs.source_path OR the global dataset_dir.")
        sys.exit(1)
    sp_str, _ = prepare_xgrids_dataset(Path(dataset_dir).expanduser().resolve())
    return Path(sp_str)


# ---------------------------------------------------------------------------
# Depth-map file naming
# ---------------------------------------------------------------------------

def _image_stem(name: str) -> str:
    """'camera_0/1776246628.862644_0.jpg' -> '1776246628.862644_0'.

    Matches the `image_name` attribute the 2DGS Camera class stores
    (see scene/dataset_readers.py:100).
    """
    return Path(name).stem


def _depth_map_key(cam_id: int, colmap_name: str) -> str:
    """Naming scheme for the per-frame .npz files.

    In this dataset the filename stem alone is NOT unique (441 stem
    collisions across camera_0/ vs camera_1/ subdirs), but (stem, cam_id)
    IS unique. The 2DGS Camera class exposes both as `image_name` and
    `colmap_id`, so we use both to key depth maps.
    """
    return f"cam{cam_id}_{_image_stem(colmap_name)}"


def _assert_unique_keys(images):
    """Bail out if two COLMAP entries produce the same depth-map key --
    would silently overwrite otherwise."""
    seen = {}
    for entry in images:
        cam_id = entry[8]
        name = entry[-1]
        key = _depth_map_key(cam_id, name)
        if key in seen:
            print(f"ERROR: depth-map key collision '{key}' between:")
            print(f"  {seen[key]}")
            print(f"  {name}")
            sys.exit(1)
        seen[key] = name
