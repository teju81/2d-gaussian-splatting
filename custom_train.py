#
# custom_train.py -- LiDAR-constrained 2DGS training with Direct Geometric
# Supervision (DGS) and Soft Phase B. Reproduces the v8k configuration from
# the project reference document.
#
# This script is a siblings-of-train.py copy modified for:
#   Component 2 -- image-space LiDAR depth + normal losses (utils/custom_loss.py)
#   Component 3 -- two-phase training schedule (Phase A / Phase B)
#   Component 4 -- Direct Geometric Supervision (utils/direct_geometric_supervision.py)
#   Component 5 -- Soft Phase B (xyz_lr=1e-6, DGS tether at lambda=0.05)
#
# All parameters are read from train_and_eval.train.custom_train_2dgs in the
# YAML config supplied via --config. The original train.py is untouched.
#
# CLI:
#   python custom_train.py --config <path/to/v0_1_release.yaml> \
#                          --dataset_root <path/to/dataset_root>
#
# Depth maps must already exist at <source_path>/depth_maps_lidar/. To run the
# LiDAR preprocessing step too, use run_training_with_lidar_preprocess.py.
#

import os
import sys
import argparse
from argparse import Namespace
from pathlib import Path
from random import randint

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr, render_net_image
from utils.general_utils import safe_state, get_expon_lr_func, build_rotation
from utils.dataset_utils import find_las_file, prepare_xgrids_dataset, resolve_dataset_root
from utils.custom_loss import compute_depth_loss, compute_normal_loss
from utils.direct_geometric_supervision import LiDARSurfaceField

from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# =============================================================================
# Config loading helpers
# =============================================================================

def _load_yaml(config_path: Path):
    with open(config_path, "r") as f:
        root = yaml.safe_load(f)
    try:
        ct = root["train_and_eval"]["train"]["custom_train_2dgs"]
    except KeyError as e:
        print(f"ERROR: config missing train_and_eval.train.custom_train_2dgs: {e}")
        sys.exit(1)
    return root, ct


def _build_base_args(ct_cfg, dataset_root, is_quiet):
    """Construct an argparse.Namespace with ModelParams/OptParams/PipelineParams
    defaults, then override the fields that custom_train_2dgs controls.

    Returns (lp, op, pp, args, source_path_resolved, model_path_resolved).
    """
    # Parse an empty argv to get registered defaults.
    p = argparse.ArgumentParser()
    lp = ModelParams(p)
    op = OptimizationParams(p)
    pp = PipelineParams(p)
    p.add_argument("--ip", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=6009)
    args = p.parse_args([])

    source_path_str, model_path_str = prepare_xgrids_dataset(
        Path(dataset_root).expanduser().resolve()
    )
    source_path = str(Path(source_path_str).resolve())
    model_path = str(Path(model_path_str).resolve())

    # ModelParams fields
    args.source_path = source_path
    args.model_path = model_path
    args.images = "images"
    args.white_background = bool(ct_cfg.get("white_background", True))
    args.eval = False

    # OptimizationParams: override iteration count + densification window only.
    args.iterations = int(ct_cfg.get("iterations", 30000))
    args.densify_until_iter = int(ct_cfg.get("densify_until_iter", 10000))
    # Do NOT change other optimisation params -- they are the vetted 2DGS
    # defaults (feature_lr, opacity_lr, etc.). The user can edit v0_1_release.yaml
    # to expose more if needed later.

    return lp, op, pp, args, source_path, model_path


# =============================================================================
# Depth-map loading
# =============================================================================

def _cam_depth_key(cam) -> str:
    """Build the depth-map key for a 2DGS Camera.

    Prefer `Path(cam.image_path).stem` (preserves the full `1776246628.862644_0`
    timestamp), because 2DGS's default `cam.image_name` truncates at the first
    dot (dataset_readers.py:100) and is NOT unique within a given camera_id.
    """
    full_stem = None
    if getattr(cam, "image_path", None):
        full_stem = Path(cam.image_path).stem
    if not full_stem:
        full_stem = cam.image_name
    return f"cam{cam.colmap_id}_{full_stem}"


def _attach_lidar_paths_to_cameras(cameras, depth_maps_dir: Path):
    """Attach the .npz path (string) to each camera, NOT the tensors.

    We lazy-load per iteration to avoid ~32 GB of VRAM/CPU-RAM for ~2k frames
    at 1024x1024 (each frame is ~16 MB of depth+normal tensor).

    Depth-map files are named `cam{colmap_id}_{image_path_stem}.npz` -- see
    scripts/lidar_to_depth_maps.py.
    """
    found = 0
    missing = []
    for cam in cameras:
        key = _cam_depth_key(cam)
        npz_path = depth_maps_dir / f"{key}.npz"
        if npz_path.is_file():
            cam.lidar_npz_path = str(npz_path)
            found += 1
        else:
            cam.lidar_npz_path = None
            missing.append(key)
    print(f"[custom_train] LiDAR maps: {found}/{len(cameras)} cameras matched "
          f"in {depth_maps_dir}")
    if missing:
        print(f"[custom_train]   first few missing: {missing[:5]}")
    return found


def _load_lidar_npz_gpu(npz_path: str, expected_h: int, expected_w: int):
    """Read one .npz and return (lidar_depth_HW, lidar_normal_3HW) on CUDA.

    Returns (None, None) if the file is missing or shape-mismatched.
    """
    if npz_path is None:
        return None, None
    with np.load(npz_path) as d:
        depth_np = d["depth"].astype(np.float32)               # (H, W)
        normal_np = d["normal"].astype(np.float32)              # (H, W, 3)
    if depth_np.shape != (expected_h, expected_w):
        return None, None
    lidar_depth = torch.from_numpy(depth_np).cuda(non_blocking=True)                      # (H, W)
    lidar_normal = torch.from_numpy(normal_np).permute(2, 0, 1).cuda(non_blocking=True)   # (3, H, W)
    return lidar_depth, lidar_normal


def _camera_centroid(cameras) -> np.ndarray:
    """Mean camera centre in world frame (used for consistent normal orientation
    between preprocessing and DGS)."""
    centres = []
    for cam in cameras:
        # Camera stores R, T as world-to-camera. World position = -R^T @ T.
        R = cam.R if isinstance(cam.R, np.ndarray) else np.asarray(cam.R)
        T = cam.T if isinstance(cam.T, np.ndarray) else np.asarray(cam.T)
        centres.append(-R.T @ T)
    return np.stack(centres, axis=0).mean(axis=0)


# =============================================================================
# Phase B LR schedule
# =============================================================================

def _apply_phase_b_lrs(optimizer, soft: bool, xyz_lr: float):
    """Override per-group learning rates for Phase B.

    Hard: xyz=0, rotation=0, scaling=0, opacity=0; SH (f_dc, f_rest) untouched.
    Soft: xyz=xyz_lr (e.g. 1e-6), rotation=0, scaling=0, opacity=0; SH untouched.
    """
    for pg in optimizer.param_groups:
        name = pg.get("name", "")
        if name == "xyz":
            pg["lr"] = xyz_lr if soft else 0.0
        elif name in ("rotation", "scaling", "opacity"):
            pg["lr"] = 0.0
        # f_dc, f_rest: unchanged


# =============================================================================
# Output folder / TB setup (mirrors train.py.prepare_output_and_logger)
# =============================================================================

def _prepare_output_and_logger(args):
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


# =============================================================================
# Training
# =============================================================================

def training(dataset, opt, pipe, ct_cfg, dataset_root,
             testing_iterations, saving_iterations, checkpoint_iterations,
             checkpoint):
    tb_writer = _prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    first_iter = 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # --- Phase / DGS config --------------------------------------------------
    iterations = opt.iterations
    phase_b_start = int(ct_cfg["phase_b_start"])
    soft_phase_b = bool(ct_cfg["soft_phase_b"])
    phase_b_xyz_lr = float(ct_cfg["phase_b_xyz_lr"])
    phase_b_dgs_lambda = float(ct_cfg["phase_b_dgs_lambda"])
    phase_b_keep_image_losses = bool(ct_cfg.get("phase_b_keep_image_losses", False))

    lambda_depth = float(ct_cfg["lambda_depth"])
    lambda_lidar_normal = float(ct_cfg["lambda_lidar_normal"])
    lambda_dgs = float(ct_cfg["lambda_dgs"])
    lambda_dgs_normal = float(ct_cfg["lambda_dgs_normal"])

    dgs_start_iter = int(ct_cfg["dgs_start_iter"])
    dgs_interval = int(ct_cfg["dgs_interval"])
    dgs_k = int(ct_cfg["dgs_k"])
    dgs_radius = float(ct_cfg["dgs_radius"])
    dgs_voxel_size = float(ct_cfg["dgs_voxel_size"])
    dgs_normal_radius = float(ct_cfg["dgs_normal_radius"])

    assert dgs_start_iter == opt.densify_until_iter, (
        f"dgs_start_iter ({dgs_start_iter}) must equal densify_until_iter "
        f"({opt.densify_until_iter}). Enabling DGS during densification "
        f"causes scene explosion (see project_dgs.pdf Section 7.5).")

    # --- Attach LiDAR maps to cameras ---------------------------------------
    dm_path = Path(dataset.source_path) / "depth_maps_lidar"
    train_cams = scene.getTrainCameras().copy()
    n_with_lidar = _attach_lidar_paths_to_cameras(train_cams, dm_path)
    use_image_losses = (lambda_depth > 0 or lambda_lidar_normal > 0) and n_with_lidar > 0
    if (lambda_depth > 0 or lambda_lidar_normal > 0) and n_with_lidar == 0:
        print("[custom_train] WARNING: no LiDAR depth maps found; image-space "
              "depth and normal losses will be disabled.")

    # --- DGS surface field ---------------------------------------------------
    surface_field = None
    if lambda_dgs > 0:
        las_path = find_las_file(Path(dataset_root).expanduser().resolve())
        centroid = _camera_centroid(scene.getTrainCameras())
        surface_field = LiDARSurfaceField(
            las_path=str(las_path),
            voxel_size=dgs_voxel_size,
            normal_radius=dgs_normal_radius,
            scene_centroid=centroid,
            device="cuda",
        )
    last_cached_count = -1

    # --- EMA log state -------------------------------------------------------
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_depth_for_log = 0.0
    ema_lnormal_for_log = 0.0
    ema_dgs_for_log = 0.0
    ema_dgsn_for_log = 0.0

    viewpoint_stack = None

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    progress_bar = tqdm(range(first_iter, iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, iterations + 1):
        iter_start.record()

        # Update xyz schedule, then override if we're in Phase B.
        gaussians.update_learning_rate(iteration)
        in_phase_b = iteration >= phase_b_start
        if in_phase_b:
            _apply_phase_b_lrs(gaussians.optimizer, soft_phase_b, phase_b_xyz_lr)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random training Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        # Black-border mask derived from GT (same logic as train.py:113).
        mask_2d = (gt_image.max(dim=0).values > 5.0 / 255.0)        # (H, W) bool
        mask_3ch = mask_2d.unsqueeze(0).float()                     # (1, H, W)
        masked_image = image * mask_3ch
        masked_gt = gt_image * mask_3ch

        Ll1 = (masked_image - masked_gt).abs().sum() / (mask_2d.float().sum() * 3 + 1e-8)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(masked_image, masked_gt))

        # --- Existing 2DGS regularisers (from train.py) ---------------------
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_reg_loss = lambda_normal * normal_error.mean()
        dist_loss = lambda_dist * rend_dist.mean()

        total_loss = loss + dist_loss + normal_reg_loss

        # --- Image-space LiDAR losses (Component 2) -------------------------
        L_depth_val = 0.0
        L_lidar_normal_val = 0.0
        image_losses_on = use_image_losses and (
            (not in_phase_b) or phase_b_keep_image_losses
        )
        lidar_depth_t = None
        lidar_normal_t = None
        if image_losses_on and getattr(viewpoint_cam, "lidar_npz_path", None):
            lidar_depth_t, lidar_normal_t = _load_lidar_npz_gpu(
                viewpoint_cam.lidar_npz_path,
                viewpoint_cam.image_height,
                viewpoint_cam.image_width,
            )
        if image_losses_on and lidar_depth_t is not None:
            if lambda_depth > 0:
                L_depth = compute_depth_loss(
                    render_pkg["rend_depth_median"],
                    lidar_depth_t,
                    mask_2d,
                )
                total_loss = total_loss + lambda_depth * L_depth
                L_depth_val = float(L_depth.detach().item())
            if lambda_lidar_normal > 0:
                L_lidar_normal = compute_normal_loss(
                    rend_normal,
                    lidar_normal_t,
                    mask_2d,
                )
                total_loss = total_loss + lambda_lidar_normal * L_lidar_normal
                L_lidar_normal_val = float(L_lidar_normal.detach().item())

        # --- DGS (Components 4 + 5) -----------------------------------------
        L_dgs_val = 0.0
        L_dgsn_val = 0.0
        dgs_active = (
            surface_field is not None
            and iteration >= dgs_start_iter
            and ((not in_phase_b) or soft_phase_b)
        )
        if dgs_active:
            N = gaussians.get_xyz.shape[0]
            count_changed = (N != last_cached_count)
            due_interval = ((iteration - dgs_start_iter) % dgs_interval == 0)
            if count_changed or due_interval:
                surface_field.update_cache(
                    gaussians.get_xyz,
                    k=dgs_k,
                    radius=dgs_radius,
                )
                last_cached_count = N
            R_all = build_rotation(gaussians.get_rotation)            # (N, 3, 3)
            surfel_normals_world = R_all[:, :, 2]                      # (N, 3)
            L_dgs, L_dgs_n = surface_field.compute_loss(
                gaussians.get_xyz, surfel_normals_world,
            )
            lam_dgs = phase_b_dgs_lambda if in_phase_b else lambda_dgs
            total_loss = total_loss + lam_dgs * L_dgs + lambda_dgs_normal * L_dgs_n
            L_dgs_val = float(L_dgs.detach().item())
            L_dgsn_val = float(L_dgs_n.detach().item())

        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_reg_loss.item() + 0.6 * ema_normal_for_log
            ema_depth_for_log = 0.4 * L_depth_val + 0.6 * ema_depth_for_log
            ema_lnormal_for_log = 0.4 * L_lidar_normal_val + 0.6 * ema_lnormal_for_log
            ema_dgs_for_log = 0.4 * L_dgs_val + 0.6 * ema_dgs_for_log
            ema_dgsn_for_log = 0.4 * L_dgsn_val + 0.6 * ema_dgsn_for_log

            if iteration % 10 == 0:
                phase_tag = "B" if in_phase_b else "A"
                progress_bar.set_postfix({
                    "L": f"{ema_loss_for_log:.4f}",
                    "dpt": f"{ema_depth_for_log:.4f}",
                    "lnrm": f"{ema_lnormal_for_log:.4f}",
                    "dgs": f"{ema_dgs_for_log:.4f}",
                    "N": f"{len(gaussians.get_xyz)}",
                    "ph": phase_tag,
                })
                progress_bar.update(10)
            if iteration == iterations:
                progress_bar.close()

            if tb_writer is not None:
                tb_writer.add_scalar("train/reg_l1",           Ll1.item(), iteration)
                tb_writer.add_scalar("train/total_loss",       total_loss.item(), iteration)
                tb_writer.add_scalar("train/dist",             ema_dist_for_log, iteration)
                tb_writer.add_scalar("train/normal_reg",       ema_normal_for_log, iteration)
                tb_writer.add_scalar("train/L_depth",          ema_depth_for_log, iteration)
                tb_writer.add_scalar("train/L_lidar_normal",   ema_lnormal_for_log, iteration)
                tb_writer.add_scalar("train/L_dgs",            ema_dgs_for_log, iteration)
                tb_writer.add_scalar("train/L_dgs_normal",     ema_dgsn_for_log, iteration)
                tb_writer.add_scalar("train/total_points",     gaussians.get_xyz.shape[0], iteration)
                tb_writer.add_scalar("train/iter_time_ms",     iter_start.elapsed_time(iter_end), iteration)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # --- Densification (Phase A only, iteration < densify_until_iter) ---
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opt.opacity_cull,
                        scene.cameras_extent, size_threshold,
                    )

                if (iteration % opt.opacity_reset_interval == 0
                        or (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            # --- Optimizer step ---------------------------------------------
            if iteration < iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LiDAR-constrained 2DGS training")
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to v0_1_release.yaml (or similar).")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Root folder containing the las file and perspective/ folder. "
                             "Overrides custom_train_2dgs.dataset_root in YAML.")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[7_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[7_000, 20_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    cli_args = parser.parse_args()

    _, ct_cfg = _load_yaml(cli_args.config)
    ct_cfg = dict(ct_cfg)
    dataset_root = resolve_dataset_root(cli_args.config, cli_args.dataset_root)

    lp, op, pp, args, source_path, model_path = _build_base_args(
        ct_cfg, dataset_root, cli_args.quiet,
    )
    args.ip = cli_args.ip
    args.port = cli_args.port

    # Make sure save/test iterations include final iteration.
    save_iterations = list(cli_args.save_iterations)
    if args.iterations not in save_iterations:
        save_iterations.append(args.iterations)

    # Checkpoint iterations: CLI takes precedence if explicitly set; otherwise
    # fall back to YAML custom_train_2dgs.checkpoint_iterations; otherwise [].
    if cli_args.checkpoint_iterations:
        checkpoint_iterations = list(cli_args.checkpoint_iterations)
    else:
        checkpoint_iterations = list(ct_cfg.get("checkpoint_iterations") or [])

    print(f"[custom_train] Optimizing {model_path}")
    print(f"[custom_train] Source:   {source_path}")
    print(f"[custom_train] Iters:    {args.iterations}  "
          f"densify_until={args.densify_until_iter}  "
          f"phase_b={ct_cfg['phase_b_start']}  "
          f"soft={ct_cfg['soft_phase_b']}")
    print(f"[custom_train] Lambdas:  depth={ct_cfg['lambda_depth']}  "
          f"lidar_normal={ct_cfg['lambda_lidar_normal']}  "
          f"dgs={ct_cfg['lambda_dgs']}  dgs_n={ct_cfg['lambda_dgs_normal']}  "
          f"phase_b_dgs={ct_cfg['phase_b_dgs_lambda']}")
    print(f"[custom_train] Saves:    ply={save_iterations}  "
          f"ckpt={checkpoint_iterations or 'disabled'}  "
          f"resume_from={cli_args.start_checkpoint or 'none'}")

    safe_state(cli_args.quiet)

    network_gui.init(cli_args.ip, cli_args.port)
    torch.autograd.set_detect_anomaly(cli_args.detect_anomaly)

    training(
        lp.extract(args), op.extract(args), pp.extract(args), ct_cfg, dataset_root,
        cli_args.test_iterations, save_iterations,
        checkpoint_iterations, cli_args.start_checkpoint,
    )
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
