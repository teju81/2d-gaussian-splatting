#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from pathlib import Path
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, incremental=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    # Incremental mode: shift densification window and position LR schedule
    # relative to the checkpoint iteration so new Gaussians can be created
    # for scene changes.
    checkpoint_iter = 0
    if incremental and checkpoint and first_iter > 0:
        checkpoint_iter = first_iter
        opt.densify_from_iter = checkpoint_iter + 500
        opt.densify_until_iter = checkpoint_iter + int(
            (opt.iterations - checkpoint_iter) * 0.5
        )
        opt.opacity_reset_interval = 3000
        # Rebuild position LR schedule relative to checkpoint iteration
        gaussians.xyz_scheduler_args = get_expon_lr_func(
            lr_init=opt.position_lr_init * gaussians.spatial_lr_scale,
            lr_final=opt.position_lr_final * gaussians.spatial_lr_scale,
            lr_delay_mult=opt.position_lr_delay_mult,
            max_steps=opt.iterations - checkpoint_iter,
        )
        # Reset gradient accumulators so densification starts fresh
        gaussians.xyz_gradient_accum = torch.zeros(
            (gaussians.get_xyz.shape[0], 1), device="cuda"
        )
        gaussians.denom = torch.zeros(
            (gaussians.get_xyz.shape[0], 1), device="cuda"
        )
        gaussians.max_radii2D = torch.zeros(
            (gaussians.get_xyz.shape[0]), device="cuda"
        )
        print(f"[Incremental] Checkpoint at iter {checkpoint_iter}")
        print(f"[Incremental] Densification window: {opt.densify_from_iter} - {opt.densify_until_iter}")
        print(f"[Incremental] Position LR schedule reset for {opt.iterations - checkpoint_iter} steps")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        # In incremental mode, offset iteration for LR schedule so it
        # starts from lr_init at the checkpoint boundary.
        if incremental and checkpoint_iter > 0:
            gaussians.update_learning_rate(iteration - checkpoint_iter)
        else:
            gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()

        # Mask out black stitching boundary regions (pixels where all channels < threshold)
        mask = (gt_image.max(dim=0).values > 5.0 / 255.0).float()  # [H, W]
        mask_3ch = mask.unsqueeze(0)  # [1, H, W]
        masked_image = image * mask_3ch
        masked_gt = gt_image * mask_3ch

        Ll1 = (torch.abs(masked_image - masked_gt)).sum() / (mask.sum() * 3 + 1e-8)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(masked_image, masked_gt))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    # Mask out black stitching boundary regions for evaluation
                    eval_mask = (gt_image.max(dim=0).values > 5.0 / 255.0).float().unsqueeze(0)
                    l1_test += l1_loss(image * eval_mask, gt_image * eval_mask).mean().double()
                    psnr_test += psnr(image * eval_mask, gt_image * eval_mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

def find_perspective_folder(xgrids_path):
    """Recursively find the perspective folder in the xgrids dataset.

    Looks for a directory named 'perspective' containing cameras.txt,
    images.txt, and an images/ subdirectory.
    """
    xgrids_path = Path(xgrids_path)
    for candidate in sorted(xgrids_path.rglob("perspective")):
        if not candidate.is_dir():
            continue
        has_cameras = (candidate / "cameras.txt").exists()
        has_images_txt = (candidate / "images.txt").exists()
        has_images_dir = (candidate / "images").is_dir()
        if has_cameras and has_images_txt and has_images_dir:
            return candidate
    return None

def prepare_xgrids_dataset(xgrids_path):
    """Prepare a COLMAP-compatible dataset from an xgrids dataset.

    Creates a new folder <xgrids_parent>/<dataset_name>_2DGS/ with:
      - images/ : symlinks preserving camera_0/camera_1 subdirectory structure
      - sparse/0/ : symlinks to cameras.txt, images.txt, points3D.txt

    Returns (source_path, model_path) to use for training.
    """
    xgrids_path = Path(xgrids_path).resolve()
    if not xgrids_path.is_dir():
        print(f"Error: xgrids dataset path does not exist: {xgrids_path}")
        sys.exit(1)

    perspective_path = find_perspective_folder(xgrids_path)
    if perspective_path is None:
        print(f"Error: Could not find a valid 'perspective' folder in {xgrids_path}")
        print("Expected: a 'perspective/' directory containing cameras.txt, images.txt, and images/ subdirectory")
        sys.exit(1)

    print(f"Found perspective folder: {perspective_path}")

    # Verify required COLMAP files
    required_files = ["cameras.txt", "images.txt", "points3D.txt"]
    for f in required_files:
        fpath = perspective_path / f
        if not fpath.exists():
            print(f"Error: Required file '{f}' not found in {perspective_path}")
            sys.exit(1)
        if fpath.stat().st_size == 0:
            print(f"Error: Required file '{f}' is empty in {perspective_path}")
            sys.exit(1)

    # Verify images exist
    images_src = perspective_path / "images"
    image_files = list(images_src.rglob("*.jpg")) + list(images_src.rglob("*.png")) + list(images_src.rglob("*.jpeg"))
    if not image_files:
        print(f"Error: No images found in {images_src}")
        sys.exit(1)
    print(f"Found {len(image_files)} images in perspective folder")

    # Create output directories inside the dataset folder
    parent_dir = xgrids_path
    dataset_name = xgrids_path.name
    colmap_dir = parent_dir / f"{dataset_name}_2DGS"
    output_dir = parent_dir / f"{dataset_name}_2DGS_output"
    sparse_dir = colmap_dir / "sparse" / "0"
    colmap_images_dir = colmap_dir / "images"

    sparse_dir.mkdir(parents=True, exist_ok=True)
    colmap_images_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Symlink images preserving subdirectory structure (camera_0/, camera_1/)
    for img_path in image_files:
        rel_path = img_path.relative_to(images_src)
        link_path = colmap_images_dir / rel_path
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if not link_path.exists():
            link_path.symlink_to(img_path)

    # Symlink COLMAP files into sparse/0/
    for f in required_files:
        src = perspective_path / f
        dst = sparse_dir / f
        if not dst.exists():
            dst.symlink_to(src)

    print(f"Created COLMAP dataset: {colmap_dir}")
    print(f"Output directory: {output_dir}")

    return str(colmap_dir), str(output_dir)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("-x", "--xgrids", type=str, default=None,
                        help="Path to xgrids dataset root. Auto-creates COLMAP-compatible structure from perspective folder.")
    parser.add_argument("--incremental", action="store_true",
                        help="Incremental training mode: re-enable densification and reset "
                             "position LR schedule relative to checkpoint iteration.")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # If xgrids mode, prepare dataset and override source/model paths
    if args.xgrids:
        source_path, model_path = prepare_xgrids_dataset(args.xgrids)
        args.source_path = source_path
        args.model_path = model_path

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, incremental=args.incremental)

    # All done
    print("\nTraining complete.")
