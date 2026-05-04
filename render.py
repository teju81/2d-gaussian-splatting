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

import torch
from scene import Scene
import os
import sys
from pathlib import Path
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
from utils.dataset_utils import prepare_xgrids_dataset

import open3d as o3d
import yaml


def _resolve_model_path_from_yaml(yaml_path: Path, cli_dataset_root=None):
    """Resolve (source_path, model_path) from the YAML config and optional CLI override.

    Priority: CLI --dataset_root > YAML custom_train_2dgs.dataset_root.
    Both outputs are derived via prepare_xgrids_dataset(dataset_root).
    """
    with open(yaml_path, "r") as f:
        root = yaml.safe_load(f)
    ct = root.get("train_and_eval", {}).get("train", {}).get("custom_train_2dgs", {}) or {}

    dataset_root = cli_dataset_root or ct.get("dataset_root")
    if not dataset_root:
        print("ERROR: dataset_root not set -- pass --dataset_root or set "
              "custom_train_2dgs.dataset_root in the YAML (or pass -m explicitly).")
        sys.exit(1)

    sp, mp = prepare_xgrids_dataset(Path(dataset_root).expanduser().resolve())
    return sp, str(Path(mp).expanduser().resolve())


def _load_render_defaults_from_yaml(yaml_path: Path):
    """Read mesh.render_2dgs from the YAML; return a dict of flag defaults.
    Missing block is treated as empty (caller keeps hardcoded argparse defaults)."""
    with open(yaml_path, "r") as f:
        root = yaml.safe_load(f)
    return (root.get("mesh", {}) or {}).get("render_2dgs", {}) or {}

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--nvblox", action="store_true", help='Mesh: use nvblox GPU TSDF instead of Open3D')
    parser.add_argument("--max_integration_distance", default=5.0, type=float, help='Mesh: max depth for nvblox integration')
    parser.add_argument("--skimage_mc", action="store_true", help='Mesh: use scikit-image marching cubes on dense TSDF grid instead of nvblox built-in extraction (requires --nvblox)')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')

    # Config-driven mode: defaults come from mesh.render_2dgs in the YAML, and
    # model_path is auto-resolved from train_and_eval.train.custom_train_2dgs.
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to v0_1_release.yaml (or similar). When set, "
                        "defaults come from mesh.render_2dgs; CLI still overrides.")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Root folder containing the las file and perspective/ folder. "
                             "Overrides custom_train_2dgs.dataset_root in YAML.")
    # CLI-only: skip Stage A (reconstruction + TSDF -> fuse.ply). Requires the
    # raw mesh to already exist. Not read from the YAML by design -- always
    # opt-in on the command line so the intent is explicit.
    parser.add_argument("--skip_fusion", action="store_true",
                        help="Skip mesh fusion (Stage A) and load existing fuse.ply "
                             "from disk. Errors out if the raw mesh is missing. "
                             "CLI only -- not read from YAML.")

    # Peek at --config before full parse so YAML can override argparse defaults
    # via parser.set_defaults(). Explicit CLI values still win at parse time.
    pre_args, _ = parser.parse_known_args()
    if pre_args.config is not None:
        render_defaults = _load_render_defaults_from_yaml(pre_args.config)
        # Only keys parser already knows about -- silently skip unknowns.
        known = {a.dest for a in parser._actions}
        # CLI-only flags: never let YAML override, even if the user adds them there.
        CLI_ONLY = {"skip_fusion"}
        parser.set_defaults(**{
            k: v for k, v in render_defaults.items()
            if k in known and k not in CLI_ONLY
        })

        # Resolve model_path from the training block so the user doesn't have to.
        source_path, model_path = _resolve_model_path_from_yaml(
            pre_args.config,
            cli_dataset_root=pre_args.dataset_root,
        )
        parser.set_defaults(source_path=source_path, model_path=model_path)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    
    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.export_image(train_dir)
        
    
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir)
    
    
    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames)

    if not args.skip_mesh:
        import gc
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0

        # Decide output filename up front so we can check for a resumable run.
        if args.nvblox:
            name = 'fuse_nvblox_skimage.ply' if args.skimage_mc else 'fuse_nvblox.ply'
        elif args.unbounded:
            name = 'fuse_unbounded.ply'
        else:
            name = 'fuse.ply'
        raw_mesh_path = os.path.join(train_dir, name)
        post_mesh_path = os.path.join(train_dir, name.replace('.ply', '_post.ply'))

        # ---- Stage A: render + TSDF (SKIPPED iff --skip_fusion) ------------
        if args.skip_fusion:
            if not os.path.exists(raw_mesh_path):
                print(f"ERROR: --skip_fusion was set but {raw_mesh_path} does not exist.")
                print(f"       Run render.py without --skip_fusion first, or "
                      f"confirm you are pointing at the correct model path.")
                sys.exit(1)
            print(f"[skip_fusion] loading existing {raw_mesh_path} -- "
                  f"Stage A (reconstruction + TSDF) skipped.")
            mesh = o3d.io.read_triangle_mesh(raw_mesh_path)
        else:
            gaussExtractor.reconstruction(scene.getTrainCameras())
            # extract the mesh and save
            if args.nvblox:
                voxel_size = 0.02 if args.voxel_size < 0 else args.voxel_size
                mesh = gaussExtractor.extract_mesh_nvblox(
                    voxel_size=voxel_size,
                    max_integration_distance=args.max_integration_distance,
                    use_skimage_mc=args.skimage_mc,
                )
            elif args.unbounded:
                mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
            else:
                depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
                voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
                sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
                mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

            # Merge close vertices for nvblox meshes to connect block boundaries
            if args.nvblox:
                mesh.merge_close_vertices(voxel_size * 0.5)

            o3d.io.write_triangle_mesh(raw_mesh_path, mesh)
            print("mesh saved at {}".format(raw_mesh_path))

        # ---- Free the big intermediate buffers before Stage B --------------
        # reconstruction() holds ~14 GB of per-frame rgb+depth tensors on CPU
        # that post_process_mesh no longer needs. Drop them + ask the GC to
        # run before cluster_connected_triangles (itself memory-hungry).
        gaussExtractor.clean()
        torch.cuda.empty_cache()
        gc.collect()

        # ---- Stage B: post-processing (wrapped so a crash leaves fuse.ply intact)
        try:
            mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
            o3d.io.write_triangle_mesh(post_mesh_path, mesh_post)
            print("mesh post processed saved at {}".format(post_mesh_path))
        except Exception as e:
            print(f"[WARN] post_process_mesh failed: {type(e).__name__}: {e}")
            print(f"[WARN] the raw mesh is preserved at {raw_mesh_path}.")
            print(f"[WARN] to recover, run:")
            print(f"    python3 scripts/post_process_mesh.py \"{raw_mesh_path}\"")
            raise