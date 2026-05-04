#!/usr/bin/env python3
"""One-command 2DGS reconstruction pipeline (LiDAR preprocess + train + mesh).

    python run_pipeline.py \\
        --config configs/v0_1_release.yaml \\
        --dataset_root /path/to/mm1_data

Stages:
  1. scripts/lidar_to_depth_maps_cuda.py -- LAS -> per-frame depth maps
  2. custom_train.py                     -- 2DGS training
  3. render.py                           -- TSDF fusion -> fuse.ply + fuse_post.ply

Extra flags after --dataset_root are forwarded to custom_train.py only.
TSDF / mesh parameters live under mesh.render_2dgs in the YAML.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset_root", required=True)
    args, extra = ap.parse_known_args()

    stages = [
        [sys.executable, str(HERE / "scripts" / "lidar_to_depth_maps_cuda.py"),
         "--config", args.config, "--dataset_root", args.dataset_root],
        [sys.executable, str(HERE / "custom_train.py"),
         "--config", args.config, "--dataset_root", args.dataset_root, *extra],
        [sys.executable, str(HERE / "render.py"),
         "--config", args.config, "--dataset_root", args.dataset_root,
         "--skip_train", "--skip_test"],
    ]
    for stage in stages:
        print(f"\n>>> {' '.join(stage)}", flush=True)
        rc = subprocess.run(stage).returncode
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
