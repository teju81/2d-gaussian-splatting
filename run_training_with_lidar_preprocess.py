#!/usr/bin/env python3
"""One-command 2DGS training (LiDAR preprocess + training).

    python run_training_with_lidar_preprocess.py \\
        --config configs/v0_1_release.yaml \\
        --dataset_root /path/to/mm1_data

Extra flags after --dataset_root are forwarded to custom_train.py
(e.g. --quiet, --test_iterations, --start_checkpoint).
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
    ]
    for stage in stages:
        print(f"\n>>> {' '.join(stage)}", flush=True)
        rc = subprocess.run(stage).returncode
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
