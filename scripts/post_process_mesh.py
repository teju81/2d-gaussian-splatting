"""Run the post-processing stage of render.py on an existing raw mesh.

Purpose: if the initial TSDF extraction succeeded (fuse.ply on disk) but the
post-processing crashed, you don't want to re-render and re-fuse 882 views to
recover. This script loads the raw .ply and runs only the cluster-pruning /
cleanup stage to produce the _post.ply output.

CLI:
    python scripts/post_process_mesh.py <path_to_raw_mesh.ply> [--num_cluster 50] [--out OUT_PLY]

Examples:
    # Default: writes fuse_post.ply next to fuse.ply, matching render.py's convention.
    python scripts/post_process_mesh.py <model>/train/ours_30000/fuse.ply

    # Explicit output path + fewer clusters for a more aggressive cleanup.
    python scripts/post_process_mesh.py <model>/train/ours_30000/fuse.ply \
        --num_cluster 25 --out /tmp/fuse_clean.ply
"""

import argparse
import sys
from pathlib import Path

# Make utils.mesh_utils importable when run from scripts/
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent  # 2dgs_inria/
sys.path.insert(0, str(_REPO_ROOT))

import open3d as o3d  # noqa: E402
from utils.mesh_utils import post_process_mesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", type=Path, help="Raw mesh to post-process (e.g. fuse.ply).")
    ap.add_argument("--num_cluster", type=int, default=50,
                    help="Keep the N largest connected components (render.py default: 50).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path. Default: <input_stem>_post.ply next to the input.")
    args = ap.parse_args()

    if not args.mesh.is_file():
        print(f"ERROR: mesh not found: {args.mesh}")
        sys.exit(1)

    out = args.out
    if out is None:
        # Match render.py's convention: fuse.ply -> fuse_post.ply
        out = args.mesh.with_name(args.mesh.stem + "_post.ply")

    print(f"[post_process] loading {args.mesh}")
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    print(f"[post_process]   vertices: {len(mesh.vertices):,}  triangles: {len(mesh.triangles):,}")

    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)

    print(f"[post_process] writing {out}")
    o3d.io.write_triangle_mesh(str(out), mesh_post)
    print("[post_process] done.")


if __name__ == "__main__":
    main()
