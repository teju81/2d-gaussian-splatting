"""Dataset preparation helpers shared by train.py and custom_train.py."""

import sys
from pathlib import Path

import yaml


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


def find_las_file(dataset_root):
    """Recursively find exactly one *.las file inside dataset_root."""
    dataset_root = Path(dataset_root)
    candidates = sorted(dataset_root.rglob("*.las"))
    if not candidates:
        print(f"Error: no .las file found under {dataset_root}")
        sys.exit(1)
    if len(candidates) > 1:
        print(f"Error: multiple .las files found under {dataset_root}:")
        for c in candidates:
            print(f"  {c}")
        print("Expected exactly one.")
        sys.exit(1)
    return candidates[0]


def resolve_dataset_root(config_path, cli_value):
    """CLI value > YAML custom_train_2dgs.dataset_root. Exit if neither is set."""
    if cli_value:
        return cli_value
    with open(config_path, "r") as f:
        root = yaml.safe_load(f)
    ct = root.get("train_and_eval", {}).get("train", {}).get("custom_train_2dgs", {}) or {}
    dr = ct.get("dataset_root")
    if not dr:
        print("ERROR: dataset_root must be set via --dataset_root or in YAML "
              "custom_train_2dgs.dataset_root.")
        sys.exit(1)
    return dr


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

    def _ensure_symlink(dst, src):
        # Stale symlinks from a previous mount/path get replaced; real files
        # are left alone to avoid clobbering user data.
        if dst.is_symlink():
            if dst.resolve() == src.resolve():
                return
            dst.unlink()
        elif dst.exists():
            return
        dst.symlink_to(src)

    # Symlink images preserving subdirectory structure (camera_0/, camera_1/)
    for img_path in image_files:
        rel_path = img_path.relative_to(images_src)
        link_path = colmap_images_dir / rel_path
        link_path.parent.mkdir(parents=True, exist_ok=True)
        _ensure_symlink(link_path, img_path)

    # Symlink COLMAP files into sparse/0/
    for f in required_files:
        _ensure_symlink(sparse_dir / f, perspective_path / f)

    print(f"Created COLMAP dataset: {colmap_dir}")
    print(f"Output directory: {output_dir}")

    return str(colmap_dir), str(output_dir)
