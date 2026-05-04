"""Image-space LiDAR supervision losses for custom_train.py.

Both losses operate in world frame (matches 2DGS's `rend_normal` convention
and the normals stored by scripts/lidar_to_depth_maps.py), are masked to
pixels where (a) the alpha/border mask is valid AND (b) the LiDAR projection
landed on that pixel.
"""

import torch
import torch.nn.functional as F


# =============================================================================
# DEPTH LOSS -- Image-space LiDAR depth supervision
# =============================================================================

def compute_depth_loss(rendered_depth: torch.Tensor,
                       lidar_depth: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """L1 between median rendered depth and LiDAR projected depth.

    Args:
        rendered_depth: (1, H, W) or (H, W) -- median rendered depth per pixel.
        lidar_depth:    (H, W) -- LiDAR projected depth (0 where no hit).
        mask:           (H, W) bool -- alpha/border mask (True = valid).

    Returns:
        Scalar tensor. If no pixel is valid, returns 0 with gradient link.
    """
    rd = rendered_depth.squeeze()                      # (H, W)
    valid = mask & (lidar_depth > 0)
    if valid.sum() == 0:
        return rd.sum() * 0.0
    return (rd - lidar_depth).abs()[valid].mean()


# =============================================================================
# NORMAL LOSS -- Image-space LiDAR normal supervision (world frame)
# =============================================================================

def compute_normal_loss(rendered_normal: torch.Tensor,
                        lidar_normal: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """Cosine alignment loss between rendered surfel normals and LiDAR normals.

    Args:
        rendered_normal: (3, H, W) -- world-frame rendered normals (rend_normal
                         from the 2DGS renderer).
        lidar_normal:    (3, H, W) -- world-frame LiDAR normals (0 where no hit).
        mask:            (H, W) bool -- alpha/border mask.

    Returns:
        Scalar tensor. If no pixel is valid, returns 0 with gradient link.
    """
    rn = F.normalize(rendered_normal, dim=0, eps=1e-8)
    ln = F.normalize(lidar_normal,    dim=0, eps=1e-8)
    cos = (rn * ln).sum(dim=0)                         # (H, W)
    has_lidar = lidar_normal.abs().sum(dim=0) > 1e-6
    valid = mask & has_lidar
    if valid.sum() == 0:
        return rendered_normal.sum() * 0.0
    return (1.0 - cos)[valid].mean()
