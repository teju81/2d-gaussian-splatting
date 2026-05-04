"""Direct Geometric Supervision (DGS) for 2D Gaussian Surfels.

World-space plane-distance loss: each surfel is pulled toward the local
plane fit of its k-nearest LiDAR neighbours, with no alpha-compositing
dilution. See Section 7 of project_dgs.pdf for the math.

Critical timing: DGS must only be active AFTER densification ends. Enabling
it earlier creates an oscillation loop (densify -> DGS yank -> photometric
push back) that explodes the scene.
"""

from typing import Tuple

import numpy as np
import torch

try:
    import laspy
except ImportError as e:
    raise ImportError("laspy is required for DGS. Install it in the container.") from e

try:
    import open3d as o3d
except ImportError as e:
    raise ImportError("open3d is required for DGS. Install it in the container.") from e

try:
    from scipy.spatial import cKDTree
except ImportError as e:
    raise ImportError("scipy is required for DGS.") from e


class LiDARSurfaceField:
    """KDTree + batched-PCA plane fitting against a voxel-downsampled LAS.

    Typical usage inside the training loop:
        field = LiDARSurfaceField(las_path, voxel_size=0.005)
        # every dgs_interval iterations (and on surfel-count change):
        field.update_cache(gaussians.get_xyz, k=8, radius=0.05)
        L_dgs, L_dgs_n = field.compute_loss(gaussians.get_xyz, surfel_normals)
    """

    def __init__(self,
                 las_path: str,
                 voxel_size: float = 0.005,
                 normal_radius: float = 0.05,
                 scene_centroid: np.ndarray = None,
                 device: str = "cuda"):
        self.device = device

        # Load LAS
        las = laspy.read(str(las_path))
        xyz_np = np.stack([np.asarray(las.x, dtype=np.float64),
                           np.asarray(las.y, dtype=np.float64),
                           np.asarray(las.z, dtype=np.float64)], axis=1)

        # Voxel downsample + estimate normals via Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_np)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamRadius(radius=normal_radius)
        )
        lidar_pts = np.asarray(pcd.points, dtype=np.float64)
        lidar_nrm = np.asarray(pcd.normals, dtype=np.float64)

        # Orient normals toward scene_centroid if supplied (consistent with
        # scripts/lidar_to_depth_maps.py). If not supplied, orient toward the
        # LAS centroid -- a reasonable default that keeps signs stable.
        if scene_centroid is None:
            scene_centroid = lidar_pts.mean(axis=0)
        to_centroid = scene_centroid[None, :] - lidar_pts
        flip = np.einsum("ij,ij->i", lidar_nrm, to_centroid) < 0.0
        lidar_nrm[flip] *= -1.0

        print(f"[DGS] LAS={las_path}")
        print(f"[DGS]   voxel={voxel_size}m -> {lidar_pts.shape[0]:,} points; "
              f"flipped {flip.sum():,} normals toward centroid {scene_centroid}")

        self.lidar_points_np = lidar_pts
        self.lidar_normals_np = lidar_nrm
        self.kdtree = cKDTree(lidar_pts)

        # GPU copies (allocated lazily on first cache update)
        self._lidar_normals_gpu = None

        # Cache state (populated by update_cache)
        self.valid_mask: torch.Tensor = None        # (N,) bool on GPU
        self.plane_normals: torch.Tensor = None     # (M, 3) on GPU
        self.plane_offsets: torch.Tensor = None     # (M,)   on GPU
        self.cached_count: int = -1

    # -----------------------------------------------------------------------
    # Cache update (expensive; call every dgs_interval iters + on N change)
    # -----------------------------------------------------------------------
    def update_cache(self,
                     surfel_xyz: torch.Tensor,
                     k: int,
                     radius: float) -> None:
        """Rebuild the plane cache for the current surfel positions.

        Args:
            surfel_xyz: (N, 3) tensor -- surfel centres. Gradient not needed.
            k: number of LiDAR neighbours to fit the plane on.
            radius: if the nearest LiDAR point is farther than this, the
                    surfel is marked invalid (no DGS gradient for it).
        """
        N = surfel_xyz.shape[0]
        xyz_np = surfel_xyz.detach().cpu().numpy()

        # Filter out NaN/Inf surfels (would propagate into KNN + eigh).
        finite_np = np.isfinite(xyz_np).all(axis=1)        # (N,) bool
        n_bad = int((~finite_np).sum())
        if n_bad > 0:
            print(f"[DGS] WARNING: {n_bad} surfels with non-finite xyz; "
                  f"excluding from DGS this iteration.")

        # KNN against the voxelised LAS (query only finite points to avoid
        # propagating NaNs through the KDTree).
        distances = np.full((N, k), np.inf, dtype=np.float64)
        indices = np.zeros((N, k), dtype=np.int64)
        if finite_np.any():
            d_fin, i_fin = self.kdtree.query(xyz_np[finite_np], k=k, workers=-1)
            distances[finite_np] = d_fin
            indices[finite_np] = i_fin

        valid_np = finite_np & (distances[:, 0] < radius)  # (N,) bool
        M = int(valid_np.sum())

        if M == 0:
            # Degenerate: no surfels close to any LiDAR -- empty cache.
            self.valid_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
            self.plane_normals = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            self.plane_offsets = torch.zeros((0,), dtype=torch.float32, device=self.device)
            self.cached_count = N
            print(f"[DGS] update_cache: N={N}, M=0 (no surfel within {radius}m of LiDAR)")
            return

        idx_v = indices[valid_np]                          # (M, k) int
        nbrs_np = self.lidar_points_np[idx_v]              # (M, k, 3)
        nbrs_nrm_np = self.lidar_normals_np[idx_v]         # (M, k, 3)

        # Batched PCA on GPU
        nbrs = torch.from_numpy(nbrs_np).to(self.device, dtype=torch.float32)
        nbrs_nrm = torch.from_numpy(nbrs_nrm_np).to(self.device, dtype=torch.float32)
        centroid = nbrs.mean(dim=1)                        # (M, 3)
        diff = nbrs - centroid.unsqueeze(1)                # (M, k, 3)
        C = torch.einsum("mki,mkj->mij", diff, diff) / float(k)  # (M, 3, 3)

        # Sanity: drop covariances with any non-finite entries before eigh.
        C_finite = torch.isfinite(C).view(C.shape[0], -1).all(dim=1)
        if not C_finite.all():
            n_drop = int((~C_finite).sum().item())
            print(f"[DGS] WARNING: {n_drop} non-finite covariance matrices; dropping.")
            C = C[C_finite]
            nbrs_nrm = nbrs_nrm[C_finite]
            centroid = centroid[C_finite]
            # Also shrink valid_mask: map back to original surfel indices.
            valid_indices = np.where(valid_np)[0]
            keep_local = C_finite.detach().cpu().numpy()
            valid_np = np.zeros_like(valid_np)
            valid_np[valid_indices[keep_local]] = True
            M = int(valid_np.sum())
            if M == 0:
                self.valid_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
                self.plane_normals = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
                self.plane_offsets = torch.zeros((0,), dtype=torch.float32, device=self.device)
                self.cached_count = N
                return

        # eigh returns eigenvalues ascending; smallest -> plane normal.
        # Fall back to CPU if cuSOLVER trips (happens on some driver/matrix combos).
        try:
            eigvals, eigvecs = torch.linalg.eigh(C)
        except Exception as e:
            print(f"[DGS] GPU eigh failed ({type(e).__name__}); falling back to CPU.")
            eigvals_cpu, eigvecs_cpu = torch.linalg.eigh(C.cpu())
            eigvals = eigvals_cpu.to(self.device)
            eigvecs = eigvecs_cpu.to(self.device)
        plane_normals = eigvecs[:, :, 0]                   # (M, 3)

        # Orient planes consistently with the local LiDAR normals so signs
        # are stable across cache rebuilds.
        mean_nbr_nrm = nbrs_nrm.mean(dim=1)                # (M, 3)
        flip = (plane_normals * mean_nbr_nrm).sum(dim=1) < 0.0
        plane_normals = torch.where(flip.unsqueeze(1), -plane_normals, plane_normals)

        plane_offsets = (plane_normals * centroid).sum(dim=1)  # (M,)

        valid_mask = torch.from_numpy(valid_np).to(self.device)
        self.valid_mask = valid_mask
        self.plane_normals = plane_normals.contiguous()
        self.plane_offsets = plane_offsets.contiguous()
        self.cached_count = N

        print(f"[DGS] update_cache: N={N}, valid M={M} "
              f"({100.0 * M / max(N, 1):.1f}%), nearest-dist p50={np.median(distances[:, 0]):.4f}m")

    # -----------------------------------------------------------------------
    # Differentiable loss
    # -----------------------------------------------------------------------
    def compute_loss(self,
                     surfel_xyz: torch.Tensor,
                     surfel_normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (L_dgs, L_dgs_normal). Gradients flow through both inputs.

        Args:
            surfel_xyz:     (N, 3) -- gaussians.get_xyz
            surfel_normals: (N, 3) -- z-column of build_rotation(get_rotation)
        """
        if self.valid_mask is None or self.plane_normals.shape[0] == 0:
            zero = surfel_xyz.sum() * 0.0 + surfel_normals.sum() * 0.0
            return zero, zero

        xyz_v = surfel_xyz[self.valid_mask]                # (M, 3)
        n_v = surfel_normals[self.valid_mask]              # (M, 3)

        # Signed distance to the plane: n . x - d
        delta = (self.plane_normals * xyz_v).sum(dim=1) - self.plane_offsets
        L_dgs = delta.abs().mean()

        # Normal alignment (abs handles sign ambiguity)
        L_dgs_n = (1.0 - (n_v * self.plane_normals).sum(dim=1).abs()).mean()

        return L_dgs, L_dgs_n
