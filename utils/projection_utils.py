import torch
import math
from utils.general_utils import build_rotation


def compute_cov2D(xyz: torch.Tensor, scaling: torch.Tensor,
                  rotation: torch.Tensor, viewpoint_camera) -> tuple:

    device = xyz.device
    N = xyz.shape[0]

    # Build 3D covariance: Sigma = R * diag(s^2) * R^T
    R = build_rotation(rotation)  # [N, 3, 3]
    S = scaling  # [N, 3]
    # L = R * diag(s)
    L = R * S[:, None, :]  # [N, 3, 3] broadcast: R[:,:,k] * s[k]
    cov3D = L @ L.transpose(1, 2)  # [N, 3, 3]

    # World-to-camera transform
    W2C = viewpoint_camera.world_view_transform.T.to(device)  # [4, 4]
    R_cam = W2C[:3, :3]  # [3, 3]
    t_cam = W2C[:3, 3]   # [3]

    # Transform positions to camera space
    xyz_cam = (xyz @ R_cam.T) + t_cam[None, :]  # [N, 3]

    # Transform covariance to camera space: Sigma' = W * Sigma * W^T
    cov3D_cam = R_cam[None, :, :] @ cov3D @ R_cam.T[None, :, :]  # [N, 3, 3]

    # Focal lengths in pixels
    fx = viewpoint_camera.image_width / (2.0 * math.tan(viewpoint_camera.FoVx / 2.0))
    fy = viewpoint_camera.image_height / (2.0 * math.tan(viewpoint_camera.FoVy / 2.0))

    # Frustum clamping (matches rasterizer forward.cu)
    limx = 1.3 * math.tan(viewpoint_camera.FoVx / 2.0)
    limy = 1.3 * math.tan(viewpoint_camera.FoVy / 2.0)
    z = xyz_cam[:, 2].clamp(min=0.2)
    x = (xyz_cam[:, 0] / z).clamp(-limx, limx) * z
    y = (xyz_cam[:, 1] / z).clamp(-limy, limy) * z

    # Jacobian of perspective projection
    inv_z = 1.0 / z
    inv_z2 = inv_z * inv_z

    # J = [[fx/z, 0, -fx*x/z^2],
    #      [0, fy/z, -fy*y/z^2]]
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = fx * inv_z
    J[:, 0, 2] = -fx * x * inv_z2
    J[:, 1, 1] = fy * inv_z
    J[:, 1, 2] = -fy * y * inv_z2

    # Project: Sigma_2D = J * Sigma' * J^T
    cov2D = J @ cov3D_cam @ J.transpose(1, 2)  # [N, 2, 2]

    return cov2D, xyz_cam


def compute_screenspace_size(cov2D: torch.Tensor) -> tuple:

    a = cov2D[:, 0, 0]
    b = cov2D[:, 0, 1]
    d = cov2D[:, 1, 1]

    trace = a + d
    det = (a * d - b * b).clamp(min=1e-10)

    half_trace = trace * 0.5
    discriminant = (half_trace * half_trace - det).clamp(min=0.0)
    sqrt_disc = torch.sqrt(discriminant)

    lambda1 = half_trace + sqrt_disc  # larger
    lambda2 = (half_trace - sqrt_disc).clamp(min=1e-10)  # smaller

    # 3-sigma radius from largest eigenvalue
    radius_px = 3.0 * torch.sqrt(lambda1.clamp(min=1e-10))

    # Ellipse area = pi * 3*sqrt(lambda1) * 3*sqrt(lambda2)
    area_px = 9.0 * math.pi * torch.sqrt((lambda1 * lambda2).clamp(min=1e-10))

    return radius_px, area_px, lambda1, lambda2


def scale_upper_loss(gaussians, scope="all", current_lod=None, include_lod1=False,
                     lod_iteration_weights=None, active_mask=None):

    if gaussians._birth_lod is None:
        return torch.tensor(0.0, device=gaussians._xyz.device)
    if scope not in ("all", "current_only"):
        raise ValueError(f"scale_upper_loss: unknown scope={scope!r} (expected 'all' or 'current_only')")

    device = gaussians._birth_lod.device
    N = gaussians._birth_lod.shape[0]

    # Build the per-Gaussian upper-bound tensor. inf = "no bound, do not penalize".
    upper = torch.full((N,), float('inf'), device=device)

    offsets = getattr(gaussians, "_lod_offsets", None)
    offsets_cpu = offsets.tolist() if offsets is not None else None
    start_lod = 1 if include_lod1 else 2
    for lod in range(start_lod, gaussians.max_lod + 1):
        if lod == 1:
            bound = gaussians.lod1_scaling_lower_bound * gaussians.lod_scaling_ratio
        else:
            bound = gaussians.lod1_scaling_lower_bound / (gaussians.lod_scaling_ratio ** (lod - 2))
        if offsets_cpu is not None:
            lo = offsets_cpu[lod - 1]
            hi = offsets_cpu[lod]
            if hi > lo:
                upper[lo:hi] = bound
        else:
            mask = (gaussians._birth_lod == lod)
            if mask.any():
                upper[mask] = bound

    finite_mask = upper < float('inf')
    if scope == "current_only" and current_lod is not None:
        finite_mask = finite_mask & (gaussians._birth_lod.to(device) == current_lod)
    if active_mask is not None:
        finite_mask = finite_mask & active_mask.to(device)
    if not finite_mask.any():
        return torch.tensor(0.0, device=device)

    def _lod_weight(lod_val):
        if lod_iteration_weights is None:
            return 1.0
        if isinstance(lod_iteration_weights, dict):
            return float(lod_iteration_weights.get(int(lod_val), 1.0))
        idx = int(lod_val) - 1
        if 0 <= idx < len(lod_iteration_weights):
            return float(lod_iteration_weights[idx])
        return 1.0

    # Activated scale (with lower bound, without upper clamp)
    raw_scaling = (
        gaussians._cat_lod_attribute("scaling")
        if hasattr(gaussians, "_cat_lod_attribute") and gaussians._using_separated_storage()
        else gaussians._scaling
    )
    raw_scale = gaussians.scaling_activation(raw_scaling)
    if gaussians._per_gaussian_lb is not None:
        raw_scale = raw_scale + gaussians._per_gaussian_lb.unsqueeze(-1)
    elif gaussians.scaling_lower_bound > 0:
        raw_scale = raw_scale + gaussians.scaling_lower_bound

    excess = torch.relu(raw_scale[finite_mask] - upper[finite_mask].unsqueeze(-1))
    if lod_iteration_weights is not None:
        weights = torch.ones((excess.shape[0],), device=device, dtype=excess.dtype)
        lods = gaussians._birth_lod[finite_mask]
        for lod_val in torch.unique(lods).tolist():
            weights[lods == lod_val] = _lod_weight(lod_val)
        excess = excess * weights.unsqueeze(-1)
    return excess.mean()


def scale_ratio_loss(gaussians, threshold=1.0, active_mask=None):

    scaling = gaussians.get_scaling  # (N, 3), activated with lower bound
    device = scaling.device

    if gaussians._parent_idx is not None:
        # Hierarchical mode: live parent lookup
        raw_parent_ids = gaussians._parent_idx.to(device).long()
        N = scaling.shape[0]
        has_parent = (raw_parent_ids >= 0) & (raw_parent_ids < N)
        # Exclude LOD 1 (no meaningful parent constraint)
        if gaussians._birth_lod is not None:
            has_parent = has_parent & (gaussians._birth_lod > 1)
        if active_mask is not None:
            has_parent = has_parent & active_mask.to(device)
        if has_parent.sum() == 0:
            return torch.tensor(0.0, device=device)

        parent_ids = raw_parent_ids[has_parent]
        child_max_scale = scaling[has_parent].max(dim=1).values       # (M,)
        parent_max_scale = scaling[parent_ids].max(dim=1).values.detach()  # (M,)

    else:
        return torch.tensor(0.0, device=device)

    ratio = child_max_scale / (parent_max_scale + 1e-8)
    return torch.relu(ratio - threshold).mean()
