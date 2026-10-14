#
# gsplat-backed renderer for HLoD.
#
# Drop-in replacement for gaussian_renderer.render() that uses gsplat's
# rasterizer instead of the INRIA diff-gaussian-rasterization one. Same
# call signature and same return dict, so train.py / render.py work unchanged.
#
# The return contract that the training loop relies on (per active Gaussian,
# aligned to `active_indices` order):
#   render              : image [3, H, W]
#   viewspace_points    : a leaf tensor whose .grad is [M, 2] screen-space
#                         gradient after backward (used for densification)
#   visibility_filter   : bool [M], radii > 0
#   radii               : float [M], per-Gaussian screen radius (px)
#
# gsplat keeps its own means2d inside the info dict; we expose an INRIA-shaped
# [M, 2] carrier and copy the means2d gradient into carrier.grad via a backward
# hook, so the densification code needs no changes.
#

import math
import weakref
import torch
import gsplat

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def _intrinsics(cam):
    W = int(cam.image_width)
    H = int(cam.image_height)
    fx = W / (2.0 * math.tan(cam.FoVx * 0.5))
    fy = H / (2.0 * math.tan(cam.FoVy * 0.5))
    K = torch.tensor(
        [[fx, 0.0, W * 0.5],
         [0.0, fy, H * 0.5],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32, device="cuda",
    )
    return K, W, H


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, override_xyz=None, active_mask=None,
           active_indices=None, opacity_scale=None):
    """gsplat drop-in for gaussian_renderer.render(). See module docstring."""

    if active_mask is not None and active_indices is not None:
        raise ValueError("render received both active_mask and active_indices")
    if active_indices is None and active_mask is not None:
        active_indices = active_mask.nonzero(as_tuple=True)[0]

    if pipe.compute_cov3D_python:
        raise NotImplementedError("gsplat renderer: compute_cov3D_python path not supported")
    if pipe.convert_SHs_python:
        raise NotImplementedError("gsplat renderer: convert_SHs_python path not supported")

    # --- gather the active Gaussians (same getters as the INRIA path) ---
    if active_indices is not None:
        active_indices = active_indices.to(device=pc.get_xyz.device, dtype=torch.long)
        means3D = pc.get_xyz_at(active_indices)
        opacity = pc.get_opacity_at(active_indices)
        scales = pc.get_scaling_at(active_indices)
        rotations = pc.get_rotation_at(active_indices)
        if opacity_scale is not None:
            opacity = opacity * opacity_scale[active_indices].unsqueeze(-1).clamp(0.0, 1.0)
        if override_color is not None:
            colors = override_color[active_indices]
            shs = None
        else:
            shs = pc.get_features_at(active_indices)
            colors = None
        if override_xyz is not None:
            means3D = override_xyz[active_indices]
    else:
        means3D = pc.get_xyz if override_xyz is None else override_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if opacity_scale is not None:
            opacity = opacity * opacity_scale.unsqueeze(-1).clamp(0.0, 1.0)
        if override_color is not None:
            colors = override_color
            shs = None
        else:
            shs = pc.get_features
            colors = None

    M = means3D.shape[0]
    scales = scales * scaling_modifier

    # gsplat wants w2c (OpenCV) [1,4,4]; HLoD stores its transpose.
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0).contiguous()
    K, W, H = _intrinsics(viewpoint_camera)
    Ks = K.unsqueeze(0)

    if shs is not None:
        sh_degree = int(pc.active_sh_degree)
        need = (sh_degree + 1) ** 2
        color_input = shs[:, :need, :].contiguous()   # gsplat expands SH internally
    else:
        sh_degree = None
        color_input = colors.contiguous()

    use_absgrad = bool(getattr(pipe, "absgrad", False))

    render_colors, render_alphas, info = gsplat.rasterization(
        means=means3D.contiguous(),
        quats=rotations.contiguous(),
        scales=scales.contiguous(),
        opacities=opacity.squeeze(-1).contiguous(),
        colors=color_input,
        viewmats=viewmat,
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=sh_degree,
        render_mode="RGB",
        packed=False,
        near_plane=float(viewpoint_camera.znear),
        far_plane=float(viewpoint_camera.zfar),
        eps2d=float(getattr(pipe, "low_pass", 0.3)),   # matches INRIA's low_pass (2D-cov dilation)
        absgrad=use_absgrad,                           # AbsGS: abs gradients -> means2d.absgrad after backward
        backgrounds=bg_color.reshape(1, -1).contiguous(),
    )

    image = render_colors[0].permute(2, 0, 1).contiguous()   # [3, H, W]

    radii = info["radii"][0].amax(dim=-1).float()            # [M] per-Gaussian radius
    visibility_filter = radii > 0

    # INRIA-shaped screen-space gradient carrier: a leaf whose .grad we fill
    # from gsplat's internal means2d gradient during backward.
    screenspace_points = torch.zeros((M, 2), dtype=torch.float32,
                                     device=means3D.device, requires_grad=True)
    means2d = info["means2d"]

    # gsplat's means2d is in PIXEL coords; INRIA's screenspace grad (which the
    # densify_grad_threshold is calibrated against) is in NDC [-1,1].
    # Rescale grad by (W/2, H/2) so the existing threshold applies unchanged.
    grad_scale = torch.tensor([W * 0.5, H * 0.5], dtype=torch.float32,
                              device=means3D.device)

    # With absgrad, gsplat's rasterize backward stashes the absolute image-plane
    # gradient on means2d.absgrad BEFORE the gradient reaches means2d (so it is
    # already set when this hook fires). Prefer it; fall back to the plain grad.
    #
    # IMPORTANT: reach means2d.absgrad via a weakref, never a direct capture. The
    # hook lives in means2d._backward_hooks, so capturing means2d in the closure
    # would form the cycle means2d -> hook -> closure -> means2d, pinning the whole
    # per-iteration backward graph until the cyclic GC runs -> CUDA OOM within an
    # epoch. During backward means2d is guaranteed alive, so the weakref resolves.
    m2d_ref = weakref.ref(means2d) if use_absgrad else None

    def _copy_means2d_grad(grad):
        src = None
        if m2d_ref is not None:
            m = m2d_ref()
            if m is not None:
                src = getattr(m, "absgrad", None)
        if src is None:
            src = grad
        screenspace_points.grad = (src.reshape(M, 2) * grad_scale).detach()
        return grad

    if means2d.requires_grad:
        means2d.register_hook(_copy_means2d_grad)

    return {"render": image,
            "viewspace_points": screenspace_points,
            "visibility_filter": visibility_filter,
            "radii": radii,
            "active_mask": active_mask,
            "active_indices": active_indices,
            }
