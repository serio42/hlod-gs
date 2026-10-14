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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
           override_color = None, override_xyz = None, active_mask = None,
           active_indices = None, opacity_scale = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    Args:
        active_mask: Optional boolean mask [N] selecting which Gaussians to render.
        active_indices: Optional long tensor [M] selecting Gaussians to render.
                        Prefer this for hierarchical rendering; it avoids full
                        boolean masks and full-model activated gathers.
        opacity_scale: Optional float tensor [N] multiplying Gaussian opacity.
                       This is intended for render-time LoD blending/fading.
    """

    if active_mask is not None and active_indices is not None:
        raise ValueError("render received both active_mask and active_indices")
    if active_indices is None and active_mask is not None:
        active_indices = active_mask.nonzero(as_tuple=True)[0]

    if active_indices is not None:
        active_indices = active_indices.to(device=pc.get_xyz.device, dtype=torch.long)
        means3D = pc.get_xyz_at(active_indices)
        opacity = pc.get_opacity_at(active_indices)
        if opacity_scale is not None:
            opacity = opacity * opacity_scale[active_indices].unsqueeze(-1).clamp(0.0, 1.0)
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance_at(active_indices, scaling_modifier)
            scales = None
            rotations = None
        else:
            cov3D_precomp = None
            scales = pc.get_scaling_at(active_indices)
            rotations = pc.get_rotation_at(active_indices)
        if override_color is not None:
            colors_precomp = override_color[active_indices]
            shs = None
        elif pipe.convert_SHs_python:
            features = pc.get_features_at(active_indices)
            shs_view = features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            shs = None
        else:
            shs = pc.get_features_at(active_indices)
            colors_precomp = None
        if override_xyz is not None:
            means3D = override_xyz[active_indices]
    else:
        if override_xyz is None:
            means3D = pc.get_xyz
        else:
            means3D = override_xyz
        opacity = pc.get_opacity
        if opacity_scale is not None:
            opacity = opacity * opacity_scale.unsqueeze(-1).clamp(0.0, 1.0)
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        low_pass=getattr(pipe, "low_pass", 0.3),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "active_mask": active_mask,
            "active_indices": active_indices,
            }
