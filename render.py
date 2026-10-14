#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr 

import torch
from scene import Scene, GaussianModel
import os
import re
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, DEFAULT_SP_THRESH
import time
import numpy as np

from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips, _criterion as lpips_criterion
from utils.render_utils import generate_path, create_video_from_frames
from utils.sh_utils import RGB2SH


def _frustum_near_mask(g, view, near):

    if near is None or near <= 0.0:
        return None
    W2C = view.world_view_transform.T.to(g._xyz.device)
    R = W2C[:3, :3]
    t = W2C[:3, 3]
    z_view = g._xyz @ R[2, :] + t[2]
    return z_view >= near

def _frustum_near_indices(g, view, indices, near):
    if near is None or near <= 0.0 or indices is None or indices.numel() == 0:
        return indices
    W2C = view.world_view_transform.T.to(g._xyz.device)
    R = W2C[:3, :3]
    t = W2C[:3, 3]
    xyz = g._xyz[indices.to(device=g._xyz.device, dtype=torch.long)]
    z_view = xyz @ R[2, :] + t[2]
    return indices[z_view >= near]


def _combine_masks(a, b):
    if a is None: return b
    if b is None: return a
    return a & b


def _near_tag(near):
    near = float(near or 0.0)
    return "" if near <= 0.0 else f"_near{near:g}"


def _cull_tag(cull_near):
    cull_near = float(cull_near or 0.0)
    return "" if cull_near <= 0.0 else f"_cull{cull_near:g}"


def _cull_frustum_tag(cull_frustum):
    cull_frustum = float(cull_frustum or 0.0)
    return "" if cull_frustum <= 0.0 else f"_cullf{cull_frustum:g}"


def _opacity_tag(opacity_min):
    opacity_min = float(opacity_min or 0.0)
    return "" if opacity_min <= 0.0 else f"_opac{opacity_min:g}"


def _apply_render_opacity_filter(g, active_mask, active_indices, opacity_min, profile=None):
    opacity_min = float(opacity_min or 0.0)
    if opacity_min <= 0.0:
        return active_mask, active_indices

    t_opacity = _profile_start(profile)
    if active_indices is not None:
        if active_indices.numel() > 0:
            idx = active_indices.to(device=g._opacity.device, dtype=torch.long)
            keep = torch.sigmoid(g._opacity[idx].flatten()) >= opacity_min
            active_indices = active_indices[keep]
        _profile_stop(profile, "filter_opacity_s", t_opacity)
        return active_mask, active_indices

    opacity_mask = torch.sigmoid(g._opacity.flatten()) >= opacity_min
    active_mask = _combine_masks(active_mask, opacity_mask)
    _profile_stop(profile, "filter_opacity_s", t_opacity)
    return active_mask, active_indices


def _render_video_loop(views, pipeline, background, select_fn,
                       video_path, n_frames=240, trajectory='ellipse', radius_offset=0.0, near=0.0,
                       dolly_distance=0.0, dolly_view_index=0):
    video_cameras = generate_path(views, n_frames=n_frames, trajectory=trajectory,
                                  radius_offset=radius_offset,
                                  dolly_distance=dolly_distance,
                                  dolly_view_index=dolly_view_index)
    frames = []
    for idx, view in enumerate(tqdm(video_cameras, desc="Rendering video")):
        g = select_fn(view)
        near_mask = _frustum_near_mask(g, view, near)
        rasterize_results = render(view, g, pipeline, background, active_mask=near_mask) if near_mask is not None else render(view, g, pipeline, background)
        rendering = torch.clamp(rasterize_results["render"], 0.0, 1.0)
        frame = (rendering.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)
    create_video_from_frames(frames, out_path=video_path)
    print(f"Video saved: {video_path}")

def _render_eval_loop(views, pipeline, background, select_fn,
                      render_path, gt_path, label="Rendering", near=0.0):
    rendering_times, gnums, visible_gnums = [], [], []
    psnrs_list, ssims_list, lpipss_list = [], [], []

    for idx, view in enumerate(tqdm(views, desc=label)):
        torch.cuda.synchronize()
        t_start = time.time()
        g = select_fn(view)
        near_mask = _frustum_near_mask(g, view, near)
        if near_mask is not None:
            rasterize_results = render(view, g, pipeline, background, active_mask=near_mask)
        else:
            rasterize_results = render(view, g, pipeline, background)
        torch.cuda.synchronize()
        t_end = time.time()

        rendering = torch.clamp(rasterize_results["render"], 0.0, 1.0)
        if hasattr(view, "alpha_mask"):
            rendering *= view.alpha_mask.cuda()
        gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)

        ssims_list.append(ssim(rendering, gt.cuda()).mean().item())
        psnrs_list.append(psnr(rendering, gt.cuda()).mean().item())
        lpipss_list.append(lpips(rendering, gt.cuda(), net_type='vgg').mean().item())
        rendering_times.append(t_end - t_start)
        gnums.append(int(near_mask.sum().item()) if near_mask is not None else g._xyz.shape[0])
        visible_gnums.append(int(rasterize_results["visibility_filter"].sum().item()))

        torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
        torchvision.utils.save_image(gt, os.path.join(gt_path, f'{idx:05d}.png'))

    return rendering_times, gnums, psnrs_list, ssims_list, lpipss_list, visible_gnums

def _write_metrics(model_path, name, render_name, rendering_times, gnums,
                   psnrs_list, ssims_list, lpipss_list, extra="", visible_gnums=None,
                   mem_model_bytes=None, render_peaks=None, run_peak_bytes=None):
    if len(rendering_times) > 5:
        fps = 1.0 / (sum(rendering_times[5:]) / len(rendering_times[5:]))
    else:
        fps = 1.0 / (sum(rendering_times) / len(rendering_times))

    mean_psnr = torch.tensor(psnrs_list).mean().item()
    mean_ssim = torch.tensor(ssims_list).mean().item()
    mean_lpips = torch.tensor(lpipss_list).mean().item()
    mean_gnum = np.mean(gnums)
    max_gnum = np.max(gnums)
    gpu_name = torch.cuda.get_device_name()
    mb = 1024.0 ** 2
    # Legacy "peak memory" = whole-frame peak (incl. LPIPS/VGG eval buffers).
    # Preserved via run_peak_bytes so pre-existing numbers stay comparable.
    memory_used = (run_peak_bytes / mb) if run_peak_bytes else (torch.cuda.max_memory_allocated() / mb)

    with open(os.path.join(model_path, name, "metrics.txt"), "a") as f:
        f.write(f"\n({gpu_name} {render_name}) "
                f"PSNR: {mean_psnr:.5f} / SSIM: {mean_ssim:.5f} / "
                f"LPIPS: {mean_lpips:.5f}\n")
        extra_str = f" / {extra}" if extra else ""
        f.write(f"mean Gnum: {mean_gnum:.0f} / max Gnum: {max_gnum:.0f} / "
                f"peak memory {memory_used:.0f}MB / FPS: {fps:.3f}{extra_str}\n")
        # Isolated render-time VRAM breakdown (excludes the VGG/LPIPS quality pass).
        # resident model = whole hierarchy loaded before the loop; render peak =
        # peak allocated during rasterization; rasterizer scratch = peak − model.
        if render_peaks:
            rp = np.asarray(render_peaks, dtype=np.float64) / mb
            reserved_mb = torch.cuda.memory_reserved() / mb
            line = (f"VRAM render: peak {rp.mean():.0f}MB (max {rp.max():.0f}MB) / "
                    f"reserved {reserved_mb:.0f}MB")
            if mem_model_bytes:
                model_mb = mem_model_bytes / mb
                scratch = rp - model_mb
                line += (f" / resident model {model_mb:.0f}MB / "
                         f"rasterizer scratch {scratch.mean():.0f}MB "
                         f"(max {scratch.max():.0f}MB)")
            f.write(line + "\n")
        if visible_gnums is not None and len(visible_gnums) > 0:
            # Post-rasterizer survivor count: Gaussians from the active set that
            # ended up with a non-zero on-screen radius (radii > 0), i.e. that
            # actually touched the image. Comparable to Octree-GS's per-view "GS"
            # count, unlike "Gnum" above which is the pre-rasterizer active set.
            f.write(f"mean visible Gnum: {np.mean(visible_gnums):.0f} / "
                    f"max visible Gnum: {np.max(visible_gnums):.0f} "
                    f"(post-rasterizer, radii>0)\n")

    print(f"\n[{render_name}] PSNR: {mean_psnr:.3f} dB | SSIM: {mean_ssim:.4f} | "
          f"FPS: {fps:.1f} | avg {mean_gnum:.0f} Gaussians/Frame")


def _profile_start(profile):
    if profile is None:
        return None
    torch.cuda.synchronize()
    return time.time()


def _profile_stop(profile, key, start):
    if profile is None or start is None:
        return None
    torch.cuda.synchronize()
    end = time.time()
    profile[key] = profile.get(key, 0.0) + (end - start)
    return end


def _write_hierarchical_profile(model_path, name, render_name, profile_rows, max_lod):
    if not profile_rows:
        return

    time_keys = [
        "selection_total_s",
        "select_core_s",
        "projected_size_s",
        "active_traversal_s",
        "subthreshold_s",
        "filter_lod_s",
        "filter_near_s",
        "filter_opacity_s",
        "indices_to_mask_s",
        "mask_to_indices_s",
        "render_s",
        "metric_timed_s",
        "lod_count_s",
        "postprocess_s",
        "quality_metrics_s",
        "save_images_s",
        "heatmap_s",
        "frame_total_s",
    ]
    lod_fields = [f"lod{lod}" for lod in range(1, max_lod + 1)]
    fields = [
        "view_idx", "active_total", "visible_total", "cuda_backend_requested", "cuda_backend_used"
    ] + lod_fields + [key[:-2] + "_ms" for key in time_keys]

    log_path = os.path.join(model_path, name, "renders", f"{render_name}_profile.csv")
    with open(log_path, "w") as f:
        f.write(",".join(fields) + "\n")
        for row in profile_rows:
            per_lod = row.get("per_lod", [])
            values = [
                str(row.get("view_idx", "")),
                str(row.get("active_total", "")),
                str(row.get("visible_total", "")),
                str(row.get("cuda_backend_requested", "")),
                str(row.get("cuda_backend_used", "")),
            ]
            values.extend(str(per_lod[lod - 1]) if lod - 1 < len(per_lod) else "0"
                          for lod in range(1, max_lod + 1))
            values.extend(f"{row.get(key, 0.0) * 1000.0:.4f}" for key in time_keys)
            f.write(",".join(values) + "\n")

    sample = profile_rows[5:] if len(profile_rows) > 5 else profile_rows

    def _mean_ms(key):
        return float(np.mean([row.get(key, 0.0) for row in sample]) * 1000.0)

    print(f"Profile log: {log_path}")
    print(
        f"[profile {render_name}] avg ms: "
        f"selection={_mean_ms('selection_total_s'):.3f}, "
        f"projection={_mean_ms('projected_size_s'):.3f}, "
        f"traversal={_mean_ms('active_traversal_s'):.3f}, "
        f"render={_mean_ms('render_s'):.3f}, "
        f"metrics={_mean_ms('quality_metrics_s'):.3f}, "
        f"save={_mean_ms('save_images_s'):.3f}"
    )

@torch.no_grad()
def render_sets_single_level(args, model, pipeline):
    dataset = model.extract(args)
    pipeline_params = pipeline.extract(args)
    gaussian_model_args = {
        'sh_degree': dataset.sh_degree,
        'lod1_scaling_lower_bound': dataset.lod1_scaling_lower_bound,
        'lod_scaling_ratio': dataset.lod_scaling_ratio,
        'increase_lod_num_childs': dataset.increase_lod_num_childs,
        'current_lod': args.render_lod,
        'max_lod': args.lod,
    }
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if not args.skip_train:
        scene = Scene(dataset, gaussians=None, load_iteration=args.iteration, load_lod=args.lod, shuffle=False, load_only_test_images=False, load_image_device=args.load_image_device, load_ply=False)
        render_set_single_level(dataset.model_path, "train", scene.loaded_iter, args.render_lod, scene.getTrainCameras(), pipeline_params, background, gaussian_model_args, args.n_frames, trajectory=args.trajectory, radius_offset=args.orbit_radius_offset, args=args)
    
    if not args.skip_test:
        scene = Scene(dataset, gaussians=None, load_iteration=args.iteration, load_lod=args.lod, shuffle=False, load_only_test_images=True, load_image_device=args.load_image_device, load_ply=False)
        test_cameras = scene.getTestCameras()
        if test_cameras:
            render_set_single_level(dataset.model_path, "test", scene.loaded_iter, args.render_lod, test_cameras, pipeline_params, background, gaussian_model_args, args.n_frames, trajectory=args.trajectory, radius_offset=args.orbit_radius_offset, args=args)

def render_set_single_level(model_path, name, iteration, render_lod, views, pipeline, background, gaussian_model_args, n_frames, trajectory="ellipse", radius_offset=0.0, args=None):
    near = getattr(args, 'near', 0.0) if args else 0.0
    _res = getattr(args, 'resolution', -1) if args else -1
    res_tag = f"_r{int(_res)}" if _res and _res > 0 else ""
    render_name = f"level{render_lod}{_near_tag(near)}{res_tag}"
    render_path = os.path.join(model_path, name, "renders", render_name)
    gt_path = os.path.join(model_path, name, f"gt{res_tag}")
    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    
    gaussian_select_lod = _load_single_lod_gaussians(
        model_path, args.iteration if args is not None else iteration, render_lod, gaussian_model_args)

    def _select(view):
        return gaussian_select_lod

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    if args and args.save_video:
        video_dir = os.path.join(model_path, name, "renders", "videos")
        makedirs(video_dir, exist_ok=True)
        _render_video_loop(views, pipeline, background, _select,
                           os.path.join(video_dir, f"{render_name}.mp4"),
                           n_frames, trajectory, radius_offset, near=near,
                           dolly_distance=getattr(args, 'dolly_distance', 0.0),
                           dolly_view_index=getattr(args, 'dolly_view_index', 0))
        return

    rendering_times, gnums, psnrs_list, ssims_list, lpipss_list, visible_gnums = \
        _render_eval_loop(views, pipeline, background, _select,
                          render_path, gt_path, near=near)
    # single_level renders one fixed LOD with no per-view selection, so the
    # timed loop is rasterization only.
    extra = f"near={near:g}, timing=render_only" if near > 0.0 else "near=0, timing=render_only"
    _write_metrics(model_path, name, render_name, rendering_times, gnums,
                   psnrs_list, ssims_list, lpipss_list, extra=extra,
                   visible_gnums=visible_gnums)


LOD_COLORS = {
    1: torch.tensor([0.2, 0.2, 1.0]),
    2: torch.tensor([0.0, 0.8, 0.8]),
    3: torch.tensor([0.0, 0.9, 0.2]),
    4: torch.tensor([1.0, 0.8, 0.0]),
    5: torch.tensor([1.0, 0.1, 0.1]),
}

def _make_lod_colors(gaussians):
    N = gaussians._xyz.shape[0]
    colors = torch.zeros(N, 3, device="cuda")
    for lod, rgb in LOD_COLORS.items():
        mask = (gaussians._birth_lod == lod)
        if mask.any():
            colors[mask] = rgb.to("cuda")
    return colors

def _hierarchy_leaf_mask(gaussians):
    N = gaussians._xyz.shape[0]
    device = gaussians._xyz.device
    is_parent = torch.zeros(N, dtype=torch.bool, device=device)
    if gaussians._parent_idx is not None:
        valid_refs = gaussians._parent_idx[gaussians._parent_idx >= 0]
        if valid_refs.numel() > 0:
            is_parent[valid_refs.unique()] = True
    return ~is_parent

def _hierarchy_orphan_mask(gaussians):
    N = gaussians._xyz.shape[0]
    device = gaussians._xyz.device
    if gaussians._parent_idx is None or gaussians._birth_lod is None:
        return torch.zeros(N, dtype=torch.bool, device=device)
    return (gaussians._parent_idx == -1) & (gaussians._birth_lod > 1)

def _make_leaf_colors(gaussians, color_by="id"):
    if color_by == "lod":
        return _make_lod_colors(gaussians)

    N = gaussians._xyz.shape[0]
    idx = torch.arange(N, device=gaussians._xyz.device, dtype=torch.float32)
    # Deterministic hash-like palette: stable per Gaussian, visually separated.
    colors = torch.stack((
        torch.frac(torch.sin((idx + 1.0) * 12.9898) * 43758.5453),
        torch.frac(torch.sin((idx + 1.0) * 78.2330) * 24634.6345),
        torch.frac(torch.sin((idx + 1.0) * 37.7190) * 13515.8732),
    ), dim=1)
    return colors * 0.85 + 0.15

def _hsv_to_rgb(h, s, v):
    h6 = h * 6.0
    i = torch.floor(h6).to(torch.long)
    f = h6 - i.to(dtype=h.dtype)
    i = torch.remainder(i, 6)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    rgb = torch.empty(h.shape + (3,), device=h.device, dtype=h.dtype)
    rgb[i == 0] = torch.stack((v, t, p), dim=1)[i == 0]
    rgb[i == 1] = torch.stack((q, v, p), dim=1)[i == 1]
    rgb[i == 2] = torch.stack((p, v, t), dim=1)[i == 2]
    rgb[i == 3] = torch.stack((p, q, v), dim=1)[i == 3]
    rgb[i == 4] = torch.stack((t, p, v), dim=1)[i == 4]
    rgb[i == 5] = torch.stack((v, p, q), dim=1)[i == 5]
    return rgb.clamp(0.0, 1.0)

def _make_group_colors(group_ids):
    group_ids = group_ids.to(dtype=torch.long)
    _, group_rank = torch.unique(group_ids, sorted=True, return_inverse=True)
    rank = group_rank.to(device=group_ids.device, dtype=torch.float32)
    hue = torch.remainder(rank * 0.61803398875, 1.0)
    sat = 0.62 + 0.28 * torch.remainder(rank * 0.38196601125, 1.0)
    val = 0.82 + 0.18 * torch.remainder(rank * 0.75487766625, 1.0)
    return _hsv_to_rgb(hue, sat, val)

def _make_sibling_group_colors(gaussians):
    N = gaussians._xyz.shape[0]
    device = gaussians._xyz.device
    if gaussians._parent_idx is None:
        return _make_group_colors(torch.arange(N, device=device, dtype=torch.long))

    parent_idx = gaussians._parent_idx.to(device=device, dtype=torch.long)
    group_ids = parent_idx.clone()
    invalid_parent = (group_ids < 0) | (group_ids >= N)
    if invalid_parent.any():
        group_ids[invalid_parent] = torch.arange(N, device=device, dtype=torch.long)[invalid_parent]
    return _make_group_colors(group_ids)

def _make_family_colors(gaussians):
    N = gaussians._xyz.shape[0]
    device = gaussians._xyz.device
    if gaussians._parent_idx is None:
        return _make_group_colors(torch.arange(N, device=device, dtype=torch.long))

    parent_idx = gaussians._parent_idx.to(device=device, dtype=torch.long)
    family_ids = torch.arange(N, device=device, dtype=torch.long)
    max_depth = int(getattr(gaussians, "current_lod", 1) or 1)
    if gaussians._birth_lod is not None and gaussians._birth_lod.numel() > 0:
        max_depth = max(max_depth, int(gaussians._birth_lod.max().item()))

    for _ in range(max_depth + 1):
        candidate_parent = parent_idx[family_ids]
        valid_parent = (candidate_parent >= 0) & (candidate_parent < N) & (candidate_parent != family_ids)
        if not valid_parent.any():
            break
        family_ids[valid_parent] = candidate_parent[valid_parent]

    return _make_group_colors(family_ids)

def _make_train_hit_colors(gaussians, max_hits=0.0, percentile=99.0):
    N = gaussians._xyz.shape[0]
    device = gaussians._xyz.device
    if gaussians._train_hits is None:
        return torch.full((N, 3), 0.25, device=device)

    hits = gaussians._train_hits.to(device=device, dtype=torch.float32)
    if hits.numel() == 0:
        return torch.full((N, 3), 0.25, device=device)
    if hits.max() <= 0:
        print("[train_hits] model contains only zero hit counts; rendering neutral gray.")
        return torch.full((N, 3), 0.25, device=device)

    if max_hits and max_hits > 0:
        denom = torch.tensor(float(max_hits), device=device)
    else:
        positive = hits[hits > 0]
        if positive.numel() > 0:
            q = float(percentile) / 100.0
            denom = torch.quantile(positive, min(max(q, 0.0), 1.0)).clamp(min=1.0)
        else:
            denom = torch.tensor(1.0, device=device)

    t = (hits / denom).clamp(0.0, 1.0)
    colors = torch.stack((
        t,
        0.10 + 0.20 * (1.0 - t),
        1.0 - t,
    ), dim=1)
    colors[hits <= 0] = torch.tensor([0.02, 0.02, 0.08], device=device)
    return colors

def _make_hierarchy_debug_colors(gaussians, args, hier_mode):
    if hier_mode == "leaf_colors":
        return _make_leaf_colors(gaussians, getattr(args, 'leaf_color_by', 'id'))
    if hier_mode == "train_hits":
        return _make_train_hit_colors(
            gaussians,
            max_hits=getattr(args, 'train_hit_color_max', 0.0),
            percentile=getattr(args, 'train_hit_color_percentile', 99.0))
    if hier_mode == "sibling_groups":
        return _make_sibling_group_colors(gaussians)
    if hier_mode == "family_colors":
        return _make_family_colors(gaussians)
    return None


def _resolve_posthoc_hierarchy_path(model_path, path):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.join(model_path, path)


def _sanitize_tag(tag):
    tag = str(tag or "").strip()
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag)
    return tag.strip("-") or "posthoc"


def _apply_posthoc_hierarchy(gaussians, hierarchy_path):

    hierarchy = torch.load(hierarchy_path, map_location="cpu", weights_only=False)
    if gaussians._birth_lod is None:
        raise ValueError("--posthoc_hierarchy requires _birth_lod metadata")

    birth_lod = gaussians._birth_lod.detach().cpu().long().view(-1)
    n_total = int(birth_lod.numel())
    new_parent_idx = torch.full((n_total,), -1, dtype=torch.long)

    lod_min = int(hierarchy.get("lod_min", int(birth_lod.min().item())))
    lod_max = int(hierarchy.get("lod_max", int(birth_lod.max().item())))
    parent_indices = hierarchy.get("parent_indices", {})
    lod_counts = hierarchy.get("lod_counts", {})

    for lod in range(lod_min, lod_max + 1):
        n_actual = int((birth_lod == lod).sum().item())
        n_expected = int(lod_counts.get(lod, n_actual))
        if n_actual != n_expected:
            raise ValueError(
                f"Post-hoc hierarchy count mismatch for LOD {lod}: "
                f"hierarchy has {n_expected}, renderer model has {n_actual}"
            )

    for child_lod in range(lod_min + 1, lod_max + 1):
        if child_lod not in parent_indices:
            raise ValueError(f"Post-hoc hierarchy is missing parent_indices for LOD {child_lod}")

        child_global = (birth_lod == child_lod).nonzero(as_tuple=True)[0]
        parent_global = (birth_lod == child_lod - 1).nonzero(as_tuple=True)[0]
        local_parent = parent_indices[child_lod].detach().cpu().long().view(-1)

        if local_parent.shape[0] != child_global.shape[0]:
            raise ValueError(
                f"Post-hoc hierarchy length mismatch for LOD {child_lod}: "
                f"{local_parent.shape[0]} parents for {child_global.shape[0]} children"
            )

        valid = (local_parent >= 0) & (local_parent < parent_global.shape[0])
        if (~valid).any():
            print(
                f"[render] WARNING: post-hoc hierarchy has {(~valid).sum().item()} "
                f"invalid/orphan parents at LOD {child_lod}; keeping them as roots."
            )
        new_parent_idx[child_global[valid]] = parent_global[local_parent[valid]]

    gaussians._parent_idx = new_parent_idx
    gaussians.current_lod = lod_max
    gaussians.max_lod = max(int(getattr(gaussians, "max_lod", lod_max)), lod_max)
    gaussians._invalidate_topology_caches()
    print(
        f"[render] Applied post-hoc hierarchy {hierarchy_path}: "
        f"LOD {lod_min}->{lod_max}, mode={hierarchy.get('mode', 'unknown')}"
    )
    return hierarchy

def _select_hierarchical_training_mask(gaussians, view, expand_factor, args,
                                       hier_mode, lod_range_mask=None,
                                       profile=None):
    opacity_scale = None
    active_indices = None
    use_indices = True
    lod_min = getattr(args, 'render_lod_min', None) or 1
    lod_max = getattr(args, 'render_lod_max', None) or gaussians.current_lod
    if not 1 <= lod_min <= lod_max <= gaussians.current_lod:
        raise ValueError("Render LoD range must be within the loaded hierarchy")
    if lod_min == lod_max and lod_min > 1:
        active_indices = gaussians._indices_at_lod(lod_min)
        near = getattr(args, 'near', 0.0)
        if near > 0:
            active_indices = _frustum_near_indices(gaussians, view, active_indices, near)
        return None, active_indices, None
    if lod_min > 1 and hier_mode not in ('leaf_colors', 'orphans', 'lod_set'):
        raise ValueError("Adaptive traversal supports cumulative prefixes (min=1) or a single birth LoD (min=max)")

    def _profiled_core(fn):
        t_core = _profile_start(profile)
        result = fn()
        _profile_stop(profile, "select_core_s", t_core)
        return result

    def _filter_indices(indices):
        if lod_range_mask is not None and indices is not None and indices.numel() > 0:
            t_lod = _profile_start(profile)
            indices = indices[lod_range_mask[indices]]
            _profile_stop(profile, "filter_lod_s", t_lod)
        near = getattr(args, 'near', 0.0)
        if near is not None and near > 0.0 and indices is not None and indices.numel() > 0:
            t_near = _profile_start(profile)
            indices = _frustum_near_indices(gaussians, view, indices, near)
            _profile_stop(profile, "filter_near_s", t_near)
        return indices

    if hier_mode == "leaf_colors":
        active_mask = _profiled_core(lambda: _hierarchy_leaf_mask(gaussians))
    elif hier_mode == "orphans":
        active_mask = _profiled_core(lambda: _hierarchy_orphan_mask(gaussians))
    elif hier_mode == "lod_set":
        active_mask = _profiled_core(
            lambda: torch.ones(gaussians._xyz.shape[0], dtype=torch.bool, device=gaussians._xyz.device))
    elif hier_mode in ("sibling_groups", "family_colors"):
        active_mode = getattr(args, 'hierarchy_color_active_mode', 'leafes_to_coarse')
        if active_mode == "all":
            active_mask = _profiled_core(
                lambda: torch.ones(gaussians._xyz.shape[0], dtype=torch.bool, device=gaussians._xyz.device))
        else:
            active_indices = _profiled_core(lambda: gaussians.get_active_indices(
                view, expand_factor, mode=active_mode,
                prefix_prob=getattr(args, 'prefix_prob_render', 0.0),
                current_prob=getattr(args, 'current_prob_render', 0.0),
                prefix_min_lod=getattr(args, 'prefix_min_lod_render', 1),
                profile=profile,
                low_pass=getattr(args, 'low_pass', 0.3),
                cull_near=getattr(args, 'cull_near', 0.0),
                cull_frustum=getattr(args, 'cull_frustum', 0.0),
                cull_backend=getattr(args, 'renderer', 'inria'),
                max_lod=lod_max))
            active_indices = _filter_indices(active_indices)
            if use_indices:
                active_mask = None
            else:
                t_mask = _profile_start(profile)
                active_mask = gaussians.active_indices_to_mask(active_indices)
                _profile_stop(profile, "indices_to_mask_s", t_mask)
    elif hier_mode == "train_hits":
        active_mode = getattr(args, 'train_hit_active_mode', 'leafes_to_coarse')
        if active_mode == "all":
            active_mask = _profiled_core(
                lambda: torch.ones(gaussians._xyz.shape[0], dtype=torch.bool, device=gaussians._xyz.device))
        else:
            active_indices = _profiled_core(lambda: gaussians.get_active_indices(
                view, expand_factor, mode=active_mode,
                prefix_prob=getattr(args, 'prefix_prob_render', 0.0),
                current_prob=getattr(args, 'current_prob_render', 0.0),
                prefix_min_lod=getattr(args, 'prefix_min_lod_render', 1),
                profile=profile,
                low_pass=getattr(args, 'low_pass', 0.3),
                cull_near=getattr(args, 'cull_near', 0.0),
                cull_frustum=getattr(args, 'cull_frustum', 0.0),
                cull_backend=getattr(args, 'renderer', 'inria'),
                max_lod=lod_max))
            active_indices = _filter_indices(active_indices)
            if use_indices:
                active_mask = None
            else:
                t_mask = _profile_start(profile)
                active_mask = gaussians.active_indices_to_mask(active_indices)
                _profile_stop(profile, "indices_to_mask_s", t_mask)
    else:
        active_indices = _profiled_core(lambda: gaussians.get_active_indices(
            view, expand_factor, mode=hier_mode,
            prefix_prob=getattr(args, 'prefix_prob_render', 0.0),
            current_prob=getattr(args, 'current_prob_render', 0.0),
            prefix_min_lod=getattr(args, 'prefix_min_lod_render', 1),
            profile=profile,
            low_pass=getattr(args, 'low_pass', 0.3),
            cull_near=getattr(args, 'cull_near', 0.0),
            cull_frustum=getattr(args, 'cull_frustum', 0.0),
            cull_backend=getattr(args, 'renderer', 'inria'),
            max_lod=lod_max))
        active_indices = _filter_indices(active_indices)
        if use_indices:
            active_mask = None
        else:
            t_mask = _profile_start(profile)
            active_mask = gaussians.active_indices_to_mask(active_indices)
            _profile_stop(profile, "indices_to_mask_s", t_mask)

    if active_indices is None:
        near = getattr(args, 'near', 0.0)
        if near is not None and near > 0.0:
            t_near = _profile_start(profile)
            near_mask = _frustum_near_mask(gaussians, view, near)
            _profile_stop(profile, "filter_near_s", t_near)
            active_mask = _combine_masks(active_mask, near_mask)
        if lod_range_mask is not None:
            t_lod = _profile_start(profile)
            active_mask = _combine_masks(active_mask, lod_range_mask)
            _profile_stop(profile, "filter_lod_s", t_lod)
        if use_indices:
            t_mask = _profile_start(profile)
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            active_mask = None
            _profile_stop(profile, "mask_to_indices_s", t_mask)

    return active_mask, active_indices, opacity_scale

def _build_hierarchical_gaussians(dataset, gaussian_model_args, args):
    
    # Load a flat hierarchical GaussianModel for the hierarchical render paths

    gaussians = GaussianModel(**gaussian_model_args)
    gaussians.load(model_path=dataset.model_path, load_iteration=args.iteration,
                   load_lod=args.lod, device="cpu",
                   lod_cap=None if getattr(args, "posthoc_hierarchy", None) else getattr(args, "render_lod_max", None))

    pc_path = os.path.join(dataset.model_path, "point_cloud",
                           f"lod_{gaussians.loaded_lod}_iteration_{gaussians.loaded_iter}")
    meta_path = os.path.join(pc_path, "hierarchy_meta.pt")
    posthoc_hierarchy = getattr(args, 'posthoc_hierarchy', None)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"HLoD rendering requires hierarchy_meta.pt: {meta_path}")

    if posthoc_hierarchy:
        posthoc_path = _resolve_posthoc_hierarchy_path(dataset.model_path, posthoc_hierarchy)
        if not os.path.exists(posthoc_path):
            raise FileNotFoundError(f"--posthoc_hierarchy not found: {posthoc_path}")
        _apply_posthoc_hierarchy(gaussians, posthoc_path)
        tag = getattr(args, 'posthoc_hierarchy_tag', None)
        if tag is None:
            tag = os.path.splitext(os.path.basename(posthoc_path))[0]
        args._posthoc_hierarchy_tag_resolved = _sanitize_tag(tag)

    cap = getattr(args, 'render_lod_max', None)
    if cap is not None and gaussians.current_lod > cap:
        gaussians.retain_indices((gaussians._birth_lod <= cap).nonzero(as_tuple=True)[0])
        gaussians.current_lod = cap
    gaussians.to_cuda()
    gaussians._apply_mip_floor = bool(dataset.mip_floor)
    # Shapes are static at render time, so cache the [N,3,3] shape factors
    # (build_rotation(rotation) * scaling) instead of rebuilding them for every
    # Gaussian on every frame inside the per-view LOD selection.
    gaussians.enable_shape_factors_cache(True)
    return gaussians


@torch.no_grad()
def render_sets_hierarchical_training(args, model, pipeline):
    dataset = model.extract(args)
    pipeline_params = pipeline.extract(args)

    gaussian_model_args = {
        'sh_degree': dataset.sh_degree,
        'lod1_scaling_lower_bound': dataset.lod1_scaling_lower_bound,
        'lod_scaling_ratio': dataset.lod_scaling_ratio,
        'increase_lod_num_childs': dataset.increase_lod_num_childs,
        'current_lod': args.lod,
        'max_lod': args.lod,
    }
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = _build_hierarchical_gaussians(dataset, gaussian_model_args, args)
    expand_factor = args.sp_thresh

    if not args.skip_train:
        scene = Scene(dataset, gaussians=None, load_iteration=args.iteration,
                      load_lod=args.lod, shuffle=False, load_only_test_images=False,
                      load_image_device=args.load_image_device, load_ply=False)
        render_set_hierarchical_training(
            dataset.model_path, "train", scene.loaded_iter,
            scene.getTrainCameras(), pipeline_params, background,
            gaussians, expand_factor, args)
            
    if not args.skip_test:
        scene = Scene(dataset, gaussians=None, load_iteration=args.iteration,
                      load_lod=args.lod, shuffle=False, load_only_test_images=True,
                      load_image_device=args.load_image_device, load_ply=False)
        test_cameras = scene.getTestCameras()
        if test_cameras:
            render_set_hierarchical_training(
                dataset.model_path, "test", scene.loaded_iter,
                test_cameras, pipeline_params, background,
                gaussians, expand_factor, args)

def render_set_hierarchical_training(model_path, name, iteration, views, pipeline,
                                     background, gaussians, expand_factor, args):
    lod = gaussians.current_lod
    hier_mode = getattr(args, 'hierarchical_mode', 'leafes_to_coarse')
    metric_timing = getattr(args, 'metric_timing', 'selection_render')
    if metric_timing not in ('selection_render', 'render_only'):
        raise ValueError(f"unknown metric_timing: {metric_timing}")

    mode_tag = f"_{hier_mode}"
    metric_tag = ""
    color_active_mode = getattr(args, 'hierarchy_color_active_mode', 'leafes_to_coarse')
    if hier_mode in ("sibling_groups", "family_colors") and color_active_mode != "leafes_to_coarse":
        mode_tag += f"_{color_active_mode}"
    posthoc_tag = getattr(args, '_posthoc_hierarchy_tag_resolved', None)
    if posthoc_tag:
        mode_tag += f"_posthoc-{posthoc_tag}"

    lod_min = getattr(args, 'render_lod_min', None)
    lod_max = getattr(args, 'render_lod_max', None)
    if lod_min is None: lod_min = 1
    if lod_max is None: lod_max = lod
    if lod_min > lod_max:
        raise ValueError(f"--render_lod_min ({lod_min}) > --render_lod_max ({lod_max})")
    lod_filter_active = (lod_min > 1) or (lod_max < lod)
    if lod_filter_active:
        lod_tag = f"_lod{lod_min}" if lod_min == lod_max else f"_lod{lod_min}-{lod_max}"
    else:
        lod_tag = ""

    birth_lod = gaussians._birth_lod
    lod_range_mask = (birth_lod >= lod_min) & (birth_lod <= lod_max) if lod_filter_active else None

    near = getattr(args, 'near', 0.0)
    cull_near = getattr(args, 'cull_near', 0.0)
    cull_frustum = getattr(args, 'cull_frustum', 0.0)
    opacity_min = getattr(args, 'render_opacity_min', 0.0)
    _res = getattr(args, 'resolution', -1)
    res_tag = f"_r{int(_res)}" if _res and _res > 0 else ""
    render_name = f"hlod{lod}_sp{expand_factor}{metric_tag}{mode_tag}{lod_tag}{_cull_tag(cull_near)}{_cull_frustum_tag(cull_frustum)}{_near_tag(near)}{_opacity_tag(opacity_min)}{res_tag}"
    render_path = os.path.join(model_path, name, "renders", render_name)
    gt_path = os.path.join(model_path, name, f"gt{res_tag}")
    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)

    heatmap_path = None
    if args.lod_heatmap:
        heatmap_path = os.path.join(model_path, name, "renders", f"{render_name}_heatmap")
        makedirs(heatmap_path, exist_ok=True)
        lod_colors = _make_lod_colors(gaussians)
    debug_colors = _make_hierarchy_debug_colors(gaussians, args, hier_mode)

    rendering_times, gnums, visible_gnums = [], [], []
    psnrs_list, ssims_list, lpipss_list = [], [], []

    max_lod = int(gaussians._birth_lod.max().item()) if gaussians._birth_lod is not None else lod
    traversal_rows = []
    profile_enabled = bool(getattr(args, 'profile', False))
    profile_rows = []

    video_only = bool(getattr(args, 'save_video', False) and getattr(args, 'video_only', False))
    eval_views = [] if video_only else views

    # VRAM instrumentation: measure the render() peak in isolation from the VGG
    # LPIPS pass. Baseline = resident hierarchy before any frame.
    render_peaks = []
    run_peak_bytes = 0
    if not video_only:
        lpips_criterion('vgg', '0.1', str(background.device))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_model_bytes = torch.cuda.memory_allocated()

    for idx, view in enumerate(tqdm(eval_views, desc=f"Rendering {render_name}")):
        profile = {"view_idx": idx} if profile_enabled else None
        t_frame = _profile_start(profile)
        if profile_enabled:
            t_start = t_frame
            t_select = t_start
        else:
            torch.cuda.synchronize()
            t_start = time.time()
            t_select = None

        active_mask, active_indices, opacity_scale = _select_hierarchical_training_mask(
            gaussians, view, expand_factor, args, hier_mode, lod_range_mask,
            profile=profile)
        active_mask, active_indices = _apply_render_opacity_filter(
            gaussians, active_mask, active_indices, opacity_min, profile=profile)
        _profile_stop(profile, "selection_total_s", t_select)
        render_select = {'active_indices': active_indices} if active_indices is not None else {'active_mask': active_mask}
        t_render = None
        if metric_timing == 'render_only':
            if profile_enabled:
                t_render = _profile_start(profile)
                t_start = t_render
            else:
                torch.cuda.synchronize()
                t_start = time.time()
        if t_render is None:
            t_render = _profile_start(profile)
        torch.cuda.reset_peak_memory_stats()
        rasterize_results = render(view, gaussians, pipeline, background,
                                   **render_select, opacity_scale=opacity_scale,
                                   override_color=debug_colors)
        render_peaks.append(torch.cuda.max_memory_allocated())

        if profile_enabled:
            t_end = _profile_stop(profile, "render_s", t_render)
        else:
            torch.cuda.synchronize()
            t_end = time.time()
        if profile is not None:
            profile["metric_timed_s"] = t_end - t_start

        t_lod_count = _profile_start(profile)
        if active_indices is not None:
            active_total = int(active_indices.numel())
            selected_lods = gaussians._birth_lod[active_indices].to(torch.long)
        else:
            active_total = int(active_mask.sum().item())
            selected_lods = gaussians._birth_lod[active_mask].to(torch.long)
        lod_counts = torch.bincount(selected_lods, minlength=max_lod + 1)
        per_lod = [int(v) for v in lod_counts[1:max_lod + 1].detach().cpu().tolist()]
        # Post-rasterizer survivor count (radii > 0)
        visible_total = int(rasterize_results["visibility_filter"].sum().item())
        _profile_stop(profile, "lod_count_s", t_lod_count)
        if profile is not None:
            profile["active_total"] = active_total
            profile["visible_total"] = visible_total
            profile["per_lod"] = per_lod
        traversal_rows.append((idx, active_total, visible_total, per_lod))

        t_post = _profile_start(profile)
        rendering = torch.clamp(rasterize_results["render"], 0.0, 1.0)
        if hasattr(view, "alpha_mask"):
            rendering *= view.alpha_mask.cuda()
        gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)
        _profile_stop(profile, "postprocess_s", t_post)

        t_quality = _profile_start(profile)
        ssims_list.append(ssim(rendering, gt.cuda()).mean().item())
        psnrs_list.append(psnr(rendering, gt.cuda()).mean().item())
        lpipss_list.append(lpips(rendering, gt.cuda(), net_type='vgg').mean().item())
        _profile_stop(profile, "quality_metrics_s", t_quality)
        rendering_times.append(t_end - t_start)
        gnums.append(active_total)
        visible_gnums.append(visible_total)

        t_save = _profile_start(profile)
        torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
        torchvision.utils.save_image(gt, os.path.join(gt_path, f'{idx:05d}.png'))
        _profile_stop(profile, "save_images_s", t_save)

        if heatmap_path is not None:
            t_heatmap = _profile_start(profile)
            hm_result = render(view, gaussians, pipeline, background,
                               **render_select, override_color=lod_colors,
                               opacity_scale=opacity_scale)
            hm_img = torch.clamp(hm_result["render"], 0.0, 1.0)
            torchvision.utils.save_image(hm_img, os.path.join(heatmap_path, f'{idx:05d}.png'))
            _profile_stop(profile, "heatmap_s", t_heatmap)
        _profile_stop(profile, "frame_total_s", t_frame)
        run_peak_bytes = max(run_peak_bytes, torch.cuda.max_memory_allocated())
        if profile is not None:
            profile_rows.append(profile)

    if not video_only:
        active_extra = ""
        if hier_mode in ("sibling_groups", "family_colors"):
            active_extra = f", active_mode={color_active_mode}"
        renderer_name = str(getattr(args, "renderer", "inria"))
        extra = f"renderer={renderer_name}, expand_factor={expand_factor}, metric=cuda_radius, mode={hier_mode}{active_extra}, near={near:g}, render_opacity_min={opacity_min:g}, timing={metric_timing}"
        _write_metrics(model_path, name, render_name, rendering_times, gnums,
                       psnrs_list, ssims_list, lpipss_list, extra=extra,
                       visible_gnums=visible_gnums, mem_model_bytes=mem_model_bytes,
                       render_peaks=render_peaks, run_peak_bytes=run_peak_bytes)

        log_path = os.path.join(model_path, name, "renders", f"{render_name}_traversal.csv")
        with open(log_path, "w") as f:
            f.write("view_idx,active_total,visible_total," + ",".join(f"lod{L}" for L in range(1, max_lod + 1)) + "\n")
            for idx, total, visible_total, per_lod in traversal_rows:
                f.write(f"{idx},{total},{visible_total}," + ",".join(str(c) for c in per_lod) + "\n")
        print(f"Traversal log: {log_path}")
        _write_hierarchical_profile(model_path, name, render_name, profile_rows, max_lod)

    if args.save_video:
        video_dir = os.path.join(model_path, name, "renders", "videos")
        makedirs(video_dir, exist_ok=True)
        video_cameras = generate_path(views, n_frames=args.n_frames,
                                      trajectory=args.trajectory,
                                      radius_offset=args.orbit_radius_offset,
                                      dolly_distance=getattr(args, 'dolly_distance', 0.0),
                                      dolly_view_index=getattr(args, 'dolly_view_index', 0))
        frames = []
        heatmap_frames = [] if args.lod_heatmap else None
        debug_colors = _make_hierarchy_debug_colors(gaussians, args, hier_mode)
        video_traversal = []
        for f_idx, view in enumerate(tqdm(video_cameras, desc="Rendering video")):
            active_mask, active_indices, opacity_scale = _select_hierarchical_training_mask(
                gaussians, view, expand_factor, args, hier_mode, lod_range_mask)
            active_mask, active_indices = _apply_render_opacity_filter(
                gaussians, active_mask, active_indices, opacity_min)
            render_select = {'active_indices': active_indices} if active_indices is not None else {'active_mask': active_mask}
            result = render(view, gaussians, pipeline, background,
                            **render_select, opacity_scale=opacity_scale,
                            override_color=debug_colors)
            frame = torch.clamp(result["render"], 0.0, 1.0)
            frames.append((frame.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))
            if active_indices is not None:
                active_total = int(active_indices.numel())
                selected_lods = gaussians._birth_lod[active_indices].to(torch.long)
            else:
                active_total = int(active_mask.sum().item())
                selected_lods = gaussians._birth_lod[active_mask].to(torch.long)
            lod_counts = torch.bincount(selected_lods, minlength=max_lod + 1)
            per_lod = [int(v) for v in lod_counts[1:max_lod + 1].detach().cpu().tolist()]
            video_traversal.append((f_idx, active_total, per_lod))
            if heatmap_frames is not None:
                hm = render(view, gaussians, pipeline, background,
                            **render_select, override_color=lod_colors,
                            opacity_scale=opacity_scale)
                hm_img = torch.clamp(hm["render"], 0.0, 1.0)
                heatmap_frames.append((hm_img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))

        create_video_from_frames(frames, out_path=os.path.join(video_dir, f"{render_name}.mp4"))
        if heatmap_frames:
            create_video_from_frames(heatmap_frames,
                                     out_path=os.path.join(video_dir, f"{render_name}_heatmap.mp4"))

        video_log_path = os.path.join(video_dir, f"{render_name}_traversal.csv")
        with open(video_log_path, "w") as f:
            f.write("frame_idx,active_total," + ",".join(f"lod{L}" for L in range(1, max_lod + 1)) + "\n")
            for f_idx, total, per_lod in video_traversal:
                f.write(f"{f_idx},{total}," + ",".join(str(c) for c in per_lod) + "\n")
        print(f"Video traversal log: {video_log_path}")


def _load_single_lod_gaussians(model_path, iteration, render_lod, gaussian_model_args):

    g = GaussianModel(**gaussian_model_args)
    g.load_level(model_path, render_lod, iteration, device="cpu", source_lod=gaussian_model_args["max_lod"])
    g.to_cuda()
    return g


def _write_n_tiles_metrics(model_path, name, render_name, n_tiles_list, gnums,
                           render_times, extra=""):
    """Append the aggregated N-tiles metric to <split>/n_tiles.txt and print it."""
    n_tiles_arr = np.asarray(n_tiles_list, dtype=np.float64)
    gnum_arr = np.asarray(gnums, dtype=np.float64)
    mean_tiles = float(n_tiles_arr.mean()) if n_tiles_arr.size else 0.0
    max_tiles = float(n_tiles_arr.max()) if n_tiles_arr.size else 0.0
    min_tiles = float(n_tiles_arr.min()) if n_tiles_arr.size else 0.0
    total_tiles = float(n_tiles_arr.sum())
    mean_gnum = float(gnum_arr.mean()) if gnum_arr.size else 0.0
    max_gnum = float(gnum_arr.max()) if gnum_arr.size else 0.0
    tiles_per_gauss = (mean_tiles / mean_gnum) if mean_gnum > 0 else 0.0

    if len(render_times) > 5:
        fps = 1.0 / (sum(render_times[5:]) / len(render_times[5:]))
    elif render_times:
        fps = 1.0 / (sum(render_times) / len(render_times))
    else:
        fps = 0.0

    gpu_name = torch.cuda.get_device_name()
    memory_used = torch.cuda.max_memory_allocated() / 1024**2

    makedirs(os.path.join(model_path, name), exist_ok=True)
    with open(os.path.join(model_path, name, "n_tiles.txt"), "a") as f:
        extra_str = f" / {extra}" if extra else ""
        f.write(f"\n({gpu_name} {render_name}) "
                f"mean N tiles: {mean_tiles:.1f} / max: {max_tiles:.0f} / "
                f"min: {min_tiles:.0f} / total: {total_tiles:.0f}\n")
        f.write(f"mean Gnum: {mean_gnum:.0f} / max Gnum: {max_gnum:.0f} / "
                f"tiles/Gaussian: {tiles_per_gauss:.2f} / peak memory {memory_used:.0f}MB / "
                f"FPS: {fps:.3f}{extra_str}\n")

    print(f"\n[{render_name}] mean N tiles: {mean_tiles:.1f} | max: {max_tiles:.0f} | "
          f"avg {mean_gnum:.0f} Gaussians/frame | {tiles_per_gauss:.2f} tiles/Gaussian | FPS: {fps:.1f}")


def _write_n_tiles_csv(model_path, name, render_name, rows, header):
    """Write a per-view N-tiles CSV under <split>/renders/<render_name>_ntiles.csv."""
    out_dir = os.path.join(model_path, name, "renders")
    makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{render_name}_ntiles.csv")
    with open(log_path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"N tiles log: {log_path}")


@torch.no_grad()
def render_sets_n_tiles(args, model, pipeline):

    from gaussian_renderer.ntiles import render_ntiles

    dataset = model.extract(args)
    pipeline_params = pipeline.extract(args)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    source = getattr(args, 'n_tiles_source', 'single_level')
    near = getattr(args, 'near', 0.0)
    _res = getattr(args, 'resolution', -1)
    res_tag = f"_r{int(_res)}" if _res and _res > 0 else ""

    if source == 'single_level':
        gaussian_model_args = {
            'sh_degree': dataset.sh_degree,
            'lod1_scaling_lower_bound': dataset.lod1_scaling_lower_bound,
            'lod_scaling_ratio': dataset.lod_scaling_ratio,
            'increase_lod_num_childs': dataset.increase_lod_num_childs,
            'current_lod': args.render_lod, 'max_lod': args.lod,
        }
        for split, skip in [("train", args.skip_train), ("test", args.skip_test)]:
            if skip:
                continue
            scene = Scene(dataset, gaussians=None, load_iteration=args.iteration,
                          load_lod=args.lod, shuffle=False,
                          load_only_test_images=(split == "test"),
                          load_image_device=args.load_image_device, load_ply=False)
            cameras = scene.getTestCameras() if split == "test" else scene.getTrainCameras()
            if not cameras:
                continue
            gaussians = _load_single_lod_gaussians(
                dataset.model_path, args.iteration, args.render_lod, gaussian_model_args)
            render_name = f"ntiles_level{args.render_lod}{_near_tag(near)}{res_tag}"
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            n_tiles_list, gnums, render_times, rows = [], [], [], []
            for idx, view in enumerate(tqdm(cameras, desc=f"N tiles {render_name}")):
                near_mask = _frustum_near_mask(gaussians, view, near)
                torch.cuda.synchronize()
                t0 = time.time()
                if near_mask is not None:
                    res = render_ntiles(view, gaussians, pipeline_params, bg, active_mask=near_mask)
                else:
                    res = render_ntiles(view, gaussians, pipeline_params, bg)
                torch.cuda.synchronize()
                t1 = time.time()
                gnum = int(near_mask.sum().item()) if near_mask is not None else gaussians._xyz.shape[0]
                n_tiles_list.append(res["n_tiles"])
                gnums.append(gnum)
                render_times.append(t1 - t0)
                rows.append((idx, res["n_tiles"], gnum))
            extra = f"source=single_level, render_lod={args.render_lod}, near={near:g}"
            _write_n_tiles_metrics(dataset.model_path, split, render_name,
                                   n_tiles_list, gnums, render_times, extra=extra)
            _write_n_tiles_csv(dataset.model_path, split, render_name, rows,
                               header="view_idx,n_tiles,gnum")
        return

    if source != 'hierarchical':
        raise ValueError(f"unknown --n_tiles_source: {source}")

    gaussian_model_args = {
        'sh_degree': dataset.sh_degree,
        'lod1_scaling_lower_bound': dataset.lod1_scaling_lower_bound,
        'lod_scaling_ratio': dataset.lod_scaling_ratio,
        'increase_lod_num_childs': dataset.increase_lod_num_childs,
        'current_lod': args.lod, 'max_lod': args.lod,
    }
    gaussians = _build_hierarchical_gaussians(dataset, gaussian_model_args, args)
    expand_factor = args.sp_thresh
    hier_mode = getattr(args, 'hierarchical_mode', 'leafes_to_coarse')
    opacity_min = getattr(args, 'render_opacity_min', 0.0)
    lod = gaussians.current_lod
    max_lod = int(gaussians._birth_lod.max().item()) if gaussians._birth_lod is not None else lod

    lod_min = getattr(args, 'render_lod_min', None)
    lod_max = getattr(args, 'render_lod_max', None)
    if lod_min is None:
        lod_min = 1
    if lod_max is None:
        lod_max = lod
    if lod_min > lod_max:
        raise ValueError(f"--render_lod_min ({lod_min}) > --render_lod_max ({lod_max})")
    lod_filter_active = (lod_min > 1) or (lod_max < lod)
    birth_lod = gaussians._birth_lod
    lod_range_mask = (birth_lod >= lod_min) & (birth_lod <= lod_max) if lod_filter_active else None
    lod_tag = ""
    if lod_filter_active:
        lod_tag = f"_lod{lod_min}" if lod_min == lod_max else f"_lod{lod_min}-{lod_max}"

    for split, skip in [("train", args.skip_train), ("test", args.skip_test)]:
        if skip:
            continue
        scene = Scene(dataset, gaussians=None, load_iteration=args.iteration,
                      load_lod=args.lod, shuffle=False,
                      load_only_test_images=(split == "test"),
                      load_image_device=args.load_image_device, load_ply=False)
        cameras = scene.getTestCameras() if split == "test" else scene.getTrainCameras()
        if not cameras:
            continue
        render_name = (f"ntiles_hier_lod{lod}_ef{expand_factor}_{hier_mode}"
                       f"{lod_tag}{_near_tag(near)}{_opacity_tag(opacity_min)}{res_tag}")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        n_tiles_list, gnums, render_times, rows = [], [], [], []
        for idx, view in enumerate(tqdm(cameras, desc=f"N tiles {render_name}")):
            active_mask, active_indices, opacity_scale = _select_hierarchical_training_mask(
                gaussians, view, expand_factor, args, hier_mode, lod_range_mask)
            active_mask, active_indices = _apply_render_opacity_filter(
                gaussians, active_mask, active_indices, opacity_min)
            render_select = ({'active_indices': active_indices} if active_indices is not None
                             else {'active_mask': active_mask})
            torch.cuda.synchronize()
            t0 = time.time()
            res = render_ntiles(view, gaussians, pipeline_params, bg,
                                opacity_scale=opacity_scale, **render_select)
            torch.cuda.synchronize()
            t1 = time.time()
            if active_indices is not None:
                gnum = int(active_indices.numel())
                selected_lods = gaussians._birth_lod[active_indices].to(torch.long)
            else:
                gnum = int(active_mask.sum().item())
                selected_lods = gaussians._birth_lod[active_mask].to(torch.long)
            lod_counts = torch.bincount(selected_lods, minlength=max_lod + 1)
            per_lod = [int(v) for v in lod_counts[1:max_lod + 1].detach().cpu().tolist()]
            n_tiles_list.append(res["n_tiles"])
            gnums.append(gnum)
            render_times.append(t1 - t0)
            rows.append((idx, res["n_tiles"], gnum) + tuple(per_lod))
        extra = (f"source=hierarchical, mode={hier_mode}, expand_factor={expand_factor}, "
                 f"near={near:g}, render_opacity_min={opacity_min:g}")
        _write_n_tiles_metrics(dataset.model_path, split, render_name,
                               n_tiles_list, gnums, render_times, extra=extra)
        header = "view_idx,n_tiles,gnum," + ",".join(f"lod{L}" for L in range(1, max_lod + 1))
        _write_n_tiles_csv(dataset.model_path, split, render_name, rows, header=header)


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument('--render_mode', choices=['hierarchical_training', 'single_level', 'n_tiles'], help='Rendering mode to use.', default = 'hierarchical_training')
    parser.add_argument('--n_tiles_source', choices=['single_level', 'hierarchical'], default='hierarchical',
                        help='n_tiles mode: which per-view Gaussian set to measure. '
                             'single_level measures a fixed LOD (--render_lod); '
                             'hierarchical measures the per-view active selection '
                             '(uses --sp_thresh, --hierarchical_mode, --render_lod_min/max, etc.).')
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--lod', default=5, type=int, help="The maximum LOD value set during training")
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--load_image_device', default='cuda', type=str)
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--video_only', action='store_true',
                        help='With --save_video: skip per-view eval/metrics and only render the trajectory video.')
    parser.add_argument('--n_frames', default=240, type=int)
    parser.add_argument('--trajectory', choices=['ellipse', 'orbit', 'dolly'], default='ellipse')
    parser.add_argument('--orbit_radius_offset', type=float, default=0.0)
    parser.add_argument('--dolly_distance', type=float, default=0.0,
                        help='Forward travel distance for dolly trajectory (world units). 0 = median scene radius.')
    parser.add_argument('--dolly_view_index', type=int, default=0,
                        help='Which input view to start the dolly from.')

    parser.add_argument('--render_lod', default=5, type=int)

    parser.add_argument('--sp_thresh', type=float, default=DEFAULT_SP_THRESH,
                        help='Screen-space traversal threshold in pixels. Inherits the saved training value; otherwise defaults to 3.')
    parser.add_argument('--hierarchical_mode', type=str, default='leafes_to_coarse',
                        choices=['leafes_to_coarse', 'fine_to_coarse',
                                 'coarse_to_fine',
                                 'leaf_colors', 'sibling_groups', 'family_colors',
                                 'train_hits', 'orphans', 'lod_set'])
    parser.add_argument('--metric_timing', '--hierarchical_metric_timing',
                        dest='metric_timing', type=str, default='selection_render',
                        choices=['selection_render', 'render_only'],
                        help='How FPS is measured. selection_render times the per-view '
                             'active-Gaussian selection plus rasterization; render_only '
                             'times only rasterization. single_level performs no per-view '
                             'selection, so it is always render-only. '
                             '(--hierarchical_metric_timing is a deprecated alias.)')
    parser.add_argument('--profile', '--hierarchical_profile',
                        dest='profile', action='store_true',
                        help='Write a per-view timing CSV. Adds CUDA synchronizations, so '
                             'use it for diagnosis rather than FPS reporting. '
                             '(--hierarchical_profile is a deprecated alias.)')
    parser.add_argument('--prefix_prob_render', type=float, default=0.0,
                        help='Probability of a random-prefix traversal at render time: run the '
                             'normal adaptive traversal but treat a random LoD in '
                             '[--prefix_min_lod_render, current LoD] as the finest level.')
    parser.add_argument('--current_prob_render', type=float, default=0.0,
                        help='Probability of rendering only the current (finest) LoD. Checked '
                             'before --prefix_prob_render; the two must sum to <= 1.0.')
    parser.add_argument('--prefix_min_lod_render', type=int, default=1,
                        help='Lowest LoD considered by the random-prefix traversal at render time.')
    parser.add_argument('--lod_heatmap', action='store_true')
    parser.add_argument('--leaf_color_by', choices=['id', 'lod'], default='id',
                        help='leaf_colors mode: color each leaf by stable Gaussian id or by birth LoD.')
    parser.add_argument('--hierarchy_color_active_mode', type=str, default='leafes_to_coarse',
                        choices=['all', 'leafes_to_coarse', 'fine_to_coarse'],
                        help='sibling_groups/family_colors mode: traversal used to choose active Gaussians. all renders every Gaussian.')
    parser.add_argument('--train_hit_active_mode', type=str, default='leafes_to_coarse',
                        help='train_hits mode: active set to color. Use all to render every Gaussian.')
    parser.add_argument('--train_hit_color_max', type=float, default=0.0,
                        help='train_hits mode: hit count mapped to full red. 0 uses percentile normalization.')
    parser.add_argument('--train_hit_color_percentile', type=float, default=99.0,
                        help='train_hits mode: positive-hit percentile mapped to full red when max is 0.')
    parser.add_argument('--render_lod_min', type=int, default=None,
                        help='Cumulative traversal: use 1. Set min=max for a single birth LoD; other ranges are diagnostic-only.')
    parser.add_argument('--render_lod_max', type=int, default=None,
                        help='Load and traverse a prefix of the final hierarchy. Default: --lod. Set min=max for a single birth LoD.')
    parser.add_argument('--posthoc_hierarchy', type=str, default=None,
                        help='hierarchical_training: replace parent_idx with a build_hierarchy.py hierarchy file. Relative paths are resolved first from cwd, then from --model_path.')
    parser.add_argument('--posthoc_hierarchy_tag', type=str, default=None,
                        help='Optional short tag appended to output render names when --posthoc_hierarchy is used.')
    parser.add_argument('--near', type=float, default=0.0,
                        help='Frustum near-plane distance (world units, along view-space z). Gaussians with view-space z < near are dropped before rasterization.')
    parser.add_argument('--cull_near', type=float, default=0.0,
                        help='In-kernel near-cull for the HLoD traversal (0 = off). Gaussians with view-space z < cull_near are dropped during the traversal instead of being clamped and distorting the LoD choice.')
    parser.add_argument('--cull_frustum', type=float, default=1.0,
                        help='Post-traversal renderer-aware visibility cull (default: 1; 0 = off). Independent of the saved training setting. With INRIA the value is an extra pixel margin; with gsplat any positive value enables its exact projection visibility test.')
    parser.add_argument('--render_opacity_min', type=float, default=0.0,
                        help='Final render-time opacity cutoff. Keeps only active Gaussians with sigmoid(opacity) >= this value before rasterization.')

    args = get_combined_args(parser, exclude_saved=("cull_frustum",))

    print(f"Rendering {args.model_path} with mode {args.render_mode}")
    safe_state(args.quiet)

    # Renderer backend selection (default INRIA diff-gaussian-rasterization).
    if str(getattr(args, "renderer", "inria")) == "gsplat":
        from gaussian_renderer.gsplat_render import render as _gsplat_render
        globals()["render"] = _gsplat_render
        print("[renderer] using gsplat backend")

    if args.render_mode == 'single_level':
        render_sets_single_level(args, model, pipeline)
    elif args.render_mode == 'hierarchical_training':
        render_sets_hierarchical_training(args, model, pipeline)
    elif args.render_mode == 'n_tiles':
        render_sets_n_tiles(args, model, pipeline)
    else:
        raise ValueError(f"Unknown render_mode: {args.render_mode}")
