import os
import torch
import random
import sys
import uuid
import time
import csv
import shlex
from tqdm import tqdm
from argparse import ArgumentParser
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, DEFAULT_SP_THRESH
from utils.training_utils import expand_list_to_match_lods
from utils.maha_utils import compute_maha_regularizer
from utils.projection_utils import scale_upper_loss
from torch.utils.tensorboard import SummaryWriter
from utils.sh_utils import SH2RGB
from utils.config_utils import save_config

try:
    import rerun as rr
    _RR_AVAILABLE = True
except ImportError:
    _RR_AVAILABLE = False


TRAIN_PROFILE_FIELDS = [
    "phase",
    "lod",
    "iteration",
    "global_step",
    "num_gaussians_start",
    "num_gaussians_end",
    "active_total",
    "visible_total",
    "cuda_backend_requested",
    "cuda_backend_used",
    "setup_s",
    "selection_s",
    "subthreshold_s",
    "projected_size_s",
    "active_traversal_s",
    "render_forward_s",
    "debug_s",
    "reconstruction_loss_s",
    "regularizer_setup_s",
    "maha_loss_s",
    "scale_upper_loss_s",
    "backward_setup_s",
    "backward_s",
    "grad_postprocess_s",
    "logging_s",
    "save_s",
    "visibility_stats_s",
    "densify_prune_s",
    "optimizer_s",
    "lod_setup_s",
    "iteration_total_s",
    "loss",
]


def _train_profile_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _train_profile_start(row):
    if row is None:
        return None
    _train_profile_sync()
    return time.perf_counter()


def _train_profile_mark(row, key, start):
    if row is None or start is None:
        return start
    _train_profile_sync()
    now = time.perf_counter()
    row[key] = row.get(key, 0.0) + (now - start)
    return now


def _train_profile_elapsed(row, key, start):
    if row is None or start is None:
        return
    _train_profile_sync()
    row[key] = row.get(key, 0.0) + (time.perf_counter() - start)


def _train_profile_memory(row, gaussians):
    # Breakdown of GPU memory: model params, Adam states (exp_avg + exp_avg_sq),
    # grads, plus resident/peak/reserved totals. Peak captures the rasterizer
    # forward/backward buffers (reset at iteration start).
    if row is None or not torch.cuda.is_available():
        return
    param_bytes = grad_bytes = state_bytes = 0
    param_ids = set()
    opt = getattr(gaussians, "optimizer", None)
    if opt is not None:
        for group in opt.param_groups:
            for p in group["params"]:
                if p is None:
                    continue
                param_ids.add(id(p))
                param_bytes += p.numel() * p.element_size()
                if p.grad is not None:
                    grad_bytes += p.grad.numel() * p.grad.element_size()
        for st in opt.state.values():
            for v in st.values():
                if torch.is_tensor(v) and v.is_floating_point():
                    state_bytes += v.numel() * v.element_size()
    # Non-optimizer CUDA tensors on the model = hierarchy metadata + caches
    # (_parent_idx, _birth_lod, _per_gaussian_lb, hit counters, ...).
    def _t_bytes(t):
        return t.numel() * t.element_size() if (torch.is_tensor(t) and t.is_cuda
                                                and id(t) not in param_ids) else 0
    meta_bytes = 0
    for v in gaussians.__dict__.values():
        if torch.is_tensor(v):
            meta_bytes += _t_bytes(v)
        elif isinstance(v, dict):
            meta_bytes += sum(_t_bytes(x) for x in v.values())
        elif isinstance(v, (list, tuple)):
            meta_bytes += sum(_t_bytes(x) for x in v)
    mb = 1024.0 * 1024.0
    n = int(gaussians.get_xyz.shape[0])
    row["mem_params_mb"] = param_bytes / mb
    row["mem_adam_state_mb"] = state_bytes / mb
    row["mem_metadata_mb"] = meta_bytes / mb
    row["mem_grads_mb"] = grad_bytes / mb
    row["mem_resident_mb"] = torch.cuda.memory_allocated() / mb
    row["mem_peak_mb"] = torch.cuda.max_memory_allocated() / mb
    row["mem_reserved_mb"] = torch.cuda.memory_reserved() / mb
    row["mem_bytes_per_gaussian"] = (param_bytes / n) if n > 0 else 0.0


def _write_train_profile(path, rows):
    if not rows:
        return
    fieldnames = list(TRAIN_PROFILE_FIELDS)
    seen = set(fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rr_log_lineage(gaussians, selected_idx, step):
    with torch.no_grad():
        rr.set_time("iteration", sequence=step)
        n = gaussians.get_xyz.shape[0]
        if selected_idx < 0 or selected_idx >= n:
            rr.log("lineage", rr.Clear(recursive=True))
            rr.log("lineage/status", rr.TextLog(f"selected={selected_idx} invalid (n={n})"))
            return
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        birth_lod = gaussians._birth_lod.cpu().numpy() if gaussians._birth_lod is not None else None
        if gaussians._parent_idx is None:
            chain = [int(selected_idx)]
            msg = f"no hierarchy yet (LOD 1 only); selected idx={selected_idx}"
        else:
            parent_idx = gaussians._parent_idx.cpu().numpy()
            chain = [int(selected_idx)]
            cur = int(selected_idx)
            for _ in range(64):
                p = int(parent_idx[cur])
                if p < 0:
                    break
                chain.append(p)
                cur = p
            msg = f"lineage len={len(chain)} idx={selected_idx} -> root={chain[-1]}"
        chain_xyz = xyz[chain]
        if len(chain) >= 2:
            rr.log("lineage/path", rr.LineStrips3D([chain_xyz], colors=[[255, 230, 0]], radii=0.002))
        else:
            rr.log("lineage/path", rr.Clear(recursive=False))
        labels = [f"idx={i} lod={birth_lod[i] if birth_lod is not None else '?'}" for i in chain]
        rr.log("lineage/nodes", rr.Points3D(chain_xyz, colors=[[255, 230, 0]] * len(chain), radii=0.01, labels=labels))
        rr.log("lineage/status", rr.TextLog(msg))


def rr_log_gaussians(gaussians, step, current_lod, log_ellipsoids=False, max_ellipsoids=50000):
    with torch.no_grad():
        xyz_t = gaussians.get_xyz.detach()
        scales_t = gaussians.get_scaling.detach()
        sh_dc = gaussians._features_dc.detach().squeeze(1).cpu()
        rgb = SH2RGB(sh_dc).clamp(0.0, 1.0).numpy()
        colors = (rgb * 255).astype("uint8")
        xyz = xyz_t.cpu().numpy()
        scales = scales_t.cpu().numpy()
        radii = scales.mean(axis=1)
        rr.set_time("iteration", sequence=step)
        rr.log(f"gaussians/lod_{current_lod}", rr.Points3D(xyz, colors=colors, radii=radii))
        if log_ellipsoids:
            n = xyz.shape[0]
            if n > max_ellipsoids:
                idx = torch.randperm(n, device=xyz_t.device)[:max_ellipsoids].cpu().numpy()
                xyz_e = xyz[idx]; scales_e = scales[idx]; colors_e = colors[idx]
                quat_wxyz = gaussians.get_rotation.detach()[idx].cpu().numpy()
            else:
                xyz_e, scales_e, colors_e = xyz, scales, colors
                quat_wxyz = gaussians.get_rotation.detach().cpu().numpy()
            quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
            rr.log(
                f"gaussians_ellipsoids/lod_{current_lod}",
                rr.Ellipsoids3D(
                    centers=xyz_e,
                    half_sizes=scales_e,
                    quaternions=[rr.Quaternion(xyzw=q) for q in quat_xyzw],
                    colors=colors_e,
                ),
            )


def _screenspace_metric_label(opt):
    return "cuda_radius", "CUDA rasterizer radius"


def print_per_lod_screensize(gaussians, viewpoint_cam, opt, label, low_pass=0.3):
    with torch.no_grad():
        radius_px = gaussians._compute_projected_size_values(viewpoint_cam, low_pass=low_pass)
        threshold = float(getattr(opt, 'sp_thresh', DEFAULT_SP_THRESH))
        print(f"[DEBUG] per-LOD CUDA rasterizer radius (px) {label}, cam '{viewpoint_cam.image_name}' (threshold={threshold:.1f}):")
        if gaussians._birth_lod is not None:
            for lod_val in range(opt.lod_min, opt.lod_max + 1):
                mask = (gaussians._birth_lod == lod_val)
                n = int(mask.sum().item())
                if n == 0:
                    continue
                r = radius_px[mask]
                sub = int((r < threshold).sum().item())
                pct = 100.0 * sub / n
                print(f"  LOD{lod_val} (N={n}, below-threshold={sub} [{pct:.1f}%]): min={r.min().item():.2f}  mean={r.mean().item():.2f}  median={r.median().item():.2f}  max={r.max().item():.2f}")
        else:
            r = radius_px
            sub = int((r < threshold).sum().item())
            pct = 100.0 * sub / r.numel() if r.numel() > 0 else 0.0
            print(f"  all (N={r.numel()}, below-threshold={sub} [{pct:.1f}%]): min={r.min().item():.2f}  mean={r.mean().item():.2f}  median={r.median().item():.2f}  max={r.max().item():.2f}")


def print_per_lod_subpixel_summary(gaussians, train_cameras, opt, label, max_cams=8, low_pass=0.3):
    """Average below-threshold counts across multiple training cameras."""
    with torch.no_grad():
        threshold = float(getattr(opt, 'sp_thresh', DEFAULT_SP_THRESH))
        cams = train_cameras if len(train_cameras) <= max_cams else random.sample(train_cameras, max_cams)
        N = gaussians.get_xyz.shape[0]
        sub_counts = torch.zeros(N, dtype=torch.float32, device=gaussians.get_xyz.device)
        for cam in cams:
            r = gaussians._compute_projected_size_values(cam, low_pass=low_pass)
            sub_counts += (r < threshold).float()
        avg_subpx = sub_counts / len(cams)
        always_sub = (avg_subpx >= 0.99)
        ever_over = (avg_subpx <= 0.01)
        print(f"[SCREENSPACE] {label} | metric=cuda_radius, sampled {len(cams)} cams, threshold={threshold:.1f}px:")
        if gaussians._birth_lod is not None:
            for lod_val in range(opt.lod_min, opt.lod_max + 1):
                mask = (gaussians._birth_lod == lod_val)
                n = int(mask.sum().item())
                if n == 0:
                    continue
                always = int((mask & always_sub).sum().item())
                over = int((mask & ever_over).sum().item())
                mean_frac = float(avg_subpx[mask].mean().item())
                print(f"  LOD{lod_val}: N={n}  always-below={always} [{100*always/n:.1f}%]  always-over={over} [{100*over/n:.1f}%]  mean-below-frac={mean_frac:.2f}")
        else:
            n = N
            always = int(always_sub.sum().item())
            over = int(ever_over.sum().item())
            mean_frac = float(avg_subpx.mean().item())
            print(f"  all: N={n}  always-below={always} [{100*always/n:.1f}%]  always-over={over} [{100*over/n:.1f}%]  mean-below-frac={mean_frac:.2f}")

def training(dataset, opt, pipe, saving_iterations,
             debug_from, start_point_cloud=None, start_lod=None, run_args=None):
    if opt.lod_min != 1 or opt.lod_max < 1:
        raise ValueError("HLoD training requires lod_min=1 and lod_max>=1")
    opt.lod_iterations = expand_list_to_match_lods(opt.lod_iterations, opt.lod_max)
    if getattr(opt, "iterations", None) is not None:
        opt.lod_iterations = [opt.iterations] * opt.lod_max
    if any(value < 1 for value in opt.lod_iterations):
        raise ValueError("Every LoD must have at least one training iteration")
    global_step = 0

    # Renderer backend selection (default INRIA diff-gaussian-rasterization).
    if str(getattr(pipe, "renderer", "inria")) == "gsplat":
        from gaussian_renderer.gsplat_render import render as render
        print("[renderer] using gsplat backend"
              + (" (absgrad ON)" if getattr(pipe, "absgrad", False) else " (absgrad OFF)"))
        # AbsGS sums |grad| over the image plane (~4x the INRIA-style grad), so it
        # needs a ~4x higher densify threshold or it over-densifies. When absgrad is
        # on and the threshold is still the INRIA default (2e-4), auto-bump it to
        # 8e-4. Explicitly passing --densify_grad_threshold overrides this.
        if getattr(pipe, "absgrad", False) and opt.densify_grad_threshold == [0.0002]:
            opt.densify_grad_threshold = [0.0008]
            print("[renderer] gsplat+absgrad: densify_grad_threshold 2e-4 -> 8e-4 (AbsGS default)")
    else:
        from gaussian_renderer import render as render

    prepare_output(dataset, opt, pipe, run_args=run_args)
    tb_writer = SummaryWriter(log_dir=dataset.model_path)
    rr_enabled = _RR_AVAILABLE and os.environ.get("RERUN", "0") != "0"
    rr_every = int(os.environ.get("RERUN_EVERY", "100"))
    rr_ellipsoids = os.environ.get("RERUN_ELLIPSOIDS", "0") != "0"
    rr_ellipsoid_max = int(os.environ.get("RERUN_ELLIPSOID_MAX", "50000"))
    if rr_enabled:
        rr.init("hlod_training", spawn=True)

    gaussians = GaussianModel(
        sh_degree=dataset.sh_degree,
        lod1_scaling_lower_bound=dataset.lod1_scaling_lower_bound,
        lod_scaling_ratio=dataset.lod_scaling_ratio,
        increase_lod_num_childs=dataset.increase_lod_num_childs,
        current_lod=opt.lod_min,
        max_lod=opt.lod_max,
        use_voxel_sampling=dataset.use_voxel_sampling,
        voxel_sampling_size=dataset.voxel_sampling_size,
    )

    loaded_start_lod = None
    if start_point_cloud:
        scene = Scene(dataset, gaussians=None, load_image_device=dataset.data_device)
        if os.path.isdir(start_point_cloud):
            pc_dir = start_point_cloud
            ply_path = os.path.join(pc_dir, "point_cloud.ply")
            meta_path = os.path.join(pc_dir, "hierarchy_meta.pt")
            if os.path.exists(ply_path):
                gaussians.load_ply(ply_path)
            elif os.path.exists(meta_path):
                gaussians.load_hierarchy_chunks(pc_dir)
            else:
                raise FileNotFoundError(
                    f"--start_point_cloud directory must contain point_cloud.ply or hierarchy_meta.pt: {pc_dir}"
                )
        else:
            gaussians.load_ply(start_point_cloud)

        if start_lod is not None:
            loaded_start_lod = int(start_lod)
            gaussians.current_lod = loaded_start_lod
            gaussians.setup_scaling_activation()
        else:
            loaded_start_lod = int(gaussians.current_lod)
        if gaussians._birth_lod is None:
            gaussians._birth_lod = torch.full((gaussians._xyz.shape[0],), loaded_start_lod,
                                              dtype=torch.long, device=gaussians._xyz.device)
            gaussians._parent_idx = torch.full_like(gaussians._birth_lod, -1)
            gaussians._per_gaussian_lb = gaussians._compute_per_gaussian_lb()
            gaussians._invalidate_topology_caches()
            gaussians._resort_by_birth_lod()
        if gaussians.max_lod != opt.lod_max:
            raise ValueError("Warm-start max_lod differs from the requested hierarchy; preserve its scale bands")
        scene.gaussians = gaussians
        gaussians.spatial_lr_scale = scene.cameras_extent
        print(
            f"[warm_start] loaded {start_point_cloud} at LOD{loaded_start_lod}; "
            f"training will continue from LOD{loaded_start_lod + 1}",
            flush=True,
        )
    else:
        scene = Scene(dataset, gaussians, load_image_device=dataset.data_device)
    prefix_prob = float(getattr(opt, "prefix_prob", 0.0))
    current_prob = float(getattr(opt, "current_prob", 0.0))
    assert prefix_prob >= 0.0 and current_prob >= 0.0,\
        "--prefix_prob and --current_prob must be non-negative"
    assert prefix_prob + current_prob <= 1.0,\
        "--prefix_prob + --current_prob must be <= 1.0"
    if prefix_prob > 0.0 or current_prob > 0.0:
        print(
            f"[prefix_traversal] current_prob={current_prob:.3f} "
            f"prefix_prob={prefix_prob:.3f} "
            f"min_lod={int(getattr(opt, 'prefix_min_lod', opt.lod_min))}",
            flush=True,
        )
    gaussians._save_hierarchy_chunks = bool(getattr(opt, "save_hierarchy_chunks", True))
    gaussians._export_flat_legacy = bool(getattr(opt, "export_flat_legacy", False))
    if opt.per_parent_budget and opt.per_parent_budget > 0:
        gaussians._per_parent_budget = int(opt.per_parent_budget)
        gaussians._budget_debug = bool(opt.per_parent_budget_debug)
    if getattr(opt, "densify_current_lod_only", False):
        gaussians._densify_current_lod_only = True
    gaussians.training_setup(opt)
    gaussians._apply_mip_floor = bool(dataset.mip_floor)
    gaussians._mip_floor_smooth_beta = 0.0
    gaussians._mip_floor_clamp_raw = bool(getattr(dataset, "mip_floor_clamp_raw", False))
    if bool(dataset.mip_floor) and not (start_point_cloud and gaussians._mip_floor is not None):
        gaussians.compute_mip_floor(scene.getTrainCameras(), mode=dataset.mip_floor_mode)

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    lods_count = opt.lod_max - opt.lod_min + 1
    train_profile_enabled = bool(getattr(opt, "train_profile", False))
    train_profile_interval = max(1, int(getattr(opt, "train_profile_interval", 100)))
    train_profile_warmup = max(0, int(getattr(opt, "train_profile_warmup", 0)))
    train_profile_rows = []
    train_profile_path = os.path.join(dataset.model_path, "train_profile.csv")
    if train_profile_enabled:
        print(
            f"[train_profile] writing {train_profile_path} "
            f"(interval={train_profile_interval}, warmup={train_profile_warmup})",
            flush=True,
        )

    densify_grad_thresholds = expand_list_to_match_lods(opt.densify_grad_threshold, lods_count)
    densification_intervals = expand_list_to_match_lods(opt.densification_interval, lods_count)
    densify_from_iters = expand_list_to_match_lods(opt.densify_from_iter, lods_count)
    densify_until_iters = expand_list_to_match_lods(opt.densify_until_iter, lods_count)


    lod_density_budgets = [0] * lods_count
    prune_opacity_thresholds = expand_list_to_match_lods(opt.prune_opacity_threshold, lods_count)
    pruning_intervals = expand_list_to_match_lods(opt.pruning_interval, lods_count)
    prune_from_iters = expand_list_to_match_lods(opt.prune_from_iter, lods_count)
    prune_until_iters = expand_list_to_match_lods(opt.prune_until_iter, lods_count)
    prune_overlap_thresholds = expand_list_to_match_lods(opt.prune_overlap_threshold, lods_count)
    pruning_overlap_intervals = expand_list_to_match_lods(opt.pruning_overlap_interval, lods_count)
    prune_max_radii2D_thresholds = expand_list_to_match_lods(opt.prune_max_radii2D, lods_count)
    opacity_reset_intervals = expand_list_to_match_lods(opt.opacity_reset_interval, lods_count)
    lambda_dssims = expand_list_to_match_lods(opt.lambda_dssim, lods_count)
    eol_prune_mode = str(getattr(opt, "end_of_lod_prune_mode", "standard")).lower()
    legacy_eol_prune_modes = {"standard", "legacy"}
    family_cleanup_eol_prune_modes = {
        "family_cleanup",
        "low_siblings_dead_families",
        "sibling_then_family",
    }
    valid_eol_prune_modes = legacy_eol_prune_modes | family_cleanup_eol_prune_modes
    if eol_prune_mode not in valid_eol_prune_modes:
        raise ValueError(
            f"--end_of_lod_prune_mode must be one of {sorted(valid_eol_prune_modes)}, "
            f"got {eol_prune_mode!r}"
        )
    scale_upper_iter_weights = None
    scale_upper_iter_weight_power = float(getattr(opt, "scale_upper_iter_weight_power", 0.0))
    if scale_upper_iter_weight_power > 0:
        max_lod_iterations = max(max(1, int(iters)) for iters in opt.lod_iterations)
        scale_upper_iter_weights = {}
        for lod_val, lod_iters in zip(range(opt.lod_min, opt.lod_max + 1), opt.lod_iterations):
            ratio = max_lod_iterations / max(1, int(lod_iters))
            scale_upper_iter_weights[lod_val] = ratio ** scale_upper_iter_weight_power
        print("[scale_upper_iter_weight] "
              + ", ".join(f"LOD{lod}={weight:.3f}" for lod, weight in scale_upper_iter_weights.items()))

    loop_start_lod = opt.lod_min
    if loaded_start_lod is not None:
        loop_start_lod = loaded_start_lod + 1
        if loop_start_lod > opt.lod_max:
            raise ValueError(
                f"Warm-start LOD{loaded_start_lod} is already >= --lod_max {opt.lod_max}; nothing to train."
            )
        if loop_start_lod <= opt.lod_min:
            raise ValueError(
                f"Warm-start LOD{loaded_start_lod} is not compatible with --lod_min {opt.lod_min}."
            )

    for current_lod in range(loop_start_lod, opt.lod_max + 1):
        i = current_lod - opt.lod_min
        iterations = opt.lod_iterations[i]
        first_iteration = 1
        allow_previous_lod_prune = bool(getattr(opt, "allow_previous_lod_prune", False))
        progress_bar = tqdm(total=iterations, initial=first_iteration - 1, desc=f"LOD:{current_lod}")
        lod_setup_row = None
        lod_setup_start = None
        if train_profile_enabled:
            lod_setup_row = {
                "phase": "lod_setup",
                "lod": current_lod,
                "iteration": 0,
                "global_step": global_step,
                "num_gaussians_start": int(gaussians.get_xyz.shape[0]),
            }
            lod_setup_start = _train_profile_start(lod_setup_row)


        if (
            opt.end_of_lod_prune
            and eol_prune_mode in legacy_eol_prune_modes
            and current_lod > opt.lod_min
        ):
            ha_eol = opt.hierarchy_aware_prune if opt.hierarchy_aware_prune else 'protect'
            eol_threshold = prune_opacity_thresholds[i - 1]
            scope_all = getattr(opt, 'end_of_lod_prune_scope', 'frozen') == 'all'
            gaussians.prune(
                eol_threshold,
                scene.cameras_extent,
                None,
                0.0,
                hierarchy_aware=ha_eol,
                protect_lod1=True,
                restrict_to_current_lod=((not scope_all) and not allow_previous_lod_prune),
                overlap_same_lod_only=opt.prune_overlap_same_lod_only,
                min_hits_for_opacity_prune=opt.min_hits_for_opacity_prune,
            )

        if current_lod > opt.lod_min:
            gaussians.increase_lod_hierarchical(training_args=opt)
            if dataset.mip_floor:
                gaussians.compute_mip_floor(scene.getTrainCameras(), mode=dataset.mip_floor_mode)

        if lod_setup_row is not None:
            _train_profile_elapsed(lod_setup_row, "lod_setup_s", lod_setup_start)
            lod_setup_row["num_gaussians_end"] = int(gaussians.get_xyz.shape[0])
            train_profile_rows.append(lod_setup_row)
            _write_train_profile(train_profile_path, train_profile_rows)

        def previous_lod_mask():
            if current_lod <= opt.lod_min or gaussians._birth_lod is None:
                return None
            return gaussians._birth_lod < current_lod

        def mutation_protect_mask():
            if getattr(opt, "freeze_previous_lod_mutations", True):
                return previous_lod_mask()
            return None

        def prune_protect_mask():
            if allow_previous_lod_prune:
                return None
            return mutation_protect_mask()

        def optimizable_params():
            return [
                ("xyz", gaussians._xyz),
                ("f_dc", gaussians._features_dc),
                ("f_rest", gaussians._features_rest),
                ("opacity", gaussians._opacity),
                ("scaling", gaussians._scaling),
                ("rotation", gaussians._rotation),
            ]

        def combine_masks(*masks):
            out = None
            for mask in masks:
                if mask is None:
                    continue
                out = mask.clone() if out is None else (out | mask)
            return out

        def backward_component(weighted_loss, protect_mask=None, retain_graph=True, param_protect_masks=None):
            if weighted_loss is None:
                return
            if not getattr(weighted_loss, "requires_grad", False):
                return
            params_for_grad_restore = [
                ("xyz", gaussians._xyz),
                ("features_dc", gaussians._features_dc),
                ("features_rest", gaussians._features_rest),
                ("opacity", gaussians._opacity),
                ("scaling", gaussians._scaling),
                ("rotation", gaussians._rotation),
            ]
            saved_grads = []
            has_saved_grads = False
            for name, param in params_for_grad_restore:
                mask = protect_mask
                if param_protect_masks is not None and name in param_protect_masks:
                    mask = param_protect_masks[name]
                if mask is not None:
                    mask = mask.to(device=param.device, dtype=torch.bool)
                if mask is not None and mask.any():
                    saved_grads.append((param, mask, param.grad[mask].clone() if param.grad is not None else None))
                    has_saved_grads = True
                else:
                    saved_grads.append((param, None, None))
            weighted_loss.backward(retain_graph=retain_graph)
            if has_saved_grads:
                with torch.no_grad():
                    for param, mask, grad_values in saved_grads:
                        if mask is None or param.grad is None:
                            continue
                        if grad_values is None:
                            param.grad[mask] = 0
                        else:
                            param.grad[mask] = grad_values

        for iteration in range(first_iteration, iterations + 1):
            profile_row = None
            profile_iter_start = None
            profile_t = None
            if (
                train_profile_enabled
                and iteration > train_profile_warmup
                and (iteration % train_profile_interval == 0 or iteration == iterations)
            ):
                profile_row = {
                    "phase": "iteration",
                    "lod": current_lod,
                    "iteration": iteration,
                    "global_step": global_step + 1,
                    "num_gaussians_start": int(gaussians.get_xyz.shape[0]),
                }
                profile_iter_start = _train_profile_start(profile_row)
                profile_t = profile_iter_start
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

            gaussians.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))

            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background
            profile_t = _train_profile_mark(profile_row, "setup_s", profile_t)

            active_mask = None
            active_indices = None
            if current_lod > opt.lod_min:
                active_indices = gaussians.get_active_indices(
                    viewpoint_cam,
                    opt.sp_thresh,
                    mode=opt.hierarchical_mode,
                    prefix_prob=opt.prefix_prob,
                    current_prob=getattr(opt, "current_prob", 0.0),
                    prefix_min_lod=getattr(opt, "prefix_min_lod", opt.lod_min),
                    profile=profile_row,
                    low_pass=pipe.low_pass,
                    cull_near=getattr(opt, "cull_near", 0.0),
                    cull_frustum=getattr(opt, "cull_frustum", 0.0),
                    cull_backend=getattr(pipe, "renderer", "inria"),
                )
            if profile_row is not None:
                if active_indices is not None:
                    profile_row["active_total"] = int(active_indices.numel())
                else:
                    profile_row["active_total"] = int(gaussians.get_xyz.shape[0])
            profile_t = _train_profile_mark(profile_row, "selection_s", profile_t)

            def ensure_active_mask():
                nonlocal active_mask
                if active_mask is None and active_indices is not None:
                    active_mask = gaussians.active_indices_to_mask(active_indices)
                return active_mask

            render_kwargs = {"active_indices": active_indices}
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, **render_kwargs)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            if profile_row is not None:
                profile_row["visible_total"] = int(visibility_filter.sum().item())
            profile_t = _train_profile_mark(profile_row, "render_forward_s", profile_t)

            if iteration == 1 and bool(getattr(opt, "screenspace_debug", False)):
                print_per_lod_screensize(gaussians, viewpoint_cam, opt, f"at LOD{current_lod} iter 1",
                                         low_pass=pipe.low_pass)
                max_debug_cams = max(1, int(getattr(opt, "screenspace_debug_cams", 8)))
                print_per_lod_subpixel_summary(gaussians, scene.getTrainCameras(), opt,
                                               f"LOD{current_lod} iter 1",
                                               max_cams=max_debug_cams,
                                               low_pass=pipe.low_pass)
            profile_t = _train_profile_mark(profile_row, "debug_s", profile_t)

            gt_image = viewpoint_cam.original_image.cuda()
            if hasattr(viewpoint_cam, "alpha_mask"):
                image = image * viewpoint_cam.alpha_mask.to(image.device)
            Ll1 = l1_loss(image, gt_image)
            reconstruction_loss = (1.0 - lambda_dssims[i]) * Ll1 + lambda_dssims[i] * (1.0 - ssim(image, gt_image))
            loss = reconstruction_loss
            regularizer_loss = None
            hier_regularizer_loss = None
            non_hier_regularizer_loss = None
            profile_t = _train_profile_mark(profile_row, "reconstruction_loss_s", profile_t)

            def add_regularizer(weighted_loss, hierarchical=False):
                nonlocal loss, regularizer_loss, hier_regularizer_loss, non_hier_regularizer_loss
                loss = loss + weighted_loss
                regularizer_loss = weighted_loss if regularizer_loss is None else regularizer_loss + weighted_loss
                if hierarchical:
                    hier_regularizer_loss = weighted_loss if hier_regularizer_loss is None else hier_regularizer_loss + weighted_loss
                else:
                    non_hier_regularizer_loss = weighted_loss if non_hier_regularizer_loss is None else non_hier_regularizer_loss + weighted_loss

            reg_scope = str(getattr(opt, "regularizer_scope", "all"))
            if reg_scope not in ("all", "active"):
                raise ValueError(f"regularizer_scope must be 'all' or 'active', got {reg_scope!r}")
            reg_active_mask = ensure_active_mask() if reg_scope == "active" else None
            profile_t = _train_profile_mark(profile_row, "regularizer_setup_s", profile_t)

            maha_loss_val = None
            if opt.lambda_maha > 0 and current_lod > opt.lod_min:
                maha_loss = compute_maha_regularizer(
                    gaussians,
                    margin=opt.maha_margin,
                    scope=getattr(opt, "maha_scope", "immediate"),
                    ancestor_decay=getattr(opt, "maha_ancestor_decay", 1.0),
                    max_ancestors=getattr(opt, "maha_max_ancestors", 0),
                    active_mask=reg_active_mask,
                    current_lod_only=getattr(opt, "maha_current_lod_children_only", False),
                    reduce=getattr(opt, "maha_reduce", "mean"),
                )
                maha_loss_val = maha_loss.item()
                add_regularizer(opt.lambda_maha * maha_loss, hierarchical=True)
            profile_t = _train_profile_mark(profile_row, "maha_loss_s", profile_t)

            scale_upper_loss_val = None
            if opt.lambda_scale_upper > 0 and gaussians._birth_lod is not None:
                su_loss = scale_upper_loss(
                    gaussians,
                    scope=opt.scale_upper_scope,
                    current_lod=current_lod,
                    include_lod1=getattr(opt, "scale_upper_include_lod1", False),
                    lod_iteration_weights=scale_upper_iter_weights,
                    active_mask=reg_active_mask,
                )
                scale_upper_loss_val = su_loss.item()
                add_regularizer(opt.lambda_scale_upper * su_loss, hierarchical=False)
            profile_t = _train_profile_mark(profile_row, "scale_upper_loss_s", profile_t)

            regularizer_protect = previous_lod_mask() if getattr(opt, "freeze_previous_lod_regularizers", True) else None

            backward_parts = [(reconstruction_loss, None)]
            backward_parts.extend([
                (non_hier_regularizer_loss, regularizer_protect, None),
                (hier_regularizer_loss, regularizer_protect, None),
            ])
            normalized_backward_parts = []
            for part in backward_parts:
                if len(part) == 2:
                    part_loss, protect_mask = part
                    param_protect_masks = None
                else:
                    part_loss, protect_mask, param_protect_masks = part
                if part_loss is not None:
                    normalized_backward_parts.append((part_loss, protect_mask, param_protect_masks))
            backward_parts = normalized_backward_parts
            profile_t = _train_profile_mark(profile_row, "backward_setup_s", profile_t)
            for part_idx, (part_loss, protect_mask, param_protect_masks) in enumerate(backward_parts):
                backward_component(
                    part_loss,
                    protect_mask=protect_mask,
                    param_protect_masks=param_protect_masks,
                    retain_graph=(part_idx < len(backward_parts) - 1),
                )
            profile_t = _train_profile_mark(profile_row, "backward_s", profile_t)


            profile_t = _train_profile_mark(profile_row, "grad_postprocess_s", profile_t)

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                global_step += 1
                if iteration % 10 == 0:
                    tb_writer.add_scalar(f"loss/total_lod{current_lod}", loss.item(), global_step)
                    tb_writer.add_scalar(f"loss/l1_lod{current_lod}", Ll1.item(), global_step)
                    tb_writer.add_scalar(f"loss/ema_lod{current_lod}", ema_loss_for_log, global_step)
                    tb_writer.add_scalar("stats/num_gaussians", gaussians.get_xyz.shape[0], global_step)
                    tb_writer.add_scalar("stats/current_lod", current_lod, global_step)
                    if maha_loss_val is not None: tb_writer.add_scalar("loss/maha", maha_loss_val, global_step)
                    if scale_upper_loss_val is not None: tb_writer.add_scalar("loss/scale_upper", scale_upper_loss_val, global_step)
                if rr_enabled and (global_step % rr_every == 0 or iteration == 1):
                    rr_log_gaussians(gaussians, global_step, current_lod, log_ellipsoids=rr_ellipsoids, max_ellipsoids=rr_ellipsoid_max)
                    sel_path = os.path.join(dataset.model_path, "selected.txt")
                    if os.path.exists(sel_path):
                        try:
                            with open(sel_path) as f:
                                sel = int(f.read().strip())
                            rr_log_lineage(gaussians, sel, global_step)
                        except (ValueError, OSError):
                            pass
                if iteration % 10 == 0:
                    postfix = {"Loss": f"{ema_loss_for_log:.7f}"}
                    if maha_loss_val: postfix["Maha"] = f"{maha_loss_val:.4f}"
                    if scale_upper_loss_val is not None: postfix["SU"] = f"{scale_upper_loss_val:.4f}"
                    if active_indices is not None: postfix["Active"] = f"{active_indices.numel()}/{gaussians.get_xyz.shape[0]}"
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)

                if iteration == iterations:
                    progress_bar.close()
                profile_t = _train_profile_mark(profile_row, "logging_s", profile_t)

                if (iteration in saving_iterations) or iteration == iterations:
                    scene.save(iteration, current_lod)
                profile_t = _train_profile_mark(profile_row, "save_s", profile_t)

                if gaussians._train_hits is not None:
                    if active_indices is not None:
                        hit_filter = visibility_filter
                        gaussians._train_hits[active_indices[hit_filter]] += 1
                    else:
                        hit_filter = visibility_filter
                        gaussians._train_hits[hit_filter] += 1

                if gaussians._recent_active_hits is not None:
                    if active_indices is not None:
                        hit_filter = visibility_filter
                        gaussians._recent_active_hits[active_indices[hit_filter]] += 1
                    else:
                        hit_filter = visibility_filter
                        gaussians._recent_active_hits[hit_filter] += 1

                _densify_window = iteration < densify_until_iters[i]
                if _densify_window:
                    if active_indices is not None:
                        stats_filter = visibility_filter
                        global_vis_indices = active_indices[stats_filter]
                        gaussians.max_radii2D[global_vis_indices] = torch.max(gaussians.max_radii2D[global_vis_indices], radii[stats_filter])
                        if viewspace_point_tensor.grad is not None and stats_filter.any():
                            grad_norms = torch.norm(viewspace_point_tensor.grad[stats_filter, :2], dim=-1, keepdim=True)
                            gaussians.xyz_gradient_accum[global_vis_indices] += grad_norms
                            gaussians.denom[global_vis_indices] += 1
                    else:
                        stats_filter = visibility_filter
                        gaussians.max_radii2D[stats_filter] = torch.max(gaussians.max_radii2D[stats_filter], radii[stats_filter])
                        if stats_filter.any():
                            gaussians.add_densification_stats(viewspace_point_tensor, stats_filter)
                profile_t = _train_profile_mark(profile_row, "visibility_stats_s", profile_t)

                if _densify_window:
                    if iteration > densify_from_iters[i] and iteration % densification_intervals[i] == 0:
                        budget_ok = True
                        current_budget = lod_density_budgets[i]
                        if current_budget > 0 and gaussians._birth_lod is not None:
                            if (gaussians._birth_lod == current_lod).sum().item() >= current_budget:
                                budget_ok = False
                        if budget_ok:
                            gaussians.densify(densify_grad_thresholds[i], extent=scene.cameras_extent, restrict_to_lod=current_lod)
                        if dataset.mip_floor:
                            gaussians.compute_mip_floor(scene.getTrainCameras(), mode=dataset.mip_floor_mode)

                _prune_window = iteration < prune_until_iters[i]
                _maha_warmup_active = False
                if _maha_warmup_active:
                    _prune_window = False
                if _prune_window and iteration > prune_from_iters[i] and iteration % pruning_intervals[i] == 0:
                    size_threshold = None
                    if iteration > opacity_reset_intervals[i]:
                        if prune_max_radii2D_thresholds[i] > 0:
                            size_threshold = prune_max_radii2D_thresholds[i]
                        elif current_lod >= opt.prune_max_radii2D_lod:
                            size_threshold = 20
                    ha = opt.hierarchy_aware_prune if (opt.hierarchy_aware_prune and current_lod > opt.lod_min) else False
                    overlap = prune_overlap_thresholds[i] if iteration % pruning_overlap_intervals[i] == 0 else 0.0
                    gaussians.prune(
                        prune_opacity_thresholds[i], scene.cameras_extent, size_threshold, overlap,
                        hierarchy_aware=ha,
                        protect_lod1=current_lod > opt.lod_min,
                        restrict_to_current_lod=(opt.prune_current_lod_only and not allow_previous_lod_prune),
                        overlap_same_lod_only=opt.prune_overlap_same_lod_only,
                        min_hits_for_opacity_prune=opt.min_hits_for_opacity_prune,
                        protect_mask=prune_protect_mask(),
                    )

                if iteration < prune_until_iters[i] and not _maha_warmup_active and (iteration % opacity_reset_intervals[i] == 0 or (dataset.white_background and iteration == densify_from_iters[i])):
                    gaussians.reset_opacity(
                        current_lod_only=False,
                        protect_inactive=getattr(opt, "reset_opacity_protect_inactive", False),
                        protect_mask=mutation_protect_mask(),
                    )
                profile_t = _train_profile_mark(profile_row, "densify_prune_s", profile_t)

                if iteration < iterations:
                    gaussians.mip_floor_active_set_step(gaussians.optimizer)
                    gaussians.optimizer.step()
                    gaussians.mip_floor_clamp_raw_post_step()

                    gaussians.optimizer.zero_grad(set_to_none=True)

                profile_t = _train_profile_mark(profile_row, "optimizer_s", profile_t)

                if profile_row is not None:
                    profile_row["num_gaussians_end"] = int(gaussians.get_xyz.shape[0])
                    profile_row["loss"] = float(loss.detach().item())
                    _train_profile_memory(profile_row, gaussians)
                    _train_profile_elapsed(profile_row, "iteration_total_s", profile_iter_start)
                    train_profile_rows.append(profile_row)
                    if len(train_profile_rows) % 50 == 0:
                        _write_train_profile(train_profile_path, train_profile_rows)
        lod_final_save_row = None
        lod_final_save_start = None
        if train_profile_enabled:
            lod_final_save_row = {
                "phase": "lod_final_save",
                "lod": current_lod,
                "iteration": iterations,
                "global_step": global_step,
                "num_gaussians_start": int(gaussians.get_xyz.shape[0]),
            }
            lod_final_save_start = _train_profile_start(lod_final_save_row)
        if opt.end_of_lod_prune and eol_prune_mode in family_cleanup_eol_prune_modes:
            eol_threshold = prune_opacity_thresholds[i]
            eol_prune_dead_families = bool(getattr(opt, "end_of_lod_prune_dead_families", True))
            eol_stats = gaussians.prune_low_opacity_families(
                eol_threshold,
                protect_lod1=True,
                restrict_sibling_prune_to_current_lod=not allow_previous_lod_prune,
                min_hits_for_opacity_prune=opt.min_hits_for_opacity_prune,
                protect_mask=prune_protect_mask(),
                dead_family_protect_mask=None,
                prune_dead_families=eol_prune_dead_families,
            )
            eol_pruned = eol_stats["before"] - eol_stats["after"]
            if eol_pruned > 0 or eol_stats["low_before"] > 0:
                print(
                    f"[end_of_lod_prune:{eol_prune_mode}] LOD{current_lod} "
                    f"thr={eol_threshold:g} before={eol_stats['before']} "
                    f"low={eol_stats['low_before']}->{eol_stats['low_after']} "
                    f"pruned={eol_pruned} "
                    f"siblings={eol_stats['sibling_total']} "
                    f"dead_family_prune={'on' if eol_prune_dead_families else 'off'} "
                    f"dead_families={eol_stats['dead_family_roots']}/{eol_stats['dead_family_total']} "
                    f"after={eol_stats['after']}",
                    flush=True,
                )

        scene.save(iterations, current_lod)
        if lod_final_save_row is not None:
            _train_profile_elapsed(lod_final_save_row, "save_s", lod_final_save_start)
            lod_final_save_row["num_gaussians_end"] = int(gaussians.get_xyz.shape[0])
            train_profile_rows.append(lod_final_save_row)
        if train_profile_enabled:
            _write_train_profile(train_profile_path, train_profile_rows)

    final_cam = scene.getTrainCameras()[0]
    print_per_lod_screensize(gaussians, final_cam, opt, "at end of training",
                             low_pass=pipe.low_pass)

    tb_writer.close()
    return scene.model_path

def prepare_output(args, opt=None, pipe=None, run_args=None):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID', str(uuid.uuid4()))[:10]
        args.model_path = os.path.join("./output/", unique_str)
    os.makedirs(args.model_path, exist_ok=True)
    values = vars(run_args).copy() if run_args is not None else {}
    for group in (args, opt, pipe):
        if group is not None:
            values.update(vars(group))
    save_config(args.model_path, values)
    with open(os.path.join(args.model_path, "train_command.txt"), 'w') as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_point_cloud", type=str, default=None,
                        help="Warm-start from a saved point_cloud/lod_X_iteration_Y directory or point_cloud.ply, then continue at the next LOD.")
    parser.add_argument("--start_lod", type=int, default=None,
                        help="Override the LOD recorded in --start_point_cloud metadata.")
    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    start_time = time.time()
    model_path = training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.save_iterations,
        args.debug_from, start_point_cloud=args.start_point_cloud,
        start_lod=args.start_lod, run_args=args,
    )
    train_time = time.time() - start_time

    h, r = divmod(int(train_time), 3600)
    m, s = divmod(r, 60)
    res_path = os.path.join(model_path, f"train_time_{h:02d}H{m:02d}M{s:02d}S.txt")
    with open(res_path, "w") as f:
        f.write(f"Training time: {train_time:.2f} seconds")
    print("\nTraining complete.")
