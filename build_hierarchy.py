import os
import argparse
import time
import torch
import numpy as np
from scipy.spatial import cKDTree

from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxLod, searchForMaxIterationGivenLod
from utils.config_utils import load_config


def quaternion_to_rotation_matrix(q):
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty(q.shape[0], 3, 3, dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2*(y*y + z*z); R[:, 0, 1] = 2*(x*y - w*z); R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z);     R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y);     R[:, 2, 1] = 2*(y*z + w*x);     R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def load_lod_level(model_path, lod, sh_degree, lod1_scaling_lower_bound, lod_scaling_ratio, max_lod,
                   device="cpu"):
    gaussians = GaussianModel(sh_degree=sh_degree, lod1_scaling_lower_bound=lod1_scaling_lower_bound,
                              lod_scaling_ratio=lod_scaling_ratio, max_lod=max_lod)
    pc_root = os.path.join(model_path, "point_cloud")
    final_iter = searchForMaxIterationGivenLod(pc_root, max_lod)
    final_dir = os.path.join(pc_root, f"lod_{max_lod}_iteration_{final_iter}")
    chunk_path = os.path.join(final_dir, f"point_cloud_lod_{lod}.ply")
    meta_path = os.path.join(final_dir, "hierarchy_meta.pt")

    if os.path.exists(chunk_path) and os.path.exists(meta_path):
        # HLoD model exports store the final model as per-LOD
        # chunks plus a flat hierarchy_meta.pt. Loading point_cloud.ply through
        # GaussianModel.load would concatenate all chunks up to load_lod; for
        # post-hoc hierarchy construction we need exactly one birth-LOD chunk.
        gaussians.load_ply(chunk_path, device=device, load_meta=False)
        gaussians.loaded_lod = lod
        gaussians.loaded_iter = final_iter
        gaussians.current_lod = lod
        gaussians.setup_scaling_activation()

        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        birth_lod = meta.get("birth_lod", None)
        per_lb = meta.get("per_gaussian_lb", None)
        if birth_lod is not None and per_lb is not None:
            mask = birth_lod.long() == lod
            if int(mask.sum().item()) == gaussians._xyz.shape[0]:
                gaussians._per_gaussian_lb = per_lb[mask].to(device=device, dtype=torch.float32)
            else:
                print(
                    f"  WARNING: per_gaussian_lb count mismatch for LOD {lod}; "
                    f"using scalar lower bound instead."
                )
        mip_floor = meta.get("mip_floor")
        if mip_floor is not None:
            gaussians._mip_floor = mip_floor[birth_lod.long() == lod].to(device=device)
        gaussians._apply_mip_floor = bool(meta.get("apply_mip_floor", True))
    else:
        gaussians.load_level(model_path, lod, -1, device=device, source_lod=max_lod)

    scaling = gaussians.get_scaling.detach()

    result = {
        'xyz':      gaussians._xyz.detach(),
        'scaling':  scaling,
        'rotation': gaussians._rotation.detach(),
        'opacity':  gaussians.opacity_activation(gaussians._opacity.detach()),
        'count':    gaussians._xyz.shape[0],
        'scaling_lower_bound': gaussians.scaling_lower_bound,
    }

    return result


def _scale_aware_candidates(coarse_xyz_np, coarse_scaling, fine_xyz_np, k_candidates=64, sigma_mult=3.0):
    N_coarse = coarse_xyz_np.shape[0]
    N_fine = fine_xyz_np.shape[0]
    K = min(k_candidates, N_coarse)

    tree = cKDTree(coarse_xyz_np)
    knn_dists, knn_idx = tree.query(fine_xyz_np, k=K, workers=-1)

    if knn_idx.ndim == 1:
        knn_idx = knn_idx[:, None]
        knn_dists = knn_dists[:, None]

    # Extend search for fine points near large anisotropic coarse Gaussians
    max_scale_per_coarse = coarse_scaling.cpu().float().numpy().max(axis=1)
    nearest_extent = max_scale_per_coarse[knn_idx[:, 0]] * sigma_mult
    kth_dist = knn_dists[:, -1]
    needs_extension = nearest_extent > kth_dist
    n_extend = needs_extension.sum()

    if n_extend > 0 and n_extend < N_fine * 0.5:
        K_ext = min(K * 2, N_coarse)
        extend_idx = np.where(needs_extension)[0]
        _, ext_cand = tree.query(fine_xyz_np[extend_idx], k=K_ext, workers=-1)
        if ext_cand.ndim == 1:
            ext_cand = ext_cand[:, None]

        cand_idx_np = np.zeros((N_fine, K_ext), dtype=np.intp)
        cand_idx_np[:, :K] = knn_idx
        for i in range(K, K_ext):
            cand_idx_np[:, i] = knn_idx[:, -1]
        cand_idx_np[extend_idx] = ext_cand
        return cand_idx_np
    elif n_extend > 0:
        print(f"  WARNING: {n_extend}/{N_fine} ({100*n_extend/N_fine:.0f}%) fine points need extended search, "
              f"skipping (consider increasing k_candidates)")

    return knn_idx


def _prepare_gpu(coarse_data, fine_data, device):
    coarse_xyz   = coarse_data['xyz'].to(device, torch.float32)
    coarse_scl   = coarse_data['scaling'].to(device, torch.float32)
    coarse_rot   = coarse_data['rotation'].to(device, torch.float32)
    coarse_R     = quaternion_to_rotation_matrix(coarse_rot)
    coarse_inv_s = 1.0 / coarse_scl.clamp(min=1e-8)
    coarse_log_s = coarse_scl.clamp(min=1e-8).log()

    fine_xyz   = fine_data['xyz'].to(device, torch.float32)
    fine_scl   = fine_data['scaling'].to(device, torch.float32)
    fine_rot   = fine_data['rotation'].to(device, torch.float32)
    fine_R     = quaternion_to_rotation_matrix(fine_rot)
    fine_inv_s = 1.0 / fine_scl.clamp(min=1e-8)
    fine_log_s = fine_scl.clamp(min=1e-8).log()

    coarse = dict(xyz=coarse_xyz, scl=coarse_scl, R=coarse_R,
                  inv_s=coarse_inv_s, log_s=coarse_log_s)
    fine = dict(xyz=fine_xyz, scl=fine_scl, R=fine_R,
                inv_s=fine_inv_s, log_s=fine_log_s)
    return coarse, fine


def _kl_chunk(cf_xyz, cf_R, cf_s, cf_ls, cc_xyz, cc_R, cc_inv_s, cc_ls):
    """KL(fine || coarse) — how well coarse covers fine."""
    delta = cf_xyz[:, None, :] - cc_xyz
    local = torch.einsum('...ji,...j->...i', cc_R, delta)
    maha = (local * cc_inv_s).pow(2).sum(dim=-1)

    R_rel = cc_R.transpose(-2, -1) @ cf_R[:, None]
    M = cc_inv_s[..., :, None] * R_rel * cf_s[:, None, None, :]
    trace_term = (M * M).sum(dim=(-2, -1))

    log_det = 2.0 * (cc_ls.sum(-1) - cf_ls.sum(-1)[:, None])
    return (0.5 * (trace_term + maha - 3.0 + log_det)).clamp(min=0.0)


def _maha_chunk(cf_xyz, cc_xyz, cc_R, cc_inv_s):
    """Mahalanobis distance: is fine center within coarse Gaussian's spatial extent?"""
    delta = cf_xyz[:, None, :] - cc_xyz
    local = torch.einsum('...ji,...j->...i', cc_R, delta)
    return (local * cc_inv_s).pow(2).sum(dim=-1)


def compute_kl_batched(fine, coarse, candidate_indices, chunk_size=8192):
    N_fine, _ = candidate_indices.shape
    device = fine['xyz'].device
    parent_idx = torch.empty(N_fine, dtype=torch.long)
    best_dist = torch.empty(N_fine, dtype=torch.float32)

    for start in range(0, N_fine, chunk_size):
        end = min(start + chunk_size, N_fine)
        cands = candidate_indices[start:end].to(device=device, non_blocking=True)
        kl = _kl_chunk(
            fine['xyz'][start:end], fine['R'][start:end],
            fine['scl'][start:end], fine['log_s'][start:end],
            coarse['xyz'][cands], coarse['R'][cands],
            coarse['inv_s'][cands], coarse['log_s'][cands])
        best_k = kl.argmin(dim=1)
        arange = torch.arange(end - start, device=device)
        parent_idx[start:end] = cands[arange, best_k].cpu()
        best_dist[start:end] = kl[arange, best_k].cpu()

    return parent_idx, best_dist


def compute_maha_batched(fine, coarse, candidate_indices, chunk_size=8192):
    N_fine, _ = candidate_indices.shape
    device = fine['xyz'].device
    parent_idx = torch.empty(N_fine, dtype=torch.long)
    best_dist = torch.empty(N_fine, dtype=torch.float32)

    for start in range(0, N_fine, chunk_size):
        end = min(start + chunk_size, N_fine)
        cands = candidate_indices[start:end].to(device=device, non_blocking=True)
        maha = _maha_chunk(
            fine['xyz'][start:end],
            coarse['xyz'][cands], coarse['R'][cands], coarse['inv_s'][cands])
        best_k = maha.argmin(dim=1)
        arange = torch.arange(end - start, device=device)
        parent_idx[start:end] = cands[arange, best_k].cpu()
        best_dist[start:end] = maha[arange, best_k].cpu()

    return parent_idx, best_dist


def build_pair(coarse_data, fine_data, k_candidates=64, chunk_size=8192, device='cuda'):
    coarse, fine = _prepare_gpu(coarse_data, fine_data, device)
    coarse_xyz_np = coarse_data['xyz'].cpu().float().numpy()
    fine_xyz_np = fine_data['xyz'].cpu().float().numpy()

    print(f"  KL divergence: computing on {device} ...")
    cand_np = _scale_aware_candidates(coarse_xyz_np, coarse_data['scaling'],
                                       fine_xyz_np, k_candidates=k_candidates)
    cand = torch.from_numpy(cand_np).long()

    parent_indices, kl_distances = compute_kl_batched(fine, coarse, cand, chunk_size)
    return parent_indices.cpu(), kl_distances.cpu()


def build_pair_maha(coarse_data, fine_data, k_candidates=64, chunk_size=8192, device='cuda'):
    coarse, fine = _prepare_gpu(coarse_data, fine_data, device)
    coarse_xyz_np = coarse_data['xyz'].cpu().float().numpy()
    fine_xyz_np = fine_data['xyz'].cpu().float().numpy()

    print(f"  Mahalanobis: computing on {device} ...")
    cand_np = _scale_aware_candidates(coarse_xyz_np, coarse_data['scaling'],
                                       fine_xyz_np, k_candidates=k_candidates)
    cand = torch.from_numpy(cand_np).long()

    parent_indices, maha_distances = compute_maha_batched(fine, coarse, cand, chunk_size)
    return parent_indices.cpu(), maha_distances.cpu()


# =============================================================================
# Main hierarchy builder
# =============================================================================
def build_hierarchy(model_path, lod_min=1, lod_max=None,
                    k_candidates=64, chunk_size=8192, device='cuda',
                    mode='kl'):
    cfg = load_config(model_path)

    sh_degree = cfg.sh_degree
    lod1_scaling_lower_bound = cfg.lod1_scaling_lower_bound
    lod_scaling_ratio = cfg.lod_scaling_ratio

    if lod_max is None:
        lod_max = searchForMaxLod(os.path.join(model_path, "point_cloud"))

    print(f"\n{'='*60}")
    print(f"HLoD Hierarchy Builder  [mode={mode}]")
    print(f"  model: {model_path}  LOD {lod_min}->{lod_max}  K={k_candidates}")
    print(f"{'='*60}")

    lod_data = {}
    for lod in range(lod_min, lod_max + 1):
        lod_data[lod] = load_lod_level(model_path, lod, sh_degree,
                                        lod1_scaling_lower_bound, lod_scaling_ratio,
                                        lod_max, device="cpu")
        d = lod_data[lod]
        print(f"  LOD {lod}: {d['count']:>8,} G  scale_lb={d['scaling_lower_bound']:.6f}")

    parent_indices, distances = {}, {}

    for child_lod in range(lod_min + 1, lod_max + 1):
        parent_lod = child_lod - 1
        print(f"\n--- LOD {parent_lod} ({lod_data[parent_lod]['count']:,}) -> LOD {child_lod} ({lod_data[child_lod]['count']:,}) ---")

        t0 = time.time()

        if mode == 'kl':
            pidx, dist = build_pair(
                lod_data[parent_lod], lod_data[child_lod],
                k_candidates=k_candidates, chunk_size=chunk_size, device=device)
        elif mode == 'maha':
            pidx, dist = build_pair_maha(
                lod_data[parent_lod], lod_data[child_lod],
                k_candidates=k_candidates, chunk_size=chunk_size, device=device)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        dt = time.time() - t0
        parent_indices[child_lod] = pidx
        distances[child_lod] = dist

        N_coarse = lod_data[parent_lod]['count']
        children_count = torch.zeros(N_coarse, dtype=torch.long)
        children_count.scatter_add_(0, pidx, torch.ones_like(pidx))
        childless = (children_count == 0).sum().item()
        print(f"  {dt:.1f}s | dist med={dist.median().item():.4f} p95={dist.quantile(0.95).item():.4f}")
        print(f"  children/parent: med={children_count.float().median().item():.0f} "
              f"max={children_count.max().item()} | childless={childless}")

    hierarchy = {
        'parent_indices': parent_indices,
        'distances': distances,
        'lod_counts': {lod: lod_data[lod]['count'] for lod in range(lod_min, lod_max + 1)},
        'lod_scaling_lower_bounds': {lod: lod_data[lod]['scaling_lower_bound']
                                     for lod in range(lod_min, lod_max + 1)},
        'lod_min': lod_min, 'lod_max': lod_max,
        'k_candidates': k_candidates,
        'model_path': os.path.abspath(model_path),
        'mode': mode,
    }

    return hierarchy, lod_data


def save_hierarchy(hierarchy, model_path, filename=None):
    out_path = os.path.join(model_path, filename or "hierarchy.pt")
    torch.save(hierarchy, out_path)
    print(f"Saved: {out_path} ({os.path.getsize(out_path) / 1024**2:.2f} MB)")


def load_hierarchy(model_path):
    return torch.load(os.path.join(model_path, "hierarchy.pt"), weights_only=False)


def load_hierarchy_meta_as_hierarchy(model_path):
    """Assemble a hierarchy dict from hierarchical-training `hierarchy_meta.pt`.

    Replace-mode hierarchical training stores all LODs in a single combined
    model; lineage lives in `hierarchy_meta.pt` as global `parent_idx` and
    `birth_lod`. This converts that flat global mapping into per-LOD local
    parent indices matching the format built by `build_hierarchy`.
    """
    import glob as _glob

    pc_dir = os.path.join(model_path, "point_cloud")

    best = None
    for d in _glob.glob(os.path.join(pc_dir, "lod_*_iteration_*")):
        parts = os.path.basename(d).split("_")
        if len(parts) < 4:
            continue
        try:
            lod = int(parts[1]); it = int(parts[3])
        except ValueError:
            continue
        meta_path = os.path.join(d, "hierarchy_meta.pt")
        if not os.path.exists(meta_path):
            continue
        key = (lod, it)
        if best is None or key > best[0]:
            best = (key, meta_path)

    if best is None:
        raise FileNotFoundError(f"No hierarchy_meta.pt found under {pc_dir}")

    meta = torch.load(best[1], map_location='cpu', weights_only=False)
    parent_idx_global = meta['parent_idx'].long()
    birth_lod = meta['birth_lod'].long()

    lod_min = int(birth_lod.min().item())
    lod_max = int(birth_lod.max().item())

    lod_counts = {}
    global_to_local = torch.full((birth_lod.shape[0],), -1, dtype=torch.long)
    for L in range(lod_min, lod_max + 1):
        mask_L = (birth_lod == L)
        n = int(mask_L.sum().item())
        lod_counts[L] = n
        global_to_local[mask_L] = torch.arange(n, dtype=torch.long)

    parent_indices = {}
    n_orphans_total = 0
    for L in range(lod_min + 1, lod_max + 1):
        child_mask = (birth_lod == L)
        parents_global = parent_idx_global[child_mask]
        local = torch.full((int(child_mask.sum().item()),), -1, dtype=torch.long)
        valid = parents_global >= 0
        if valid.any():
            mapped = global_to_local[parents_global[valid]]
            local[valid] = mapped
        n_orphans_total += int((local < 0).sum().item())
        parent_indices[L] = local

    print(f"Loaded hierarchy_meta from {os.path.relpath(best[1], model_path)}: "
          f"LOD {lod_min}->{lod_max}  "
          f"({', '.join(f'LOD{l}:{lod_counts[l]:,}' for l in sorted(lod_counts))})"
          f"  orphans={n_orphans_total}")

    return {
        'parent_indices': parent_indices,
        'distances': {},
        'lod_counts': lod_counts,
        'lod_scaling_lower_bounds': {},
        'lod_min': lod_min,
        'lod_max': lod_max,
        'mode': 'hierarchy_meta',
        'model_path': os.path.abspath(model_path),
    }


def print_hierarchy_summary(hierarchy):
    print(f"\n{'='*60}")
    print(f"Hierarchy: LOD {hierarchy['lod_min']} -> {hierarchy['lod_max']}  [mode={hierarchy.get('mode', 'kl')}]")
    total = 0
    for lod in range(hierarchy['lod_min'], hierarchy['lod_max'] + 1):
        cnt = hierarchy['lod_counts'][lod]
        total += cnt
        extra = ""
        if lod > hierarchy['lod_min'] and hierarchy.get('distances', {}).get(lod) is not None:
            dist = hierarchy['distances'][lod]
            extra = f"  med_dist={dist.median().item():.3f}"
        print(f"  LOD {lod}: {cnt:>9,} G{extra}")
    print(f"  Total: {total:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or export a Gaussian LoD hierarchy")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--mode", choices=["kl", "maha", "hierarchy_meta"], default="kl",
                        help="Assignment metric: kl=full KL divergence, maha=Mahalanobis only, "
                             "hierarchy_meta=read hierarchical-training hierarchy_meta.pt (replace-mode)")
    parser.add_argument("--lod_max", type=int, default=None)
    parser.add_argument("--lod_min", type=int, default=1)
    parser.add_argument("--k_candidates", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "hierarchy_meta":
        hierarchy = load_hierarchy_meta_as_hierarchy(args.model_path)
    else:
        hierarchy, _ = build_hierarchy(
            model_path=args.model_path, lod_min=args.lod_min, lod_max=args.lod_max,
            k_candidates=args.k_candidates, chunk_size=args.chunk_size,
            device=args.device, mode=args.mode)

    print_hierarchy_summary(hierarchy)
    save_hierarchy(hierarchy, args.model_path, filename=args.output)
