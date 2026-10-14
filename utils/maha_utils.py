import torch
from utils.general_utils import build_rotation


def _maha_hinge(child_xyz, parent_xyz, parent_scaling, parent_rotation, margin):
    R_p = build_rotation(parent_rotation)
    inv_s2 = 1.0 / (parent_scaling ** 2 + 1e-8)

    delta = child_xyz - parent_xyz
    delta_local = (R_p.transpose(1, 2) @ delta.unsqueeze(2)).squeeze(2)

    maha_sq = (delta_local ** 2 * inv_s2).sum(dim=1)
    # Clamp per-Gaussian Maha to prevent explosion at finer LODs where
    # parent scales are small and 1/scale^2 amplifies tiny displacements.
    maha_sq = maha_sq.clamp(max=100.0)
    return torch.relu(maha_sq - margin)


def _reduce_per_child_loss(per_child_loss, parent_ids, mode):

    if per_child_loss.numel() == 0:
        return per_child_loss.new_tensor(0.0)
    if mode == "mean":
        return per_child_loss.mean()
    if mode == "worst_child":
        unique_parents, inverse = torch.unique(parent_ids, return_inverse=True)
        out = torch.full(
            (unique_parents.numel(),), float("-inf"),
            device=per_child_loss.device, dtype=per_child_loss.dtype,
        )
        out = out.scatter_reduce(0, inverse, per_child_loss, reduce="amax", include_self=True)
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out.mean()
    raise ValueError(f"unknown maha reduce mode: {mode!r}")


def compute_maha_regularizer(gaussians, margin=1.0, scope="immediate",
                             ancestor_decay=1.0, max_ancestors=0, active_mask=None,
                             current_lod_only=False, reduce="mean"):

    child_xyz_all = gaussians.get_xyz
    device = child_xyz_all.device

    if current_lod_only and gaussians._birth_lod is not None:
        current_lod_mask = (gaussians._birth_lod.to(device) == int(gaussians.current_lod))
        if active_mask is None:
            active_mask = current_lod_mask
        else:
            active_mask = active_mask.to(device).bool() & current_lod_mask

    if gaussians._parent_idx is None:
        return torch.tensor(0.0, device=device)

    raw_parent_ids = gaussians._parent_idx.to(device).long()
    N = child_xyz_all.shape[0]
    max_depth = int(max_ancestors) if max_ancestors is not None else 0

    if scope == "oldest":
        if active_mask is not None:
            original_indices = active_mask.to(device).nonzero(as_tuple=True)[0]
            parent_ids = raw_parent_ids[original_indices]
        else:
            original_indices = torch.arange(N, device=device)
            parent_ids = raw_parent_ids
        oldest_parent_ids = torch.full((N,), -1, device=device, dtype=torch.long)
        depth = 1
        while True:
            valid = (parent_ids >= 0) & (parent_ids < N)
            if not valid.any():
                break
            child_idx = original_indices[valid]
            parent_idx = parent_ids[valid]
            oldest_parent_ids[child_idx] = parent_idx
            if max_depth > 0 and depth >= max_depth:
                break
            original_indices = child_idx
            parent_ids = raw_parent_ids[parent_idx]
            depth += 1
            if depth > N:
                break

        valid = (oldest_parent_ids >= 0) & (oldest_parent_ids < N)
        if not valid.any():
            return torch.tensor(0.0, device=device)
        child_idx = torch.arange(N, device=device)[valid]
        parent_idx = oldest_parent_ids[valid]
        per_child = _maha_hinge(
            child_xyz_all[child_idx],
            child_xyz_all[parent_idx].detach(),
            gaussians.get_scaling[parent_idx].detach(),
            gaussians._rotation[parent_idx].detach(),
            margin,
        )
        return _reduce_per_child_loss(per_child, parent_idx, reduce)

    if scope != "immediate":
        raise ValueError(f"compute_maha_regularizer: unknown scope={scope!r}")

    has_parent = (raw_parent_ids >= 0) & (raw_parent_ids < N)
    if active_mask is not None:
        has_parent = has_parent & active_mask.to(device)
    if has_parent.sum() == 0:
        return torch.tensor(0.0, device=device)

    child_xyz = child_xyz_all[has_parent]
    parent_ids = raw_parent_ids[has_parent]

    parent_xyz = child_xyz_all[parent_ids].detach()
    parent_scaling = gaussians.get_scaling[parent_ids].detach()
    parent_rotation = gaussians._rotation[parent_ids].detach()

    per_child = _maha_hinge(child_xyz, parent_xyz, parent_scaling, parent_rotation, margin)
    return _reduce_per_child_loss(per_child, parent_ids, reduce)
