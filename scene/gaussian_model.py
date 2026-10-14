import math
import time
import torch
import numpy as np
from dataclasses import dataclass
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import random

from utils.system_utils import searchForMaxIterationGivenLod, searchForMaxLod, checkFileSize

try:
    import hierarchical_traversal_cuda as _hier_trav_cuda
except Exception:
    _hier_trav_cuda = None

@dataclass
class HierarchyIndex:
    lod_offsets: torch.Tensor
    lod_offsets_host: tuple
    lod_counts: torch.Tensor
    parent_indices_int32: torch.Tensor
    root_active_uint8: torch.Tensor
    leaf_active_uint8: torch.Tensor
    finest_active_uint8: torch.Tensor
    child_offsets: torch.Tensor
    child_indices: torch.Tensor
    version: int


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


class GaussianModel:
    def setup_scaling_activation(self):
        if self.current_lod < self.max_lod:
            self.scaling_lower_bound = self.lod1_scaling_lower_bound / self.lod_scaling_ratio ** (self.current_lod - 1)
        else:
            self.scaling_lower_bound = 0.0

    def _compute_per_gaussian_lb(self):
        if self._birth_lod is None:
            return None
        # Birth levels are authoritative, including newly appended/unsorted rows.
        # Cached offsets can still describe the model before densification.
        lod = self._birth_lod.to(dtype=torch.float32)
        lb = self.lod1_scaling_lower_bound / (self.lod_scaling_ratio ** (lod - 1))
        return torch.where(lod < self.max_lod, lb, torch.zeros_like(lb))

    def compute_mip_floor(self, train_cameras, mode="fixed", chunk_size=262144):

        with torch.no_grad():
            xyz = self._xyz.detach()
            device = xyz.device
            n_points = xyz.shape[0]
            chunk_size = max(1, int(chunk_size))

            if mode == "worst_case":
                floor = torch.zeros((n_points,), device=device)
                reduce_fn = torch.maximum
            else:
                floor = torch.full((n_points,), float('inf'), device=device)
                reduce_fn = torch.minimum

            for cam in train_cameras:
                f = cam.image_width / (2.0 * math.tan(cam.FoVx / 2.0))
                cam_center = cam.camera_center.to(device)
                denom = 2.0 * f
                for start in range(0, n_points, chunk_size):
                    end = min(start + chunk_size, n_points)
                    d = torch.norm(xyz[start:end] - cam_center, dim=1)
                    d_over_2f = d / denom
                    floor[start:end] = reduce_fn(floor[start:end], d_over_2f)

            if mode == "lod_scaled" and self._birth_lod is not None:
                lod_factor = torch.pow(
                    torch.tensor(float(self.lod_scaling_ratio), device=device),
                    (self.max_lod - self._birth_lod.to(device).float()).clamp(min=0.0),
                )
                floor = floor * lod_factor

            self._mip_floor = floor.detach()

    def _mip_floor_log_threshold(self):
        
        if self._mip_floor is None:
            return None
        if self._per_gaussian_lb is not None:
            room = self._mip_floor - self._per_gaussian_lb
        else:
            room = self._mip_floor - float(self.scaling_lower_bound)
        return torch.log(room.clamp(min=1e-12)).unsqueeze(-1)

    def mip_floor_active_set_step(self, optimizer):

        if (self._mip_floor is None
                or not getattr(self, '_mip_floor_clamp_raw', False)
                or not getattr(self, '_apply_mip_floor', True)
                or self._scaling.grad is None):
            return
        log_floor = self._mip_floor_log_threshold()
        if log_floor is None:
            return
        at_floor = self._scaling.data <= log_floor
        push_into_wall = at_floor & (self._scaling.grad > 0)
        if not push_into_wall.any():
            return
        self._scaling.grad[push_into_wall] = 0
        for group in optimizer.param_groups:
            if group.get("name") != "scaling":
                continue
            state = optimizer.state.get(group["params"][0], None)
            if state is None:
                return
            if "exp_avg" in state and state["exp_avg"].shape == push_into_wall.shape:
                state["exp_avg"][push_into_wall] = 0
            if "exp_avg_sq" in state and state["exp_avg_sq"].shape == push_into_wall.shape:
                state["exp_avg_sq"][push_into_wall] = 0
            return

    def mip_floor_clamp_raw_post_step(self):
        
        if (self._mip_floor is None
                or not getattr(self, '_mip_floor_clamp_raw', False)
                or not getattr(self, '_apply_mip_floor', True)):
            return
        log_floor = self._mip_floor_log_threshold()
        if log_floor is None:
            return
        with torch.no_grad():
            self._scaling.data.clamp_(min=log_floor)

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.setup_scaling_activation()
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.shape_activation = torch.sigmoid

    def __init__(self,
                 sh_degree                  : int,
                 lod1_scaling_lower_bound   : float,
                 lod_scaling_ratio          : float,
                 increase_lod_num_childs    : int = 1,
                 current_lod                : int = 1,
                 max_lod                    : int = 5,
                 use_voxel_sampling         : bool = False,
                 voxel_sampling_size        : float = 0.2,
                 ):

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.lod1_scaling_lower_bound = lod1_scaling_lower_bound
        self.lod_scaling_ratio = lod_scaling_ratio
        self.increase_lod_num_childs = increase_lod_num_childs
        self.current_lod = current_lod
        self.max_lod = max_lod
        self.use_voxel_sampling = use_voxel_sampling
        self.voxel_sampling_size = voxel_sampling_size


        self._mip_floor = None
        self._apply_mip_floor = True
        self._birth_lod = None
        self._parent_idx = None
        self._per_gaussian_lb = None
        self._train_hits = None
        self._recent_active_hits = None
        self._shape_factors_cache = None
        self._shape_factors_cache_version = None
        self._shape_factors_cache_enabled = False
        self._densify_current_lod_only = False
        self._per_parent_budget = None

        self._gaussian_version = 0
        self._is_parent_cache = None
        self._is_parent_cache_version = None
        self._hierarchy_index = None
        self._hierarchy_index_version = None
        self._top_down_candidate_idx_cache = {}
        self._top_down_candidate_idx_cache_version = None
        self._last_training_args = None
        self._optimizer_needs_rebuild = False

        self._lod_offsets = None

        self.setup_functions()

    def _invalidate_topology_caches(self):
        self._gaussian_version += 1
        self._is_parent_cache = None
        self._is_parent_cache_version = None
        self._hierarchy_index = None
        self._hierarchy_index_version = None
        self._top_down_candidate_idx_cache = {}
        self._top_down_candidate_idx_cache_version = None

    _LOD_ATTR_FIELDS = {
        "xyz": "_xyz",
        "f_dc": "_features_dc",
        "f_rest": "_features_rest",
        "opacity": "_opacity",
        "scaling": "_scaling",
        "rotation": "_rotation",
    }

    def _gather_lod_attribute(self, attr, indices):
        field = self._LOD_ATTR_FIELDS[attr]
        indices = indices.to(device=getattr(self, field).device, dtype=torch.long)
        return getattr(self, field)[indices]


    def _refresh_lod_offsets(self):

        if self._birth_lod is None or self._birth_lod.numel() == 0:
            self._lod_offsets = torch.zeros(self.max_lod + 1, dtype=torch.long,
                                            device=self._xyz.device if self._xyz.is_cuda else "cpu")
            return self._lod_offsets
        device = self._birth_lod.device
        bl = self._birth_lod.to(torch.long)
        counts = torch.bincount(bl.clamp(min=0, max=self.max_lod), minlength=self.max_lod + 1)
        # counts[0] holds 0-or-out-of-range; we ignore it. cumsum over LOD 1..max_lod.
        offsets = torch.zeros(self.max_lod + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(counts[1:], dim=0)
        self._lod_offsets = offsets
        return offsets

    def _ensure_hierarchy_index(self):
    
        N = self._xyz.shape[0]
        device = self._xyz.device
        cached = self._hierarchy_index
        if (
            cached is not None
            and self._hierarchy_index_version == self._gaussian_version
            and cached.lod_offsets.device == device
            and cached.child_offsets.shape[0] == N + 1
        ):
            return cached

        lod_offsets = self._refresh_lod_offsets()
        if lod_offsets is None:
            lod_offsets = torch.zeros(self.max_lod + 1, dtype=torch.long, device=device)
        lod_offsets = lod_offsets.to(device=device, dtype=torch.long)
        lod_counts = lod_offsets[1:] - lod_offsets[:-1]

        child_offsets = torch.zeros(N + 1, dtype=torch.int32, device=device)
        child_indices = torch.empty(0, dtype=torch.int32, device=device)
        parent_indices_int32 = torch.full((N,), -1, dtype=torch.int32, device=device)
        is_parent = torch.zeros(N, dtype=torch.bool, device=device)
        if self._parent_idx is not None and N > 0:
            parent_idx = self._parent_idx.to(device=device, dtype=torch.long)
            if N > torch.iinfo(torch.int32).max:
                raise RuntimeError("CUDA hierarchy traversal requires fewer than 2^31 nodes")
            parent_indices_int32 = parent_idx.to(dtype=torch.int32)
            valid_child = (parent_idx >= 0) & (parent_idx < N)
            if valid_child.any():
                raw_child_indices = valid_child.nonzero(as_tuple=True)[0]
                raw_parent_indices = parent_idx[raw_child_indices]
                is_parent[raw_parent_indices] = True
                order = torch.argsort(raw_parent_indices, stable=True)
                sorted_parent = raw_parent_indices[order]
                child_indices = raw_child_indices[order].to(dtype=torch.int32)
                counts = torch.bincount(sorted_parent, minlength=N)
                child_offsets[1:] = torch.cumsum(counts, dim=0)

        birth_lod = self._birth_lod.to(device=device)
        included = birth_lod <= int(self.current_lod)
        root_active_uint8 = (included & (parent_indices_int32 < 0)).to(torch.uint8)
        leaf_active_uint8 = (included & ~is_parent).to(torch.uint8)
        finest_active_uint8 = (birth_lod == int(self.current_lod)).to(torch.uint8)
        self._is_parent_cache = is_parent
        self._is_parent_cache_version = self._gaussian_version

        self._hierarchy_index = HierarchyIndex(
            lod_offsets=lod_offsets,
            lod_offsets_host=tuple(int(value) for value in lod_offsets.detach().cpu().tolist()),
            lod_counts=lod_counts,
            parent_indices_int32=parent_indices_int32,
            root_active_uint8=root_active_uint8,
            leaf_active_uint8=leaf_active_uint8,
            finest_active_uint8=finest_active_uint8,
            child_offsets=child_offsets,
            child_indices=child_indices,
            version=self._gaussian_version,
        )
        self._hierarchy_index_version = self._gaussian_version
        return self._hierarchy_index

    def rebuild_hierarchy_index(self):
        self._hierarchy_index = None
        return self._ensure_hierarchy_index()

    def _indices_in_lod_range(self, lod_lo, lod_hi, device=None):
        if device is None:
            device = self._xyz.device
        if self._birth_lod is None:
            return torch.empty(0, dtype=torch.long, device=device)
        idx = self._ensure_hierarchy_index()
        lod_lo = max(1, int(lod_lo))
        lod_hi = min(int(lod_hi), self.max_lod)
        if lod_hi < lod_lo:
            return torch.empty(0, dtype=torch.long, device=device)
        lo = int(idx.lod_offsets[lod_lo - 1].item())
        hi = int(idx.lod_offsets[lod_hi].item())
        if hi <= lo:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.arange(lo, hi, device=device, dtype=torch.long)

    def active_indices_to_mask(self, active_indices, N=None, device=None):
        if device is None:
            device = self._xyz.device
        if N is None:
            N = self._xyz.shape[0]
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        if active_indices is not None and active_indices.numel() > 0:
            mask[active_indices.to(device=device, dtype=torch.long)] = True
        return mask

    def _lod_slice(self, lod_lo, lod_hi=None):

        if self._lod_offsets is None:
            return None
        if lod_hi is None:
            lod_hi = lod_lo
        lo = int(self._lod_offsets[lod_lo - 1].item()) if lod_lo > 0 else 0
        hi = int(self._lod_offsets[lod_hi].item())
        return slice(lo, hi)

    def _indices_at_lod(self, lod, device=None):

        if device is None:
            device = self._xyz.device
        if self._lod_offsets is not None:
            sl = self._lod_slice(lod)
            if sl is None:
                return torch.empty(0, dtype=torch.long, device=device)
            if sl.stop <= sl.start:
                return torch.empty(0, dtype=torch.long, device=device)
            return torch.arange(sl.start, sl.stop, device=device, dtype=torch.long)
        if self._birth_lod is None:
            return torch.empty(0, dtype=torch.long, device=device)
        return (self._birth_lod == lod).nonzero(as_tuple=True)[0]

    def _is_birth_lod_sorted(self):
        if self._birth_lod is None or self._birth_lod.numel() <= 1:
            return True
        diffs = self._birth_lod[1:].to(torch.long) - self._birth_lod[:-1].to(torch.long)
        return bool((diffs >= 0).all().item())

    def _apply_permutation(self, perm):

        device = self._xyz.device
        perm = perm.to(device=device, dtype=torch.long)
        N = self._xyz.shape[0]
        assert perm.shape == (N,), f"perm shape {perm.shape} mismatch N={N}"

        # Build inverse permutation for parent-value remap.
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(N, device=device, dtype=torch.long)

        # --- nn.Parameters in the optimizer ---
        param_attrs = [
            ("_xyz", "xyz"),
            ("_features_dc", "f_dc"),
            ("_features_rest", "f_rest"),
            ("_opacity", "opacity"),
            ("_scaling", "scaling"),
            ("_rotation", "rotation"),
        ]
        opt = self.optimizer
        if opt is not None:
            # Build name -> group lookup
            name_to_group = {g["name"]: g for g in opt.param_groups}
            for attr, name in param_attrs:
                old_param = getattr(self, attr)
                group = name_to_group.get(name)
                if group is None or group["params"][0] is not old_param:
                    # Optimizer is stale relative to attributes; just permute the
                    # parameter without touching optimizer state.
                    new_param = nn.Parameter(old_param.detach()[perm].requires_grad_(True))
                    setattr(self, attr, new_param)
                    continue
                state = opt.state.get(old_param, None)
                new_data = old_param.detach()[perm].contiguous()
                new_param = nn.Parameter(new_data.requires_grad_(True))
                if state is not None:
                    new_state = {k: v for k, v in state.items()}
                    if "exp_avg" in new_state:
                        new_state["exp_avg"] = new_state["exp_avg"][perm].contiguous()
                    if "exp_avg_sq" in new_state:
                        new_state["exp_avg_sq"] = new_state["exp_avg_sq"][perm].contiguous()
                    del opt.state[old_param]
                    opt.state[new_param] = new_state
                group["params"][0] = new_param
                setattr(self, attr, new_param)
        else:
            for attr, _ in param_attrs:
                old_param = getattr(self, attr)
                if isinstance(old_param, nn.Parameter):
                    new_param = nn.Parameter(old_param.detach()[perm].requires_grad_(True))
                else:
                    new_param = old_param[perm]
                setattr(self, attr, new_param)

        # --- Plain per-row tensors ---
        if self.max_radii2D is not None and self.max_radii2D.nelement() != 0:
            self.max_radii2D = self.max_radii2D[perm]
        if self.xyz_gradient_accum is not None and self.xyz_gradient_accum.nelement() != 0:
            self.xyz_gradient_accum = self.xyz_gradient_accum[perm]
        if self.denom is not None and self.denom.nelement() != 0:
            self.denom = self.denom[perm]
        if self._birth_lod is not None:
            self._birth_lod = self._birth_lod[perm]
        if self._mip_floor is not None:
            self._mip_floor = self._mip_floor[perm]
        if self._per_gaussian_lb is not None:
            self._per_gaussian_lb = self._per_gaussian_lb[perm]
        if self._train_hits is not None:
            self._train_hits = self._train_hits[perm]
        if self._recent_active_hits is not None:
            self._recent_active_hits = self._recent_active_hits[perm]
        # --- _parent_idx: row-permute AND value-remap ---
        if self._parent_idx is not None:
            pidx = self._parent_idx[perm]
            valid = pidx >= 0
            if valid.any():
                pidx = pidx.clone()
                pidx[valid] = inv_perm[pidx[valid]]
            self._parent_idx = pidx

        self._invalidate_topology_caches()

    def _resort_by_birth_lod(self, force=False):

        if self._birth_lod is None or self._birth_lod.numel() == 0:
            self._refresh_lod_offsets()
            self._rebuild_optimizer_if_needed()
            return False
        if not force and self._is_birth_lod_sorted():
            self._refresh_lod_offsets()
            self._rebuild_optimizer_if_needed()
            return False

        sorted_keys, perm = torch.sort(self._birth_lod.to(torch.long), stable=True)
        self._apply_permutation(perm)
        self._refresh_lod_offsets()
        self._rebuild_optimizer_if_needed()
        return True

    def _get_is_parent(self):
        if self._parent_idx is None:
            return None
        N = self._xyz.shape[0]
        cached = self._is_parent_cache
        if (cached is not None
                and self._is_parent_cache_version == self._gaussian_version
                and cached.shape[0] == N
                and cached.device == self._xyz.device):
            return cached
        device = self._xyz.device
        is_parent = torch.zeros(N, dtype=torch.bool, device=device)
        valid_refs = self._parent_idx[self._parent_idx >= 0]
        if valid_refs.numel() > 0:
            is_parent[valid_refs.unique()] = True
        self._is_parent_cache = is_parent
        self._is_parent_cache_version = self._gaussian_version
        return is_parent

    def to_cpu(self):
        self._xyz = self._xyz.to("cpu")
        self._features_dc = self._features_dc.to("cpu")
        self._features_rest = self._features_rest.to("cpu")
        self._opacity = self._opacity.to("cpu")
        self._scaling = self._scaling.to("cpu")
        self._rotation = self._rotation.to("cpu")
        if self._mip_floor is not None:
            self._mip_floor = self._mip_floor.to("cpu")
        if self._birth_lod is not None:
            self._birth_lod = self._birth_lod.to("cpu")
        if self._parent_idx is not None:
            self._parent_idx = self._parent_idx.to("cpu")
        if self._per_gaussian_lb is not None:
            self._per_gaussian_lb = self._per_gaussian_lb.to("cpu")
        if self._train_hits is not None:
            self._train_hits = self._train_hits.to("cpu")
        if self._recent_active_hits is not None:
            self._recent_active_hits = self._recent_active_hits.to("cpu")
        if self._lod_offsets is not None:
            self._lod_offsets = self._lod_offsets.to("cpu")
        self._invalidate_topology_caches()

    def to_cuda(self):
        self._xyz = self._xyz.to("cuda")
        self._features_dc = self._features_dc.to("cuda")
        self._features_rest = self._features_rest.to("cuda")
        self._opacity = self._opacity.to("cuda")
        self._scaling = self._scaling.to("cuda")
        self._rotation = self._rotation.to("cuda")
        if self._mip_floor is not None:
            self._mip_floor = self._mip_floor.to("cuda")
        if self._birth_lod is not None:
            self._birth_lod = self._birth_lod.to("cuda")
        if self._parent_idx is not None:
            self._parent_idx = self._parent_idx.to("cuda")
        if self._per_gaussian_lb is not None:
            self._per_gaussian_lb = self._per_gaussian_lb.to("cuda")
        if self._train_hits is not None:
            self._train_hits = self._train_hits.to("cuda")
        if self._recent_active_hits is not None:
            self._recent_active_hits = self._recent_active_hits.to("cuda")
        if self._lod_offsets is not None:
            self._lod_offsets = self._lod_offsets.to("cuda")
        self._invalidate_topology_caches()


    @property
    def get_scaling(self):
        raw_scaling = self._scaling
        if self._per_gaussian_lb is not None:
            base = self.scaling_activation(raw_scaling) + self._per_gaussian_lb.unsqueeze(-1)
        else:
            base = self.scaling_activation(raw_scaling) + self.scaling_lower_bound


        if self._mip_floor is not None and getattr(self, '_apply_mip_floor', True):
            floor = self._mip_floor.unsqueeze(-1)
            beta = float(getattr(self, '_mip_floor_smooth_beta', 0.0))
            if beta > 0.0:
                # Smooth max via softplus: final = floor + softplus(β(base − floor))/β.
                # β → ∞ recovers torch.max; finite β gives a globally differentiable,
                # monotone lift with no zero-gradient region below the floor.
                base = floor + torch.nn.functional.softplus(beta * (base - floor)) / beta
            else:
                base = torch.max(base, floor)
        return base

    def get_scaling_at(self, indices=None):
        if indices is None:
            return self.get_scaling
        indices = indices.to(device=self._scaling.device, dtype=torch.long)
        raw_scaling = self._gather_lod_attribute("scaling", indices)
        if self._per_gaussian_lb is not None:
            base = self.scaling_activation(raw_scaling) + self._per_gaussian_lb[indices].unsqueeze(-1)
        else:
            base = self.scaling_activation(raw_scaling) + self.scaling_lower_bound

        if self._mip_floor is not None and getattr(self, '_apply_mip_floor', True):
            floor = self._mip_floor[indices].unsqueeze(-1)
            beta = float(getattr(self, '_mip_floor_smooth_beta', 0.0))
            if beta > 0.0:
                base = floor + torch.nn.functional.softplus(beta * (base - floor)) / beta
            else:
                base = torch.max(base, floor)
        return base

    def get_rotation_at(self, indices=None):
        if indices is None:
            return self.get_rotation
        indices = indices.to(device=self._rotation.device, dtype=torch.long)
        return self.rotation_activation(self._gather_lod_attribute("rotation", indices))

    def get_xyz_at(self, indices=None):
        if indices is None:
            return self.get_xyz
        indices = indices.to(device=self._xyz.device, dtype=torch.long)
        return self._gather_lod_attribute("xyz", indices)

    def get_features_at(self, indices=None):
        if indices is None:
            return self.get_features
        indices = indices.to(device=self._features_dc.device, dtype=torch.long)
        features_dc = self._gather_lod_attribute("f_dc", indices)
        features_rest = self._gather_lod_attribute("f_rest", indices)
        return torch.cat((features_dc, features_rest), dim=1)

    def get_opacity_at(self, indices=None):
        if indices is None:
            return self.get_opacity
        indices = indices.to(device=self._opacity.device, dtype=torch.long)
        return self.opacity_activation(self._gather_lod_attribute("opacity", indices))

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)


    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_covariance_at(self, indices=None, scaling_modifier=1):
        if indices is None:
            return self.get_covariance(scaling_modifier)
        indices = indices.to(device=self._rotation.device, dtype=torch.long)
        rotation = self._gather_lod_attribute("rotation", indices)
        return self.covariance_activation(self.get_scaling_at(indices), scaling_modifier, rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, init_opacity : float):
        self.spatial_lr_scale = spatial_lr_scale

        if self.use_voxel_sampling:
            voxel_points = np.round(np.asarray(pcd.points)/self.voxel_sampling_size)
            unique_voxel_points = np.unique(voxel_points, axis=0)
            sampled_indices = [random.choice(np.where((voxel_points == voxel_point).all(axis=1))[0]) for voxel_point in unique_voxel_points]
            fused_point_cloud = torch.tensor(np.asarray(pcd.points[sampled_indices])).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors[sampled_indices])).float().cuda())
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points[sampled_indices])).float().cuda()), 0.0000001)
        else:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(init_opacity * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        dist = torch.clamp_min(torch.sqrt(dist2) - self.scaling_lower_bound, 0.0000001)
        scales = torch.log(dist)[...,None].repeat(1, 3)
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._birth_lod = torch.full((fused_point_cloud.shape[0],), self.current_lod, dtype=torch.int, device="cuda")
        self._invalidate_topology_caches()
        self._resort_by_birth_lod()

    def _make_optimizer_groups(self, training_args):
        return [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

    def _build_optimizer(self, training_args):
        self._last_training_args = training_args
        groups = self._make_optimizer_groups(training_args)
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self._optimizer_needs_rebuild = False

    def _rebuild_optimizer_if_needed(self):
        if self._optimizer_needs_rebuild and self._last_training_args is not None:
            self._build_optimizer(self._last_training_args)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        N = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.denom = torch.zeros((N, 1), device="cuda")
        self.max_radii2D = torch.zeros((N,), device="cuda")
        self._train_hits = torch.zeros((N,), dtype=torch.int32, device="cuda")
        self._recent_active_hits = torch.zeros((N,), dtype=torch.int32, device="cuda")

        self._build_optimizer(training_args)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = float(self.xyz_scheduler_args(iteration))
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l

    def load(self, model_path, load_lod=None, load_iteration=None, device="cuda", lod_cap=None):
        self.model_path = model_path

        if load_lod != -1:
            self.loaded_lod = load_lod
        else:
            self.loaded_lod = searchForMaxLod(os.path.join(self.model_path, "point_cloud"))

        if load_iteration == -1:
            self.loaded_iter = searchForMaxIterationGivenLod(os.path.join(self.model_path, "point_cloud"), self.loaded_lod)
        else:
            self.loaded_iter = load_iteration

        model_export_path = os.path.join(self.model_path,
                                    "point_cloud",
                                    f"lod_{self.loaded_lod}_iteration_{self.loaded_iter}")
        meta_path = os.path.join(model_export_path, "hierarchy_meta.pt")
        if os.path.exists(meta_path):
            self.load_hierarchy_chunks(model_export_path, device=device, lod_cap=lod_cap)
        else:
            self.load_ply(os.path.join(model_export_path, "point_cloud.ply"), device=device)

        self.current_lod = min(self.loaded_lod, lod_cap) if lod_cap is not None else self.loaded_lod
        self.setup_scaling_activation()

    def load_level(self, model_path, lod, iteration=-1, device="cuda", source_lod=None):
        """Load one birth level from the selected HLoD model export."""
        if source_lod is None:
            source_lod = searchForMaxLod(os.path.join(model_path, "point_cloud"))
        source_iter = searchForMaxIterationGivenLod(os.path.join(model_path, "point_cloud"), source_lod) if iteration == -1 else iteration
        source_dir = os.path.join(model_path, "point_cloud", f"lod_{source_lod}_iteration_{source_iter}")
        if os.path.exists(os.path.join(source_dir, "hierarchy_meta.pt")):
            self.load(model_path, source_lod, source_iter, device=device, lod_cap=lod)
            self.retain_indices((self._birth_lod == lod).nonzero(as_tuple=True)[0])
        else:
            raise FileNotFoundError(f"HLoD birth-level loading requires hierarchy_meta.pt: {source_dir}")
        self.current_lod = lod

    def save_ply(self, path, indices=None):
        mkdir_p(os.path.dirname(path))

        if indices is not None:
            indices = indices.to(device=self._xyz.device, dtype=torch.long)
            xyz_t = self._xyz.detach()[indices]
            f_dc_t = self._features_dc.detach()[indices]
            f_rest_t = self._features_rest.detach()[indices]
            opacity_t = self._opacity.detach()[indices]
            scaling_t = self._scaling.detach()[indices]
            rotation_t = self._rotation.detach()[indices]
        else:
            xyz_t = self._xyz.detach()
            f_dc_t = self._features_dc.detach()
            f_rest_t = self._features_rest.detach()
            opacity_t = self._opacity.detach()
            scaling_t = self._scaling.detach()
            rotation_t = self._rotation.detach()

        xyz = xyz_t.cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = f_dc_t.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = f_rest_t.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity_t.cpu().numpy()

        if self._per_gaussian_lb is not None:
            save_lb = self._per_gaussian_lb[indices].unsqueeze(-1) if indices is not None else self._per_gaussian_lb.unsqueeze(-1)
        else:
            save_lb = self.scaling_lower_bound
        scale = scaling_t.cpu().numpy()

        rotation = rotation_t.cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def _hierarchy_meta_payload(self, chunk_paths=None, storage_format="flat"):
        hidx = self._ensure_hierarchy_index() if self._birth_lod is not None else None
        return {
            "version": 2,
            "storage_format": storage_format,
            "chunk_paths": chunk_paths or [],
            "current_lod": int(self.current_lod),
            "max_lod": int(self.max_lod),
            "parent_idx": self._parent_idx.detach().cpu() if self._parent_idx is not None else None,
            "birth_lod": self._birth_lod.detach().cpu() if self._birth_lod is not None else None,
            "per_gaussian_lb": self._per_gaussian_lb.detach().cpu() if self._per_gaussian_lb is not None else None,
            "mip_floor": self._mip_floor.detach().cpu() if self._mip_floor is not None else None,
            "train_hits": self._train_hits.detach().cpu() if self._train_hits is not None else None,
            "recent_active_hits": self._recent_active_hits.detach().cpu() if self._recent_active_hits is not None else None,
            "lod_offsets": hidx.lod_offsets.detach().cpu() if hidx is not None else None,
            "lod_counts": hidx.lod_counts.detach().cpu() if hidx is not None else None,
            "child_offsets": hidx.child_offsets.detach().cpu() if hidx is not None else None,
            "child_indices": hidx.child_indices.detach().cpu() if hidx is not None else None,
            "lod1_scaling_lower_bound": float(self.lod1_scaling_lower_bound),
            "lod_scaling_ratio": float(self.lod_scaling_ratio),
            "spatial_lr_scale": float(self.spatial_lr_scale),
            "apply_mip_floor": bool(self._apply_mip_floor),
            "dtype": {
                "parent_idx": str(self._parent_idx.dtype) if self._parent_idx is not None else None,
                "birth_lod": str(self._birth_lod.dtype) if self._birth_lod is not None else None,
                "xyz": str(self._xyz.dtype),
                "features": str(self._features_dc.dtype),
            },
        }

    def save_hierarchy_chunks(self, path, export_flat_legacy=False):
        mkdir_p(path)
        if self._birth_lod is None:
            self.save_ply(os.path.join(path, "point_cloud.ply"))
            return

        chunk_paths = []
        for lod in range(1, self.max_lod + 1):
            indices = self._indices_at_lod(lod, device=self._xyz.device)
            if indices.numel() == 0:
                continue
            rel_path = f"point_cloud_lod_{lod}.ply"
            self.save_ply(os.path.join(path, rel_path), indices=indices)
            chunk_paths.append({"lod": lod, "path": rel_path, "count": int(indices.numel())})

        if export_flat_legacy:
            self.save_ply(os.path.join(path, "point_cloud.ply"))

        torch.save(
            self._hierarchy_meta_payload(chunk_paths=chunk_paths, storage_format="lod_chunks"),
            os.path.join(path, "hierarchy_meta.pt"),
        )

    def _load_hierarchy_meta(self, meta_path, device="cuda", row_indices=None):
        if not os.path.exists(meta_path):
            return None
        meta = torch.load(meta_path, map_location=device, weights_only=True)
        self.current_lod = int(meta.get("current_lod", self.current_lod))
        self.max_lod = int(meta.get("max_lod", self.max_lod))
        self.lod1_scaling_lower_bound = float(meta.get("lod1_scaling_lower_bound", self.lod1_scaling_lower_bound))
        self.lod_scaling_ratio = float(meta.get("lod_scaling_ratio", self.lod_scaling_ratio))
        self.spatial_lr_scale = float(meta.get("spatial_lr_scale", self.spatial_lr_scale))
        self._apply_mip_floor = bool(meta.get("apply_mip_floor", True))
        self._parent_idx = meta.get("parent_idx", None)
        self._birth_lod = meta.get("birth_lod", None)
        self._per_gaussian_lb = meta.get("per_gaussian_lb", None)
        self._mip_floor = meta.get("mip_floor", None)
        self._train_hits = meta.get("train_hits", None)
        self._recent_active_hits = meta.get("recent_active_hits", None)
        for name in ("_parent_idx", "_birth_lod", "_per_gaussian_lb", "_mip_floor",
                     "_train_hits", "_recent_active_hits"):
            tensor = getattr(self, name)
            if tensor is not None:
                if row_indices is not None:
                    indices = row_indices.to(device=tensor.device, dtype=torch.long)
                    if name == "_parent_idx":
                        mapping = torch.full((tensor.shape[0],), -1, dtype=torch.long, device=tensor.device)
                        mapping[indices] = torch.arange(indices.numel(), device=tensor.device)
                        tensor = tensor[indices]
                        valid = tensor >= 0
                        tensor = tensor.clone()
                        tensor[valid] = mapping[tensor[valid]]
                    else:
                        tensor = tensor[indices]
                setattr(self, name, tensor.to(device))
        if self._train_hits is not None:
            self._train_hits = self._train_hits.to(dtype=torch.int32)
        if self._recent_active_hits is not None:
            self._recent_active_hits = self._recent_active_hits.to(dtype=torch.int32)
        self.setup_scaling_activation()
        self._invalidate_topology_caches()
        if self._birth_lod is not None:
            self._resort_by_birth_lod()
        return meta

    def load_hierarchy_chunks(self, path, device="cuda", lod_cap=None):
        meta_path = os.path.join(path, "hierarchy_meta.pt")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"hierarchy_meta.pt not found at {meta_path}")
        meta = torch.load(meta_path, map_location="cpu", weights_only=True)
        if lod_cap is not None and not 1 <= lod_cap <= int(meta["current_lod"]):
            raise ValueError(f"lod_cap must be in [1, {meta['current_lod']}]")
        chunk_paths = meta.get("chunk_paths", [])
        if not chunk_paths:
            legacy_path = os.path.join(path, "point_cloud.ply")
            if not os.path.exists(legacy_path):
                raise FileNotFoundError(f"no hierarchy chunks or point_cloud.ply found in {path}")
            self.load_ply(legacy_path, device=device, load_meta=True)
            if lod_cap is not None:
                self.retain_indices((self._birth_lod <= lod_cap).nonzero(as_tuple=True)[0])
                self.current_lod = lod_cap
            return

        xyz, fdc, frest, opacity, scaling, rotation = [], [], [], [], [], []
        for chunk in chunk_paths:
            if lod_cap is not None and int(chunk["lod"]) > lod_cap:
                continue
            rel_path = chunk["path"] if isinstance(chunk, dict) else str(chunk)
            chunk_path = rel_path if os.path.isabs(rel_path) else os.path.join(path, rel_path)
            tmp = GaussianModel(
                self.max_sh_degree,
                self.lod1_scaling_lower_bound,
                self.lod_scaling_ratio,
                self.increase_lod_num_childs,
                self.current_lod,
                self.max_lod,
                self.use_voxel_sampling,
                self.voxel_sampling_size,
            )
            tmp.load_ply(chunk_path, device=device, load_meta=False)
            xyz.append(tmp._xyz.detach())
            fdc.append(tmp._features_dc.detach())
            frest.append(tmp._features_rest.detach())
            opacity.append(tmp._opacity.detach())
            scaling.append(tmp._scaling.detach())
            rotation.append(tmp._rotation.detach())

        self._xyz = nn.Parameter(torch.cat(xyz, dim=0).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.cat(fdc, dim=0).requires_grad_(True))
        self._features_rest = nn.Parameter(torch.cat(frest, dim=0).requires_grad_(True))
        self._opacity = nn.Parameter(torch.cat(opacity, dim=0).requires_grad_(True))
        self._scaling = nn.Parameter(torch.cat(scaling, dim=0).requires_grad_(True))
        self._rotation = nn.Parameter(torch.cat(rotation, dim=0).requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0],), device=device)
        self.scaling_activation = torch.exp
        self.scaling_lower_bound = 0.0
        self.active_sh_degree = self.max_sh_degree
        row_indices = None
        if lod_cap is not None:
            row_indices = (meta["birth_lod"] <= lod_cap).nonzero(as_tuple=True)[0]
            if row_indices.numel() != self._xyz.shape[0]:
                raise ValueError("Hierarchy chunk counts do not match the birth-level metadata")
        self._load_hierarchy_meta(meta_path, device=device, row_indices=row_indices)
        if lod_cap is not None:
            self.current_lod = lod_cap
            self._invalidate_topology_caches()

    def retain_indices(self, indices):
        """Select rows for offline rendering while preserving activated attributes."""
        if self.optimizer is not None:
            raise ValueError("retain_indices is for offline models, not optimizer mutations")
        indices = indices.to(device=self._xyz.device, dtype=torch.long)
        n = self._xyz.shape[0]
        old_parents = self._parent_idx
        for field in self._LOD_ATTR_FIELDS.values():
            setattr(self, field, nn.Parameter(getattr(self, field).detach()[indices]))
        for field in ("_birth_lod", "_per_gaussian_lb", "_mip_floor", "_train_hits",
                      "_recent_active_hits", "max_radii2D", "xyz_gradient_accum", "denom"):
            value = getattr(self, field)
            if value is not None and value.ndim > 0 and value.shape[0] == n:
                setattr(self, field, value[indices])
        if old_parents is not None:
            mapping = torch.full((n,), -1, device=indices.device, dtype=torch.long)
            mapping[indices] = torch.arange(indices.numel(), device=indices.device)
            parents = old_parents[indices].clone()
            valid = parents >= 0
            parents[valid] = mapping[parents[valid]]
            self._parent_idx = parents
        self._shape_factors_cache = None
        self._invalidate_topology_caches()
        self._resort_by_birth_lod()

    def load_ply(self, path, device="cuda", load_meta=True):
        if not os.path.exists(path):
            meta_path = os.path.join(os.path.dirname(path), "hierarchy_meta.pt")
            if os.path.exists(meta_path):
                self.load_hierarchy_chunks(os.path.dirname(path), device=device)
                return
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))

        # Standard 3DGS/HLoD PLYs store log scales. Positive log scales are
        # valid for large splats; their sign cannot identify a linear encoding.
        if scales.size and "scale_encoding=linear" in plydata.comments:
            scales = np.log(np.maximum(scales, 1e-7))
            self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))

        self.scaling_activation = torch.exp
        self.scaling_lower_bound = 0.0
        self.active_sh_degree = self.max_sh_degree
        self._invalidate_topology_caches()
        # Loaded plies carry _birth_lod only when they were saved with hierarchical
        # metadata. Sort if metadata is present; otherwise this is a no-op.
        if load_meta:
            self._load_hierarchy_meta(os.path.join(os.path.dirname(path), "hierarchy_meta.pt"), device=device)
        elif self._birth_lod is not None:
            self._resort_by_birth_lod()

    def reset_opacity(self, current_lod_only=False, protect_inactive=False, protect_mask=None):
        capped = torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        opacities_new = inverse_sigmoid(capped)
        if current_lod_only and self._birth_lod is not None:
            keep = (self._birth_lod != self.current_lod).unsqueeze(-1)
            opacities_new = torch.where(keep, self._opacity.detach(), opacities_new)
        if protect_inactive and self._recent_active_hits is not None:
            keep_inactive = (self._recent_active_hits == 0).unsqueeze(-1)
            opacities_new = torch.where(keep_inactive, self._opacity.detach(), opacities_new)
        if protect_mask is not None:
            keep_protected = protect_mask.to(self._opacity.device).unsqueeze(-1)
            opacities_new = torch.where(keep_protected, self._opacity.detach(), opacities_new)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        self._rebuild_optimizer_if_needed()
        if self._recent_active_hits is not None:
            self._recent_active_hits.zero_()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        self._shape_factors_cache = None
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


        if self._mip_floor is not None:
            self._mip_floor = self._mip_floor[valid_points_mask]

        if self._birth_lod is not None:
            self._birth_lod = self._birth_lod[valid_points_mask]


        if self._per_gaussian_lb is not None:
            self._per_gaussian_lb = self._per_gaussian_lb[valid_points_mask]

        if self._parent_idx is not None:

            idx_map = torch.full((mask.shape[0],), -1, dtype=torch.long, device=mask.device)
            idx_map[valid_points_mask] = torch.arange(valid_points_mask.sum(), device=mask.device)

            pruned_parent_idx = self._parent_idx[valid_points_mask]
            had_parent = (pruned_parent_idx >= 0)

            new_parent_idx = torch.full_like(pruned_parent_idx, -1)
            if had_parent.any():
                new_parent_idx[had_parent] = idx_map[pruned_parent_idx[had_parent]]

            # Children whose parent was pruned are left as roots (parent_idx == -1).
            self._parent_idx = new_parent_idx

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self._train_hits is not None:
            self._train_hits = self._train_hits[valid_points_mask]
        if self._recent_active_hits is not None:
            self._recent_active_hits = self._recent_active_hits[valid_points_mask]
        self._invalidate_topology_caches()
        # Pruning preserves relative order of survivors; sort is a no-op,
        # but offsets must be refreshed to reflect new per-LOD counts.
        self._resort_by_birth_lod()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_mip_floor=None, new_birth_lod=None, new_parent_idx=None):
        self._shape_factors_cache = None
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


        if self._mip_floor is not None and new_mip_floor is not None:
            self._mip_floor = torch.cat([self._mip_floor, new_mip_floor], dim=0)

        if self._birth_lod is not None:
            if new_birth_lod is not None:
                self._birth_lod = torch.cat([self._birth_lod, new_birth_lod], dim=0)
            else:
                self._birth_lod = torch.cat([self._birth_lod,
                    torch.full((new_xyz.shape[0],), self.current_lod, dtype=torch.int, device="cuda")], dim=0)

        if self._parent_idx is not None:
            if new_parent_idx is not None:
                self._parent_idx = torch.cat([self._parent_idx, new_parent_idx])
            else:
                self._parent_idx = torch.cat([self._parent_idx,
                    torch.full((new_xyz.shape[0],), -1, dtype=torch.long, device="cuda")])

        if self._per_gaussian_lb is not None:
            self._per_gaussian_lb = self._compute_per_gaussian_lb()

        N = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.denom = torch.zeros((N, 1), device="cuda")
        self.max_radii2D = torch.zeros((N,), device="cuda")
        if self._train_hits is not None:
            new_hits = torch.zeros((new_xyz.shape[0],), dtype=torch.int32, device=self._train_hits.device)
            self._train_hits = torch.cat([self._train_hits, new_hits], dim=0)
        if self._recent_active_hits is not None:
            new_recent = torch.zeros((new_xyz.shape[0],), dtype=torch.int32, device=self._recent_active_hits.device)
            self._recent_active_hits = torch.cat([self._recent_active_hits, new_recent], dim=0)
        self._invalidate_topology_caches()
        # New rows are appended at the end with current_lod birth (>= existing
        # max birth_lod), so the array is still sorted; this just refreshes
        # offsets. If new_birth_lod was supplied with a coarser value, the
        # sorted-check forces a real permutation.
        self._resort_by_birth_lod()

    def _apply_per_parent_budget(self, selected_pts_mask, N):


        budget = self._per_parent_budget
        parent_ids = self._parent_idx
        has_parent = parent_ids >= 0
        if not has_parent.any():
            return selected_pts_mask

        n_buckets = int(parent_ids.max().item()) + 1
        descendant_count = torch.zeros(n_buckets, dtype=torch.long, device=parent_ids.device)
        valid = has_parent.nonzero(as_tuple=True)[0]
        descendant_count.scatter_add_(0, parent_ids[valid].long(),
                                      torch.ones(valid.numel(), dtype=torch.long, device=parent_ids.device))

        over_budget = torch.zeros_like(selected_pts_mask)
        cand = selected_pts_mask & has_parent
        cand_idx = cand.nonzero(as_tuple=True)[0]
        if cand_idx.numel():
            cand_parents = parent_ids[cand_idx].long()

            over = descendant_count[cand_parents] + N > budget
            over_budget[cand_idx[over]] = True

        if getattr(self, "_budget_debug", False):

            active_counts = descendant_count[descendant_count > 0]
            n_at_cap = (active_counts >= budget).sum().item()
            n_active = active_counts.numel()
            n_cand = int(cand_idx.numel())
            n_rej = int(over_budget.sum().item())
            if n_active > 0:
                max_c = active_counts.max().item()
                mean_c = active_counts.float().mean().item()
                q = torch.quantile(active_counts.float(),
                                   torch.tensor([0.5, 0.9, 0.99], device=active_counts.device))
                print(f"  [budget] LOD={self.current_lod} N={N} cap={budget} | "
                      f"parents_at_cap={n_at_cap}/{n_active} ({100*n_at_cap/n_active:.1f}%) | "
                      f"desc/parent mean={mean_c:.1f} p50={q[0].item():.0f} p90={q[1].item():.0f} "
                      f"p99={q[2].item():.0f} max={max_c} | "
                      f"rejected={n_rej}/{n_cand}", flush=True)
        return selected_pts_mask & ~over_budget

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        max_scale_per_gaussian = torch.max(self.get_scaling, dim=1).values
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              max_scale_per_gaussian > self.percent_dense*scene_extent)

        if getattr(self, "_densify_current_lod_only", False) and self._birth_lod is not None:
            selected_pts_mask &= (self._birth_lod == self.current_lod)


        if self._per_parent_budget is not None and self._parent_idx is not None:
            selected_pts_mask = self._apply_per_parent_budget(selected_pts_mask, N)


        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)


        if self._per_gaussian_lb is not None:
            split_lb = self._per_gaussian_lb[selected_pts_mask].unsqueeze(-1).repeat(N, 1)
        else:
            split_lb = self.scaling_lower_bound

        new_scaling = self.scaling_inverse_activation(torch.clamp_min(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N) - split_lb, 0.0000001))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_mip_floor = self._mip_floor[selected_pts_mask].repeat(N) if self._mip_floor is not None else None
        new_parent_idx = self._parent_idx[selected_pts_mask].repeat(N) if self._parent_idx is not None else None
        new_birth_lod = self._birth_lod[selected_pts_mask].repeat(N) if self._birth_lod is not None else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_mip_floor, new_parent_idx=new_parent_idx, new_birth_lod=new_birth_lod)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))

        if self._parent_idx is not None:
            split_indices = selected_pts_mask.nonzero(as_tuple=True)[0]
            if split_indices.numel() > 0:
                n_sel = split_indices.numel()


                replacement_idx = torch.arange(n_init_points, n_init_points + N * n_sel,
                                               device="cuda", dtype=torch.long).view(N, n_sel)


                # Children of a split parent are re-pointed to its first replicate.
                remap = torch.arange(self._parent_idx.shape[0], device="cuda", dtype=torch.long)
                remap[split_indices] = replacement_idx[0]
                valid = (self._parent_idx >= 0)
                self._parent_idx[valid] = remap[self._parent_idx[valid]]

        self.prune_points(prune_filter)

    def _hierarchy_cascade_mask(self, root_mask):
        prune_mask = root_mask.to(device=self._xyz.device, dtype=torch.bool).clone()
        if self._parent_idx is None or prune_mask.numel() == 0 or not prune_mask.any():
            return prune_mask

        n_total = prune_mask.shape[0]
        parent_ids = self._parent_idx.to(device=prune_mask.device, dtype=torch.long)
        valid_child = (parent_ids >= 0) & (parent_ids < n_total)
        if not valid_child.any():
            return prune_mask

        max_steps = max(1, int(max(self.current_lod, self.max_lod)) + 2)
        for _ in range(max_steps):
            parent_pruned = torch.zeros_like(prune_mask)
            parent_pruned[valid_child] = prune_mask[parent_ids[valid_child]]
            cascade = parent_pruned & ~prune_mask
            if not cascade.any():
                break
            prune_mask[cascade] = True
        return prune_mask

    def _subtree_contains_mask(self, seed_mask):
        contains = seed_mask.to(device=self._xyz.device, dtype=torch.bool).clone()
        if self._parent_idx is None or contains.numel() == 0 or not contains.any():
            return contains

        n_total = contains.shape[0]
        parent_ids = self._parent_idx.to(device=contains.device, dtype=torch.long)
        valid_child = (parent_ids >= 0) & (parent_ids < n_total)
        if not valid_child.any():
            return contains

        max_steps = max(1, int(max(self.current_lod, self.max_lod)) + 2)
        for _ in range(max_steps):
            flagged_child = valid_child & contains
            if not flagged_child.any():
                break
            parent_counts = torch.zeros(n_total, device=contains.device, dtype=torch.long)
            parent_counts.scatter_add_(
                0,
                parent_ids[flagged_child],
                torch.ones(flagged_child.sum(), device=contains.device, dtype=torch.long),
            )
            new_contains = contains | (parent_counts > 0)
            if torch.equal(new_contains, contains):
                break
            contains = new_contains
        return contains

    def prune_low_opacity_families(self, min_opacity, protect_lod1=False,
                                   restrict_sibling_prune_to_current_lod=False,
                                   min_hits_for_opacity_prune=0,
                                   protect_mask=None,
                                   dead_family_protect_mask=None,
                                   prune_dead_families=True):
        #End-of-LOD cleanup: prune low leaf siblings, then optional full-low subtrees.
        stats = {
            "before": int(self._xyz.shape[0]),
            "low_before": 0,
            "sibling_roots": 0,
            "sibling_total": 0,
            "dead_family_roots": 0,
            "dead_family_total": 0,
            "after": int(self._xyz.shape[0]),
            "low_after": 0,
        }
        if self._parent_idx is None or self._birth_lod is None or self._xyz.shape[0] == 0:
            return stats

        sibling_protect = None
        if protect_mask is not None:
            sibling_protect = protect_mask.to(device=self._xyz.device, dtype=torch.bool).clone()

        dead_family_protect = None
        if dead_family_protect_mask is not None:
            dead_family_protect = dead_family_protect_mask.to(device=self._xyz.device, dtype=torch.bool).clone()

        def opacity_low_mask():
            low = (self.get_opacity < min_opacity).squeeze()
            if low.ndim == 0:
                low = low.view(1)
            if min_hits_for_opacity_prune > 0 and self._train_hits is not None:
                low = low & (self._train_hits >= min_hits_for_opacity_prune)
            return low

        def protected_mask(extra_protect=None, mask_name="protect_mask"):
            n_total = self._xyz.shape[0]
            protected = torch.zeros(n_total, device=self._xyz.device, dtype=torch.bool)
            if extra_protect is not None:
                if extra_protect.shape[0] != n_total:
                    raise ValueError(f"{mask_name} shape no longer matches Gaussian count")
                protected |= extra_protect
            if protect_lod1 and self._birth_lod is not None:
                protected |= (self._birth_lod == 1)
            return protected

        def parent_info():
            n_total = self._xyz.shape[0]
            parent_ids = self._parent_idx.to(device=self._xyz.device, dtype=torch.long)
            valid_child = (parent_ids >= 0) & (parent_ids < n_total)
            child_count = torch.zeros(n_total, device=self._xyz.device, dtype=torch.long)
            if valid_child.any():
                child_count.scatter_add_(
                    0,
                    parent_ids[valid_child],
                    torch.ones(valid_child.sum(), device=self._xyz.device, dtype=torch.long),
                )
            return parent_ids, valid_child, child_count

        def finish_stats():
            stats["after"] = int(self._xyz.shape[0])
            if self._xyz.shape[0] > 0:
                stats["low_after"] = int(opacity_low_mask().sum().item())
            else:
                stats["low_after"] = 0
            return stats

        low = opacity_low_mask()
        stats["low_before"] = int(low.sum().item())
        protected = protected_mask(sibling_protect, "protect_mask")

        parent_ids, valid_child, child_count = parent_info()
        leaf_child = valid_child & (child_count == 0)
        prunable_child = leaf_child & low & ~protected
        if restrict_sibling_prune_to_current_lod:
            prunable_child &= (self._birth_lod == int(self.current_lod))
        if sibling_protect is not None and prunable_child.any():
            prunable_child &= ~self._subtree_contains_mask(protected)

        if prunable_child.any():
            n_total = self._xyz.shape[0]
            prunable_count = torch.zeros(n_total, device=self._xyz.device, dtype=torch.long)
            prunable_count.scatter_add_(
                0,
                parent_ids[prunable_child],
                torch.ones(prunable_child.sum(), device=self._xyz.device, dtype=torch.long),
            )

            parent_lookup = parent_ids.clamp(min=0, max=n_total - 1)
            removes_every_child = prunable_child & (prunable_count[parent_lookup] == child_count[parent_lookup])
            if removes_every_child.any():
                opacity_flat = self.get_opacity.squeeze().detach()
                if opacity_flat.ndim == 0:
                    opacity_flat = opacity_flat.view(1)
                best_opacity = torch.full((n_total,), -float("inf"), device=self._xyz.device)
                best_opacity.scatter_reduce_(
                    0,
                    parent_ids[removes_every_child],
                    opacity_flat[removes_every_child],
                    reduce="amax",
                )
                best_opacity_child = removes_every_child & (opacity_flat == best_opacity[parent_lookup])
                row_idx = torch.arange(n_total, device=self._xyz.device, dtype=torch.long)
                best_row = torch.full((n_total,), n_total, device=self._xyz.device, dtype=torch.long)
                best_row.scatter_reduce_(
                    0,
                    parent_ids[best_opacity_child],
                    row_idx[best_opacity_child],
                    reduce="amin",
                )
                keep_child = best_opacity_child & (row_idx == best_row[parent_lookup])
                prunable_child &= ~keep_child

            sibling_mask = prunable_child.clone()
            stats["sibling_roots"] = int(prunable_child.sum().item())
            stats["sibling_total"] = int(sibling_mask.sum().item())
            if sibling_mask.any():
                valid_after_sibling = ~sibling_mask
                if sibling_protect is not None:
                    sibling_protect = sibling_protect[valid_after_sibling]
                if dead_family_protect is not None:
                    dead_family_protect = dead_family_protect[valid_after_sibling]
                self.prune_points(sibling_mask)
                torch.cuda.empty_cache()

        if not prune_dead_families or self._xyz.shape[0] == 0:
            return finish_stats()

        low = opacity_low_mask()
        protected = protected_mask(dead_family_protect, "dead_family_protect_mask")
        parent_ids, valid_child, child_count = parent_info()
        full_low_subtree = low & ~protected
        max_steps = max(1, int(max(self.current_lod, self.max_lod)) + 2)
        for _ in range(max_steps):
            full_child = valid_child & full_low_subtree
            full_child_count = torch.zeros(self._xyz.shape[0], device=self._xyz.device, dtype=torch.long)
            if full_child.any():
                full_child_count.scatter_add_(
                    0,
                    parent_ids[full_child],
                    torch.ones(full_child.sum(), device=self._xyz.device, dtype=torch.long),
                )
            all_children_full = (child_count > 0) & (full_child_count == child_count)
            new_full_low_subtree = low & ~protected & ((child_count == 0) | all_children_full)
            if torch.equal(new_full_low_subtree, full_low_subtree):
                break
            full_low_subtree = new_full_low_subtree

        parent_full = torch.zeros_like(full_low_subtree)
        if valid_child.any():
            parent_full[valid_child] = full_low_subtree[parent_ids[valid_child]]
        dead_family_roots = full_low_subtree & (child_count > 0) & ~parent_full
        dead_family_mask = self._hierarchy_cascade_mask(dead_family_roots)
        stats["dead_family_roots"] = int(dead_family_roots.sum().item())
        stats["dead_family_total"] = int(dead_family_mask.sum().item())
        if dead_family_mask.any():
            self.prune_points(dead_family_mask)
            torch.cuda.empty_cache()

        return finish_stats()

    def prune(self, min_opacity, extent, max_screen_size, prune_overlap_threshold, hierarchy_aware=False, protect_lod1=False, restrict_to_current_lod=False, overlap_same_lod_only=False, min_hits_for_opacity_prune=0, protect_mask=None):


        opacity_low = (self.get_opacity < min_opacity).squeeze()
        if min_hits_for_opacity_prune > 0 and self._train_hits is not None:
            sufficient_evidence = (self._train_hits >= min_hits_for_opacity_prune)
            opacity_low = opacity_low & sufficient_evidence
        prune_mask = opacity_low

        if prune_overlap_threshold:
            if overlap_same_lod_only and self._birth_lod is not None:


                dist = torch.full((self._xyz.shape[0],), float('inf'), device=self._xyz.device)
                for lod_val in torch.unique(self._birth_lod):
                    mask = (self._birth_lod == lod_val)
                    if mask.sum() >= 2:
                        sub_dist = torch.sqrt(distCUDA2(self._xyz[mask].contiguous()))
                        dist[mask] = sub_dist
            else:
                dist = torch.sqrt(distCUDA2(self._xyz))
            prune_overlap_mask = (dist < prune_overlap_threshold)
            if self._parent_idx is not None and not hierarchy_aware and not overlap_same_lod_only:


                has_parent = (self._parent_idx >= 0)
                prune_overlap_mask[has_parent] = False
            prune_mask = torch.logical_or(prune_mask, prune_overlap_mask)

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if hierarchy_aware and self._parent_idx is not None:
            mode = hierarchy_aware if isinstance(hierarchy_aware, str) else 'protect'

            parent_ids = self._parent_idx.to(prune_mask.device).long()
            has_parent = parent_ids >= 0
            n_total = self._xyz.shape[0]

            n_parents = parent_ids.max().item() + 1 if has_parent.any() else 0

            if n_parents > 0 and has_parent.any():
                if mode in ('protect', 'protect_last_child'):
                    surviving = ~prune_mask & has_parent
                    surviving_count = torch.zeros(n_parents, device=prune_mask.device, dtype=torch.long)
                    surviving_count.scatter_add_(0, parent_ids[surviving], torch.ones(surviving.sum(), device=prune_mask.device, dtype=torch.long))
                    would_orphan = prune_mask & has_parent & (surviving_count[parent_ids.clamp(min=0)] == 0)
                    if would_orphan.any():
                        opacity_flat = self.get_opacity.squeeze()
                        candidate_opacity = torch.full_like(opacity_flat, -float('inf'))
                        candidate_opacity[would_orphan] = opacity_flat[would_orphan]
                        best_opacity = torch.full((n_parents,), -float('inf'), device=prune_mask.device)
                        best_opacity.scatter_reduce_(0, parent_ids[would_orphan], candidate_opacity[would_orphan], reduce='amax')
                        is_best = would_orphan & (opacity_flat == best_opacity[parent_ids.clamp(min=0)])
                        prune_mask[is_best] = False

                    if mode == 'protect':


                        internal_parent_mask = torch.zeros_like(prune_mask)
                        valid_parent_refs = parent_ids[has_parent & (parent_ids < n_total)]
                        if valid_parent_refs.numel() > 0:
                            internal_parent_mask[valid_parent_refs.unique()] = True
                            prune_mask[internal_parent_mask] = False

        if protect_lod1 and self._birth_lod is not None:
            prune_mask[self._birth_lod == 1] = False

        if restrict_to_current_lod and self._birth_lod is not None:
            prune_mask &= (self._birth_lod == self.current_lod)

        if protect_mask is not None:
            prune_mask[protect_mask.to(prune_mask.device)] = False

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def increase_lod_hierarchical(self, training_args=None):
        n_existing = self._xyz.shape[0]

        finest_mask = (self._birth_lod == self.current_lod)
        finest_indices = finest_mask.nonzero(as_tuple=True)[0]
        n_parents = finest_indices.shape[0]


        parent_scaling_activated = self.get_scaling[finest_mask].detach()
        parent_rotation_raw = self._rotation[finest_mask].detach()

        self.current_lod += 1
        self._mip_floor = None

        if self.current_lod < self.max_lod:
            child_lb = self.lod1_scaling_lower_bound / self.lod_scaling_ratio ** (self.current_lod - 1)
        else:
            child_lb = 0.0

        N = self.increase_lod_num_childs
        child_scaling = self.scaling_inverse_activation(
            (parent_scaling_activated.repeat(N, 1) - child_lb).clamp(min=1e-7)
        )
        child_opacity = self._opacity[finest_mask].detach().repeat(N, 1)
        child_xyz = self._xyz[finest_mask].detach().repeat(N, 1)
        child_rotation = parent_rotation_raw.repeat(N, 1)
        child_features_dc = self._features_dc[finest_mask].detach().repeat(N, 1, 1)
        child_features_rest = self._features_rest[finest_mask].detach().repeat(N, 1, 1)
        child_parent_idx = finest_indices.repeat(N)
        n_children = N * n_parents

        self._xyz = nn.Parameter(torch.cat([self._xyz.detach(), child_xyz], dim=0).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc.detach(), child_features_dc], dim=0).requires_grad_(True))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest.detach(), child_features_rest], dim=0).requires_grad_(True))
        self._opacity = nn.Parameter(torch.cat([self._opacity.detach(), child_opacity], dim=0).requires_grad_(True))
        self._scaling = nn.Parameter(torch.cat([self._scaling.detach(), child_scaling], dim=0).requires_grad_(True))
        self._rotation = nn.Parameter(torch.cat([self._rotation.detach(), child_rotation], dim=0).requires_grad_(True))

        child_birth = torch.full((n_children,), self.current_lod, dtype=torch.int, device="cuda")
        self._birth_lod = torch.cat([self._birth_lod, child_birth])

        if self._parent_idx is None:
            existing_parent_idx = torch.full((n_existing,), -1, dtype=torch.long, device="cuda")
            self._parent_idx = torch.cat([existing_parent_idx, child_parent_idx])
        else:
            self._parent_idx = torch.cat([self._parent_idx, child_parent_idx])

        self._per_gaussian_lb = self._compute_per_gaussian_lb()

        if training_args is not None:
            self.training_setup(training_args)

        self.setup_scaling_activation()
        self._invalidate_topology_caches()
        self._resort_by_birth_lod()

    def enable_shape_factors_cache(self, enabled=True):
        #Enable/disable caching of shape factors. Safe to enable during rendering only.
        self._shape_factors_cache_enabled = bool(enabled)
        self._shape_factors_cache = None
        self._shape_factors_cache_version = None

    def _projected_shape_factors(self, candidate_idx=None):
        """[N, 3, 3] Cholesky factor L = R @ diag(s) for CUDA traversal/projection.

        Cache is opt-in (disabled during training to avoid stale shapes after optimizer steps).
        """
        from utils.general_utils import build_rotation
        use_cache = bool(getattr(self, "_shape_factors_cache_enabled", False))
        if use_cache:
            cache = self._shape_factors_cache
            if (
                cache is None
                or self._shape_factors_cache_version != self._gaussian_version
                or cache.device != self._xyz.device
                or cache.shape[0] != self._xyz.shape[0]
            ):
                scaling = self.get_scaling.detach()
                rotation = self.get_rotation.detach()
                cache = (build_rotation(rotation) * scaling.unsqueeze(1)).contiguous()
                self._shape_factors_cache = cache
                self._shape_factors_cache_version = self._gaussian_version
            if candidate_idx is None:
                return cache
            return cache[candidate_idx.to(device=cache.device, dtype=torch.long)]

        if candidate_idx is None:
            scaling = self.get_scaling.detach()
            rotation = self.get_rotation.detach()
        else:
            candidate_idx = candidate_idx.to(device=self._xyz.device, dtype=torch.long)
            scaling = self.get_scaling_at(candidate_idx).detach()
            rotation = self.get_rotation_at(candidate_idx).detach()
        return (build_rotation(rotation) * scaling.unsqueeze(1)).contiguous()

    _SCREENSPACE_METRIC_ID = 2  # cuda_radius: ceil(3*sqrt(lambda_max(cov2D)))

    def _compute_projected_size_values_cuda(self, viewpoint_cam, xyz, Vrk, low_pass=0.3, cull_near=0.0):
        if _hier_trav_cuda is None:
            return None
        if xyz.device.type != "cuda" or xyz.dtype != torch.float32:
            return None
        if Vrk.device != xyz.device or Vrk.dtype != torch.float32:
            return None
        try:
            viewmatrix = viewpoint_cam.world_view_transform.to(
                device=xyz.device, dtype=torch.float32).contiguous()
            return _hier_trav_cuda.projected_size_from_shape(
                xyz,
                Vrk,
                viewmatrix,
                int(viewpoint_cam.image_width),
                int(viewpoint_cam.image_height),
                math.tan(viewpoint_cam.FoVx * 0.5),
                math.tan(viewpoint_cam.FoVy * 0.5),
                float(low_pass),
                self._SCREENSPACE_METRIC_ID,
                cull_near=float(cull_near),
            )
        except Exception as exc:
            self._cuda_projected_size_error = repr(exc)
            return None

    def _compute_projected_size_values(self, viewpoint_cam, candidate_idx=None, low_pass=0.3, cull_near=0.0):
        if candidate_idx is None:
            xyz = self._xyz.detach()
        else:
            candidate_idx = candidate_idx.to(device=self._xyz.device, dtype=torch.long)
            xyz = self._xyz.detach()[candidate_idx]
        Vrk = self._projected_shape_factors(candidate_idx)

        values = self._compute_projected_size_values_cuda(
            viewpoint_cam, xyz, Vrk, low_pass=low_pass, cull_near=cull_near)
        if values is None:
            # Mirror compute_metric_from_shape, including its rasterizer eigenvalue
            # floor and ceil. This also makes the extension-disable flag usable.
            w2c = viewpoint_cam.world_view_transform.T.to(xyz)
            camera_xyz = xyz @ w2c[:3, :3].T + w2c[:3, 3]
            raw_z = camera_xyz[:, 2]
            z = raw_z.clamp(min=float(cull_near) if cull_near > 0 else 0.2)
            tx = (camera_xyz[:, 0] / z).clamp(
                -1.3 * math.tan(viewpoint_cam.FoVx / 2),
                1.3 * math.tan(viewpoint_cam.FoVx / 2))
            ty = (camera_xyz[:, 1] / z).clamp(
                -1.3 * math.tan(viewpoint_cam.FoVy / 2),
                1.3 * math.tan(viewpoint_cam.FoVy / 2))
            shape = w2c[:3, :3].unsqueeze(0) @ Vrk
            fx = viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx / 2))
            fy = viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy / 2))
            row_x = fx / z[:, None] * (shape[:, 0] - tx[:, None] * shape[:, 2])
            row_y = fy / z[:, None] * (shape[:, 1] - ty[:, None] * shape[:, 2])
            a = row_x.square().sum(-1) + low_pass
            b = (row_x * row_y).sum(-1)
            d = row_y.square().sum(-1) + low_pass
            mid = (a + d) * 0.5
            eigenvalue = mid + torch.sqrt((mid.square() - (a * d - b.square())).clamp(min=0.1))
            values = torch.ceil(3 * torch.sqrt(eigenvalue.clamp(min=0)))
            if cull_near > 0:
                values = torch.where(raw_z < cull_near, -torch.ones_like(values), values)
        return values

    def _onscreen_mask(self, indices, viewpoint_cam, margin=0.0, low_pass=0.3,
                       frustum_near=None):
        """Boolean mask over `indices`: True where the splat overlaps a screen tile.
        Torch mirror of INRIA's getRect test (== cull_frustum_kernel). `margin`
        is an extra pixel margin on the splat radius. `frustum_near` must match
        the selected renderer; INRIA uses 0.2 while gsplat uses camera.znear."""
        device = self._xyz.device
        idx = indices.to(device=device, dtype=torch.long)
        if idx.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=device)

        xyz_sel = self._xyz.detach()[idx].to(dtype=torch.float32)
        W2C = viewpoint_cam.world_view_transform.T.to(device=device, dtype=torch.float32)
        z_view = xyz_sel @ W2C[2, :3] + W2C[2, 3]
        FPT = viewpoint_cam.full_proj_transform.to(device=device, dtype=torch.float32)
        xyz_h = torch.cat([xyz_sel, torch.ones_like(xyz_sel[:, :1])], dim=1)
        p_hom = xyz_h @ FPT
        inv_w = 1.0 / (p_hom[:, 3] + 1e-7)
        W = int(viewpoint_cam.image_width)
        H = int(viewpoint_cam.image_height)
        point_x = ((p_hom[:, 0] * inv_w + 1.0) * W - 1.0) * 0.5
        point_y = ((p_hom[:, 1] * inv_w + 1.0) * H - 1.0) * 0.5

        if frustum_near is None:
            frustum_near = max(0.0, float(getattr(viewpoint_cam, "znear", 0.01)))
        else:
            frustum_near = max(0.0, float(frustum_near))
        radius = self._compute_projected_size_values(
            viewpoint_cam, candidate_idx=idx, low_pass=low_pass,
            cull_near=frustum_near) + float(margin)

        grid_x = (W + 15) // 16
        grid_y = (H + 15) // 16
        rmin_x = ((point_x - radius) / 16).to(torch.int32).clamp(0, grid_x)
        rmin_y = ((point_y - radius) / 16).to(torch.int32).clamp(0, grid_y)
        rmax_x = ((point_x + radius + 15) / 16).to(torch.int32).clamp(0, grid_x)
        rmax_y = ((point_y + radius + 15) / 16).to(torch.int32).clamp(0, grid_y)
        onscreen = (rmax_x - rmin_x) * (rmax_y - rmin_y) != 0
        return (z_view > frustum_near) & onscreen

    def _cull_offscreen_gsplat(self, indices, viewpoint_cam, low_pass=0.3):
        """Return exactly the subset accepted by gsplat's projection stage."""
        idx = indices.to(device=self._xyz.device, dtype=torch.long)
        if idx.numel() == 0:
            return idx

        try:
            import gsplat
        except Exception as exc:
            raise RuntimeError(
                "gsplat frustum culling requested, but gsplat is unavailable"
            ) from exc

        device = self._xyz.device
        dtype = torch.float32
        W = int(viewpoint_cam.image_width)
        H = int(viewpoint_cam.image_height)
        fx = W / (2.0 * math.tan(viewpoint_cam.FoVx * 0.5))
        fy = H / (2.0 * math.tan(viewpoint_cam.FoVy * 0.5))
        K = torch.tensor(
            [[fx, 0.0, W * 0.5],
             [0.0, fy, H * 0.5],
             [0.0, 0.0, 1.0]],
            dtype=dtype, device=device,
        ).unsqueeze(0)
        viewmat = viewpoint_cam.world_view_transform.transpose(
            0, 1
        ).to(device=device, dtype=dtype).unsqueeze(0).contiguous()

        projection = gsplat.fully_fused_projection(
            means=self.get_xyz_at(idx).detach().to(dtype=dtype).contiguous(),
            covars=None,
            quats=self.get_rotation_at(idx).detach().to(dtype=dtype).contiguous(),
            scales=self.get_scaling_at(idx).detach().to(dtype=dtype).contiguous(),
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            eps2d=float(low_pass),
            near_plane=float(getattr(viewpoint_cam, "znear", 0.01)),
            far_plane=float(getattr(viewpoint_cam, "zfar", 1.0e10)),
            radius_clip=0.0,
            packed=True,
            sparse_grad=False,
            calc_compensations=False,
            camera_model="pinhole",
            opacities=self.get_opacity_at(idx).detach().squeeze(-1).to(
                dtype=dtype
            ).contiguous(),
        )
        return idx[projection[2].to(device=device, dtype=torch.long)]

    CUDA_HIERARCHY_MODES = (
        "leafes_to_coarse",
        "fine_to_coarse",
        "coarse_to_fine",
    )

    def get_active_indices(self, viewpoint_cam, expand_factor=1.0, mode="leafes_to_coarse",
                           prefix_prob=0.0,
                           current_prob=0.0,
                           prefix_min_lod=1,
                           profile=None,
                           low_pass=0.3,
                           cull_near=0.0,
                           cull_frustum=0.0,
                           cull_backend="inria", max_lod=None):
        if mode not in self.CUDA_HIERARCHY_MODES:
            raise ValueError(f"Unknown hierarchy traversal mode: {mode}")
        if expand_factor <= 0:
            raise ValueError("The screen-space threshold must be positive")
        cull_backend = str(cull_backend).strip().lower()
        if cull_backend not in ("inria", "gsplat"):
            raise ValueError(
                f"cull_backend must be 'inria' or 'gsplat', got {cull_backend!r}"
            )
        # INRIA hard-codes 0.2 in in_frustum(); gsplat uses camera.znear.
        frustum_near = (
            0.2 if cull_backend == "inria"
            else max(0.0, float(getattr(viewpoint_cam, "znear", 0.01)))
        )

        N = self._xyz.shape[0]
        device = self._xyz.device
        if N == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if self._birth_lod is None:
            return torch.arange(N, dtype=torch.long, device=device)

        px_threshold = expand_factor
        with torch.no_grad():
            lod_min = max(1, int(prefix_min_lod))
            lod_max = int(self.current_lod if max_lod is None else max_lod)
            if not 1 <= lod_max <= self.current_lod:
                raise ValueError(f"max_lod must be in [1, {self.current_lod}]")
            lod_min = min(lod_min, lod_max)

            current_prob = max(0.0, float(current_prob))
            prefix_prob = max(0.0, float(prefix_prob))
            current_lod_arg = lod_max

            if current_prob > 0.0 or prefix_prob > 0.0:
                t_subthreshold = _profile_start(profile)
                sample = torch.rand((), device=device).item()
                if sample < current_prob:
                    result = self._indices_at_lod(lod_max, device=device)
                    _profile_stop(profile, "subthreshold_s", t_subthreshold)
                    if cull_near and float(cull_near) > 0.0 and result.numel() > 0:
                        W2C = viewpoint_cam.world_view_transform.T.to(
                            device=device, dtype=torch.float32)
                        z_view = (
                            self._xyz.detach()[result].to(dtype=torch.float32)
                            @ W2C[2, :3] + W2C[2, 3]
                        )
                        result = result[z_view >= float(cull_near)]
                    if cull_frustum and float(cull_frustum) > 0.0 and result.numel() > 0:
                        if cull_backend == "gsplat":
                            result = self._cull_offscreen_gsplat(
                                result, viewpoint_cam, low_pass=low_pass)
                        else:
                            keep = self._onscreen_mask(
                                result, viewpoint_cam,
                                margin=float(cull_frustum),
                                low_pass=low_pass,
                                frustum_near=frustum_near)
                            result = result[keep]
                    return result

                if sample < current_prob + prefix_prob:
                    # random-prefix traversal: run the normal adaptive traversal but
                    # treat a randomly sampled LoD in [lod_min, lod_max] as the finest.
                    sampled_lod = torch.randint(lod_min, lod_max + 1, (), device=device).item()
                    current_lod_arg = int(sampled_lod)
                _profile_stop(profile, "subthreshold_s", t_subthreshold)

            result = self._active_indices_top_down_projected_lazy(
                viewpoint_cam, px_threshold, N, device, mode,
                current_lod=current_lod_arg,
                profile=profile,
                low_pass=low_pass,
                cull_near=cull_near,
                cull_frustum=cull_frustum,
                frustum_near=frustum_near,
                cull_backend=cull_backend)
            if profile is not None:
                profile["cull_backend"] = cull_backend
            return result

    def _active_indices_top_down_projected_lazy(self, viewpoint_cam, px_threshold, N, device, mode,
                                                current_lod=None, profile=None, low_pass=0.3,
                                                cull_near=0.0, cull_frustum=0.0,
                                                frustum_near=0.2,
                                                cull_backend="inria"):
        if current_lod is None:
            current_lod = self.current_lod
        current_lod = int(current_lod)

        cuda_result = self._active_indices_top_down_projected_cuda(
            viewpoint_cam, px_threshold, N, device, mode,
            current_lod=current_lod,
            profile=profile,
            low_pass=low_pass,
            cull_near=cull_near,
            cull_frustum=cull_frustum,
            frustum_near=frustum_near,
            cull_backend=cull_backend)
        if cuda_result is not None:
            return cuda_result

        if profile is not None:
            profile.setdefault("cuda_backend_requested", "shape")
            profile["cuda_backend_used"] = "pytorch"

        included = self._birth_lod <= current_lod
        included_idx = self._indices_in_lod_range(1, current_lod, device=device)
        if included_idx.numel() == 0:
            return included_idx

        if self._parent_idx is None:
            if mode in ("top_down_any_from_max", "fine_to_coarse"):
                return self._indices_at_lod(current_lod, device=device)
            return included_idx

        t_traversal = _profile_start(profile)
        parent_idx_all = self._parent_idx.to(device=device, dtype=torch.long)
        birth_lod = self._birth_lod.to(device=device)

        if mode == "coarse_to_fine":
            # True top-down traversal.  A parent is replaced by all of its direct
            # children when at least one child reaches the same screen-space
            # threshold used by the bottom-up collapse rule.
            active = included & (parent_idx_all < 0)
            for child_lod in range(2, current_lod + 1):
                all_child_indices = self._indices_at_lod(child_lod, device=device)
                if all_child_indices.numel() == 0:
                    continue
                all_parent_indices = parent_idx_all[all_child_indices]
                has_parent = all_parent_indices >= 0
                child_indices = all_child_indices[has_parent]
                parent_indices = all_parent_indices[has_parent]
                if child_indices.numel() == 0:
                    continue

                parent_is_active = active[parent_indices]
                if not parent_is_active.any():
                    continue
                candidate_children = child_indices[parent_is_active]
                candidate_parents = parent_indices[parent_is_active]

                t_traversal = _profile_stop(profile, "active_traversal_s", t_traversal)
                t_proj = _profile_start(profile)
                child_proj = self._compute_projected_size_values(
                    viewpoint_cam, candidate_idx=candidate_children,
                    low_pass=low_pass, cull_near=cull_near)
                _profile_stop(profile, "projected_size_s", t_proj)
                t_traversal = _profile_start(profile)

                split_parent = torch.zeros(N, dtype=torch.bool, device=device)
                big = child_proj >= px_threshold
                if big.any():
                    split_parent[candidate_parents[big]] = True
                split_children = split_parent[candidate_parents]
                if not split_children.any():
                    continue

                active[split_parent] = False
                active[candidate_children[split_children]] = True

            _profile_stop(profile, "active_traversal_s", t_traversal)
            result = active.nonzero(as_tuple=True)[0]
            if cull_near and float(cull_near) > 0.0 and result.numel() > 0:
                W2C = viewpoint_cam.world_view_transform.T.to(device=device, dtype=torch.float32)
                z_view = self._xyz.detach()[result].to(dtype=torch.float32) @ W2C[2, :3] + W2C[2, 3]
                result = result[z_view >= float(cull_near)]
            if cull_frustum and float(cull_frustum) > 0.0 and result.numel() > 0:
                if cull_backend == "gsplat":
                    result = self._cull_offscreen_gsplat(
                        result, viewpoint_cam, low_pass=low_pass)
                else:
                    keep = self._onscreen_mask(
                        result, viewpoint_cam, margin=float(cull_frustum),
                        low_pass=low_pass, frustum_near=frustum_near)
                    result = result[keep]
            return result

        if current_lod == int(self.current_lod):
            is_parent = self._get_is_parent()
        else:
            is_parent = torch.zeros(N, dtype=torch.bool, device=device)
            child_included = included & (parent_idx_all >= 0)
            valid_refs = parent_idx_all[child_included]
            if valid_refs.numel() > 0:
                is_parent[valid_refs.unique()] = True

        if mode in ("top_down_any_from_max", "fine_to_coarse"):
            active = (birth_lod == current_lod).clone()
        else:
            active = included & ~is_parent
        collapse_all = mode in ("leafes_to_coarse", "fine_to_coarse")

        for lod in range(current_lod, 1, -1):
            all_child_indices = self._indices_at_lod(lod, device=device)
            if all_child_indices.numel() == 0:
                continue
            all_parent_indices = parent_idx_all[all_child_indices]
            has_parent = all_parent_indices >= 0
            child_indices = all_child_indices[has_parent]
            parent_indices = all_parent_indices[has_parent]
            if child_indices.numel() == 0:
                continue

            is_active = active[child_indices]
            if not is_active.any():
                continue
            active_child_idx = child_indices[is_active]
            active_parent_idx = parent_indices[is_active]

            t_traversal = _profile_stop(profile, "active_traversal_s", t_traversal)
            t_proj = _profile_start(profile)
            active_child_proj = self._compute_projected_size_values(
                viewpoint_cam, candidate_idx=active_child_idx, low_pass=low_pass,
                cull_near=cull_near)
            _profile_stop(profile, "projected_size_s", t_proj)
            t_traversal = _profile_start(profile)
            too_small = active_child_proj < px_threshold

            unique_parents, inverse = active_parent_idx.unique(return_inverse=True)
            n_unique = unique_parents.shape[0]

            if collapse_all:
                blocked = torch.zeros(N, dtype=torch.bool, device=device)
                # An inactive direct child may still have active descendants.
                # Collapsing its parent would overlap those descendants.
                blocked[parent_indices[~is_active]] = True
                has_big_child = blocked[unique_parents]
                big_mask = active_child_proj >= px_threshold
                if big_mask.any():
                    has_big_child[inverse[big_mask]] = True
                has_small_child = torch.zeros(n_unique, dtype=torch.bool, device=device)
                small_mask = too_small & (active_child_proj >= 0)
                has_small_child[inverse[small_mask]] = True
                collapse = has_small_child & ~has_big_child
            else:
                has_small_child = torch.zeros(n_unique, dtype=torch.bool, device=device)
                if too_small.any():
                    has_small_child[inverse[too_small]] = True
                collapse = has_small_child

            collapsed_parents = unique_parents[collapse]
            if collapsed_parents.numel() == 0:
                continue

            parent_collapsed = torch.zeros(N, dtype=torch.bool, device=device)
            parent_collapsed[collapsed_parents] = True
            active[child_indices[parent_collapsed[parent_indices]]] = False
            active[collapsed_parents] = True

        _profile_stop(profile, "active_traversal_s", t_traversal)
        result = active.nonzero(as_tuple=True)[0]
        # Mirror the CUDA path's culling so both backends return the same set:
        # near cull first (cheap), then the on-screen tile cull (cull_frustum_kernel).
        if cull_near and float(cull_near) > 0.0 and result.numel() > 0:
            W2C = viewpoint_cam.world_view_transform.T.to(device=device, dtype=torch.float32)
            z_view = self._xyz.detach()[result].to(dtype=torch.float32) @ W2C[2, :3] + W2C[2, 3]
            result = result[z_view >= float(cull_near)]
        if cull_frustum and float(cull_frustum) > 0.0 and result.numel() > 0:
            if cull_backend == "gsplat":
                result = self._cull_offscreen_gsplat(
                    result, viewpoint_cam, low_pass=low_pass)
            else:
                keep = self._onscreen_mask(
                    result, viewpoint_cam, margin=float(cull_frustum),
                    low_pass=low_pass, frustum_near=frustum_near)
                result = result[keep]
        return result

    def _active_indices_top_down_projected_cuda(self, viewpoint_cam, px_threshold, N, device, mode,
                                                current_lod=None,
                                                profile=None, low_pass=0.3,
                                                cull_near=0.0, cull_frustum=0.0,
                                                frustum_near=0.2,
                                                cull_backend="inria"):
        if profile is not None:
            profile["cuda_backend_requested"] = "shape"
        if _hier_trav_cuda is None:
            return None
        if current_lod is None:
            current_lod = self.current_lod
        current_lod = int(current_lod)
        if mode not in self.CUDA_HIERARCHY_MODES:
            return None
        if self._parent_idx is None or self._birth_lod is None:
            return None
        if device.type != "cuda" or self._xyz.device.type != "cuda":
            return None
        if self._xyz.dtype != torch.float32:
            return None

        mode_id = {
            "leafes_to_coarse": 0,
            "fine_to_coarse": 1,
        }.get(mode)
        if mode_id is None and mode != "coarse_to_fine":
            return None

        try:
            hidx = self._ensure_hierarchy_index()
            is_parent = self._get_is_parent()
            if is_parent is None:
                return None
            if current_lod != self.current_lod:
                included = self._birth_lod <= current_lod
                is_parent = torch.zeros(N, dtype=torch.bool, device=device)
                parents = self._parent_idx[included & (self._parent_idx >= 0)]
                is_parent[parents] = True
                initial_root = (included & (self._parent_idx < 0)).to(torch.uint8)
                initial_leaf = (included & ~is_parent).to(torch.uint8)
                initial_finest = (self._birth_lod == current_lod).to(torch.uint8)
            else:
                initial_root = hidx.root_active_uint8
                initial_leaf = hidx.leaf_active_uint8
                initial_finest = hidx.finest_active_uint8
            viewmatrix = viewpoint_cam.world_view_transform.to(
                device=device, dtype=torch.float32).contiguous()
            shape = self._projected_shape_factors()
            if shape.device != self._xyz.device or shape.dtype != torch.float32:
                return None

            projmatrix = None
            if float(cull_frustum) > 0.0:
                projmatrix = viewpoint_cam.full_proj_transform.to(
                    device=device, dtype=torch.float32).contiguous()

            t_cuda = _profile_start(profile)
            if mode == "coarse_to_fine":
                result = _hier_trav_cuda.active_coarse_to_fine_from_shape(
                    self._xyz.detach(),
                    shape,
                    self._birth_lod.to(device=device),
                    hidx.parent_indices_int32,
                    hidx.lod_offsets_host,
                    viewmatrix,
                    int(viewpoint_cam.image_width),
                    int(viewpoint_cam.image_height),
                    math.tan(viewpoint_cam.FoVx * 0.5),
                    math.tan(viewpoint_cam.FoVy * 0.5),
                    float(low_pass),
                    float(px_threshold),
                    current_lod,
                    self._SCREENSPACE_METRIC_ID,
                    initial_active=initial_root,
                    cull_near=float(cull_near),
                    projmatrix=projmatrix,
                    cull_frustum=float(cull_frustum),
                    frustum_near=max(0.0, float(frustum_near)),
                    frustum_far=float(
                        getattr(viewpoint_cam, "zfar", 1.0e10)
                        if cull_backend == "gsplat" else 1.0e10
                    ),
                    cull_backend_id=1 if cull_backend == "gsplat" else 0,
                    opacity_logits=(
                        self._opacity.detach()
                        if cull_backend == "gsplat" and float(cull_frustum) > 0.0
                        else None
                    ),
                )
            else:
                result = _hier_trav_cuda.active_top_down_from_shape(
                    self._xyz.detach(),
                    shape,
                    self._birth_lod.to(device=device),
                    hidx.parent_indices_int32,
                    is_parent,
                    hidx.lod_offsets_host,
                    viewmatrix,
                    int(viewpoint_cam.image_width),
                    int(viewpoint_cam.image_height),
                    math.tan(viewpoint_cam.FoVx * 0.5),
                    math.tan(viewpoint_cam.FoVy * 0.5),
                    float(low_pass),
                    float(px_threshold),
                    current_lod,
                    mode_id,
                    self._SCREENSPACE_METRIC_ID,
                    initial_active=(
                        initial_leaf if mode_id == 0 else initial_finest
                    ),
                    cull_near=float(cull_near),
                    projmatrix=projmatrix,
                    cull_frustum=float(cull_frustum),
                    frustum_near=max(0.0, float(frustum_near)),
                    frustum_far=float(
                        getattr(viewpoint_cam, "zfar", 1.0e10)
                        if cull_backend == "gsplat" else 1.0e10
                    ),
                    cull_backend_id=1 if cull_backend == "gsplat" else 0,
                    opacity_logits=(
                        self._opacity.detach()
                        if cull_backend == "gsplat" and float(cull_frustum) > 0.0
                        else None
                    ),
                )
            _profile_stop(profile, "active_traversal_s", t_cuda)
            if profile is not None:
                profile["cuda_backend_used"] = "shape"
            return result
        except Exception as exc:
            self._cuda_hierarchy_traversal_error = repr(exc)
            return None

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        if getattr(self, "_densify_current_lod_only", False) and self._birth_lod is not None:
            selected_pts_mask &= (self._birth_lod == self.current_lod)

        if self._per_parent_budget is not None and self._parent_idx is not None:
            selected_pts_mask = self._apply_per_parent_budget(selected_pts_mask, 1)

        new_xyz = self._xyz[selected_pts_mask].clone()
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_mip_floor = self._mip_floor[selected_pts_mask] if self._mip_floor is not None else None
        new_parent_idx = self._parent_idx[selected_pts_mask] if self._parent_idx is not None else None
        new_birth_lod = self._birth_lod[selected_pts_mask] if self._birth_lod is not None else None


        # Cloning appends copies at the end; existing rows keep their indices, so
        # existing children remain correctly parented to the original Gaussian.
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_mip_floor, new_parent_idx=new_parent_idx, new_birth_lod=new_birth_lod)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify(self, max_grad, extent=None, restrict_to_lod=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if restrict_to_lod is not None and self._birth_lod is not None:
            non_target = (self._birth_lod != restrict_to_lod)
            grads[non_target] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
