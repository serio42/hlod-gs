import os
import sys
import shutil
from pathlib import Path

import torch

_EXT = None
_LOAD_ERROR = None
_DISABLED = False


def _as_host_offsets(lod_offsets):
    if torch.is_tensor(lod_offsets):
        return [int(value) for value in lod_offsets.detach().cpu().tolist()]
    return [int(value) for value in lod_offsets]


def _prepare_cuda_home():
    """Use a complete toolkit, respecting explicit CUDA paths before auto-detection."""
    candidates = []
    for name in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(name):
            candidates.append(Path(os.environ[name]))
    if torch.version.cuda:
        candidates.append(Path("/usr/local") / f"cuda-{torch.version.cuda}")
    if os.environ.get("CONDA_PREFIX"):
        candidates.append(Path(os.environ["CONDA_PREFIX"]))
    exe = Path(sys.executable).resolve()
    if len(exe.parents) >= 2 and exe.parent.name == "bin":
        candidates.append(exe.parent.parent)
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        candidates.append(Path(nvcc_path).parent.parent)

    for candidate in candidates:
        # A conda nvcc executable alone does not imply usable headers at the
        # same prefix (newer packages put them under targets/<arch>/include).
        if not all((candidate / relative).is_file() for relative in (
            "bin/nvcc", "include/cuda_runtime_api.h", "include/crt/host_defines.h"
        )):
            continue
        os.environ["CUDA_HOME"] = str(candidate)
        os.environ["CUDA_PATH"] = str(candidate)
        nvcc_bin = str(candidate / "bin")
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if path_parts[:1] != [nvcc_bin]:
            os.environ["PATH"] = os.pathsep.join([nvcc_bin] + path_parts)
        return


def _load_extension():
    global _EXT, _LOAD_ERROR, _DISABLED
    if _EXT is not None:
        return _EXT
    if _DISABLED or os.environ.get("HLOD_DISABLE_CUDA_HIER_TRAV", "0") == "1":
        return None
    if not torch.cuda.is_available():
        _DISABLED = True
        return None

    try:
        from . import _C as ext
        _EXT = ext
        return _EXT
    except Exception:
        pass

    try:
        _prepare_cuda_home()
        from torch.utils import cpp_extension

        cpp_extension.CUDA_HOME = os.environ.get("CUDA_HOME", cpp_extension.CUDA_HOME)
        load = cpp_extension.load

        root = Path(__file__).resolve().parent
        verbose = os.environ.get("HLOD_CUDA_HIER_TRAV_VERBOSE", "0") == "1"
        _EXT = load(
            name="hlod_hierarchical_traversal_cuda",
            sources=[
                str(root / "ext.cpp"),
                str(root / "hierarchical_traversal.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=verbose,
        )
        return _EXT
    except Exception as exc:
        _LOAD_ERROR = exc
        _DISABLED = True
        return None


def is_available():
    return _load_extension() is not None


def load_error():
    return _LOAD_ERROR


def cull_indices_frustum_from_shape(
    xyz,
    shape_factors,
    active_indices,
    viewmatrix,
    projmatrix,
    image_width,
    image_height,
    tan_fovx,
    tan_fovy,
    low_pass,
    cull_frustum,
    frustum_near=0.01,
):
    ext = _load_extension()
    if ext is None:
        raise RuntimeError(f"hierarchical traversal CUDA extension unavailable: {_LOAD_ERROR}")
    return ext.cull_indices_frustum_from_shape(
        xyz.contiguous(),
        shape_factors.contiguous(),
        active_indices.to(dtype=torch.long).contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        int(image_width),
        int(image_height),
        float(tan_fovx),
        float(tan_fovy),
        float(low_pass),
        float(cull_frustum),
        float(frustum_near),
    )


def projected_size_from_shape(
    xyz,
    shape_factors,
    viewmatrix,
    image_width,
    image_height,
    tan_fovx,
    tan_fovy,
    low_pass,
    metric_id,
    cull_near=0.0,
):
    ext = _load_extension()
    if ext is None:
        raise RuntimeError(f"hierarchical traversal CUDA extension unavailable: {_LOAD_ERROR}")
    return ext.projected_size_from_shape(
        xyz.contiguous(),
        shape_factors.contiguous(),
        viewmatrix.contiguous(),
        int(image_width),
        int(image_height),
        float(tan_fovx),
        float(tan_fovy),
        float(low_pass),
        float(cull_near),
        int(metric_id),
    )


def active_top_down_from_shape(
    xyz,
    shape_factors,
    birth_lod,
    parent_idx,
    is_parent,
    lod_offsets,
    viewmatrix,
    image_width,
    image_height,
    tan_fovx,
    tan_fovy,
    low_pass,
    px_threshold,
    current_lod,
    mode_id,
    metric_id,
    initial_active=None,
    cull_near=0.0,
    projmatrix=None,
    cull_frustum=0.0,
    frustum_near=0.01,
    frustum_far=1.0e10,
    cull_backend_id=0,
    opacity_logits=None,
):
    """Top-down LOD cut selection.

    cull_near > 0: drop Gaussians behind the near plane during the traversal.
    cull_frustum > 0: after the traversal, drop inactive-to-the-renderer splats.
    cull_backend_id 0 mirrors INRIA getRect and uses cull_frustum as a pixel
    margin. cull_backend_id 1 mirrors gsplat projection visibility, including
    near/far and opacity-aware anisotropic bounds; opacity_logits is required.
    """
    ext = _load_extension()
    if ext is None:
        raise RuntimeError(f"hierarchical traversal CUDA extension unavailable: {_LOAD_ERROR}")
    if float(cull_frustum) > 0.0 and projmatrix is None:
        raise ValueError("cull_frustum > 0 requires projmatrix (the camera's full_proj_transform)")
    if initial_active is None:
        included = birth_lod <= int(current_lod)
        if int(mode_id) == 1:
            initial_active = birth_lod == int(current_lod)
        else:
            initial_active = included & ~is_parent
    viewmatrix = viewmatrix.contiguous()
    if projmatrix is None:
        # Unused when cull_frustum <= 0; placeholder so the C++ side can validate it.
        projmatrix = torch.eye(4, dtype=viewmatrix.dtype, device=viewmatrix.device)
    if opacity_logits is None:
        opacity_logits = torch.empty(0, dtype=xyz.dtype, device=xyz.device)
    return ext.active_top_down_from_shape(
        xyz.contiguous(),
        shape_factors.contiguous(),
        opacity_logits.reshape(-1).contiguous(),
        parent_idx.to(dtype=torch.int32).contiguous(),
        initial_active.to(dtype=torch.uint8).contiguous(),
        _as_host_offsets(lod_offsets),
        viewmatrix,
        projmatrix.contiguous(),
        int(image_width),
        int(image_height),
        float(tan_fovx),
        float(tan_fovy),
        float(low_pass),
        float(cull_near),
        float(cull_frustum),
        float(frustum_near),
        float(frustum_far),
        int(cull_backend_id),
        float(px_threshold),
        int(current_lod),
        int(mode_id),
        int(metric_id),
    )


def active_coarse_to_fine_from_shape(
    xyz,
    shape_factors,
    birth_lod,
    parent_idx,
    lod_offsets,
    viewmatrix,
    image_width,
    image_height,
    tan_fovx,
    tan_fovy,
    low_pass,
    px_threshold,
    current_lod,
    metric_id,
    initial_active=None,
    cull_near=0.0,
    projmatrix=None,
    cull_frustum=0.0,
    frustum_near=0.01,
    frustum_far=1.0e10,
    cull_backend_id=0,
    opacity_logits=None,
):
    """True coarse-to-fine LOD traversal over the current active frontier."""
    ext = _load_extension()
    if ext is None:
        raise RuntimeError(f"hierarchical traversal CUDA extension unavailable: {_LOAD_ERROR}")
    if float(cull_frustum) > 0.0 and projmatrix is None:
        raise ValueError("cull_frustum > 0 requires projmatrix (the camera's full_proj_transform)")
    if initial_active is None:
        initial_active = (birth_lod <= int(current_lod)) & (parent_idx < 0)
    viewmatrix = viewmatrix.contiguous()
    if projmatrix is None:
        projmatrix = torch.eye(4, dtype=viewmatrix.dtype, device=viewmatrix.device)
    if opacity_logits is None:
        opacity_logits = torch.empty(0, dtype=xyz.dtype, device=xyz.device)
    return ext.active_coarse_to_fine_from_shape(
        xyz.contiguous(),
        shape_factors.contiguous(),
        opacity_logits.reshape(-1).contiguous(),
        parent_idx.to(dtype=torch.int32).contiguous(),
        initial_active.to(dtype=torch.uint8).contiguous(),
        _as_host_offsets(lod_offsets),
        viewmatrix,
        projmatrix.contiguous(),
        int(image_width),
        int(image_height),
        float(tan_fovx),
        float(tan_fovy),
        float(low_pass),
        float(cull_near),
        float(cull_frustum),
        float(frustum_near),
        float(frustum_far),
        int(cull_backend_id),
        float(px_threshold),
        int(current_lod),
        int(metric_id),
    )
