#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int THREADS = 256;

struct ProjectionConstants {
    float focal_x;
    float focal_y;
    float limit_x;
    float limit_y;
};

ProjectionConstants make_projection_constants(
    int image_width,
    int image_height,
    double tan_fovx,
    double tan_fovy)
{
    return {
        static_cast<float>(image_width / (2.0 * tan_fovx)),
        static_cast<float>(image_height / (2.0 * tan_fovy)),
        static_cast<float>(1.3 * tan_fovx),
        static_cast<float>(1.3 * tan_fovy),
    };
}

enum MetricId {
    METRIC_PROJECTED_SIZE = 0,
    METRIC_RASTERIZER_RADIUS = 1,
    METRIC_CUDA_RADIUS = 2,
};

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")

__forceinline__ __device__ float clampf(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}

__forceinline__ __device__ float compute_metric_from_shape(
    int idx,
    const float* __restrict__ xyz,
    const float* __restrict__ shape,
    const float* __restrict__ view,
    float focal_x,
    float focal_y,
    float limit_x,
    float limit_y,
    float low_pass,
    float cull_near,
    int metric_id)
{
    const float x = xyz[3 * idx + 0];
    const float y = xyz[3 * idx + 1];
    const float z0 = xyz[3 * idx + 2];

    const float tx_view = view[0] * x + view[4] * y + view[8] * z0 + view[12];
    const float ty_view = view[1] * x + view[5] * y + view[9] * z0 + view[13];
    const float tz_view_raw = view[2] * x + view[6] * y + view[10] * z0 + view[14];
    // cull_near > 0: drop Gaussians behind the near plane (return -1) instead of
    // clamping to z=0.2, which would project them huge. Negative return = culled.
    if (cull_near > 0.0f && tz_view_raw < cull_near) {
        return -1.0f;
    }
    const float projection_floor = cull_near > 0.0f ? cull_near : 0.2f;
    const float tz_view = fmaxf(tz_view_raw, projection_floor);

    const float* s = shape + 9 * idx;

    const float w00 = view[0], w01 = view[4], w02 = view[8];
    const float w10 = view[1], w11 = view[5], w12 = view[9];
    const float w20 = view[2], w21 = view[6], w22 = view[10];

    float row0[3], row1[3], row2[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const float v0 = s[0 * 3 + j];
        const float v1 = s[1 * 3 + j];
        const float v2 = s[2 * 3 + j];
        row0[j] = w00 * v0 + w01 * v1 + w02 * v2;
        row1[j] = w10 * v0 + w11 * v1 + w12 * v2;
        row2[j] = w20 * v0 + w21 * v1 + w22 * v2;
    }

    const float inv_z = 1.0f / tz_view;
    const float normalized_x = clampf(tx_view * inv_z, -limit_x, limit_x);
    const float normalized_y = clampf(ty_view * inv_z, -limit_y, limit_y);
    const float jacobian_x = focal_x * inv_z;
    const float jacobian_y = focal_y * inv_z;

    float cov00 = low_pass;
    float cov01 = 0.0f;
    float cov11 = low_pass;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const float t0 = jacobian_x * (row0[j] - normalized_x * row2[j]);
        const float t1 = jacobian_y * (row1[j] - normalized_y * row2[j]);
        cov00 += t0 * t0;
        cov01 += t0 * t1;
        cov11 += t1 * t1;
    }

    if (metric_id == METRIC_PROJECTED_SIZE) {
        const float det_term = sqrtf(fmaxf((cov00 - cov11) * (cov00 - cov11) + 4.0f * cov01 * cov01, 0.0f));
        return sqrtf(fmaxf(0.5f * (cov00 + cov11 + det_term), 0.0f));
    }

    const float det = cov00 * cov11 - cov01 * cov01;
    const float mid = 0.5f * (cov00 + cov11);
    const float lambda_max = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
    const float radius = 3.0f * sqrtf(fmaxf(lambda_max, 0.0f));
    return metric_id == METRIC_CUDA_RADIUS ? ceilf(radius) : radius;
}

__global__ void projected_size_kernel(
    int n,
    const float* __restrict__ xyz,
    const float* __restrict__ shape,
    const float* __restrict__ view,
    float focal_x,
    float focal_y,
    float limit_x,
    float limit_y,
    float low_pass,
    float cull_near,
    int metric_id,
    float* __restrict__ out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    out[idx] = compute_metric_from_shape(
        idx, xyz, shape, view, focal_x, focal_y,
        limit_x, limit_y, low_pass, cull_near, metric_id);
}

__global__ void mark_parent_flags_kernel(
    int start,
    int end,
    const int32_t* __restrict__ parent_idx,
    const uint8_t* __restrict__ active,
    const float* __restrict__ xyz,
    const float* __restrict__ shape,
    const float* __restrict__ view,
    float focal_x,
    float focal_y,
    float limit_x,
    float limit_y,
    float low_pass,
    float cull_near,
    float px_threshold,
    int metric_id,
    uint8_t* __restrict__ has_small,
    uint8_t* __restrict__ has_big)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int child = start + offset;
    if (child >= end) {
        return;
    }
    const int parent = parent_idx[child];
    if (parent < 0) {
        return;
    }
    if (!active[child]) {
        // This branch can still be represented by active deeper descendants.
        // Its ancestor must not collapse over that finer frontier.
        has_big[parent] = 1;
        return;
    }

    const float metric = compute_metric_from_shape(
        child, xyz, shape, view, focal_x, focal_y,
        limit_x, limit_y, low_pass, cull_near, metric_id);
    // Culled child (metric < 0): skip so it sets neither has_small nor has_big and
    // can't affect the parent's collapse. It stays active; the final cull removes it.
    if (metric < 0.0f) {
        return;
    }
    // These level-local flags only transition 0 -> 1. Concurrent siblings write
    // the same byte, and the next kernel launch is the visibility boundary, so an
    // atomic read-modify-write would only add serialization.
    if (metric < px_threshold) {
        has_small[parent] = 1;
    } else {
        has_big[parent] = 1;
    }
}

__global__ void collapse_update_kernel(
    int start,
    int end,
    const int32_t* __restrict__ parent_idx,
    const uint8_t* __restrict__ has_small,
    const uint8_t* __restrict__ has_big,
    uint8_t* __restrict__ active)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int child = start + offset;
    if (child >= end) {
        return;
    }

    const int parent = parent_idx[child];
    if (parent < 0) {
        return;
    }

    const bool collapsed = (has_small[parent] != 0 && has_big[parent] == 0);
    if (collapsed) {
        active[child] = 0;
        active[parent] = 1;
    }
}

__global__ void mark_split_flags_kernel(
    int start,
    int end,
    const int32_t* __restrict__ parent_idx,
    const uint8_t* __restrict__ active,
    const float* __restrict__ xyz,
    const float* __restrict__ shape,
    const float* __restrict__ view,
    float focal_x,
    float focal_y,
    float limit_x,
    float limit_y,
    float low_pass,
    float cull_near,
    float px_threshold,
    int metric_id,
    uint8_t* __restrict__ split_parent)
{
    const int child = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (child >= end) {
        return;
    }
    const int parent = parent_idx[child];
    if (parent < 0 || !active[parent]) {
        return;
    }
    const float metric = compute_metric_from_shape(
        child, xyz, shape, view, focal_x, focal_y,
        limit_x, limit_y, low_pass, cull_near, metric_id);
    if (metric >= px_threshold) {
        // Idempotent 0 -> 1 store; consumed only by the subsequent kernel launch.
        split_parent[parent] = 1;
    }
}

__global__ void apply_split_flags_kernel(
    int start,
    int end,
    const int32_t* __restrict__ parent_idx,
    const uint8_t* __restrict__ split_parent,
    uint8_t* __restrict__ active)
{
    const int child = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (child >= end) {
        return;
    }
    const int parent = parent_idx[child];
    if (parent >= 0 && split_parent[parent] != 0) {
        active[parent] = 0;
        active[child] = 1;
    }
}

// Final near-cull over the whole active set (after traversal), so behind-near
// Gaussians are dropped at any level, including level-1 roots. Purely subtractive.
__global__ void cull_active_kernel(
    int n,
    const float* __restrict__ xyz,
    const float* __restrict__ view,
    float cull_near,
    uint8_t* __restrict__ active)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (!active[idx]) {
        return;
    }
    const float x = xyz[3 * idx + 0];
    const float y = xyz[3 * idx + 1];
    const float z0 = xyz[3 * idx + 2];
    const float tz_view_raw = view[2] * x + view[6] * y + view[10] * z0 + view[14];
    if (tz_view_raw < cull_near) {
        active[idx] = 0;
    }
}

// Rebuild the dense active mask from an index set (for the standalone on-screen cull).
__global__ void set_active_indices_kernel(
    int m,
    const int64_t* __restrict__ indices,
    uint8_t* __restrict__ active)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) {
        return;
    }
    active[indices[i]] = 1;
}

// Final on-screen cull over the active set: replicates the rasterizer's
// preprocessCUDA visibility test (camera near cull, project center, same radius,
// getRect 16x16 tile-overlap). Anything dropped had zero pixel contribution, so
// the image is unchanged. Purely subtractive. margin_px widens the radius (0 = exact).
__global__ void cull_frustum_kernel(
    int n,
    const float* __restrict__ xyz,
    const float* __restrict__ shape,
    const float* __restrict__ opacity_logits,
    const float* __restrict__ view,
    const float* __restrict__ proj,
    int image_width,
    int image_height,
    float focal_x,
    float focal_y,
    float limit_x,
    float limit_y,
    float low_pass,
    float cull_near,
    float frustum_near,
    float frustum_far,
    float margin_px,
    int cull_backend_id,
    uint8_t* __restrict__ active)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (!active[idx]) {
        return;
    }

    const float x = xyz[3 * idx + 0];
    const float y = xyz[3 * idx + 1];
    const float z0 = xyz[3 * idx + 2];

    const float tz_view = view[2] * x + view[6] * y + view[10] * z0 + view[14];
    // Match the active renderer's depth interval. A user-provided cull_near may
    // only make the near test more aggressive, never less aggressive.
    const float effective_near = fmaxf(frustum_near, cull_near);
    const bool depth_culled = cull_backend_id == 1
        ? (tz_view < effective_near || tz_view > frustum_far)
        : (tz_view <= effective_near);
    if (depth_culled) {
        active[idx] = 0;
        return;
    }

    if (cull_backend_id == 1) {
        // Match gsplat's fused EWA projection visibility: opacity threshold,
        // covariance validity, opacity-aware anisotropic radii, and direct
        // pixel-boundary clipping.
        constexpr float ALPHA_THRESHOLD = 1.0f / 255.0f;
        const float opacity = 1.0f / (1.0f + expf(-opacity_logits[idx]));
        if (opacity < ALPHA_THRESHOLD) {
            active[idx] = 0;
            return;
        }

        const float* s = shape + 9 * idx;
        const float w00 = view[0], w01 = view[4], w02 = view[8];
        const float w10 = view[1], w11 = view[5], w12 = view[9];
        const float w20 = view[2], w21 = view[6], w22 = view[10];
        float row0[3], row1[3], row2[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float v0 = s[0 * 3 + j];
            const float v1 = s[1 * 3 + j];
            const float v2 = s[2 * 3 + j];
            row0[j] = w00 * v0 + w01 * v1 + w02 * v2;
            row1[j] = w10 * v0 + w11 * v1 + w12 * v2;
            row2[j] = w20 * v0 + w21 * v1 + w22 * v2;
        }

        const float tx_view = view[0] * x + view[4] * y + view[8] * z0 + view[12];
        const float ty_view = view[1] * x + view[5] * y + view[9] * z0 + view[13];
        const float tx = clampf(tx_view / tz_view, -limit_x, limit_x) * tz_view;
        const float ty = clampf(ty_view / tz_view, -limit_y, limit_y) * tz_view;
        const float inv_z = 1.0f / tz_view;
        const float inv_z2 = inv_z * inv_z;

        float cov00 = low_pass;
        float cov01 = 0.0f;
        float cov11 = low_pass;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float t0 = focal_x * (
                inv_z * row0[j] - tx * inv_z2 * row2[j]);
            const float t1 = focal_y * (
                inv_z * row1[j] - ty * inv_z2 * row2[j]);
            cov00 += t0 * t0;
            cov01 += t0 * t1;
            cov11 += t1 * t1;
        }

        const float det = cov00 * cov11 - cov01 * cov01;
        if (det <= 0.0f) {
            active[idx] = 0;
            return;
        }

        const float extend = fminf(
            3.33f,
            sqrtf(2.0f * logf(opacity / ALPHA_THRESHOLD)));
        const float radius_x = ceilf(extend * sqrtf(cov00));
        const float radius_y = ceilf(extend * sqrtf(cov11));
        if (radius_x <= 0.0f && radius_y <= 0.0f) {
            active[idx] = 0;
            return;
        }

        const float point_x = focal_x * tx_view / tz_view
            + 0.5f * static_cast<float>(image_width);
        const float point_y = focal_y * ty_view / tz_view
            + 0.5f * static_cast<float>(image_height);
        if (
            point_x + radius_x <= 0.0f
            || point_x - radius_x >= static_cast<float>(image_width)
            || point_y + radius_y <= 0.0f
            || point_y - radius_y >= static_cast<float>(image_height)
        ) {
            active[idx] = 0;
        }
        return;
    }

    // Project the center (rasterizer transformPoint4x4 convention) to pixel space.
    const float hx = proj[0] * x + proj[4] * y + proj[8] * z0 + proj[12];
    const float hy = proj[1] * x + proj[5] * y + proj[9] * z0 + proj[13];
    const float hw = proj[3] * x + proj[7] * y + proj[11] * z0 + proj[15];
    const float inv_w = 1.0f / (hw + 1e-7f);
    const float point_x = ((hx * inv_w + 1.0f) * image_width - 1.0f) * 0.5f;
    const float point_y = ((hy * inv_w + 1.0f) * image_height - 1.0f) * 0.5f;

    // Use the same near plane for the Jacobian projection. A fixed 0.2 floor
    // underestimates close Street-view splats and can cull visible geometry.
    const float radius = compute_metric_from_shape(
        idx, xyz, shape, view, focal_x, focal_y,
        limit_x, limit_y, low_pass, effective_near, METRIC_CUDA_RADIUS) + margin_px;

    // getRect tile-overlap test (BLOCK 16x16), clamped to the tile grid.
    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;
    const int grid_x = (image_width + BLOCK_X - 1) / BLOCK_X;
    const int grid_y = (image_height + BLOCK_Y - 1) / BLOCK_Y;
    const int rect_min_x = min(grid_x, max(0, static_cast<int>((point_x - radius) / BLOCK_X)));
    const int rect_min_y = min(grid_y, max(0, static_cast<int>((point_y - radius) / BLOCK_Y)));
    const int rect_max_x = min(grid_x, max(0, static_cast<int>((point_x + radius + BLOCK_X - 1) / BLOCK_X)));
    const int rect_max_y = min(grid_y, max(0, static_cast<int>((point_y + radius + BLOCK_Y - 1) / BLOCK_Y)));
    if ((rect_max_x - rect_min_x) * (rect_max_y - rect_min_y) == 0) {
        active[idx] = 0;
    }
}

void validate_project_inputs(
    const torch::Tensor& xyz,
    const torch::Tensor& shape_factors,
    const torch::Tensor& viewmatrix)
{
    CHECK_CUDA(xyz);
    CHECK_CUDA(shape_factors);
    CHECK_CUDA(viewmatrix);
    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(shape_factors);
    CHECK_CONTIGUOUS(viewmatrix);
    CHECK_FLOAT(xyz);
    CHECK_FLOAT(shape_factors);
    CHECK_FLOAT(viewmatrix);
    TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3, "xyz must have shape [N, 3]");
    TORCH_CHECK(shape_factors.dim() == 3 && shape_factors.size(1) == 3 && shape_factors.size(2) == 3,
                "shape_factors must have shape [N, 3, 3]");
    TORCH_CHECK(shape_factors.size(0) == xyz.size(0), "xyz and shape_factors must have the same N");
    TORCH_CHECK(viewmatrix.numel() == 16, "viewmatrix must contain 16 float values");
}

}  // namespace

torch::Tensor projected_size_from_shape_cuda(
    torch::Tensor xyz,
    torch::Tensor shape_factors,
    torch::Tensor viewmatrix,
    int image_width,
    int image_height,
    double tan_fovx,
    double tan_fovy,
    double low_pass,
    double cull_near,
    int metric_id)
{
    validate_project_inputs(xyz, shape_factors, viewmatrix);
    TORCH_CHECK(metric_id >= 0 && metric_id <= 2, "invalid metric_id");
    c10::cuda::CUDAGuard guard(xyz.device());

    const int n = static_cast<int>(xyz.size(0));
    auto out = torch::empty({n}, xyz.options());
    if (n == 0) {
        return out;
    }

    const dim3 block(THREADS);
    const dim3 grid((n + THREADS - 1) / THREADS);
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto projection =
        make_projection_constants(image_width, image_height, tan_fovx, tan_fovy);
    projected_size_kernel<<<grid, block, 0, stream>>>(
        n,
        xyz.data_ptr<float>(),
        shape_factors.data_ptr<float>(),
        viewmatrix.data_ptr<float>(),
        projection.focal_x,
        projection.focal_y,
        projection.limit_x,
        projection.limit_y,
        static_cast<float>(low_pass),
        static_cast<float>(cull_near),
        metric_id,
        out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor cull_indices_frustum_from_shape_cuda(
    torch::Tensor xyz,
    torch::Tensor shape_factors,
    torch::Tensor active_indices,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    int image_width,
    int image_height,
    double tan_fovx,
    double tan_fovy,
    double low_pass,
    double cull_frustum,
    double frustum_near)
{
    validate_project_inputs(xyz, shape_factors, viewmatrix);
    CHECK_CUDA(active_indices);
    CHECK_CUDA(projmatrix);
    CHECK_CONTIGUOUS(active_indices);
    CHECK_CONTIGUOUS(projmatrix);
    CHECK_FLOAT(projmatrix);
    TORCH_CHECK(active_indices.scalar_type() == torch::kInt64, "active_indices must be int64");
    TORCH_CHECK(active_indices.dim() == 1, "active_indices must have shape [M]");
    TORCH_CHECK(projmatrix.numel() == 16, "projmatrix must contain 16 float values");
    TORCH_CHECK(frustum_near >= 0.0, "frustum_near must be non-negative");
    c10::cuda::CUDAGuard guard(xyz.device());

    const int n = static_cast<int>(xyz.size(0));
    const int m = static_cast<int>(active_indices.size(0));
    auto byte_opts = xyz.options().dtype(torch::kUInt8);
    auto int_opts = xyz.options().dtype(torch::kInt32);
    auto long_opts = xyz.options().dtype(torch::kInt64);

    if (n == 0 || m == 0) {
        return torch::empty({0}, long_opts);
    }

    auto active = torch::zeros({n}, byte_opts);
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(THREADS);
    const dim3 grid_m((m + THREADS - 1) / THREADS);
    const dim3 grid_n((n + THREADS - 1) / THREADS);
    const auto projection =
        make_projection_constants(image_width, image_height, tan_fovx, tan_fovy);

    set_active_indices_kernel<<<grid_m, block, 0, stream>>>(
        m,
        active_indices.data_ptr<int64_t>(),
        active.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cull_frustum_kernel<<<grid_n, block, 0, stream>>>(
        n,
        xyz.data_ptr<float>(),
        shape_factors.data_ptr<float>(),
        nullptr,
        viewmatrix.data_ptr<float>(),
        projmatrix.data_ptr<float>(),
        image_width,
        image_height,
        projection.focal_x,
        projection.focal_y,
        projection.limit_x,
        projection.limit_y,
        static_cast<float>(low_pass),
        0.0f,  // no additional traversal near cull
        static_cast<float>(frustum_near),
        1.0e10f,
        static_cast<float>(cull_frustum),
        0,  // INRIA/getRect backend
        active.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto selected = torch::empty({n}, long_opts);
    auto num_selected = torch::empty({1}, int_opts);
    cub::CountingInputIterator<int64_t> counting_itr(0);
    const uint8_t* flag_itr = active.data_ptr<uint8_t>();

    size_t temp_storage_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr,
        temp_storage_bytes,
        counting_itr,
        flag_itr,
        selected.data_ptr<int64_t>(),
        num_selected.data_ptr<int>(),
        n,
        stream));
    auto temp_storage = torch::empty({static_cast<int64_t>(temp_storage_bytes)}, byte_opts);
    C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
        temp_storage.data_ptr<uint8_t>(),
        temp_storage_bytes,
        counting_itr,
        flag_itr,
        selected.data_ptr<int64_t>(),
        num_selected.data_ptr<int>(),
        n,
        stream));

    int count_host = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(&count_host, num_selected.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return selected.narrow(0, 0, count_host).contiguous();
}

torch::Tensor active_top_down_from_shape_cuda(
    torch::Tensor xyz,
    torch::Tensor shape_factors,
    torch::Tensor opacity_logits,
    torch::Tensor parent_idx,
    torch::Tensor initial_active,
    std::vector<int64_t> lod_offsets,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    int image_width,
    int image_height,
    double tan_fovx,
    double tan_fovy,
    double low_pass,
    double cull_near,
    double cull_frustum,
    double frustum_near,
    double frustum_far,
    int cull_backend_id,
    double px_threshold,
    int current_lod,
    int mode_id,
    int metric_id)
{
    validate_project_inputs(xyz, shape_factors, viewmatrix);
    CHECK_CUDA(projmatrix);
    CHECK_CUDA(opacity_logits);
    CHECK_CONTIGUOUS(projmatrix);
    CHECK_CONTIGUOUS(opacity_logits);
    CHECK_FLOAT(projmatrix);
    CHECK_FLOAT(opacity_logits);
    TORCH_CHECK(projmatrix.numel() == 16, "projmatrix must contain 16 float values");
    CHECK_CUDA(parent_idx);
    CHECK_CUDA(initial_active);
    CHECK_CONTIGUOUS(parent_idx);
    CHECK_CONTIGUOUS(initial_active);
    TORCH_CHECK(parent_idx.scalar_type() == torch::kInt32, "parent_idx must be int32");
    TORCH_CHECK(initial_active.scalar_type() == torch::kUInt8, "initial_active must be uint8");
    TORCH_CHECK(parent_idx.size(0) == xyz.size(0), "parent_idx length must match xyz");
    TORCH_CHECK(initial_active.size(0) == xyz.size(0), "initial_active length must match xyz");
    TORCH_CHECK(current_lod >= 1, "current_lod must be >= 1");
    TORCH_CHECK(current_lod < static_cast<int>(lod_offsets.size()),
                "lod_offsets is too short for current_lod");
    TORCH_CHECK(mode_id >= 0 && mode_id <= 1, "invalid mode_id");
    TORCH_CHECK(metric_id >= 0 && metric_id <= 2, "invalid metric_id");
    TORCH_CHECK(frustum_near >= 0.0, "frustum_near must be non-negative");
    TORCH_CHECK(frustum_far > frustum_near, "frustum_far must exceed frustum_near");
    TORCH_CHECK(cull_backend_id == 0 || cull_backend_id == 1, "invalid cull_backend_id");
    if (cull_frustum > 0.0 && cull_backend_id == 1) {
        TORCH_CHECK(
            opacity_logits.numel() == xyz.size(0),
            "gsplat culling requires one opacity logit per Gaussian");
    }
    c10::cuda::CUDAGuard guard(xyz.device());

    const int n = static_cast<int>(xyz.size(0));
    auto byte_opts = xyz.options().dtype(torch::kUInt8);
    auto int_opts = xyz.options().dtype(torch::kInt32);
    auto long_opts = xyz.options().dtype(torch::kInt64);

    auto active = torch::empty({n}, byte_opts);
    if (n == 0) {
        return torch::empty({0}, long_opts);
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(THREADS);
    const dim3 grid_n((n + THREADS - 1) / THREADS);
    const auto projection =
        make_projection_constants(image_width, image_height, tan_fovx, tan_fovy);

    C10_CUDA_CHECK(cudaMemcpyAsync(
        active.data_ptr<uint8_t>(),
        initial_active.data_ptr<uint8_t>(),
        static_cast<size_t>(n),
        cudaMemcpyDeviceToDevice,
        stream));

    const int64_t parent_flag_count = lod_offsets[current_lod - 1];
    auto has_small = torch::empty({parent_flag_count}, byte_opts);
    auto has_big = torch::empty({parent_flag_count}, byte_opts);

    for (int lod = current_lod; lod >= 2; --lod) {
        const int start = static_cast<int>(lod_offsets[lod - 1]);
        const int end = static_cast<int>(lod_offsets[lod]);
        if (end <= start) {
            continue;
        }

        const int parent_start = static_cast<int>(lod_offsets[lod - 2]);
        const int parent_end = static_cast<int>(lod_offsets[lod - 1]);
        const size_t parent_bytes = static_cast<size_t>(parent_end - parent_start);
        C10_CUDA_CHECK(cudaMemsetAsync(
            has_small.data_ptr<uint8_t>() + parent_start, 0, parent_bytes, stream));
        C10_CUDA_CHECK(cudaMemsetAsync(
            has_big.data_ptr<uint8_t>() + parent_start, 0, parent_bytes, stream));

        const int count = end - start;
        const dim3 grid_lod((count + THREADS - 1) / THREADS);
        mark_parent_flags_kernel<<<grid_lod, block, 0, stream>>>(
            start,
            end,
            parent_idx.data_ptr<int32_t>(),
            active.data_ptr<uint8_t>(),
            xyz.data_ptr<float>(),
            shape_factors.data_ptr<float>(),
            viewmatrix.data_ptr<float>(),
            projection.focal_x,
            projection.focal_y,
            projection.limit_x,
            projection.limit_y,
            static_cast<float>(low_pass),
            static_cast<float>(cull_near),
            static_cast<float>(px_threshold),
            metric_id,
            has_small.data_ptr<uint8_t>(),
            has_big.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        collapse_update_kernel<<<grid_lod, block, 0, stream>>>(
            start,
            end,
            parent_idx.data_ptr<int32_t>(),
            has_small.data_ptr<uint8_t>(),
            has_big.data_ptr<uint8_t>(),
            active.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Final culls. The frustum cull folds in the near cull (cull_near below), so
    // the standalone near pass only runs when the frustum cull is off.
    if (cull_frustum > 0.0) {
        cull_frustum_kernel<<<grid_n, block, 0, stream>>>(
            n,
            xyz.data_ptr<float>(),
            shape_factors.data_ptr<float>(),
            opacity_logits.numel() > 0 ? opacity_logits.data_ptr<float>() : nullptr,
            viewmatrix.data_ptr<float>(),
            projmatrix.data_ptr<float>(),
            image_width,
            image_height,
            projection.focal_x,
            projection.focal_y,
            projection.limit_x,
            projection.limit_y,
            static_cast<float>(low_pass),
            static_cast<float>(cull_near),
            static_cast<float>(frustum_near),
            static_cast<float>(frustum_far),
            static_cast<float>(cull_frustum),
            cull_backend_id,
            active.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (cull_near > 0.0) {
        cull_active_kernel<<<grid_n, block, 0, stream>>>(
            n,
            xyz.data_ptr<float>(),
            viewmatrix.data_ptr<float>(),
            static_cast<float>(cull_near),
            active.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto selected = torch::empty({n}, long_opts);
    auto num_selected = torch::empty({1}, int_opts);
    cub::CountingInputIterator<int64_t> counting_itr(0);
    const uint8_t* flag_itr = active.data_ptr<uint8_t>();

    size_t temp_storage_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr,
        temp_storage_bytes,
        counting_itr,
        flag_itr,
        selected.data_ptr<int64_t>(),
        num_selected.data_ptr<int>(),
        n,
        stream));

    auto temp_storage = torch::empty({static_cast<int64_t>(temp_storage_bytes)}, byte_opts);
    C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
        temp_storage.data_ptr<uint8_t>(),
        temp_storage_bytes,
        counting_itr,
        flag_itr,
        selected.data_ptr<int64_t>(),
        num_selected.data_ptr<int>(),
        n,
        stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int count_host = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(&count_host, num_selected.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return selected.narrow(0, 0, count_host);
}

torch::Tensor active_coarse_to_fine_from_shape_cuda(
    torch::Tensor xyz,
    torch::Tensor shape_factors,
    torch::Tensor opacity_logits,
    torch::Tensor parent_idx,
    torch::Tensor initial_active,
    std::vector<int64_t> lod_offsets,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    int image_width,
    int image_height,
    double tan_fovx,
    double tan_fovy,
    double low_pass,
    double cull_near,
    double cull_frustum,
    double frustum_near,
    double frustum_far,
    int cull_backend_id,
    double px_threshold,
    int current_lod,
    int metric_id)
{
    validate_project_inputs(xyz, shape_factors, viewmatrix);
    CHECK_CUDA(projmatrix);
    CHECK_CUDA(opacity_logits);
    CHECK_CONTIGUOUS(projmatrix);
    CHECK_CONTIGUOUS(opacity_logits);
    CHECK_FLOAT(projmatrix);
    CHECK_FLOAT(opacity_logits);
    TORCH_CHECK(projmatrix.numel() == 16, "projmatrix must contain 16 float values");
    CHECK_CUDA(parent_idx);
    CHECK_CUDA(initial_active);
    CHECK_CONTIGUOUS(parent_idx);
    CHECK_CONTIGUOUS(initial_active);
    TORCH_CHECK(parent_idx.scalar_type() == torch::kInt32, "parent_idx must be int32");
    TORCH_CHECK(initial_active.scalar_type() == torch::kUInt8, "initial_active must be uint8");
    TORCH_CHECK(parent_idx.size(0) == xyz.size(0), "parent_idx length must match xyz");
    TORCH_CHECK(initial_active.size(0) == xyz.size(0), "initial_active length must match xyz");
    TORCH_CHECK(current_lod >= 1, "current_lod must be >= 1");
    TORCH_CHECK(current_lod < static_cast<int>(lod_offsets.size()),
                "lod_offsets is too short for current_lod");
    TORCH_CHECK(metric_id >= 0 && metric_id <= 2, "invalid metric_id");
    TORCH_CHECK(frustum_near >= 0.0, "frustum_near must be non-negative");
    TORCH_CHECK(frustum_far > frustum_near, "frustum_far must exceed frustum_near");
    TORCH_CHECK(cull_backend_id == 0 || cull_backend_id == 1, "invalid cull_backend_id");
    if (cull_frustum > 0.0 && cull_backend_id == 1) {
        TORCH_CHECK(
            opacity_logits.numel() == xyz.size(0),
            "gsplat culling requires one opacity logit per Gaussian");
    }
    c10::cuda::CUDAGuard guard(xyz.device());

    const int n = static_cast<int>(xyz.size(0));
    auto byte_opts = xyz.options().dtype(torch::kUInt8);
    auto int_opts = xyz.options().dtype(torch::kInt32);
    auto long_opts = xyz.options().dtype(torch::kInt64);
    if (n == 0) {
        return torch::empty({0}, long_opts);
    }

    auto active = torch::empty({n}, byte_opts);
    const int64_t parent_flag_count = lod_offsets[current_lod - 1];
    auto split_parent = torch::empty({parent_flag_count}, byte_opts);

    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(THREADS);
    const dim3 grid_n((n + THREADS - 1) / THREADS);
    const auto projection =
        make_projection_constants(image_width, image_height, tan_fovx, tan_fovy);

    C10_CUDA_CHECK(cudaMemcpyAsync(
        active.data_ptr<uint8_t>(),
        initial_active.data_ptr<uint8_t>(),
        static_cast<size_t>(n),
        cudaMemcpyDeviceToDevice,
        stream));

    for (int child_lod = 2; child_lod <= current_lod; ++child_lod) {
        const int start = static_cast<int>(lod_offsets[child_lod - 1]);
        const int end = static_cast<int>(lod_offsets[child_lod]);
        if (end <= start) {
            continue;
        }
        const int parent_start = static_cast<int>(lod_offsets[child_lod - 2]);
        const int parent_end = static_cast<int>(lod_offsets[child_lod - 1]);
        C10_CUDA_CHECK(cudaMemsetAsync(
            split_parent.data_ptr<uint8_t>() + parent_start,
            0,
            static_cast<size_t>(parent_end - parent_start),
            stream));
        const int count = end - start;
        const dim3 grid_lod((count + THREADS - 1) / THREADS);
        mark_split_flags_kernel<<<grid_lod, block, 0, stream>>>(
            start,
            end,
            parent_idx.data_ptr<int32_t>(),
            active.data_ptr<uint8_t>(),
            xyz.data_ptr<float>(),
            shape_factors.data_ptr<float>(),
            viewmatrix.data_ptr<float>(),
            projection.focal_x,
            projection.focal_y,
            projection.limit_x,
            projection.limit_y,
            static_cast<float>(low_pass),
            static_cast<float>(cull_near),
            static_cast<float>(px_threshold),
            metric_id,
            split_parent.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        apply_split_flags_kernel<<<grid_lod, block, 0, stream>>>(
            start,
            end,
            parent_idx.data_ptr<int32_t>(),
            split_parent.data_ptr<uint8_t>(),
            active.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (cull_frustum > 0.0) {
        cull_frustum_kernel<<<grid_n, block, 0, stream>>>(
            n,
            xyz.data_ptr<float>(),
            shape_factors.data_ptr<float>(),
            opacity_logits.numel() > 0 ? opacity_logits.data_ptr<float>() : nullptr,
            viewmatrix.data_ptr<float>(),
            projmatrix.data_ptr<float>(),
            image_width,
            image_height,
            projection.focal_x,
            projection.focal_y,
            projection.limit_x,
            projection.limit_y,
            static_cast<float>(low_pass),
            static_cast<float>(cull_near),
            static_cast<float>(frustum_near),
            static_cast<float>(frustum_far),
            static_cast<float>(cull_frustum),
            cull_backend_id,
            active.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (cull_near > 0.0) {
        cull_active_kernel<<<grid_n, block, 0, stream>>>(
            n,
            xyz.data_ptr<float>(),
            viewmatrix.data_ptr<float>(),
            static_cast<float>(cull_near),
            active.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto selected = torch::empty({n}, long_opts);
    auto num_selected = torch::empty({1}, int_opts);
    cub::CountingInputIterator<int64_t> counting_itr(0);
    size_t select_temp_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr,
        select_temp_bytes,
        counting_itr,
        active.data_ptr<uint8_t>(),
        selected.data_ptr<int64_t>(),
        num_selected.data_ptr<int>(),
        n,
        stream));
    auto select_temp = torch::empty({static_cast<int64_t>(select_temp_bytes)}, byte_opts);
    C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
        select_temp.data_ptr<uint8_t>(),
        select_temp_bytes,
        counting_itr,
        active.data_ptr<uint8_t>(),
        selected.data_ptr<int64_t>(),
        num_selected.data_ptr<int>(),
        n,
        stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int selected_count = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(
        &selected_count,
        num_selected.data_ptr<int>(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    return selected.narrow(0, 0, selected_count);
}
