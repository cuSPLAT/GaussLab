#include "includes/knn.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <limits>

struct grid_dims_t
{
    int x, y, z;
};

// =========================================================================
//  Kernel 1A: Partial Reduction Kernel
// =========================================================================
// each block computes the min/max of a subset of the points and writes its
// partial result to a temporary global memory array.
__global__ void calculate_bounds_kernel_partial(
    const float3* points, int n,
    float3* partial_min_bounds, float3* partial_max_bounds)
{
    // shared memory for the reduction within one block
    extern __shared__ float3 sdata[];
    float3* s_min = sdata;
    float3* s_max = &sdata[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread initializes its shared memory locations
    s_min[tid] = make_float3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    s_max[tid] = make_float3(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());

    // each thread finds the min/max for the points it's responsible for (grid-stride loop)
    while (i < n)
    {
        float3 p = points[i];
        s_min[tid].x = fminf(s_min[tid].x, p.x);
        s_min[tid].y = fminf(s_min[tid].y, p.y);
        s_min[tid].z = fminf(s_min[tid].z, p.z);
        s_max[tid].x = fmaxf(s_max[tid].x, p.x);
        s_max[tid].y = fmaxf(s_max[tid].y, p.y);
        s_max[tid].z = fmaxf(s_max[tid].z, p.z);
        i += gridDim.x * blockDim.x;
    }
    __syncthreads();

    // perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_min[tid].x = fminf(s_min[tid].x, s_min[tid + s].x);
            s_min[tid].y = fminf(s_min[tid].y, s_min[tid + s].y);
            s_min[tid].z = fminf(s_min[tid].z, s_min[tid + s].z);
            s_max[tid].x = fmaxf(s_max[tid].x, s_max[tid + s].x);
            s_max[tid].y = fmaxf(s_max[tid].y, s_max[tid + s].y);
            s_max[tid].z = fmaxf(s_max[tid].z, s_max[tid + s].z);
        }
        __syncthreads();
    }

    // thread 0 of each block writes its partial result to the temporary global array.
    // safe because each block writes to a unique location (blockIdx.x).
    if (tid == 0)
    {
        partial_min_bounds[blockIdx.x] = s_min[0];
        partial_max_bounds[blockIdx.x] = s_max[0];
    }
}

// =========================================================================
//  Kernel 1B: Final Reduction Kernel
// =========================================================================
// reads all the partial results and computes the final answer.
__global__ void calculate_bounds_kernel_final(
    const float3* partial_min_bounds, const float3* partial_max_bounds, int num_partials,
    float3* final_min_bound, float3* final_max_bound)
{
    extern __shared__ float3 sdata[];
    float3* s_min = sdata;
    float3* s_max = &sdata[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    s_min[tid] = make_float3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    s_max[tid] = make_float3(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());

    // single block collectively loads all partial results
    while (i < num_partials)
    {
        float3 p_min = partial_min_bounds[i];
        float3 p_max = partial_max_bounds[i];
        s_min[tid].x = fminf(s_min[tid].x, p_min.x);
        s_min[tid].y = fminf(s_min[tid].y, p_min.y);
        s_min[tid].z = fminf(s_min[tid].z, p_min.z);
        s_max[tid].x = fmaxf(s_max[tid].x, p_max.x);
        s_max[tid].y = fmaxf(s_max[tid].y, p_max.y);
        s_max[tid].z = fmaxf(s_max[tid].z, p_max.z);
        i += blockDim.x;
    }
    __syncthreads();

    // final reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) {
            s_min[tid].x = fminf(s_min[tid].x, s_min[tid + s].x);
            s_min[tid].y = fminf(s_min[tid].y, s_min[tid + s].y);
            s_min[tid].z = fminf(s_min[tid].z, s_min[tid + s].z);
            s_max[tid].x = fmaxf(s_max[tid].x, s_max[tid + s].x);
            s_max[tid].y = fmaxf(s_max[tid].y, s_max[tid + s].y);
            s_max[tid].z = fmaxf(s_max[tid].z, s_max[tid + s].z);
        }
        __syncthreads();
    }

    // thread 0 writes the one and only final result.
    if (tid == 0)
    {
        *final_min_bound = s_min[0];
        *final_max_bound = s_max[0];
    }
}

// =========================================================================
//  other Kernels
// =========================================================================
__global__ void calculate_hashes_kernel(const float3* points, unsigned int* hashes, int n, float3 min_bound, float inv_cell_size, grid_dims_t grid_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float3 p = points[idx];
    int gx = static_cast<int>((p.x - min_bound.x) * inv_cell_size);
    int gy = static_cast<int>((p.y - min_bound.y) * inv_cell_size);
    int gz = static_cast<int>((p.z - min_bound.z) * inv_cell_size);
    gx = min(gx, grid_dim.x - 1); gy = min(gy, grid_dim.y - 1); gz = min(gz, grid_dim.z - 1);
    hashes[idx] = gz * grid_dim.y * grid_dim.x + gy * grid_dim.x + gx;
}

__global__ void build_cell_indices_kernel(const unsigned int* sorted_hashes, int2* cell_indices, int n, int num_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int current_hash = sorted_hashes[idx];
    if (idx > 0 && current_hash != sorted_hashes[idx - 1])
    {
        cell_indices[sorted_hashes[idx - 1]].y = idx;
        cell_indices[current_hash].x = idx;
    }
    if (idx == 0) { cell_indices[current_hash].x = 0; }
    if (idx == n - 1) { cell_indices[current_hash].y = n; }
}

__device__ void insert_neighbor(float* dists, int k, float new_dist)
{
    int i = k - 1;
    while (i > 0 && dists[i - 1] > new_dist) { dists[i] = dists[i - 1]; i--; }
    dists[i] = new_dist;
}

__global__ void find_neighbors_kernel(
    const float3* sorted_points, const float3* original_points, float* scales,
    const int2* cell_indices, int n, int k,
    float3 min_bound, float inv_cell_size, grid_dims_t grid_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float3 p_query = original_points[idx];
    float top_k_dists[5];
    for (int i = 0; i < k; ++i) { top_k_dists[i] = std::numeric_limits<float>::max(); }
    int gx = static_cast<int>((p_query.x - min_bound.x) * inv_cell_size);
    int gy = static_cast<int>((p_query.y - min_bound.y) * inv_cell_size);
    int gz = static_cast<int>((p_query.z - min_bound.z) * inv_cell_size);
    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                int ngx = gx + x; int ngy = gy + y; int ngz = gz + z;
                if (ngx >= 0 && ngx < grid_dim.x && ngy >= 0 && ngy < grid_dim.y && ngz >= 0 && ngz < grid_dim.z)
                {
                    unsigned int neighbor_hash = ngz * grid_dim.y * grid_dim.x + ngy * grid_dim.x + ngx;
                    int2 cell_range = cell_indices[neighbor_hash];
                    for (int j = cell_range.x; j < cell_range.y; ++j)
                    {
                        float3 p_candidate = sorted_points[j];
                        float3 diff = make_float3(p_query.x - p_candidate.x, p_query.y - p_candidate.y, p_query.z - p_candidate.z);
                        float dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                        if (dist_sq < top_k_dists[k-1]) { insert_neighbor(top_k_dists, k, dist_sq); }
                    }
                }
            }
        }
    }
    float sum_dists = 0.0f;
    for (int i = 1; i < k; ++i) { sum_dists += sqrtf(top_k_dists[i]); }
    scales[idx] = sum_dists / (k - 1);
}

// =========================================================================
//  orchestrator Function (the two-step reduction)
// =========================================================================
torch::Tensor spatial_grid_knn_scales(const torch::Tensor& points, int k)
{
    TORCH_CHECK(points.is_cuda() && points.is_contiguous() && points.scalar_type() == torch::kFloat32, "Input must be a contiguous CUDA float tensor.");
    TORCH_CHECK(points.size(1) == 3, "Input must be of shape (N, 3).");

    const int n = points.size(0);
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    auto dev = points.device();
    auto options = torch::TensorOptions().device(dev).dtype(torch::kFloat32);

    // --- 1. Calculate Scene Bounding Box (Two-Step Reduction) ---
    // Allocate temporary storage for partial results (one result per block)
    auto partial_min_t = torch::empty({grid_size, 3}, options);
    auto partial_max_t = torch::empty({grid_size, 3}, options);

    size_t shared_mem_size = 2 * block_size * sizeof(float3);
    calculate_bounds_kernel_partial<<<grid_size, block_size, shared_mem_size>>>(
        (const float3*)points.data_ptr<float>(), n,
        (float3*)partial_min_t.data_ptr<float>(),
        (float3*)partial_max_t.data_ptr<float>()
    );
    
    // Allocate final storage
    auto final_min_t = torch::empty({3}, options);
    auto final_max_t = torch::empty({3}, options);

    // Launch a single block to reduce the partial results
    calculate_bounds_kernel_final<<<1, block_size, shared_mem_size>>>(
        (const float3*)partial_min_t.data_ptr<float>(),
        (const float3*)partial_max_t.data_ptr<float>(),
        grid_size,
        (float3*)final_min_t.data_ptr<float>(),
        (float3*)final_max_t.data_ptr<float>()
    );

    // --- 2. Determine Grid Properties ---
    float3 min_b, max_b;
    cudaMemcpy(&min_b, final_min_t.data_ptr<float>(), sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_b, final_max_t.data_ptr<float>(), sizeof(float3), cudaMemcpyDeviceToHost);
    
    // ... PLACEHOLDER - messi

    float3 extent = make_float3(max_b.x - min_b.x, max_b.y - min_b.y, max_b.z - min_b.z);
    float avg_extent = (extent.x + extent.y + extent.z) / 3.0f;
    float cell_size = avg_extent / cbrtf(static_cast<float>(n)) * 2.0f;
    float inv_cell_size = (cell_size > 1e-6) ? 1.0f / cell_size : 0.0f;

    grid_dims_t grid_dim;
    grid_dim.x = std::max(1, static_cast<int>(ceilf(extent.x * inv_cell_size)));
    grid_dim.y = std::max(1, static_cast<int>(ceilf(extent.y * inv_cell_size)));
    grid_dim.z = std::max(1, static_cast<int>(ceilf(extent.z * inv_cell_size)));
    int num_cells = grid_dim.x * grid_dim.y * grid_dim.z;
    
    // --- 3. Compute Hashes and Sort ---
    auto hashes_t = torch::empty({n}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
    calculate_hashes_kernel<<<grid_size, block_size>>>(
        (const float3*)points.data_ptr<float>(), (unsigned int*)hashes_t.data_ptr<int>(), n, min_b, inv_cell_size, grid_dim
    );
    
    auto original_indices_t = torch::arange(0, n, torch::TensorOptions().device(dev).dtype(torch::kInt32));
    
    thrust::sort_by_key(
        thrust::device,
        thrust::device_ptr<int>(hashes_t.data_ptr<int>()),
        thrust::device_ptr<int>(hashes_t.data_ptr<int>()) + n,
        thrust::device_ptr<int>(original_indices_t.data_ptr<int>())
    );

    auto sorted_points_t = points.index_select(0, original_indices_t.to(torch::kInt64));

    // --- 4. Build Cell Start/End Index ---
    auto cell_indices_t = torch::zeros({num_cells, 2}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
    build_cell_indices_kernel<<<grid_size, block_size>>>(
        (const unsigned int*)hashes_t.data_ptr<int>(), (int2*)cell_indices_t.data_ptr<int>(), n, num_cells
    );

    // --- 5. Find Neighbors and Compute Scales ---
    auto scales_t = torch::empty({n}, options);
    find_neighbors_kernel<<<grid_size, block_size>>>(
        (const float3*)sorted_points_t.data_ptr<float>(),
        (const float3*)points.data_ptr<float>(),
        scales_t.data_ptr<float>(),
        (const int2*)cell_indices_t.data_ptr<int>(),
        n, k, min_b, inv_cell_size, grid_dim
    );

    return scales_t.unsqueeze(1);
}