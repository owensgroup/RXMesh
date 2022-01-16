#include "rxmesh/rxmesh_dynamic.h"

namespace rxmesh {

namespace detail {

__global__ static void calc_num_elements(const Context context,
                                         uint32_t*     sum_num_vertices,
                                         uint32_t*     sum_num_edges,
                                         uint32_t*     sum_num_faces)
{
    uint32_t thread_id = threadIdx.x + blockIdx.x * gridDim.x;

    if (thread_id < context.get_num_patches()) {
        ::atomicAdd(
            sum_num_vertices,
            uint32_t(context.get_patches_info()[thread_id].num_owned_vertices));

        ::atomicAdd(
            sum_num_edges,
            uint32_t(context.get_patches_info()[thread_id].num_owned_edges));

        ::atomicAdd(
            sum_num_faces,
            uint32_t(context.get_patches_info()[thread_id].num_owned_faces));
    }
}

}  // namespace detail

bool RXMeshDynamic::validate()
{
    CUDA_ERROR(cudaDeviceSynchronize());

    auto check_num_mesh_elements = [&]() -> bool {
        uint32_t *d_sum_num_vertices, *d_sum_num_edges, *d_sum_num_faces;
        CUDA_ERROR(cudaMalloc((void**)&d_sum_num_vertices, sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc((void**)&d_sum_num_vertices, sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc((void**)&d_sum_num_faces, sizeof(uint32_t)));

        CUDA_ERROR(cudaMemset(d_sum_num_vertices, 0, sizeof(uint32_t)));
        CUDA_ERROR(cudaMemset(d_sum_num_vertices, 0, sizeof(uint32_t)));
        CUDA_ERROR(cudaMemset(d_sum_num_faces, 0, sizeof(uint32_t)));


        uint32_t num_patches;
        CUDA_ERROR(cudaMemcpy(&num_patches,
                              m_rxmesh_context.m_num_patches,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        const uint32_t block_size = 256;
        const uint32_t grid_size  = DIVIDE_UP(num_patches, block_size);

        detail::calc_num_elements<<<grid_size, block_size>>>(m_rxmesh_context,
                                                             d_sum_num_vertices,
                                                             d_sum_num_edges,
                                                             d_sum_num_faces);

        uint32_t num_vertices, num_edges, num_faces;
        CUDA_ERROR(cudaMemcpy(&num_vertices,
                              m_rxmesh_context.m_num_vertices,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(&num_edges,
                              m_rxmesh_context.m_num_edges,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(&num_faces,
                              m_rxmesh_context.m_num_faces,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        uint32_t sum_num_vertices, sum_num_edges, sum_num_faces;

        CUDA_ERROR(cudaMemcpy(&sum_num_vertices,
                              d_sum_num_vertices,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(&sum_num_edges,
                              d_sum_num_edges,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(&sum_num_faces,
                              d_sum_num_faces,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaFree(d_sum_num_vertices));
        CUDA_ERROR(cudaFree(d_sum_num_edges));
        CUDA_ERROR(cudaFree(d_sum_num_faces));

        if (num_vertices != sum_num_vertices || num_edges != sum_num_edges ||
            num_faces != sum_num_faces) {
            return false;
        } else {
            return true;
        }
    };

    if (!check_num_mesh_elements()) {
        return false;
    }

    return true;
}
}  // namespace rxmesh