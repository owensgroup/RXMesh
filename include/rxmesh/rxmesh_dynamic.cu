#include "rxmesh/kernels/loader.cuh"
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

template <uint32_t blockThreads>
__global__ static void check_uniqueness(const Context context,
                                        uint32_t*     d_check)
{
    const uint32_t patch_id = blockIdx.x;
    if (patch_id < context.get_num_patches()) {

        PatchInfo patch_info = context.get_patches_info()[patch_id];

        extern __shared__ uint16_t shrd_mem[];
        LocalVertexT* s_ev = reinterpret_cast<LocalVertexT*>(shrd_mem);
        LocalEdgeT*   s_fe = reinterpret_cast<LocalEdgeT*>(shrd_mem);
        load_mesh<blockThreads>(patch_info, true, true, s_ev, s_fe);
        __syncthreads();

        // make sure an edge is connecting two unique vertices
        for (uint16_t e = threadIdx.x; e < patch_info.num_edges;
             e += blockThreads) {
            uint16_t v0 = s_ev[2 * e + 0].id;
            uint16_t v1 = s_ev[2 * e + 1].id;

            if (v0 >= patch_info.num_vertices ||
                v1 >= patch_info.num_vertices || v0 == v1) {
                ::atomicAdd(d_check, uint32_t(1));
            }
        }

        // make sure a face is formed by three unique edges and these edges
        // gives three unique vertices
        for (uint16_t f = threadIdx.x; f < patch_info.num_faces;
             f += blockThreads) {
            uint16_t e0, e1, e2;
            flag_t   d0(0), d1(0), d2(0);
            Context::unpack_edge_dir(s_fe[3 * f + 0].id, e0, d0);
            Context::unpack_edge_dir(s_fe[3 * f + 1].id, e1, d1);
            Context::unpack_edge_dir(s_fe[3 * f + 2].id, e2, d2);

            if (e0 >= patch_info.num_edges || e1 >= patch_info.num_edges ||
                e2 >= patch_info.num_edges || e0 == e1 || e0 == e2 ||
                e1 == e2) {
                ::atomicAdd(d_check, uint32_t(1));
            }

            uint16_t v0, v1, v2;
            v0 = s_ev[(2 * e0) + (1 * d0)].id;
            v1 = s_ev[(2 * e1) + (1 * d1)].id;
            v2 = s_ev[(2 * e2) + (1 * d2)].id;

            if (v0 >= patch_info.num_vertices ||
                v1 >= patch_info.num_vertices ||
                v2 >= patch_info.num_vertices || v0 == v1 || v0 == v2 ||
                v1 == v2) {
                ::atomicAdd(d_check, uint32_t(1));
            }
        }
    }
}

}  // namespace detail

bool RXMeshDynamic::validate()
{
    CUDA_ERROR(cudaDeviceSynchronize());

    // check that the sum of owned vertices, edges, and faces per patch is equal
    // to the number of vertices, edges, and faces respectively.

    auto check_num_mesh_elements = [&]() -> bool {
        uint32_t *d_sum_num_vertices, *d_sum_num_edges, *d_sum_num_faces;
        CUDA_ERROR(cudaMalloc((void**)&d_sum_num_vertices, sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc((void**)&d_sum_num_edges, sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc((void**)&d_sum_num_faces, sizeof(uint32_t)));

        CUDA_ERROR(cudaMemset(d_sum_num_vertices, 0, sizeof(uint32_t)));
        CUDA_ERROR(cudaMemset(d_sum_num_edges, 0, sizeof(uint32_t)));
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

    // check that each edge is composed of two unique vertices and each face is
    // composed of three unique edges
    auto check_uniqueness = [&]() -> bool {
        uint32_t num_patches;
        CUDA_ERROR(cudaMemcpy(&num_patches,
                              m_rxmesh_context.m_num_patches,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        uint32_t* d_check;
        CUDA_ERROR(cudaMalloc((void**)&d_check, sizeof(uint32_t)));
        CUDA_ERROR(cudaMemset(d_check, 0, sizeof(uint32_t)));

        const uint32_t block_size   = 256;
        const uint32_t grid_size    = num_patches;
        const uint32_t dynamic_smem = (3 * this->m_max_faces_per_patch + 1 +
                                       2 * this->m_max_edges_per_patch) *
                                      sizeof(uint16_t);

        detail::check_uniqueness<256><<<grid_size, block_size, dynamic_smem>>>(
            m_rxmesh_context, d_check);

        uint32_t h_check;
        CUDA_ERROR(cudaMemcpy(
            &h_check, d_check, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        CUDA_ERROR(cudaFree(d_check));

        if (h_check != 0) {
            return false;
        } else {
            return true;
        }
    };

    if (!check_num_mesh_elements()) {
        return false;
    }

    if (!check_uniqueness()) {
        return false;
    }

    return true;
}
}  // namespace rxmesh