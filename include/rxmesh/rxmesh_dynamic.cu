#include "rxmesh/kernels/is_deleted.cuh"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/rxmesh_dynamic.h"

namespace rxmesh {

namespace detail {

__global__ static void calc_num_elements(const Context context,
                                         uint32_t*     sum_num_vertices,
                                         uint32_t*     sum_num_edges,
                                         uint32_t*     sum_num_faces)
{
    uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

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
__global__ static void check_uniqueness(const Context           context,
                                        unsigned long long int* d_check)
{
    const uint32_t patch_id = blockIdx.x;
    if (patch_id < context.get_num_patches()) {

        PatchInfo patch_info = context.get_patches_info()[patch_id];

        extern __shared__ uint16_t shrd_mem[];
        uint16_t*                  s_ev = shrd_mem;
        uint16_t*                  s_fe = shrd_mem;

        // FV since it loads both FE and EV
        load_mesh_async<Op::FV>(patch_info, s_ev, s_fe, true);
        //__syncthreads();

        // make sure an edge is connecting two unique vertices
        for (uint16_t e = threadIdx.x; e < patch_info.num_edges;
             e += blockThreads) {
            uint16_t v0 = s_ev[2 * e + 0];
            uint16_t v1 = s_ev[2 * e + 1];

            if (is_deleted(e, patch_info.mask_e)) {
                if (v0 != INVALID16 || v1 != INVALID16) {
                    // printf(
                    //    "\n edge loop: del edge p= %u, del e= %u, v0= %u, v1=
                    //    "
                    //    "%u",
                    //    patch_id,
                    //    e,
                    //    v0,
                    //    v1);
                    ::atomicAdd(d_check, 1);
                }
            } else {
                if (v0 >= patch_info.num_vertices ||
                    v1 >= patch_info.num_vertices || v0 == v1) {
                    // printf("\n edge loop: vertex check p= %u, e= %u, v0= %u,
                    // v1= %u",
                    //       patch_id,
                    //       e,
                    //       v0,
                    //       v1);
                    ::atomicAdd(d_check, 1);
                }
            }
        }

        // make sure a face is formed by three unique edges and these edges
        // gives three unique vertices
        for (uint16_t f = threadIdx.x; f < patch_info.num_faces;
             f += blockThreads) {

            if (is_deleted(f, patch_info.mask_f)) {
                uint16_t e0, e1, e2;
                e0 = s_fe[3 * f + 0];
                e1 = s_fe[3 * f + 1];
                e2 = s_fe[3 * f + 2];
                if (e0 != INVALID16 || e1 != INVALID16 || e2 != INVALID16) {
                    // printf(
                    //    "\n face loop: del face p= %u, del f= %u, e0= %u, e1=
                    //    "
                    //    "%u, e2= %u",
                    //    patch_id,
                    //    f,
                    //    e0,
                    //    e1,
                    //    e2);
                    ::atomicAdd(d_check, 1);
                }
            } else {
                uint16_t e0, e1, e2;
                flag_t   d0(0), d1(0), d2(0);
                Context::unpack_edge_dir(s_fe[3 * f + 0], e0, d0);
                Context::unpack_edge_dir(s_fe[3 * f + 1], e1, d1);
                Context::unpack_edge_dir(s_fe[3 * f + 2], e2, d2);


                if (e0 >= patch_info.num_edges || e1 >= patch_info.num_edges ||
                    e2 >= patch_info.num_edges || e0 == e1 || e0 == e2 ||
                    e1 == e2) {
                    // printf(
                    //    "\n face loop: edge check p= %u, f= %u, e0= %u, e1= "
                    //    "%u, e2= %u",
                    //    patch_id,
                    //    f,
                    //    e0,
                    //    e1,
                    //    e2);
                    ::atomicAdd(d_check, 1);
                }

                uint16_t v0, v1, v2;
                v0 = s_ev[(2 * e0) + (1 * d0)];
                v1 = s_ev[(2 * e1) + (1 * d1)];
                v2 = s_ev[(2 * e2) + (1 * d2)];

                if (v0 >= patch_info.num_vertices ||
                    v1 >= patch_info.num_vertices ||
                    v2 >= patch_info.num_vertices || v0 == v1 || v0 == v2 ||
                    v1 == v2) {
                    // printf(
                    //    "\n face loop: vertex check p= %u, f= %u, e0= %u, e1="
                    //    "%u, e2= %u, v0= %u, v1= %u, v2= %u",
                    //    patch_id,
                    //    f,
                    //    e0,
                    //    e1,
                    //    e2,
                    //    v0,
                    //    v1,
                    //    v2);
                    ::atomicAdd(d_check, 1);
                }
            }
        }
    }
}


template <uint32_t blockThreads>
__global__ static void check_not_owned(const Context           context,
                                       unsigned long long int* d_check)
{

    const uint32_t patch_id = blockIdx.x;

    if (patch_id < context.get_num_patches()) {

        PatchInfo patch_info = context.get_patches_info()[patch_id];

        extern __shared__ uint16_t shrd_mem[];
        uint16_t*                  s_ev = shrd_mem;
        uint16_t*                  s_fe = shrd_mem;

        // FV since it loads both FE and EV
        load_mesh_async<Op::FV>(patch_info, s_ev, s_fe, true);
        //__syncthreads();

        // for every not-owned face, check that its three edges (possibly
        // not-owned) are the same as those in the face's owner patch
        for (uint16_t f = threadIdx.x + patch_info.num_owned_faces;
             f < patch_info.num_faces;
             f += blockThreads) {

            uint16_t e0, e1, e2;
            flag_t   d0(0), d1(0), d2(0);
            uint32_t p0(patch_id), p1(patch_id), p2(patch_id);
            Context::unpack_edge_dir(s_fe[3 * f + 0], e0, d0);
            Context::unpack_edge_dir(s_fe[3 * f + 1], e1, d1);
            Context::unpack_edge_dir(s_fe[3 * f + 2], e2, d2);

            // if the edge is not owned, grab its local index in the owner patch
            auto get_owned_e =
                [&](uint16_t& e, uint32_t& p, const PatchInfo pi) {
                    if (e >= pi.num_owned_edges) {
                        e -= pi.num_owned_edges;
                        p = pi.not_owned_patch_e[e];
                        e = pi.not_owned_id_e[e].id;
                    }
                };
            get_owned_e(e0, p0, patch_info);
            get_owned_e(e1, p1, patch_info);
            get_owned_e(e2, p2, patch_info);

            // get f's three edges from its owner patch
            uint16_t f_owned           = f - patch_info.num_owned_faces;
            uint32_t f_patch           = patch_info.not_owned_patch_f[f_owned];
            f_owned                    = patch_info.not_owned_id_f[f_owned].id;
            PatchInfo owner_patch_info = context.get_patches_info()[f_patch];

            if (!is_deleted(f_owned, owner_patch_info.mask_f)) {
                // TODO this is a scatter read from global that could be
                // improved by using shared memory
                uint16_t ew0, ew1, ew2;
                flag_t   dw0(0), dw1(0), dw2(0);
                uint32_t pw0(f_patch), pw1(f_patch), pw2(f_patch);
                Context::unpack_edge_dir(
                    owner_patch_info.fe[3 * f_owned + 0].id, ew0, dw0);
                Context::unpack_edge_dir(
                    owner_patch_info.fe[3 * f_owned + 1].id, ew1, dw1);
                Context::unpack_edge_dir(
                    owner_patch_info.fe[3 * f_owned + 2].id, ew2, dw2);

                get_owned_e(ew0, pw0, owner_patch_info);
                get_owned_e(ew1, pw1, owner_patch_info);
                get_owned_e(ew2, pw2, owner_patch_info);

                if (e0 != ew0 || d0 != dw0 || p0 != pw0 || e1 != ew1 ||
                    d1 != dw1 || p1 != pw1 || e2 != ew2 || d2 != dw2 ||
                    p2 != pw2) {
                    ::atomicAdd(d_check, 1);
                }
            }
        }

        // for every not-owned edge, check its two vertices (possibly not-owned)
        // are the same as those in the edge's owner patch
        for (uint16_t e = threadIdx.x + patch_info.num_owned_edges;
             e < patch_info.num_edges;
             e += blockThreads) {

            uint16_t v0 = s_ev[2 * e + 0];
            uint16_t v1 = s_ev[2 * e + 1];
            uint32_t p0(patch_id), p1(patch_id);

            auto get_owned_v =
                [&](uint16_t& v, uint32_t& p, const PatchInfo pi) {
                    if (v >= pi.num_owned_vertices) {
                        v -= pi.num_owned_vertices;
                        p = pi.not_owned_patch_v[v];
                        v = pi.not_owned_id_v[v].id;
                    }
                };
            get_owned_v(v0, p0, patch_info);
            get_owned_v(v1, p1, patch_info);

            // get e's two vertices from its owner patch
            uint16_t e_owned           = e - patch_info.num_owned_edges;
            uint32_t e_patch           = patch_info.not_owned_patch_e[e_owned];
            e_owned                    = patch_info.not_owned_id_e[e_owned].id;
            PatchInfo owner_patch_info = context.get_patches_info()[e_patch];

            if (!is_deleted(e_owned, owner_patch_info.mask_e)) {
                // TODO this is a scatter read from global that could be
                // improved by using shared memory
                uint16_t vw0 = owner_patch_info.ev[2 * e_owned + 0].id;
                uint16_t vw1 = owner_patch_info.ev[2 * e_owned + 1].id;
                uint32_t pw0(e_patch), pw1(e_patch);

                get_owned_v(vw0, pw0, owner_patch_info);
                get_owned_v(vw1, pw1, owner_patch_info);

                if (v0 != vw0 || p0 != pw0 || v1 != vw1 || p1 != pw1) {
                    ::atomicAdd(d_check, 1);
                }
            }
        }
    }
}

}  // namespace detail

bool RXMeshDynamic::validate()
{
    CUDA_ERROR(cudaDeviceSynchronize());

    // check that the sum of owned vertices, edges, and faces per patch is equal
    // to the number of vertices, edges, and faces respectively
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
    // composed of three unique edges that give three unique vertices.
    // if a face or edge is deleted, check that their connectivity is set to
    // INVALID16
    auto check_uniqueness = [&]() -> bool {
        uint32_t num_patches;
        CUDA_ERROR(cudaMemcpy(&num_patches,
                              m_rxmesh_context.m_num_patches,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        unsigned long long int* d_check;
        CUDA_ERROR(
            cudaMalloc((void**)&d_check, sizeof(unsigned long long int)));
        CUDA_ERROR(cudaMemset(d_check, 0, sizeof(unsigned long long int)));

        const uint32_t block_size   = 256;
        const uint32_t grid_size    = num_patches;
        const uint32_t dynamic_smem = (3 * this->m_max_faces_per_patch + 1 +
                                       2 * this->m_max_edges_per_patch) *
                                      sizeof(uint16_t);

        detail::check_uniqueness<256><<<grid_size, block_size, dynamic_smem>>>(
            m_rxmesh_context, d_check);

        unsigned long long int h_check(0);
        CUDA_ERROR(cudaMemcpy(&h_check,
                              d_check,
                              sizeof(unsigned long long int),
                              cudaMemcpyDeviceToHost));

        CUDA_ERROR(cudaFree(d_check));

        if (h_check != 0) {
            return false;
        } else {
            return true;
        }
    };

    // check that every not-owned mesh elements' connectivity (faces and
    // edges) is equivalent to their connectivity in their owner patch.
    // if the mesh element is deleted in the owner patch, no check is done
    auto check_not_owned = [&]() -> bool {
        uint32_t num_patches;
        CUDA_ERROR(cudaMemcpy(&num_patches,
                              m_rxmesh_context.m_num_patches,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        unsigned long long int* d_check;
        CUDA_ERROR(
            cudaMalloc((void**)&d_check, sizeof(unsigned long long int)));
        CUDA_ERROR(cudaMemset(d_check, 0, sizeof(unsigned long long int)));

        const uint32_t block_size   = 256;
        const uint32_t grid_size    = num_patches;
        const uint32_t dynamic_smem = (3 * this->m_max_faces_per_patch + 1 +
                                       2 * this->m_max_edges_per_patch) *
                                      sizeof(uint16_t);

        detail::check_not_owned<256><<<grid_size, block_size, dynamic_smem>>>(
            m_rxmesh_context, d_check);

        unsigned long long int h_check(0);
        CUDA_ERROR(cudaMemcpy(&h_check,
                              d_check,
                              sizeof(unsigned long long int),
                              cudaMemcpyDeviceToHost));

        CUDA_ERROR(cudaFree(d_check));

        if (h_check != 0) {
            return false;
        } else {
            return true;
        }
    };


    if (!check_num_mesh_elements()) {
        RXMESH_WARN("RXMeshDynamic::validate() check_num_mesh_elements failed");
        return false;
    }

    if (!check_uniqueness()) {
        RXMESH_WARN("RXMeshDynamic::validate() check_uniqueness failed");
        return false;
    }

    if (!check_not_owned()) {
        RXMESH_WARN("RXMeshDynamic::validate() check_not_owned failed");
        return false;
    }

    return true;
}
}  // namespace rxmesh