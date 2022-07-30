#include <cooperative_groups.h>

#include "rxmesh/kernels/dynamic_util.cuh"
#include "rxmesh/kernels/for_each_dispatcher.cuh"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/shmem_allocator.cuh"
#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/util/bitmask_util.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace rxmesh {

namespace detail {

template <uint32_t blockThreads>
__global__ static void calc_num_elements(const Context context,
                                         uint32_t*     sum_num_vertices,
                                         uint32_t*     sum_num_edges,
                                         uint32_t*     sum_num_faces)
{
    auto sum_v = [&](VertexHandle& v_id) { ::atomicAdd(sum_num_vertices, 1u); };
    for_each_dispatcher<Op::V, blockThreads>(context, sum_v);


    auto sum_e = [&](EdgeHandle& e_id) { ::atomicAdd(sum_num_edges, 1u); };
    for_each_dispatcher<Op::E, blockThreads>(context, sum_e);


    auto sum_f = [&](FaceHandle& f_id) { ::atomicAdd(sum_num_faces, 1u); };
    for_each_dispatcher<Op::F, blockThreads>(context, sum_f);
}

template <uint32_t blockThreads>
__global__ static void check_uniqueness(const Context           context,
                                        unsigned long long int* d_check)
{
    auto block = cooperative_groups::this_thread_block();

    const uint32_t patch_id = blockIdx.x;

    if (patch_id < context.get_num_patches()) {

        PatchInfo patch_info = context.get_patches_info()[patch_id];

        ShmemAllocator shrd_alloc;

        uint16_t* s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces);
        uint16_t* s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges);

        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.ev),
                   2 * patch_info.num_edges,
                   s_ev,
                   false);

        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.fe),
                   3 * patch_info.num_faces,
                   s_fe,
                   true);
        block.sync();

        // make sure an edge is connecting two unique vertices
        for (uint16_t e = threadIdx.x; e < patch_info.num_edges;
             e += blockThreads) {
            uint16_t v0 = s_ev[2 * e + 0];
            uint16_t v1 = s_ev[2 * e + 1];

            if (!is_deleted(e, patch_info.active_mask_e)) {

                if (v0 >= patch_info.num_vertices ||
                    v1 >= patch_info.num_vertices || v0 == v1) {
                    ::atomicAdd(d_check, 1);
                }
                if (is_deleted(v0, patch_info.active_mask_v) ||
                    is_deleted(v1, patch_info.active_mask_v)) {
                    ::atomicAdd(d_check, 1);
                }
            }
        }

        // make sure a face is formed by three unique edges and these edges
        // gives three unique vertices
        for (uint16_t f = threadIdx.x; f < patch_info.num_faces;
             f += blockThreads) {

            if (!is_deleted(f, patch_info.active_mask_f)) {
                uint16_t e0, e1, e2;
                flag_t   d0(0), d1(0), d2(0);
                Context::unpack_edge_dir(s_fe[3 * f + 0], e0, d0);
                Context::unpack_edge_dir(s_fe[3 * f + 1], e1, d1);
                Context::unpack_edge_dir(s_fe[3 * f + 2], e2, d2);

                if (e0 >= patch_info.num_edges || e1 >= patch_info.num_edges ||
                    e2 >= patch_info.num_edges || e0 == e1 || e0 == e2 ||
                    e1 == e2) {
                    ::atomicAdd(d_check, 1);
                }

                if (is_deleted(e0, patch_info.active_mask_e) ||
                    is_deleted(e1, patch_info.active_mask_e) ||
                    is_deleted(e2, patch_info.active_mask_e)) {
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
                    ::atomicAdd(d_check, 1);
                }

                if (is_deleted(v0, patch_info.active_mask_v) ||
                    is_deleted(v1, patch_info.active_mask_v) ||
                    is_deleted(v2, patch_info.active_mask_v)) {
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
    auto block = cooperative_groups::this_thread_block();

    const uint32_t patch_id = blockIdx.x;

    if (patch_id < context.get_num_patches()) {

        PatchInfo patch_info = context.get_patches_info()[patch_id];

        ShmemAllocator shrd_alloc;
        uint16_t* s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces);
        uint16_t* s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges);
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.ev),
                   2 * patch_info.num_edges,
                   s_ev,
                   false);

        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.fe),
                   3 * patch_info.num_faces,
                   s_fe,
                   true);
        block.sync();


        // for every not-owned face, check that its three edges (possibly
        // not-owned) are the same as those in the face's owner patch
        for (uint16_t f = threadIdx.x; f < patch_info.num_faces;
             f += blockThreads) {

            if (!is_deleted(f, patch_info.active_mask_f) &&
                !is_owned(f, patch_info.owned_mask_f)) {

                uint16_t e0, e1, e2;
                flag_t   d0(0), d1(0), d2(0);
                uint32_t p0(patch_id), p1(patch_id), p2(patch_id);
                Context::unpack_edge_dir(s_fe[3 * f + 0], e0, d0);
                Context::unpack_edge_dir(s_fe[3 * f + 1], e1, d1);
                Context::unpack_edge_dir(s_fe[3 * f + 2], e2, d2);

                // if the edge is not owned, grab its local index in the owner
                // patch
                auto get_owned_e =
                    [&](uint16_t& e, uint32_t& p, const PatchInfo pi) {
                        if (!is_owned(e, pi.owned_mask_e)) {
                            auto e_pair = pi.lp_e.find(e);
                            e           = e_pair.local_id_in_owner_patch();
                            p           = pi.patch_stash.get_patch(e_pair);
                        }
                    };
                get_owned_e(e0, p0, patch_info);
                get_owned_e(e1, p1, patch_info);
                get_owned_e(e2, p2, patch_info);

                // get f's three edges from its owner patch
                auto      f_pair  = patch_info.lp_f.find(f);
                uint16_t  f_owned = f_pair.local_id_in_owner_patch();
                uint32_t  f_patch = patch_info.patch_stash.get_patch(f_pair);
                PatchInfo owner_patch_info =
                    context.get_patches_info()[f_patch];

                // the owner patch should have indicate that the owned face is
                // owned by it
                if (!is_owned(f_owned, owner_patch_info.owned_mask_f)) {
                    ::atomicAdd(d_check, 1);
                }

                // If a face is deleted, it should also be deleted in the other
                // patches that have it as not-owned
                bool is_del =
                    is_deleted(f_owned, owner_patch_info.active_mask_f);
                if (is_del) {
                    ::atomicAdd(d_check, 1);
                } else {
                    // TODO this is a scattered read from global that could be
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
        }

        // for every not-owned edge, check its two vertices (possibly
        // not-owned) are the same as those in the edge's owner patch
        for (uint16_t e = threadIdx.x; e < patch_info.num_edges;
             e += blockThreads) {

            if (!is_deleted(e, patch_info.active_mask_e) &&
                !is_owned(e, patch_info.owned_mask_e)) {

                uint16_t v0 = s_ev[2 * e + 0];
                uint16_t v1 = s_ev[2 * e + 1];
                uint32_t p0(patch_id), p1(patch_id);

                auto get_owned_v =
                    [&](uint16_t& v, uint32_t& p, const PatchInfo pi) {
                        if (!is_owned(v, pi.owned_mask_v)) {
                            auto v_pair = pi.lp_v.find(v);
                            v           = v_pair.local_id_in_owner_patch();
                            p           = pi.patch_stash.get_patch(v_pair);
                        }
                    };
                get_owned_v(v0, p0, patch_info);
                get_owned_v(v1, p1, patch_info);

                // get e's two vertices from its owner patch
                auto      e_pair  = patch_info.lp_e.find(e);
                uint16_t  e_owned = e_pair.local_id_in_owner_patch();
                uint32_t  e_patch = patch_info.patch_stash.get_patch(e_pair);
                PatchInfo owner_patch_info =
                    context.get_patches_info()[e_patch];

                // the owner patch should have indicate that the owned face is
                // owned by it
                if (!is_owned(e_owned, owner_patch_info.owned_mask_e)) {
                    ::atomicAdd(d_check, 1);
                }

                // If an edge is deleted, it should also be deleted in the other
                // patches that have it as not-owned
                bool is_del =
                    is_deleted(e_owned, owner_patch_info.active_mask_e);
                if (is_del) {
                    ::atomicAdd(d_check, 1);
                } else {
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
}


template <uint32_t blockThreads>
__global__ static void check_ribbon(const Context           context,
                                    unsigned long long int* d_check)
{
    auto block = cooperative_groups::this_thread_block();

    const uint32_t patch_id = blockIdx.x;

    if (patch_id < context.get_num_patches()) {
        PatchInfo patch_info = context.get_patches_info()[patch_id];

        ShmemAllocator shrd_alloc;
        uint16_t* s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces);
        load_async(block,
                   reinterpret_cast<uint16_t*>(patch_info.fe),
                   3 * patch_info.num_faces,
                   s_fe,
                   true);
        uint16_t* s_mark_edges =
            shrd_alloc.alloc<uint16_t>(patch_info.num_edges);

        for (uint16_t e = threadIdx.x; e < patch_info.num_edges;
             e += blockThreads) {
            s_mark_edges[e] = 0;
        }

        block.sync();

        // Check that each owned edge is incident to at least one owned
        // not-deleted face. We do that by iterating over faces, each face
        // (atomically) mark its incident edges only if they are owned. Then we
        // check the marked edges where we expect all owned edges to be marked.
        // If there is an edge that is owned but not marked, then this edge is
        // not incident to any owned faces
        for (uint16_t f = threadIdx.x; f < patch_info.num_faces;
             f += blockThreads) {

            if (!is_deleted(f, patch_info.active_mask_f) &&
                is_owned(f, patch_info.owned_mask_f)) {

                uint16_t e0 = s_fe[3 * f + 0] >> 1;
                uint16_t e1 = s_fe[3 * f + 1] >> 1;
                uint16_t e2 = s_fe[3 * f + 2] >> 1;

                auto mark_if_owned = [&](uint16_t edge) {
                    if (is_owned(edge, patch_info.owned_mask_e)) {
                        ::rxmesh::atomicAdd(s_mark_edges + edge, uint16_t(1));
                    }
                };

                mark_if_owned(e0);
                mark_if_owned(e1);
                mark_if_owned(e2);
            }
        }
        block.sync();
        for (uint16_t e = threadIdx.x; e < patch_info.num_edges;
             e += blockThreads) {
            if (is_owned(e, patch_info.owned_mask_e)) {
                if (s_mark_edges[e] == 0) {
                    ::atomicAdd(d_check, 1);
                }
            }
        }
        block.sync();

        shrd_alloc.dealloc<uint16_t>(patch_info.num_edges);
    }
}
}  // namespace detail

bool RXMeshDynamic::validate()
{
    CUDA_ERROR(cudaDeviceSynchronize());

    uint32_t num_patches;
    CUDA_ERROR(cudaMemcpy(&num_patches,
                          m_rxmesh_context.m_num_patches,
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    unsigned long long int* d_check;
    CUDA_ERROR(cudaMalloc((void**)&d_check, sizeof(unsigned long long int)));

    auto is_okay = [&]() {
        unsigned long long int h_check(0);
        CUDA_ERROR(cudaMemcpy(&h_check,
                              d_check,
                              sizeof(unsigned long long int),
                              cudaMemcpyDeviceToHost));
        if (h_check != 0) {
            return false;
        } else {
            return true;
        }
    };

    // check that the sum of owned vertices, edges, and faces per patch is equal
    // to the number of vertices, edges, and faces respectively
    auto check_num_mesh_elements = [&]() -> bool {
        uint32_t *d_sum_num_vertices, *d_sum_num_edges, *d_sum_num_faces;
        thrust::device_vector<uint32_t> d_sum_vertices(1, 0);
        thrust::device_vector<uint32_t> d_sum_edges(1, 0);
        thrust::device_vector<uint32_t> d_sum_faces(1, 0);

        constexpr uint32_t block_size = 256;
        const uint32_t     grid_size  = num_patches;

        detail::calc_num_elements<block_size>
            <<<grid_size, block_size>>>(m_rxmesh_context,
                                        d_sum_vertices.data().get(),
                                        d_sum_edges.data().get(),
                                        d_sum_faces.data().get());

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
        thrust::copy(
            d_sum_vertices.begin(), d_sum_vertices.end(), &sum_num_vertices);
        thrust::copy(d_sum_edges.begin(), d_sum_edges.end(), &sum_num_edges);
        thrust::copy(d_sum_faces.begin(), d_sum_faces.end(), &sum_num_faces);

        if (num_vertices != sum_num_vertices || num_edges != sum_num_edges ||
            num_faces != sum_num_faces) {
            return false;
        } else {
            return true;
        }
    };

    // check that each edge is composed of two unique vertices and each face is
    // composed of three unique edges that give three unique vertices.
    auto check_uniqueness = [&]() -> bool {
        CUDA_ERROR(cudaMemset(d_check, 0, sizeof(unsigned long long int)));
        constexpr uint32_t block_size = 256;
        const uint32_t     grid_size  = num_patches;
        const uint32_t     dynamic_smem =
            rxmesh::detail::ShmemAllocator::default_alignment * 2 +
            (3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
            (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

        detail::check_uniqueness<block_size>
            <<<grid_size, block_size, dynamic_smem>>>(m_rxmesh_context,
                                                      d_check);

        return is_okay();
    };

    // check that every not-owned mesh elements' connectivity (faces and
    // edges) is equivalent to their connectivity in their owner patch.
    // if the mesh element is deleted in the owner patch, no check is done
    auto check_not_owned = [&]() -> bool {
        CUDA_ERROR(cudaMemset(d_check, 0, sizeof(unsigned long long int)));
        CUDA_ERROR(cudaMemset(d_check, 0, sizeof(unsigned long long int)));

        constexpr uint32_t block_size = 256;
        const uint32_t     grid_size  = num_patches;
        const uint32_t     dynamic_smem =
            rxmesh::detail::ShmemAllocator::default_alignment * 2 +
            (3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
            (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

        detail::check_not_owned<block_size>
            <<<grid_size, block_size, dynamic_smem>>>(m_rxmesh_context,
                                                      d_check);
        return is_okay();
    };

    // check if the ribbon construction is complete i.e., 1) each owned edge is
    // incident to an owned face, and 2) VF of the three vertices of an owned
    // face is inside the patch
    auto check_ribbon = [&]() {
        CUDA_ERROR(cudaMemset(d_check, 0, sizeof(unsigned long long int)));
        constexpr uint32_t block_size = 256;
        const uint32_t     grid_size  = num_patches;
        const uint32_t     dynamic_smem =
            rxmesh::detail::ShmemAllocator::default_alignment * 2 +
            (3 * this->m_max_faces_per_patch) * sizeof(uint16_t) +
            (2 * this->m_max_edges_per_patch) * sizeof(uint16_t);

        detail::check_ribbon<block_size>
            <<<grid_size, block_size, dynamic_smem>>>(m_rxmesh_context,
                                                      d_check);

        return is_okay();
    };

    bool success = true;
    if (!check_num_mesh_elements()) {
        RXMESH_WARN("RXMeshDynamic::validate() check_num_mesh_elements failed");
        success = false;
    }

    if (!check_uniqueness()) {
        RXMESH_WARN("RXMeshDynamic::validate() check_uniqueness failed");
        success = false;
    }

    if (!check_not_owned()) {
        RXMESH_WARN("RXMeshDynamic::validate() check_not_owned failed");
        success = false;
    }

    if (!check_ribbon()) {
        RXMESH_WARN("RXMeshDynamic::validate() check_ribbon failed");
        success = false;
    }

    CUDA_ERROR(cudaFree(d_check));

    return success;
}


void RXMeshDynamic::update_host()
{
    auto resize_not_owned = [&](const uint16_t in_num_elements,
                                const uint16_t in_num_owned_elements,
                                uint16_t&      out_num_elements,
                                uint16_t&      out_num_owned_elements,
                                uint32_t*&     not_owned_patch,
                                auto*&         not_owned_id) {
        const uint16_t num_not_owned = in_num_elements - in_num_owned_elements;

        using T = std::remove_pointer_t<
            std::remove_reference_t<decltype(not_owned_id)>>;

        if (num_not_owned > (out_num_elements - out_num_owned_elements)) {
            free(not_owned_id);
            free(not_owned_patch);
            not_owned_id = (T*)malloc(num_not_owned * sizeof(T));
            not_owned_patch =
                (uint32_t*)malloc(num_not_owned * sizeof(uint32_t));
        }
        out_num_owned_elements = in_num_owned_elements;
        out_num_elements       = in_num_elements;
    };

    auto resize_active_mask =
        [&](uint16_t size, uint16_t& capacity, uint32_t*& mask) {
            if (size > capacity) {
                capacity = size;
                free(mask);
                mask = (uint32_t*)malloc(detail::mask_num_bytes(size));
            }
        };

    for (uint32_t p = 0; p < m_num_patches; ++p) {
        PatchInfo d_patch;
        CUDA_ERROR(cudaMemcpy(&d_patch,
                              m_d_patches_info + p,
                              sizeof(PatchInfo),
                              cudaMemcpyDeviceToHost));

        assert(d_patch.patch_id == p);

        // resize topology
        if (d_patch.num_edges > m_h_patches_info[p].edges_capacity) {
            free(m_h_patches_info[p].ev);
            m_h_patches_info[p].ev = (LocalVertexT*)malloc(
                d_patch.num_edges * 2 * sizeof(LocalVertexT));
        }

        if (d_patch.num_faces > m_h_patches_info[p].faces_capacity) {
            free(m_h_patches_info[p].fe);
            m_h_patches_info[p].fe =
                (LocalEdgeT*)malloc(d_patch.num_faces * 3 * sizeof(LocalEdgeT));
        }

        // resize not-owned patch and local id (update num_owned_X, num_X)
        // TODO
        /*resize_not_owned(d_patch.num_vertices,
                         d_patch.num_owned_vertices,
                         m_h_patches_info[p].num_vertices,
                         m_h_patches_info[p].num_owned_vertices,
                         m_h_patches_info[p].not_owned_patch_v,
                         m_h_patches_info[p].not_owned_id_v);
        resize_not_owned(d_patch.num_edges,
                         d_patch.num_owned_edges,
                         m_h_patches_info[p].num_edges,
                         m_h_patches_info[p].num_owned_edges,
                         m_h_patches_info[p].not_owned_patch_e,
                         m_h_patches_info[p].not_owned_id_e);
        resize_not_owned(d_patch.num_faces,
                         d_patch.num_owned_faces,
                         m_h_patches_info[p].num_faces,
                         m_h_patches_info[p].num_owned_faces,
                         m_h_patches_info[p].not_owned_patch_f,
                         m_h_patches_info[p].not_owned_id_f);

        // resize mask (update capacity)
        resize_active_mask(d_patch.num_vertices,
                           m_h_patches_info[p].vertices_capacity,
                           m_h_patches_info[p].active_mask_v);
        resize_active_mask(d_patch.num_edges,
                           m_h_patches_info[p].edges_capacity,
                           m_h_patches_info[p].active_mask_e);
        resize_active_mask(d_patch.num_faces,
                           m_h_patches_info[p].faces_capacity,
                           m_h_patches_info[p].active_mask_f);

        // copy topology
        CUDA_ERROR(cudaMemcpy(m_h_patches_info[p].ev,
                              d_patch.ev,
                              2 * d_patch.num_edges * sizeof(LocalVertexT),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(m_h_patches_info[p].fe,
                              d_patch.fe,
                              3 * d_patch.num_faces * sizeof(LocalEdgeT),
                              cudaMemcpyDeviceToHost));
        // not owned patch
        CUDA_ERROR(
            cudaMemcpy(m_h_patches_info[p].not_owned_patch_v,
                       d_patch.not_owned_patch_v,
                       (d_patch.num_vertices - d_patch.num_owned_vertices) *
                           sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(
            m_h_patches_info[p].not_owned_patch_e,
            d_patch.not_owned_patch_e,
            (d_patch.num_edges - d_patch.num_owned_edges) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(
            m_h_patches_info[p].not_owned_patch_f,
            d_patch.not_owned_patch_f,
            (d_patch.num_faces - d_patch.num_owned_faces) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));

        // not owned local id
        CUDA_ERROR(
            cudaMemcpy(m_h_patches_info[p].not_owned_id_v,
                       d_patch.not_owned_id_v,
                       (d_patch.num_vertices - d_patch.num_owned_vertices) *
                           sizeof(LocalVertexT),
                       cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(
            m_h_patches_info[p].not_owned_id_e,
            d_patch.not_owned_id_e,
            (d_patch.num_edges - d_patch.num_owned_edges) * sizeof(LocalEdgeT),
            cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(
            m_h_patches_info[p].not_owned_id_f,
            d_patch.not_owned_id_f,
            (d_patch.num_faces - d_patch.num_owned_faces) * sizeof(LocalFaceT),
            cudaMemcpyDeviceToHost));

        // mask
        CUDA_ERROR(cudaMemcpy(m_h_patches_info[p].active_mask_v,
                              d_patch.active_mask_v,
                              detail::mask_num_bytes(d_patch.num_vertices),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(m_h_patches_info[p].active_mask_e,
                              d_patch.active_mask_e,
                              detail::mask_num_bytes(d_patch.num_edges),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(m_h_patches_info[p].active_mask_f,
                              d_patch.active_mask_f,
                              detail::mask_num_bytes(d_patch.num_faces),
                              cudaMemcpyDeviceToHost));*/
    }
}
}  // namespace rxmesh