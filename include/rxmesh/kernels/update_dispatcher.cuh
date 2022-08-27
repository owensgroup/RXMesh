#pragma once

#include <cooperative_groups.h>

#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/util/meta.h"


namespace rxmesh {

template <uint32_t blockThreads, CavityOp cop>
struct Cavity
{
    __device__ __inline__ Cavity()
        : m_num_cavities(nullptr),
          m_cavity_id_v(nullptr),
          m_cavity_id_e(nullptr),
          m_cavity_id_f(nullptr),
          m_cavity_edge_loop(nullptr),
          m_s_ev(nullptr),
          m_s_fe(nullptr)
    {
    }

    __device__ __inline__ Cavity(cooperative_groups::thread_block& block,
                                 ShmemAllocator&                   shrd_alloc,
                                 const PatchInfo&                  patch_info)
    {
        m_num_cavities = shrd_alloc.alloc<int>(1);
        m_cavity_id_v  = shrd_alloc.alloc<uint16_t>(patch_info.num_vertices);
        m_cavity_id_e  = shrd_alloc.alloc<uint16_t>(patch_info.num_edges);
        m_cavity_id_f  = shrd_alloc.alloc<uint16_t>(patch_info.num_faces);
        m_cavity_edge_loop = shrd_alloc.alloc<uint16_t>(patch_info.num_edges);

        if (threadIdx.x == 0) {
            m_num_cavities[0] = 0;
        }

        // TODO fix the bank conflict
        for (uint16_t v = threadIdx.x; v < patch_info.num_vertices;
             v += blockThreads) {
            m_cavity_id_v[v] = INVALID16;
        }

        for (uint16_t e = threadIdx.x; e < patch_info.num_edges;
             e += blockThreads) {
            m_cavity_id_e[e] = INVALID16;
        }

        for (uint16_t f = threadIdx.x; f < patch_info.num_faces;
             f += blockThreads) {
            m_cavity_id_f[f] = INVALID16;
        }

        block.sync();
    }

    /**
     * @brief create new cavity
     */
    template <typename HandleT>
    __device__ __inline__ void add(HandleT handle)
    {
        if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                      cop == CavityOp::VE || cop == CavityOp::VF) {
            static_assert(std::is_same_v<HandleT, VertexHandle>,
                          "Cavity::get_handle() since Cavity's template "
                          "parameter operation is Op::V/Op::VV/Op::VE/Op::VF, "
                          "get_handle() should take VertexHandle as an input");
        }

        if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                      cop == CavityOp::EE || cop == CavityOp::EF) {
            static_assert(std::is_same_v<HandleT, EdgeHandle>,
                          "Cavity::get_handle() since Cavity's template "
                          "parameter operation is Op::E/Op::EV/Op::EE/Op::EF, "
                          "get_handle() should take EdgeHandle as an input");
        }

        if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                      cop == CavityOp::FE || cop == CavityOp::FF) {
            static_assert(std::is_same_v<HandleT, FaceHandle>,
                          "Cavity::get_handle() since Cavity's template "
                          "parameter operation is Op::F/Op::FV/Op::FE/Op::FF, "
                          "get_handle() should take FaceHandle as an input");
        }

        int id = ::atomicAdd(m_num_cavities, 1);
        if constexpr (cop == CavityOp::E || cop == CavityOp::EV ||
                      cop == CavityOp::EE || cop == CavityOp::EF) {
            // TODO EV may also mark its vertices immediately
            m_cavity_id_e[handle.unpack().first] = id;
        }

        if constexpr (cop == CavityOp::V || cop == CavityOp::VV ||
                      cop == CavityOp::VE || cop == CavityOp::VF) {
            m_cavity_id_v[handle.unpack().first] = id;
        }

        if constexpr (cop == CavityOp::F || cop == CavityOp::FV ||
                      cop == CavityOp::FE || cop == CavityOp::FF) {
            // TODO FE may also mark its edges immediately
            m_cavity_id_f[handle.unpack().first] = id;
        }
    }

    /**
     * @brief delete elements by applying the cop operation
     * TODO we probably need to clear any shared memory used for queries during
     * adding elements to cavity
     */
    __device__ __inline__ void process(cooperative_groups::thread_block& block,
                                       ShmemAllocator& shrd_alloc,
                                       PatchInfo&      patch_info)
    {

        m_s_cavity_size = shrd_alloc.alloc<int>(m_num_cavities[0] + 1);

        for (uint16_t i = threadIdx.x; i < m_num_cavities[0] + 1;
             i += blockThreads) {
            m_s_cavity_size[i] = 0;
        }

        m_s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges);
        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(patch_info.ev),
                           2 * patch_info.num_edges,
                           m_s_ev,
                           false);
        m_s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces);
        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(patch_info.fe),
                           3 * patch_info.num_faces,
                           m_s_fe,
                           true);
        block.sync();


        if constexpr (cop == CavityOp::V) {
            mark_edges_through_vertices(patch_info);
            block.sync();
            mark_faces_through_edges(patch_info);
            block.sync();
        }

        if constexpr (cop == CavityOp::E) {
            mark_faces_through_edges(patch_info);
            block.sync();
        }
    }


    /**
     * @brief propagate the cavity tag from vertices to their incident edges
     */
    __device__ __inline__ void mark_edges_through_vertices(
        PatchInfo& patch_info)
    {
        for (uint16_t e = threadIdx.x; e < patch_info.num_edges; ++e) {
            if (!detail::is_deleted(e, patch_info.active_mask_e)) {
                // vertices tag
                const uint16_t v0 = m_cavity_id_v[m_s_ev[2 * e]];
                const uint16_t v1 = m_cavity_id_v[m_s_ev[2 * e + 1]];

                if (v0 != INVALID16) {
                    // TODO possible race condition
                    m_cavity_id_e[e] = v0;
                }

                if (v1 != INVALID16) {
                    // TODO possible race condition
                    m_cavity_id_e[e] = v1;
                }
            }
        }
    }


    /**
     * @brief propagate the cavity tag from edges to their incident faces
     */
    __device__ __inline__ void mark_faces_through_edges(PatchInfo& patch_info)
    {
        for (uint16_t f = threadIdx.x; f < patch_info.num_faces; ++f) {
            if (!detail::is_deleted(f, patch_info.active_mask_f)) {

                // edges tag
                const uint16_t e0 = m_cavity_id_v[m_s_fe[3 * f]];
                const uint16_t e1 = m_cavity_id_v[m_s_fe[3 * f + 1]];
                const uint16_t e2 = m_cavity_id_v[m_s_fe[3 * f + 2]];

                if (e0 != INVALID16) {
                    // TODO possible race condition
                    m_cavity_id_f[f] = e0;
                }

                if (e1 != INVALID16) {
                    // TODO possible race condition
                    m_cavity_id_f[f] = e1;
                }

                if (e2 != INVALID16) {
                    // TODO possible race condition
                    m_cavity_id_f[f] = e2;
                }
            }
        }
    }

    /**
     * @brief construct the cavities boundary loop
     */
    template <uint32_t itemPerThread = 5>
    __device__ __inline__ void construct_cavities_edge_loop(
        cooperative_groups::thread_block& block,
        PatchInfo&                        patch_info)
    {


        // Trace faces on the border of the cavity i.e., having an edge on the
        // cavity boundary loop. These faces will add how many of their edges
        // are on the boundary loop. We then do scan and then populate the
        // boundary loop
        uint16_t local_offset[itemPerThread];

        auto index = [&](uint16_t i) {
            // return itemPerThread * threadIdx.x + i;
            return threadIdx.x + blockThreads * i;
        };

        for (uint16_t i = 0; i < itemPerThread; ++i) {
            uint16_t f = index(i);

            local_offset[i] = INVALID16;

            const uint16_t face_cavity = m_cavity_id_f[f];

            // if the face is inside a cavity
            // we could check on if the face is deleted but we only mark faces
            // that are not deleted so no need to double check this
            if (face_cavity != INVALID16) {
                const uint16_t c0 = m_cavity_id_e[m_s_ev[3 * f + 0] >> 1];
                const uint16_t c1 = m_cavity_id_e[m_s_ev[3 * f + 1] >> 1];
                const uint16_t c2 = m_cavity_id_e[m_s_ev[3 * f + 2] >> 1];

                // the edge tag is supposed to be the same as the face tag
                assert(c0 == INVALID16 || c0 == face_cavity);
                assert(c1 == INVALID16 || c1 == face_cavity);
                assert(c2 == INVALID16 || c2 == face_cavity);

                // count how many edges this face contribute to the cavity
                // boundary loop
                int num_edges_on_boundary = 0;
                num_edges_on_boundary += (c0 != INVALID16);
                num_edges_on_boundary += (c1 != INVALID16);
                num_edges_on_boundary += (c2 != INVALID16);

                // it is a face on the boundary only if it has 1 or 2 edges
                // tagged with the (same) cavity id. If it is three edges, then
                // this face is in the interior of the cavity
                if (num_edges_on_boundary == 1 || num_edges_on_boundary == 2) {
                    local_offset[i] = ::atomicAdd(m_s_cavity_size + face_cavity,
                                                  num_edges_on_boundary);
                }
            }
        }


        block.sync();
        // scan
        detail::cub_block_exclusive_sum<int, blockThreads>(m_s_cavity_size,
                                                           m_num_cavities);

        for (uint16_t i = 0; i < itemPerThread; ++i) {
            if (local_offset[i] != INVALID16) {

                uint16_t f = index(i);

                const uint16_t face_cavity = m_cavity_id_f[f];

                int num_added = 0;

                const uint16_t e0 = m_s_ev[3 * f + 0] >> 1;
                const uint16_t e1 = m_s_ev[3 * f + 1] >> 1;
                const uint16_t e2 = m_s_ev[3 * f + 2] >> 1;

                const uint16_t c0 = m_cavity_id_e[e0];
                const uint16_t c1 = m_cavity_id_e[e1];
                const uint16_t c2 = m_cavity_id_e[e2];


                auto check_and_add = [&](const uint16_t c, const uint16_t e) {
                    if (c0 != INVALID16) {
                        uint16_t offset = m_s_cavity_size[face_cavity] +
                                          local_offset[i] + num_added;
                        m_cavity_edge_loop[offset] = e;
                        num_added++;
                    }
                };

                check_and_add(c0, e0);
                check_and_add(c1, e1);
                check_and_add(c2, e2);
            }
        }
    }

        

    int*      m_num_cavities;
    uint16_t *m_cavity_id_v, *m_cavity_id_e, *m_cavity_id_f;
    uint16_t* m_cavity_edge_loop;
    uint16_t *m_s_ev, *m_s_fe;
    int*      m_s_cavity_size;
};

}  // namespace rxmesh