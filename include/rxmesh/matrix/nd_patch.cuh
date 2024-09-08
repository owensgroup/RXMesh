#pragma once

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/kernels/rxmesh_queries.cuh"

namespace rxmesh {

struct VV
{
    uint16_t *offset, *value;
};

template <uint32_t blockThreads>
struct PatchND
{
    __device__ __inline__ PatchND(cooperative_groups::thread_block& block,
                                  Context&                          context,
                                  ShmemAllocator&                   shrd_alloc)
        : m_patch_info(context.m_patches_info[blockIdx.x])
    {
        m_num_v = m_patch_info.num_vertices[0];
        m_num_e = m_patch_info.num_edges[0];

        m_s_active_v_mis    = Bitmask(m_num_v, shrd_alloc);
        m_s_v_mis           = Bitmask(m_num_v, shrd_alloc);
        m_s_candidate_v_mis = Bitmask(m_num_e, shrd_alloc);


        uint16_t* s_ev = shrd_alloc.alloc<uint16_t>(
            std::max(m_num_v + 1, 2 * m_num_e) + 2 * m_num_e);
        vv_nxt.offset = shrd_alloc.alloc<uint16_t>(2 * m_num_e);
        vv_nxt.value  = shrd_alloc.alloc<uint16_t>(2 * m_num_e);

        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(m_patch_info.ev),
                           2 * m_num_e,
                           s_ev,
                           true);
        block.sync();
        vv_cur.offset = &s_ev[0];
        vv_cur.value  = &s_ev[m_num_v + 1];
        detail::v_v<blockThreads>(block,
                                  m_patch_info,
                                  shrd_alloc,
                                  vv_cur.offset,
                                  vv_cur.value,
                                  false,
                                  false);
    }

    __device__ __inline__ void coarsen(cooperative_groups::thread_block& block)
    {
        // TODO exit condition
        while (true) {
            matching(block);
        }
    }


    /**
     * @brief max matching on the patch mesh edges
     */
    __device__ __inline__ void matching(cooperative_groups::thread_block& block)
    {
        // 1) do maximal independent set for vertices,
        // 2) each vertex in the MIS is an end vertex for the max matching,
        // thus, we use this results to pair a vertex in MIS with one of its
        // neighbor vertices (possibly the one with max weight)
        mis();
    }

    /**
     * @brief compute a maximal independent set of vertices
     */
    __device__ __inline__ void mis(int max_iter = 100)
    {
        // the active vertices for MIS considerations
        __shared__ int s_num_active_v_mis[1];
        if (threadIdx.x == 0) {
            s_num_active_v_mis[0] = 0;
        }

        m_s_active_v_mis.copy(block, m_patch_info.active_mask_v);

        m_s_v_mis.reset(block);

        block.sync();

        // up to max number of iterations or until no longer have active
        // vertices
        for (int iter = 0; iter < max_iter; ++iter) {
            // calc the number of active vertices for MIS consideration
            for (int c = threadIdx.x; c < m_num_v; c += blockThreads) {
                if (m_s_active_v_mis(c)) {
                    ::atomicAdd(s_num_active_v_mis, 1);
                }
            }

            m_s_candidate_v_mis.reset(block);
            block.sync();

            if (s_num_active_v_mis[0] == 0) {
                break;
            }
        }

        // try to find candidate vertices for MIS
        for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
            // if the vertex is one of the active vertices for MIS calculation
            if (m_s_active_v_mis(v)) {
                bool add_v = true;

                // TODO generate VV
                //  for (int i = 0; i < MAX_OVERLAP_CAVITIES; ++i) {
                //      const uint16_t neighbour_c =
                //          m_s_cavity_graph[MAX_OVERLAP_CAVITIES * c + i];
                //      if (neighbour_c != INVALID16) {
                //
                //          if (m_s_active_cavity_mis(neighbour_c) &&
                //              neighbour_c > c) {
                //              add_c = false;
                //              break;
                //          }
                //      }
                //  }
                if (add_v) {
                    m_s_candidate_v_mis.set(v, true);
                }
            }
        }
        block.sync();
    }


   private:
    PatchInfo m_patch_info;
    VV        vv_cur, vv_nxt;
    Bitmask   m_s_active_v_mis;     // the active vertices for MIS
    Bitmask   m_s_candidate_v_mis;  // candidates for MIS
    Bitmask   m_s_v_mis;            // the vertices in the MIS
    uint32_t  m_num_v, m_num_e;
};
}  // namespace rxmesh