#pragma once

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/kernels/rxmesh_queries.cuh"
#include "rxmesh/kernels/util.cuh"

#include "rxmesh/matrix/mgnd_permute.cuh"

namespace rxmesh {

struct VV
{
    uint16_t *offset, *value;
};

template <uint32_t blockThreads, int maxCoarsenLevels>
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
        m_s_candidate_v_mis = Bitmask(m_num_v, shrd_alloc);
        m_s_active_v        = Bitmask(m_num_v, shrd_alloc);
        m_s_cur_active_v    = Bitmask(m_num_v, shrd_alloc);

        m_s_active_v_mis.reset(block);
        m_s_v_mis.reset(block);
        m_s_candidate_v_mis.reset(block);
        m_s_active_v.reset(block);
        m_s_cur_active_v.reset(block);

        m_v_matching = shrd_alloc.alloc<uint16_t>(maxCoarsenLevels * m_num_v);

        detail::load_async(
            block,
            reinterpret_cast<const char*>(m_patch_info.active_mask_v),
            m_s_active_v.num_bytes(),
            reinterpret_cast<char*>(m_s_active_v.m_bitmask),
            false);


        uint16_t* s_ev = shrd_alloc.alloc<uint16_t>(
            std::max(m_num_v + 1, 2 * m_num_e) + 2 * m_num_e);
        vv_nxt.offset = shrd_alloc.alloc<uint16_t>(2 * m_num_e);
        vv_nxt.value  = shrd_alloc.alloc<uint16_t>(2 * m_num_e);

        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(m_patch_info.ev),
                           2 * m_num_e,
                           s_ev,
                           true);

        // mark not-owned vertices as not-active
        uint16_t mask_num_elements = DIVIDE_UP(m_s_active_v.size(), 32);
        for (uint16_t i = threadIdx.x; i < mask_num_elements;
             i += blockThreads) {
            m_s_active_v.m_bitmask[i] =
                m_s_active_v.m_bitmask[i] & m_patch_info.owned_mask_v[i];
        }

        // create vv
        vv_cur.offset = &s_ev[0];
        vv_cur.value  = &s_ev[m_num_v + 1];
        detail::v_v<blockThreads>(block,
                                  m_patch_info,
                                  shrd_alloc,
                                  vv_cur.offset,
                                  vv_cur.value,
                                  false,
                                  false);
        block.sync();

        // mask out vertices on the separator (permuted via cross-patch
        // permutation)
        for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
            bool on_sep = false;
            if (m_s_active_v(v)) {
                uint16_t start = vv_cur.offset[v];
                uint16_t stop  = vv_cur.offset[v + 1];
                for (uint16_t i = start; i < stop; ++i) {
                    uint16_t n = vv_cur.value[i];

                    VertexHandle nh = m_patch_info.find<VertexHandle>(n);

                    if (nh.patch_id() > m_patch_info.patch_id) {
                        on_sep = true;
                        break;
                    }
                }
            }
            if (on_sep) {
                m_s_active_v.reset(v, true);
            }
        }

        block.sync();

        // copy active mask
        m_s_cur_active_v.copy(block, m_s_active_v);
        block.sync();
    }

    /**
     * @brief coarsen the graph after edge matching
     */
    __device__ __inline__ void coarsen(cooperative_groups::thread_block& block,
                                       int                               level)
    {
    }


    /**
     * @brief max matching on the patch mesh edges. Level 0 refers to the mesh
     * edges and higher levels are the coarsened graphs
     */
    __device__ __inline__ void edge_matching(
        cooperative_groups::thread_block& block,
        int                               level,
        VertexAttribute<int>&             attr_v,
        EdgeAttribute<int>&               attr_e)
    {

#if 1
        // 1) do maximal independent set for vertices,
        // 2) each vertex in the MIS is an end vertex for the max matching,
        // thus, we use this results to pair a vertex in MIS with one of its
        // neighbor vertices (possibly the one with max weight)
        maximal_independent_set(block, m_s_cur_active_v, vv_cur);

        // no need to sync here
        //  recycling m_s_candidate_v_mis
        extract_random_edge_matching(block,
                                     vv_cur,
                                     m_s_cur_active_v,
                                     m_s_v_mis,
                                     m_s_candidate_v_mis,
                                     get_matching_arr(level));
#else
        single_thread_edge_matching(block,
                                    vv_cur,
                                    m_s_cur_active_v,
                                    m_s_candidate_v_mis,
                                    get_matching_arr(level));
#endif

        block.sync();


        {
            for (uint16_t v = threadIdx.x; v < m_num_v; v += blockThreads) {
                attr_v(VertexHandle(m_patch_info.patch_id, v)) =
                    (get_matching_arr(level)[v] == INVALID16) ?
                        -1 :
                        get_matching_arr(level)[v];
            }

            for (uint16_t e = threadIdx.x; e < m_num_e; e += blockThreads) {
                uint16_t v0 = m_patch_info.ev[2 * e + 0].id;
                uint16_t v1 = m_patch_info.ev[2 * e + 1].id;
                if (get_matching_arr(level)[v0] ==
                        get_matching_arr(level)[v1] &&
                    get_matching_arr(level)[v0] != INVALID16) {
                    attr_e(EdgeHandle(m_patch_info.patch_id, e)) = 1;
                } else {
                    attr_e(EdgeHandle(m_patch_info.patch_id, e)) = -1;
                }
            }
        }

        // construct the next level graph
    }

    /**
     * @brief quick prototype of edge matching using a single thread in the
     * block
     * @return
     */
    __device__ __inline__ void single_thread_edge_matching(
        cooperative_groups::thread_block& block,
        const VV&                         vv,
        const Bitmask&                    active_v,
        Bitmask&                          matched_v,
        uint16_t*                         v_matching,
        int                               max_iter = 100)
    {
        matched_v.reset(block);
        fill_n<blockThreads>(v_matching, m_num_v, uint16_t(INVALID16));
        block.sync();

        if (threadIdx.x == 0) {
            for (uint16_t v = 0; v < m_num_v; ++v) {
                if (active_v(v) && !matched_v(v)) {

                    // TODO apply the vertex weight
                    uint16_t matched_neighbour = INVALID16;

                    uint16_t start = vv.offset[v];
                    uint16_t stop  = vv.offset[v + 1];

                    for (uint16_t i = start; i < stop; ++i) {
                        uint16_t n = vv.value[i];
                        if (!matched_v(n)) {
                            matched_neighbour = n;
                            break;
                        }
                    }


                    if (matched_neighbour != INVALID16) {
                        matched_v.set(v, true);
                        matched_v.set(matched_neighbour, true);

                        v_matching[v] = std::min(matched_neighbour, v);
                        v_matching[matched_neighbour] =
                            std::min(matched_neighbour, v);
                    }
                }
            }
        }
    }

    /**
     * @brief In every iteration, an edge will try_set (atomically) both of its
     * vertices. If successful, then both vertices will be marked as matched.
     * Then, every edge will mark itself as in-active if one of its two end
     * vertices are matched.
     * active_v and active_e should be populated with the current set of active
     * vertices and edges. They will be updated as they are used inside this
     * function
     */
    __device__ __inline__ void iterative_edge_matching(
        cooperative_groups::thread_block& block,
        const uint16_t*                   ev,
        uint16_t*                         v_matching,
        Bitmask&                          active_v,
        Bitmask&                          candidate_v,
        Bitmask&                          active_e,
        int                               max_iter = 100)
    {
        __shared__ int s_added[1];

        fill_n<blockThreads>(v_matching, m_num_v, uint16_t(INVALID16));
        for (int iter = 0; iter < max_iter; ++iter) {
            if (threadIdx.x == 0) {
                s_added[0] = 0;
            }
            candidate_v.reset(block);
            block.sync();

            for (uint16_t e = threadIdx.x; e < m_num_e; e += blockThreads) {
                // to reduce contention
                if ((e % 2 == iter % 2) && active_e(e)) {
                    const uint16_t v0 = ev[2 * e + 0];
                    const uint16_t v1 = ev[2 * e + 1];
                    assert(v0 < m_num_v);
                    assert(v1 < m_num_v);
                    if (candidate_v.try_set(v0) && candidate_v.try_set(v1)) {
                        assert(v_matching[v0] == INVALID16);
                        assert(v_matching[v1] == INVALID16);
                        assert(active_v(v0));
                        assert(active_v(v1));

                        v_matching[v0] = v1;
                        v_matching[v1] = v0;

                        active_v.reset(v0, true);
                        active_v.reset(v1, true);
                        s_added[0] = 1;
                    }
                }
            }
            block.sync();

            if (s_added[0] == 0) {
                break;
            }

            for (uint16_t e = threadIdx.x; e < m_num_e; e += blockThreads) {
                if (active_e(e)) {
                    const uint16_t v0 = ev[2 * e + 0];
                    const uint16_t v1 = ev[2 * e + 1];
                    if (!active_v(v0) && !active_v(v1)) {
                        active_e.reset(e, true);
                    }
                }
            }
        }
    }

    /**
     * @brief compute a maximal independent set of vertices
     */
    __device__ __inline__ void maximal_independent_set(
        cooperative_groups::thread_block& block,
        Bitmask&                          active_v,
        const VV&                         vv,
        int                               max_iter = 100)
    {
        // the active vertices for MIS considerations
        __shared__ int s_num_active_v_mis[1];
        if (threadIdx.x == 0) {
            s_num_active_v_mis[0] = 0;
        }

        m_s_active_v_mis.copy(block, active_v);

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
            // reset the candidate
            m_s_candidate_v_mis.reset(block);
            block.sync();

            if (s_num_active_v_mis[0] == 0) {
                break;
            }

            // try to find candidate vertices for MIS
            for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
                // if the vertex is one of the active vertices for MIS
                // calculation
                if (m_s_active_v_mis(v)) {
                    bool     add_v = true;
                    uint16_t start = vv.offset[v];
                    uint16_t stop  = vv.offset[v + 1];

                    for (int i = start; i < stop; ++i) {
                        const uint16_t n = vv.value[i];

                        if (active_v(n) && m_s_active_v_mis(n) && n > v) {
                            add_v = false;
                            break;
                        }
                    }
                    if (add_v) {
                        m_s_candidate_v_mis.set(v, true);
                    }
                }
            }
            block.sync();


            // add the candidate to the MIS and remove them from the active set
            for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
                if (m_s_candidate_v_mis(v)) {
                    m_s_active_v_mis.reset(v, true);
                    m_s_v_mis.set(v, true);

                    // remove the neighbor from the active set
                    uint16_t start = vv.offset[v];
                    uint16_t stop  = vv.offset[v + 1];
                    for (int i = start; i < stop; ++i) {
                        const uint16_t n = vv.value[i];
                        m_s_active_v_mis.reset(n, true);
                    }
                }
            }

            if (threadIdx.x == 0) {
                s_num_active_v_mis[0] = 0;
            }
            block.sync();
        }
    }

    /**
     * @brief given a maximal independent set of vertices, extract an edge
     * matching from it.
     */
    __device__ __inline__ void extract_random_edge_matching(
        cooperative_groups::thread_block& block,
        const VV&                         vv,
        const Bitmask&                    active_v,
        const Bitmask&                    mis_v,
        Bitmask&                          helper_v,
        uint16_t*                         v_matching)
    {

        fill_n<blockThreads>(v_matching, m_num_v, uint16_t(INVALID16));
        helper_v.reset(block);
        block.sync();

        for (uint16_t v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (mis_v(v)) {
                uint16_t start = vv.offset[v];
                uint16_t stop  = vv.offset[v + 1];
                for (uint16_t i = start; i < stop; ++i) {
                    uint16_t n = vv.value[i];

                    if (helper_v.try_set(n)) {
                        v_matching[v] = std::min(v, n);
                        v_matching[n] = std::min(v, n);
                        break;
                    }
                }
            }
        }
    }

    /**
     * @brief for a given level in the edge max match tree, return the pointer
     * to the array that stores the vertex matching
     */
    __device__ __inline__ uint16_t* get_matching_arr(int level)
    {
        return &m_v_matching[m_num_v * level];
    }

   private:
    PatchInfo m_patch_info;
    VV        vv_cur, vv_nxt;
    Bitmask   m_s_active_v_mis;     // the active vertices for MIS
    Bitmask   m_s_candidate_v_mis;  // candidates for MIS
    Bitmask   m_s_v_mis;            // the vertices in the MIS
    Bitmask   m_s_active_v;  // active vertices in the patch (minus not-owned)
    Bitmask   m_s_cur_active_v;  // current active vertices during coarsening
    uint32_t  m_num_v, m_num_e;
    uint16_t* m_v_matching;  // store the vertex a given vertex is matched with
};
}  // namespace rxmesh