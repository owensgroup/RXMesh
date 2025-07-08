#pragma once

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/kernels/collective.cuh"
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

        // recycle this memory
        m_s_bipart    = Bitmask(m_num_v, m_s_v_mis.m_bitmask);
        m_s_swapped_v = Bitmask(m_num_v, m_s_candidate_v_mis.m_bitmask);

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
            true);


        uint16_t* s_ev = shrd_alloc.alloc<uint16_t>(
            std::max(m_num_v + 1, 2 * m_num_e) + 2 * m_num_e);
        m_s_vv_nxt.offset =
            shrd_alloc.alloc<uint16_t>(std::max(m_num_v + 1, 2 * m_num_e));
        m_s_vv_nxt.value = shrd_alloc.alloc<uint16_t>(2 * m_num_e);

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
        m_s_vv_cur.offset = &s_ev[0];
        m_s_vv_cur.value  = &s_ev[m_num_v + 1];
        detail::v_v<blockThreads>(block,
                                  m_patch_info,
                                  shrd_alloc,
                                  m_s_vv_cur.offset,
                                  m_s_vv_cur.value,
                                  false,
                                  false);
        block.sync();

        // mask out vertices on the separator (permuted via cross-patch
        // permutation)
        for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
            bool on_sep = false;
            if (m_s_active_v(v)) {
                uint16_t start = m_s_vv_cur.offset[v];
                uint16_t stop  = m_s_vv_cur.offset[v + 1];
                for (uint16_t i = start; i < stop; ++i) {
                    uint16_t n = m_s_vv_cur.value[i];

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
        coarsen_graph(block,
                      m_s_vv_cur,
                      get_matching_arr(level),
                      m_s_vv_nxt,
                      m_s_cur_active_v);

        // set the next VV to be the current for next iteration
        swap(m_s_vv_cur.offset, m_s_vv_nxt.offset);
        swap(m_s_vv_cur.value, m_s_vv_nxt.value);
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

#if 0
        // 1) do maximal independent set for vertices,
        // 2) each vertex in the MIS is an end vertex for the max matching,
        // thus, we use this results to pair a vertex in MIS with one of its
        // neighbor vertices (possibly the one with max weight)
        maximal_independent_set(block, m_s_cur_active_v, m_s_vv_cur);

        // no need to sync here
        //  recycling m_s_candidate_v_mis
        extract_random_edge_matching(block,
                                     m_s_vv_cur,
                                     m_s_cur_active_v,
                                     m_s_v_mis,
                                     m_s_candidate_v_mis,
                                     get_matching_arr(level));
#else
        single_thread_edge_matching(block,
                                    m_s_vv_cur,
                                    m_s_cur_active_v,
                                    m_s_candidate_v_mis,
                                    get_matching_arr(level));
#endif

        block.sync();


        //{
        //    for (uint16_t v = threadIdx.x; v < m_num_v; v += blockThreads) {
        //        attr_v(VertexHandle(m_patch_info.patch_id, v)) =
        //            (get_matching_arr(level)[v] == INVALID16) ?
        //                -1 :
        //                get_matching_arr(level)[v];
        //    }
        //
        //    for (uint16_t e = threadIdx.x; e < m_num_e; e += blockThreads) {
        //        uint16_t v0 = m_patch_info.ev[2 * e + 0].id;
        //        uint16_t v1 = m_patch_info.ev[2 * e + 1].id;
        //        if (get_matching_arr(level)[v0] ==
        //                get_matching_arr(level)[v1] &&
        //            get_matching_arr(level)[v0] != INVALID16) {
        //            attr_e(EdgeHandle(m_patch_info.patch_id, e)) = 1;
        //        } else {
        //            attr_e(EdgeHandle(m_patch_info.patch_id, e)) = -1;
        //        }
        //    }
        //}
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
                    uint16_t w                 = INVALID16;

                    uint16_t start = vv.offset[v];
                    uint16_t stop  = vv.offset[v + 1];

                    for (uint16_t i = start; i < stop; ++i) {
                        uint16_t n = vv.value[i];
                        if (!matched_v(n) && active_v(n) && n < w) {
                            matched_neighbour = n;
                            w                 = n;
                            // break;
                        }
                    }


                    if (matched_neighbour != INVALID16) {
                        matched_v.set(v, true);
                        matched_v.set(matched_neighbour, true);

                        v_matching[v] = matched_neighbour;

                        v_matching[matched_neighbour] = v;
                    }
                }
            }
        }
    }


    /**
     * @brief partition the coarse graph into two partitions
     */
    __device__ __inline__ void bipartition_coarse_graph(
        cooperative_groups::thread_block& block,
        int                               max_iter = 20)
    {


        // compacting active vertices ID by recycling the memory that is
        //  used to store the next graph
        uint16_t* s_active_v_id = m_s_vv_nxt.offset;


        int num_active_v = compact_active_vertices(block, s_active_v_id);

        // KL algorithm

        assert(num_active_v > 1);

        // Recycling the memory against.
        assert(2 * num_active_v < 2 * m_num_e);

        // the vertex with which we will switch assignment
        uint16_t* s_switch = m_s_vv_nxt.value;

        // store the gain of switching the assignment
        int16_t* s_gain =
            reinterpret_cast<int16_t*>(m_s_vv_nxt.value + num_active_v);


        // Step 1: Initial partition of the vertices into two sets

        // Random assignment
        for (uint16_t v = threadIdx.x; v < num_active_v; v += blockThreads) {
            // assign partition randomly
            if (v % 2 == 0) {
                m_s_bipart.reset(s_active_v_id[v], true);
            } else {
                m_s_bipart.set(s_active_v_id[v], true);
            }
        }

        // BFS from the first vertex (to make sure that each partition is a
        // connected component)
        // m_s_bipart.reset(block);
        // block.sync();
        //__shared__ int s_num_added;
        // if (threadIdx.x == 0) {
        //    m_s_bipart.set(s_active_v_id[0], true);
        //    s_num_added = 1;
        //}
        // block.sync();
        //
        // for (int iter = 0; iter < max_iter; ++iter) {
        //    for (uint16_t v = threadIdx.x; v < num_active_v;
        //         v += blockThreads) {
        //        uint16_t vertex = s_active_v_id[v];
        //        if (!m_s_bipart(vertex)) {
        //
        //            uint16_t v_start = m_s_vv_cur.offset[vertex];
        //            uint16_t v_stop  = m_s_vv_cur.offset[vertex + 1];
        //            for (uint16_t i = v_start; i < v_stop; ++i) {
        //                uint16_t n = m_s_vv_cur.value[i];
        //
        //                if (m_s_bipart(n)) {
        //                    m_s_bipart.set(vertex, true);
        //                    int ret = ::atomicAdd(&s_num_added, 1);
        //
        //
        //                    if (ret >= DIVIDE_UP(num_active_v, 2)) {
        //                        ::atomicAdd(&s_num_added, -1);
        //                        m_s_bipart.reset(vertex, true);
        //                    }
        //                    break;
        //                }
        //            }
        //        }
        //    }
        //    block.sync();
        //    if (s_num_added >= DIVIDE_UP(num_active_v, 2)) {
        //        break;
        //    }
        //}

        //  Make vertices that has been swapped and so we don't swap them
        // anymore
        // exit if all gains are -ve

        auto calc_gain = [&](uint16_t a, uint16_t b) {
            // compute the gain of switch the assignment between u and v

            uint16_t int_a(0), ext_a(0), int_b(0), ext_b(0);
            bool     is_edge = false;
            bool     a_par   = m_s_bipart(a);
            bool     b_par   = m_s_bipart(b);

            if (a_par == b_par) {
                return std::numeric_limits<int16_t>::min();
            }

            // Calculate D(a) and D(b), where D is the difference between
            // external and internal edges

            uint16_t a_start = m_s_vv_cur.offset[a];
            uint16_t a_stop  = m_s_vv_cur.offset[a + 1];
            for (uint16_t i = a_start; i < a_stop; ++i) {
                uint16_t s = m_s_vv_cur.value[i];
                if (!m_s_cur_active_v(s)) {
                    continue;
                }
                if (s == b) {
                    is_edge = true;
                }

                if (m_s_bipart(s) == a_par) {
                    int_a++;
                } else {
                    ext_a++;
                }
            }


            uint16_t b_start = m_s_vv_cur.offset[b];
            uint16_t b_stop  = m_s_vv_cur.offset[b + 1];
            for (uint16_t i = b_start; i < b_stop; ++i) {
                uint16_t s = m_s_vv_cur.value[i];
                if (!m_s_cur_active_v(s)) {
                    continue;
                }
                if (m_s_bipart(s) == b_par) {
                    int_b++;
                } else {
                    ext_b++;
                }
            }


            int16_t D_a = ext_a - int_a;
            int16_t D_b = ext_b - int_a;

            int16_t g = D_a + D_b - 2 * int(is_edge);

            return g;
        };

        __shared__ bool s_exit;
        m_s_swapped_v.reset(block);

        int cur_edge_cut = std::numeric_limits<int>::max();
        for (int iter = 0; iter < max_iter; ++iter) {

            // if (threadIdx.x == 0) {
            //     printf("\n iter =%u, edge_cut= %d",
            //            iter,
            //            calc_edge_cut(m_s_vv_cur, m_s_cur_active_v,
            //            m_s_bipart));
            //     print_graph(m_s_vv_cur, m_s_cur_active_v);
            // }

            // Step 2: calc the gain of changing the assignment with all active
            // vertices, i.e., every thread will compute the gain of switching
            // one vertex with all other vertices and calc the max gain out of
            // all these vertices
            block.sync();


            for (uint16_t i = threadIdx.x; i < num_active_v;
                 i += blockThreads) {
                uint16_t v = s_active_v_id[i];

                // for all other vertices
                int16_t  max_g = std::numeric_limits<int16_t>::min();
                uint16_t max_u = INVALID16;
                if (!m_s_swapped_v(v)) {
                    for (uint16_t j = 0; j < num_active_v; ++j) {
                        uint16_t u = s_active_v_id[j];

                        if (v < u && !m_s_swapped_v(u)) {
                            int16_t g = calc_gain(v, u);
                            if (g > max_g) {
                                max_g = g;
                                max_u = u;
                            }
                        }
                    }
                }
                s_gain[i]   = max_g;
                s_switch[i] = max_u;
            }
            block.sync();

            // let one thread decide which two vertices to switch
            if (threadIdx.x == 0) {

                uint16_t max_g_id = INVALID16;
                int16_t  max_g    = std::numeric_limits<int16_t>::min();

                int all_swapped = true;

                for (uint16_t i = 0; i < num_active_v; ++i) {
                    if (!m_s_swapped_v(i)) {
                        all_swapped = false;
                    }
                    int16_t g = s_gain[i];
                    if (g > max_g) {
                        max_g    = g;
                        max_g_id = i;
                    }
                }

                if (max_g_id == INVALID16 || all_swapped) {
                    s_exit = true;
                } else {
                    s_exit = false;

                    uint16_t a = s_active_v_id[max_g_id];
                    uint16_t b = s_switch[max_g_id];

                    assert(m_s_bipart(a) != m_s_bipart(b));
                    assert(!m_s_swapped_v(a));
                    assert(!m_s_swapped_v(b));

                    m_s_swapped_v.set(a, true);
                    m_s_swapped_v.set(b, true);

                    if (m_s_bipart(a)) {
                        m_s_bipart.reset(a, true);
                    } else {
                        m_s_bipart.set(a, true);
                    }


                    if (m_s_bipart(b)) {
                        m_s_bipart.reset(b, true);
                    } else {
                        m_s_bipart.set(b, true);
                    }
                }
            }

            block.sync();
            if (s_exit) {
                break;
            }
        }


        //{
        //    block.sync();
        //    print_graph(m_s_vv_cur, m_s_cur_active_v);
        //}
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
        __shared__ int s_added;

        fill_n<blockThreads>(v_matching, m_num_v, uint16_t(INVALID16));
        for (int iter = 0; iter < max_iter; ++iter) {
            if (threadIdx.x == 0) {
                s_added = 0;
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
                        s_added = 1;
                    }
                }
            }
            block.sync();

            if (s_added == 0) {
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
        __shared__ int s_num_active_v_mis;
        if (threadIdx.x == 0) {
            s_num_active_v_mis = 0;
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
                    ::atomicAdd(&s_num_active_v_mis, 1);
                }
            }
            // reset the candidate
            m_s_candidate_v_mis.reset(block);
            block.sync();

            if (s_num_active_v_mis == 0) {
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
                s_num_active_v_mis = 0;
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
                    if (active_v(n)) {
                        if (helper_v.try_set(n)) {
                            v_matching[v] = n;
                            v_matching[n] = v;
                            break;
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief generate another graph given an edge matching by collapsing the
     * matched edges into a vertex
     */
    __device__ __inline__ void coarsen_graph(
        cooperative_groups::thread_block& block,
        const VV&                         vv_cur,
        const uint16_t*                   v_matching,
        VV&                               vv_next,
        Bitmask&                          active_v)
    {

        fill_n<blockThreads>(vv_next.offset, m_num_v + 1, uint16_t(0));
        block.sync();

        auto is_connected = [&](const uint16_t v,
                                const uint16_t n,
                                const uint16_t v_start,
                                const uint16_t v_stop) {
            // check if v is connected to n
            for (uint16_t i = v_start; i < v_stop; ++i) {
                uint16_t s = vv_cur.value[i];
                if (active_v(s)) {
                    if (s == n) {
                        return true;
                    }
                }
            }

            return false;
        };


        auto for_each_future_neighbour = [&](uint16_t v, auto do_something) {
            // we only consider the vertex if 1) it is not matched, i.e., it
            // will appear as itself in the coarse graph, 2) if it is matched
            // but it is matched vertex has a higher id since we only keep the
            // vertex with lesser ID


            uint16_t match = v_matching[v];

            if (v < match || match == INVALID16) {
                // if v is the smallest, it will be the one that creates the
                // new vertex
                // count how many vertices v will be connected to (be
                // careful about the duplicates)

                const uint16_t v_start = vv_cur.offset[v];
                const uint16_t v_stop  = vv_cur.offset[v + 1];

                for (uint16_t i = v_start; i < v_stop; ++i) {
                    uint16_t n = vv_cur.value[i];

                    assert(n < m_num_v);

                    // if n is active (because initial VV could be connected to
                    // not-owned vertices or vertices on the separator) and if
                    // n is not the matched vertex with v

                    if (active_v(n) && n != match) {
                        uint16_t n_match = v_matching[n];
                        if (n_match == v) {
                            continue;
                        }


                        if (n < n_match || n_match == INVALID16) {

                            do_something(n);
                        } else if (n_match != INVALID16 && n_match < n) {
                            bool is_n_match_connected =
                                is_connected(v, n_match, v_start, v_stop);


                            if (!is_n_match_connected) {

                                do_something(n_match);
                            }
                        }
                    }
                }


                if (match != INVALID16) {

                    // do the same with the match
                    const uint16_t m_start = vv_cur.offset[match];
                    const uint16_t m_stop  = vv_cur.offset[match + 1];

                    for (uint16_t i = m_start; i < m_stop; ++i) {
                        uint16_t n = vv_cur.value[i];

                        assert(n < m_num_v);


                        if (active_v(n) && n != v) {
                            uint16_t n_match = v_matching[n];


                            // if the match is v itself, then we don't wanna
                            // consider it
                            if (n_match == v) {
                                continue;
                            }


                            if (is_connected(v, n, v_start, v_stop) ||
                                is_connected(v, n_match, v_start, v_stop)) {
                                // if n (or its match) is connected to v, then
                                // we considered it already

                                continue;
                            }

                            if (n < n_match || n_match == INVALID16) {

                                do_something(n);

                            } else if (n_match != INVALID16 && n_match < n) {
                                bool is_n_match_connected = is_connected(
                                    match, n_match, m_start, m_stop);

                                if (!is_n_match_connected) {

                                    do_something(n_match);
                                }
                            }
                        }
                    }
                }
            }
        };

        // 1) Count
        for (uint16_t v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (active_v(v)) {
                uint16_t num_vv = 0;

                for_each_future_neighbour(v, [&](uint16_t n) { num_vv++; });

                vv_next.offset[v] = num_vv;
            }
        }
        block.sync();


        // 2) Compute Prefix Sum
        detail::cub_block_exclusive_sum<uint16_t, blockThreads>(vv_next.offset,
                                                                m_num_v);

        block.sync();
        assert(vv_next.offset[m_num_v] <= 2 * m_num_e);


        // 3) Populate
        for (uint16_t v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (active_v(v)) {

                uint16_t id     = 0;
                uint16_t offset = vv_next.offset[v];
                uint16_t num_v  = 0;

                for_each_future_neighbour(v, [&](uint16_t n) {
                    assert(n < m_num_v);
                    vv_next.value[offset + id] = n;
                    id++;
                    num_v++;
                });

                assert(num_v == vv_next.offset[v + 1] - vv_next.offset[v]);
            }
        }
        block.sync();

        // 4) Deactivate matched vertices with higher ID than its match
        for (uint16_t v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (active_v(v)) {
                uint16_t match = v_matching[v];
                if (v > match && match != INVALID16) {
                    // if the matched vertex (match) is the one that will create
                    // the new vertex, then deactivate v
                    active_v.reset(v, true);
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

    /**
     * @brief count the number of current active vertices
     */
    __device__ __inline__ int num_active_vertices(
        cooperative_groups::thread_block& block)
    {
        __shared__ int s_count;
        if (threadIdx.x == 0) {
            s_count = 0;
        }
        block.sync();

        const uint16_t mask_num_elements =
            DIVIDE_UP(m_s_cur_active_v.size(), 32);

        const uint16_t rem_bits = m_s_active_v.size() & 31u;

        const uint32_t last_word_mask =
            (rem_bits == 0) ? 0xFFFFFFFFu : ((1u << rem_bits) - 1u);

        for (uint16_t i = threadIdx.x; i < mask_num_elements;
             i += blockThreads) {

            unsigned int x = m_s_cur_active_v.m_bitmask[i];

            if (rem_bits && i == mask_num_elements - 1) {
                x &= last_word_mask;
            }

            int num_set_bits = __popc(x);

            ::atomicAdd(&s_count, num_set_bits);
        }
        block.sync();
        return s_count;
    }

    /**
     * @brief compact the active vertices ID
     * @return
     */
    __device__ __inline__ int compact_active_vertices(
        cooperative_groups::thread_block& block,
        uint16_t*                         s_active_v_id)
    {
        __shared__ int s_count[1];
        if (threadIdx.x == 0) {
            s_count[0] = 0;
        }
        block.sync();

        for (uint16_t v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (m_s_cur_active_v(v)) {
                s_active_v_id[::atomicAdd(s_count, 1)] = v;
            }
        }
        block.sync();
        return s_count[0];
    }

    __device__ __inline__ void print_graph(const VV&      vv,
                                           const Bitmask& active_v)
    {
        if (threadIdx.x == 0) {
            printf("\n ************ \n");
            printf("\n digraph G {");
            for (uint16_t v = 0; v < m_num_v; ++v) {
                if (active_v(v)) {
                    uint16_t start = vv.offset[v];
                    uint16_t stop  = vv.offset[v + 1];

                    if (m_s_bipart(v)) {
                        printf("\n %u [style=filled, fillcolor=lightblue]", v);
                    }

                    for (uint16_t i = start; i < stop; ++i) {
                        uint16_t n = vv.value[i];
                        assert(n < m_num_v);
                        assert(n != v);
                        if (v < n && active_v(n)) {
                            printf("\n %u -> %u", v, n);
                        }
                    }
                }
            }
            printf("\n }");
            printf("\n ************ \n");
        }
    }

    __device__ __inline__ uint16_t calc_edge_cut(const VV&      vv,
                                                 const Bitmask& active_v,
                                                 const Bitmask& bipart_v)
    {
        int ret = 0;
        if (threadIdx.x == 0) {

            for (uint16_t v = 0; v < m_num_v; ++v) {
                if (active_v(v)) {
                    uint16_t start = vv.offset[v];
                    uint16_t stop  = vv.offset[v + 1];


                    for (uint16_t i = start; i < stop; ++i) {
                        uint16_t n = vv.value[i];

                        if (v < n && active_v(n) &&
                            bipart_v(v) != bipart_v(n)) {
                            ret++;
                        }
                    }
                }
            }
        }

        return ret;
    }

    // private:
    PatchInfo m_patch_info;
    VV        m_s_vv_cur, m_s_vv_nxt;
    Bitmask   m_s_active_v_mis;     // the active vertices for MIS
    Bitmask   m_s_candidate_v_mis;  // candidates for MIS
    Bitmask   m_s_v_mis;            // the vertices in the MIS
    Bitmask   m_s_active_v;  // active vertices in the patch (minus not-owned)
    Bitmask   m_s_cur_active_v;  // current active vertices during coarsening
    Bitmask   m_s_bipart;        // the bipartition assignment
    Bitmask   m_s_swapped_v;
    uint32_t  m_num_v, m_num_e;
    uint16_t* m_v_matching;  // store the vertex a given vertex is matched with
};
}  // namespace rxmesh