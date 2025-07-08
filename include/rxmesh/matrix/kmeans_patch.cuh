#pragma once

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/rxmesh_queries.cuh"
#include "rxmesh/kernels/util.cuh"

#include "rxmesh/matrix/mgnd_permute.cuh"

namespace rxmesh {

namespace detail {

template <uint32_t blockThreads>
__device__ void bi_assignment_ggp(cooperative_groups::thread_block& block,
                                  const uint16_t  num_vertices,
                                  const Bitmask&  s_owned_v,
                                  const bool      ignore_owned_v,
                                  const Bitmask&  s_active_v,
                                  const uint16_t* m_s_vv_offset,
                                  const uint16_t* m_s_vv,
                                  Bitmask&        s_assigned_v,
                                  Bitmask&        s_current_frontier_v,
                                  Bitmask&        s_next_frontier_v,
                                  Bitmask&        s_partition_a_v,
                                  Bitmask&        s_partition_b_v,
                                  int             num_iter);
}


template <uint32_t blockThreads>
struct PatchKMeans
{
    __device__ __inline__ PatchKMeans(cooperative_groups::thread_block& block,
                                      Context&                          context,
                                      ShmemAllocator& shrd_alloc)
        : m_patch_info(context.m_patches_info[blockIdx.x])
    {
        m_num_v = m_patch_info.num_vertices[0];
        m_num_e = m_patch_info.num_edges[0];

        m_s_active_v = Bitmask(m_num_v, shrd_alloc);
        m_s_active_v.reset(block);

        detail::load_async(
            block,
            reinterpret_cast<const char*>(m_patch_info.active_mask_v),
            m_s_active_v.num_bytes(),
            reinterpret_cast<char*>(m_s_active_v.m_bitmask),
            true);


        uint16_t* s_ev = shrd_alloc.alloc<uint16_t>(
            std::max(m_num_v + 1, 2 * m_num_e) + 2 * m_num_e);

        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(m_patch_info.ev),
                           2 * m_num_e,
                           s_ev,
                           true);

        // mark not-owned vertices as not-active
        uint16_t mask_num_elements = DIVIDE_UP(m_s_active_v.size(), 32);
        for (int i = threadIdx.x; i < mask_num_elements; i += blockThreads) {
            m_s_active_v.m_bitmask[i] =
                m_s_active_v.m_bitmask[i] & m_patch_info.owned_mask_v[i];
        }

        // create vv
        m_s_vv.offset = &s_ev[0];
        m_s_vv.value  = &s_ev[2 * m_num_e];
        detail::v_v<blockThreads>(block,
                                  m_patch_info,
                                  shrd_alloc,
                                  m_s_vv.offset,
                                  m_s_vv.value,
                                  false,
                                  false);
        block.sync();

        //
        m_s_assigned_v      = Bitmask(m_num_v, shrd_alloc);
        m_s_cur_frontier_v  = Bitmask(m_num_v, shrd_alloc);
        m_s_next_frontier_v = Bitmask(m_num_v, shrd_alloc);
        m_s_partition_a_v   = Bitmask(m_num_v, shrd_alloc);
        m_s_partition_b_v   = Bitmask(m_num_v, shrd_alloc);
        m_s_v_locked        = Bitmask(m_num_v, shrd_alloc);

        m_s_separator = Bitmask(m_num_v, m_s_cur_frontier_v.m_bitmask);


        m_s_assigned_v.reset(block);
        m_s_cur_frontier_v.reset(block);
        m_s_next_frontier_v.reset(block);
        m_s_partition_a_v.reset(block);
        m_s_partition_b_v.reset(block);

        m_s_index      = shrd_alloc.alloc<uint16_t>(m_num_v);
        m_s_max_gain_v = m_s_index;

        m_s_current_gain = shrd_alloc.alloc<int16_t>(m_num_v);
        m_s_cum_gain     = shrd_alloc.alloc<int16_t>(m_num_v);

        // mask out vertices on the separator (permuted via cross-patch
        // permutation)
        for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
            bool on_sep = false;
            if (m_s_active_v(v)) {
                uint16_t start = m_s_vv.offset[v];
                uint16_t stop  = m_s_vv.offset[v + 1];
                for (int i = start; i < stop; ++i) {
                    uint16_t n = m_s_vv.value[i];

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
    }

    /**
     * @brief partition the mesh into two parts of nearly equal size and with
     */
    __device__ __inline__ void partition(
        cooperative_groups::thread_block& block)
    {


        detail::bi_assignment_ggp<blockThreads>(block,
                                                m_num_v,
                                                m_s_active_v,
                                                true,
                                                m_s_active_v,
                                                m_s_vv.offset,
                                                m_s_vv.value,
                                                m_s_assigned_v,
                                                m_s_cur_frontier_v,
                                                m_s_next_frontier_v,
                                                m_s_partition_a_v,
                                                m_s_partition_b_v,
                                                10);

#ifndef NDEBUG
        for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (m_s_active_v(v)) {
                assert(m_s_partition_a_v(v) != m_s_partition_b_v(v));
                assert(m_s_partition_a_v(v) || m_s_partition_b_v(v));
            }
        }

#endif

        //{
        //    block.sync();
        //    for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
        //        if (m_s_partition_a_v(v)) {
        //            attr_v(VertexHandle(m_patch_info.patch_id, v)) = 1;
        //        } else if (m_s_partition_b_v(v)) {
        //            attr_v(VertexHandle(m_patch_info.patch_id, v)) = 2;
        //        } else {
        //            attr_v(VertexHandle(m_patch_info.patch_id, v)) = -1;
        //        }
        //    }
        //}
    }

    /**
     * @brief implement Fiduccia–Mattheyses (FM) refinement to reduce the edge
     * cut
     */
    __device__ __inline__ void fm_refinement(
        cooperative_groups::thread_block& block,
        float                             deviation_threshold = 0.1)
    {
        m_s_v_locked.reset(block);
        fill_n<blockThreads>(m_s_max_gain_v, m_num_v, uint16_t(INVALID16));
        fill_n<blockThreads>(m_s_cum_gain, m_num_v, int16_t(0));

        int num_active_v, num_a, num_b;

        __shared__ bool s_should_exit;

        // the maximum possible number of iterations should be the number
        // of active vertices. we should exit way before this number of
        // iteration but it is an upper bound in case we get stuck in some
        // pathological case


        num_active_vertices(block, num_active_v, num_a, num_b);

        int iter = 0;

        while (iter < num_active_v) {

            fill_n<blockThreads>(m_s_current_gain,
                                 m_num_v,
                                 std::numeric_limits<int16_t>::lowest());
            block.sync();

            compute_fm_gain(block);

            const uint16_t max_gain_v = max_gain_vertex(block);
            const int16_t  max_gain   = m_s_current_gain[max_gain_v];

            if (max_gain == std::numeric_limits<int16_t>::lowest()) {
                break;
            }

            if (threadIdx.x == 0) {
                // change partition
                if (m_s_partition_a_v(max_gain_v)) {
                    // form a to b
                    assert(!m_s_partition_b_v(max_gain_v));
                    m_s_partition_a_v.reset(max_gain_v, true);
                    m_s_partition_b_v.set(max_gain_v, true);
                    num_a--;
                    num_b++;
                }

                if (m_s_partition_b_v(max_gain_v)) {
                    // from b to a
                    assert(!m_s_partition_a_v(max_gain_v));
                    m_s_partition_b_v.reset(max_gain_v, true);
                    m_s_partition_a_v.set(max_gain_v, true);
                    num_b--;
                    num_a++;
                }

                // lock the vertex
                m_s_v_locked.set(max_gain_v, true);


                // record the max gain vertex of this iteration
                m_s_max_gain_v[iter] = max_gain_v;

                // cumulative gain of this iteration                
                m_s_cum_gain[iter] = max_gain;
                if (iter > 0) {
                    m_s_cum_gain[iter] += m_s_cum_gain[iter - 1];
                }

                float eps =
                    float(std::abs(num_a - num_b)) / float(num_active_v);

                s_should_exit = eps > deviation_threshold;
            }

            iter++;

            block.sync();
            if (s_should_exit) {
                break;
            }
        }        

        if (iter > 0) {
            fm_backtracking(block, iter);
            block.sync();
        }
        
    }

    /**
     * @brief extract the separator between the two partitions
     */
    __device__ __inline__ void extract_separator(
        cooperative_groups::thread_block& block)
    {
        m_s_separator.reset(block);
        block.sync();
        for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (m_s_active_v(v)) {
                assert(m_s_partition_a_v(v) != m_s_partition_b_v(v));
                assert(m_s_partition_a_v(v) || m_s_partition_b_v(v));
                uint16_t start = m_s_vv.offset[v];
                uint16_t stop  = m_s_vv.offset[v + 1];

                if (m_s_partition_a_v(v)) {

                    for (int i = start; i < stop; ++i) {
                        uint16_t n = m_s_vv.value[i];
                        if (m_s_active_v(n)) {
                            assert(m_s_partition_a_v(n) !=
                                   m_s_partition_b_v(n));
                            assert(m_s_partition_a_v(n) ||
                                   m_s_partition_b_v(n));

                            if (m_s_partition_b_v(n)) {
                                m_s_separator.set(v, true);
                                // attr_v(VertexHandle(m_patch_info.patch_id,
                                // v)) = 1;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief given the partitions and their extracted separators, compute the
     * new permutation by putting the separator at the end, then reorder
     * everything in partition a then everything in partition b
     */
    __device__ __inline__ void assign_permutation(
        cooperative_groups::thread_block& block,
        VertexAttribute<uint16_t>&        v_permute)
    {
        __shared__ int s_num_a;
        __shared__ int s_num_b;
        __shared__ int s_num_sep;
        if (threadIdx.x == 0) {
            s_num_a   = 0;
            s_num_b   = 0;
            s_num_sep = 0;
        }
        block.sync();


        for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (m_s_active_v(v)) {

                if (m_s_separator(v)) {
                    m_s_index[v] = ::atomicAdd(&s_num_sep, 1);
                } else if (m_s_partition_b_v(v)) {
                    m_s_index[v] = ::atomicAdd(&s_num_b, 1);
                } else if (m_s_partition_a_v(v)) {
                    m_s_index[v] = ::atomicAdd(&s_num_a, 1);
                } else {
                    assert(1 != 1);
                }
            }
        }

        block.sync();

        int sum = s_num_a + s_num_b + s_num_sep;

        for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (m_s_active_v(v)) {
                VertexHandle vh(m_patch_info.patch_id, v);

                assert(v_permute(vh) == INVALID16);

                if (m_s_separator(v)) {
                    // v_permute(vh) = sum - m_s_index[v] - 1;
                    v_permute(vh) = m_s_index[v];
                } else if (m_s_partition_b_v(v)) {
                    // v_permute(vh) = sum - s_num_sep - m_s_index[v] - 1;
                    v_permute(vh) = s_num_sep + m_s_index[v];
                } else if (m_s_partition_a_v(v)) {
                    // v_permute(vh) =
                    //     sum - (s_num_sep + s_num_b) - m_s_index[v] - 1;
                    v_permute(vh) = s_num_sep + s_num_b + m_s_index[v];
                }
            }
        }
    }

    /**
     * @brief count the number of current active vertices
     */
    __device__ __inline__ void num_active_vertices(
        cooperative_groups::thread_block& block,
        int&                              num_active_v,
        int&                              num_a,
        int&                              num_b)
    {

        __shared__ int s_count, s_count_a, s_count_b;
        if (threadIdx.x == 0) {
            s_count   = 0;
            s_count_a = 0;
            s_count_b = 0;
        }
        block.sync();

        const uint16_t mask_num_elements = DIVIDE_UP(m_s_active_v.size(), 32);

        const uint16_t rem_bits = m_s_active_v.size() & 31u;

        const uint32_t last_word_mask =
            (rem_bits == 0) ? 0xFFFFFFFFu : ((1u << rem_bits) - 1u);


        for (int i = threadIdx.x; i < mask_num_elements; i += blockThreads) {

            unsigned int x   = m_s_active_v.m_bitmask[i];
            unsigned int x_a = m_s_partition_a_v.m_bitmask[i];
            unsigned int x_b = m_s_partition_b_v.m_bitmask[i];

            if (rem_bits && i == mask_num_elements - 1) {
                x &= last_word_mask;
                x_a &= last_word_mask;
                x_b &= last_word_mask;
            }

            int num_set_bits   = __popc(x);
            int num_set_bits_a = __popc(x_a);
            int num_set_bits_b = __popc(x_b);

            ::atomicAdd(&s_count, num_set_bits);
            ::atomicAdd(&s_count_a, num_set_bits_a);
            ::atomicAdd(&s_count_b, num_set_bits_b);
        }
        block.sync();
        num_active_v = s_count;
        num_a        = s_count_a;
        num_b        = s_count_b;
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

        for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (m_s_active_v(v)) {
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
            for (int v = 0; v < m_num_v; ++v) {
                if (active_v(v)) {
                    uint16_t start = vv.offset[v];
                    uint16_t stop  = vv.offset[v + 1];

                    if (m_s_partition_a_v(v)) {
                        printf("\n %u [style=filled, fillcolor=lightblue]", v);
                    }

                    for (int i = start; i < stop; ++i) {
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

    __device__ __inline__ int calc_edge_cut(const VV&      vv,
                                            const Bitmask& active_v,
                                            const Bitmask& bipart_v)
    {
        int ret = 0;
        if (threadIdx.x == 0) {

            for (int v = 0; v < m_num_v; ++v) {
                if (active_v(v)) {
                    uint16_t start = vv.offset[v];
                    uint16_t stop  = vv.offset[v + 1];


                    for (int i = start; i < stop; ++i) {
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

    /**
     * @brief compute the gain of possibly moving a vertex from its assigned
     * partition to the other partition.
     * The gain = #neighbor in other partition - #neighbor in same partition
     */
    __device__ __inline__ void compute_fm_gain(
        cooperative_groups::thread_block& block)
    {
        for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (m_s_active_v(v) && !m_s_v_locked(v)) {

                assert(m_s_partition_a_v(v) != m_s_partition_b_v(v));
                assert(m_s_partition_a_v(v) || m_s_partition_b_v(v));

                const bool v_part_a = m_s_partition_a_v(v);
                const bool v_part_b = m_s_partition_b_v(v);

                const uint16_t start = m_s_vv.offset[v];
                const uint16_t stop  = m_s_vv.offset[v + 1];

                int16_t other = 0;
                int16_t same  = 0;
                for (int i = start; i < stop; ++i) {
                    const uint16_t n = m_s_vv.value[i];

                    if (!m_s_active_v(n)) {
                        continue;
                    }
                    assert(m_s_partition_a_v(n) != m_s_partition_b_v(n));
                    assert(m_s_partition_a_v(n) || m_s_partition_b_v(n));

                    if (v_part_a && m_s_partition_a_v(n) ||
                        v_part_b && m_s_partition_b_v(n)) {
                        same++;
                    } else if (v_part_a && m_s_partition_b_v(n) ||
                               v_part_b && m_s_partition_a_v(n)) {
                        other++;
                    } else {
                        assert(false);
                    }
                }

                m_s_current_gain[v] = other - same;

            } else {
                m_s_current_gain[v] = std::numeric_limits<int16_t>::lowest();
            }
        }
        block.sync();
    }

    /**
     * @brief find the vertex with maximum gain--assuming it is stored in
     * m_s_current_gain
     */
    template <int item_per_thread = 4>
    __device__ __inline__ uint16_t max_gain_vertex(
        cooperative_groups::thread_block& block)
    {
        assert(block.size() * item_per_thread >= m_num_v);

        using BlockReduce = cub::BlockReduce<uint16_t, blockThreads>;
        __shared__ typename BlockReduce::TempStorage temp_storage;

        // Reduction results returned from cub is only valid for thread 0,
        // so we write it here so other threads can see it.
        __shared__ uint16_t s_result;

        uint16_t thread_data[item_per_thread];
        for (int i = 0; i < item_per_thread; ++i) {
            int id = i * block.size() + threadIdx.x;
            if (id < m_num_v) {
                if (m_s_active_v(id)) {
                    thread_data[i] = id;
                } else {
                    thread_data[i] = INVALID16;
                }
            } else {
                thread_data[i] = INVALID16;
            }
        }

        int res =
            BlockReduce(temp_storage)
                .Reduce(thread_data, [&](const uint16_t& i, const uint16_t& j) {
                    if (i >= m_num_v && j < m_num_v) {
                        // if i is out of bound but j is not
                        return j;
                    }
                    if (j >= m_num_v && i < m_num_v) {
                        // if j is out of bound but i is not
                        return i;
                    }

                    if (j >= m_num_v && i >= m_num_v) {
                        // if both are out of bound, does not matter
                        return i;
                    }

                    if (m_s_active_v(j) && m_s_active_v(i)) {
                        // if both in bound
                        return (m_s_current_gain[i] > m_s_current_gain[j]) ? i :
                                                                             j;
                    }

                    // does not matter is both are inactive
                    return i;
                });

        if (threadIdx.x == 0) {
            s_result = res;
        }
        block.sync();

        return s_result;
    }


    /**
     * @brief
     */
    template <int item_per_thread = 4>
    __device__ __inline__ uint16_t fm_backtracking(
        cooperative_groups::thread_block& block,
        int                               size)
    {
        assert(block.size() * item_per_thread >= size);

        using BlockReduce = cub::BlockReduce<uint16_t, blockThreads>;
        __shared__ typename BlockReduce::TempStorage temp_storage;

        // Reduction results returned from cub is only valid for thread 0,
        // so we write it here so other threads can see it.
        //
        // Iteration that caused maximum gain
        __shared__ uint16_t s_max_cum_iter;

        uint16_t thread_data[item_per_thread];
        for (int i = 0; i < item_per_thread; ++i) {
            int id = i * block.size() + threadIdx.x;
            if (id < size) {
                thread_data[i] = id;
            } else {
                thread_data[i] = INVALID16;
            }
        }

        int res =
            BlockReduce(temp_storage)
                .Reduce(thread_data, [&](const uint16_t& i, const uint16_t& j) {
                    if (i >= size && j < size) {
                        // if i is out of bound but j is not
                        return j;
                    }
                    if (j >= size && i < size) {
                        // if j is out of bound but i is not
                        return i;
                    }

                    if (j >= size && i >= size) {
                        // if both are out of bound, does not matter
                        return i;
                    }

                    return (m_s_cum_gain[i] > m_s_cum_gain[j]) ? i : j;
                });

        if (threadIdx.x == 0) {            
            s_max_cum_iter = res;
        }
        block.sync();

        // backtrack changes done on vertices after s_max_cum_iter
        for (int i = threadIdx.x; i < size; i += blockThreads) {
            int id = i + s_max_cum_iter;
            if (id >= size) {
                continue;
            }

            uint16_t v = m_s_max_gain_v[id];

            // we must have locked v before
            assert(m_s_v_locked(v));

            if (m_s_partition_a_v(v)) {
                // form a to b
                assert(!m_s_partition_b_v(v));
                m_s_partition_a_v.reset(v, true);
                m_s_partition_b_v.set(v, true);
            }

            if (m_s_partition_b_v(v)) {
                // from b to a
                assert(!m_s_partition_a_v(v));
                m_s_partition_b_v.reset(v, true);
                m_s_partition_a_v.set(v, true);
            }
        }
    }


   private:
    PatchInfo m_patch_info;
    VV        m_s_vv;
    Bitmask   m_s_active_v;  // active vertices in the patch (minus not-owned)
    Bitmask   m_s_separator;
    uint16_t* m_s_index;

    Bitmask m_s_assigned_v, m_s_cur_frontier_v, m_s_next_frontier_v,
        m_s_partition_a_v, m_s_partition_b_v;

    // is vertex locked during FM refinement
    Bitmask m_s_v_locked;
    // vertex associated with max gain in the current iteration of FM
    // overlap m_s_index
    uint16_t* m_s_max_gain_v;
    // vertex gain at the current iteration and the cumulative gain across
    // iterations
    int16_t *m_s_current_gain, *m_s_cum_gain;

    uint32_t m_num_v, m_num_e;
};
}  // namespace rxmesh