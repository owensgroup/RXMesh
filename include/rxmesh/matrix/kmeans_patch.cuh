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

        m_s_active_v     = Bitmask(m_num_v, shrd_alloc);
        m_s_cur_active_v = Bitmask(m_num_v, shrd_alloc);

        m_s_assigned_v      = Bitmask(m_num_v, shrd_alloc);
        m_s_cur_frontier_v  = Bitmask(m_num_v, shrd_alloc);
        m_s_next_frontier_v = Bitmask(m_num_v, shrd_alloc);
        m_s_partition_a_v   = Bitmask(m_num_v, shrd_alloc);
        m_s_partition_b_v   = Bitmask(m_num_v, shrd_alloc);

        m_s_separator = Bitmask(m_num_v, m_s_cur_frontier_v.m_bitmask);

        m_s_active_v.reset(block);
        m_s_cur_active_v.reset(block);

        m_s_assigned_v.reset(block);
        m_s_cur_frontier_v.reset(block);
        m_s_next_frontier_v.reset(block);
        m_s_partition_a_v.reset(block);
        m_s_partition_b_v.reset(block);

        m_s_index = shrd_alloc.alloc<uint16_t>(m_num_v);

        detail::load_async(
            block,
            reinterpret_cast<const char*>(m_patch_info.active_mask_v),
            m_s_active_v.num_bytes(),
            reinterpret_cast<char*>(m_s_active_v.m_bitmask),
            false);


        uint16_t* s_ev = shrd_alloc.alloc<uint16_t>(
            std::max(m_num_v + 1, 2 * m_num_e) + 2 * m_num_e);

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
        m_s_vv.offset = &s_ev[0];
        m_s_vv.value  = &s_ev[m_num_v + 1];
        detail::v_v<blockThreads>(block,
                                  m_patch_info,
                                  shrd_alloc,
                                  m_s_vv.offset,
                                  m_s_vv.value,
                                  false,
                                  false);
        block.sync();

        // mask out vertices on the separator (permuted via cross-patch
        // permutation)
        for (int v = threadIdx.x; v < m_num_v; v += blockThreads) {
            bool on_sep = false;
            if (m_s_active_v(v)) {
                uint16_t start = m_s_vv.offset[v];
                uint16_t stop  = m_s_vv.offset[v + 1];
                for (uint16_t i = start; i < stop; ++i) {
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

        // copy active mask
        m_s_cur_active_v.copy(block, m_s_active_v);
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
                                                m_s_cur_active_v,
                                                true,
                                                m_s_cur_active_v,
                                                m_s_vv.offset,
                                                m_s_vv.value,
                                                m_s_assigned_v,
                                                m_s_cur_frontier_v,
                                                m_s_next_frontier_v,
                                                m_s_partition_a_v,
                                                m_s_partition_b_v,
                                                10);

#ifndef NDEBUG
        for (uint16_t v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (m_s_cur_active_v(v)) {
                assert(m_s_partition_a_v(v) != m_s_partition_b_v(v));
                assert(m_s_partition_a_v(v) || m_s_partition_b_v(v));
            }
        }

#endif

        //{
        //    block.sync();
        //    for (uint16_t v = threadIdx.x; v < m_num_v; v += blockThreads) {
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
     * @brief extract the separator between the two partitions
     */
    __device__ __inline__ void extract_separator(
        cooperative_groups::thread_block& block)
    {
        m_s_separator.reset(block);
        block.sync();
        for (uint16_t v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (m_s_cur_active_v(v)) {
                assert(m_s_partition_a_v(v) != m_s_partition_b_v(v));
                assert(m_s_partition_a_v(v) || m_s_partition_b_v(v));
                uint16_t start = m_s_vv.offset[v];
                uint16_t stop  = m_s_vv.offset[v + 1];

                if (m_s_partition_a_v(v)) {


                    for (uint16_t i = start; i < stop; ++i) {
                        uint16_t n = m_s_vv.value[i];
                        if (m_s_cur_active_v(n)) {
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


        for (uint16_t v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (m_s_cur_active_v(v)) {

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

        // assert(sum == num_active_vertices(block));


        for (uint16_t v = threadIdx.x; v < m_num_v; v += blockThreads) {
            if (m_s_cur_active_v(v)) {
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
        for (uint16_t i = threadIdx.x; i < mask_num_elements;
             i += blockThreads) {
            int num_set_bits = __popc(m_s_cur_active_v.m_bitmask[i]);

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

                    if (m_s_partition_a_v(v)) {
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
    VV        m_s_vv;
    Bitmask   m_s_active_v;  // active vertices in the patch (minus not-owned)
    Bitmask   m_s_cur_active_v;
    Bitmask   m_s_separator;
    uint16_t* m_s_index;

    Bitmask m_s_assigned_v, m_s_cur_frontier_v, m_s_next_frontier_v,
        m_s_partition_a_v, m_s_partition_b_v;

    uint32_t m_num_v, m_num_e;
};
}  // namespace rxmesh