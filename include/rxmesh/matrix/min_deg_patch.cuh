#pragma once

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/rxmesh_queries.cuh"
#include "rxmesh/kernels/util.cuh"

#include "rxmesh/matrix/mgnd_permute.cuh"

#include <cub/block/block_reduce.cuh>

namespace rxmesh {

template <uint32_t blockThreads>
struct PatchMinDeg
{
    __device__ __inline__ PatchMinDeg(cooperative_groups::thread_block& block,
                                      Context&                          context,
                                      ShmemAllocator& shrd_alloc)
        : m_patch_info(context.m_patches_info[blockIdx.x])
    {
        m_num_v = m_patch_info.num_vertices[0];
        m_num_e = m_patch_info.num_edges[0];

        m_s_active_v = Bitmask(m_num_v, shrd_alloc);
        m_s_active_e = Bitmask(m_num_e, shrd_alloc);

        m_s_active_v.reset(block);
        m_s_active_e.reset(block);

        detail::load_async(
            block,
            reinterpret_cast<const char*>(m_patch_info.active_mask_v),
            m_s_active_v.num_bytes(),
            reinterpret_cast<char*>(m_s_active_v.m_bitmask),
            true);

        detail::load_async(
            block,
            reinterpret_cast<const char*>(m_patch_info.active_mask_e),
            m_s_active_e.num_bytes(),
            reinterpret_cast<char*>(m_s_active_e.m_bitmask),
            true);

        // to store the vertex valence
        m_s_valence = shrd_alloc.alloc<uint8_t>(m_num_v);

        // allocate double the number of edges to account for the new added
        // edges
        m_s_ev = shrd_alloc.alloc<uint16_t>(2 * 2 * m_num_e);

        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(m_patch_info.ev),
                           2 * m_num_e,
                           m_s_ev,
                           true);

        // mark not-owned vertices as not-active
        uint16_t mask_num_elements = DIVIDE_UP(m_s_active_v.size(), 32);
        for (uint16_t i = threadIdx.x; i < mask_num_elements;
             i += blockThreads) {
            m_s_active_v.m_bitmask[i] =
                m_s_active_v.m_bitmask[i] & m_patch_info.owned_mask_v[i];
        }
        // Note that we do not do the same for edges (i.e., mark not-owned as
        // not active) since a not-owned edge may connect two owned vertices

        block.sync();

        // mask out vertices on the separator (permuted via cross-patch
        // permutation)
        for (int e = threadIdx.x; e < m_num_e; e += blockThreads) {
            if (m_s_active_e(e)) {
                uint16_t v0(m_s_ev[2 * e]), v1(m_s_ev[2 * e + 1]);

                VertexHandle vh0 = m_patch_info.template find<VertexHandle>(v0);
                VertexHandle vh1 = m_patch_info.template find<VertexHandle>(v1);

                if (vh0.patch_id() == m_patch_info.patch_id) {
                    // v0 is owned
                    if (vh0.patch_id() != vh1.patch_id()) {
                        // v0 is owned but v1 is not owned
                        if (vh0.patch_id() < vh1.patch_id()) {
                            // v0 is neighbor to another vertex (v1) with
                            // higher patch index
                            m_s_active_v.reset(v0, true);
                        }
                    }
                }

                if (vh1.patch_id() == m_patch_info.patch_id) {
                    // v1 is owned
                    if (vh0.patch_id() != vh1.patch_id()) {
                        // v1 is owned but v0 is not owned
                        if (vh1.patch_id() < vh0.patch_id()) {
                            // v1 is neighbor to another vertex (v0) with
                            // higher patch index
                            m_s_active_v.reset(v1, true);
                        }
                    }
                }
            }
        }

        block.sync();
    }

    /**
     * @brief permute the 'active' vertices using min deg ordering
     */
    __device__ __inline__ void permute(cooperative_groups::thread_block& block,
                                       VertexAttribute<uint16_t>& v_permute)
    {
        int num_active_v = num_active_vertices(block);

        for (int i = 0; i < num_active_v; ++i) {

            // compute valence
            compute_vertex_valence(block);

            // find the vertex with min valence/degree
            int v_min = min_valence_vertex(block);

            assert(v_min < m_s_active_v.size());

            assert(m_s_active_v(v_min));


#ifndef NDEBUG
            if (threadIdx.x == 0) {
                for (int i = 0; i < m_s_active_v.size(); ++i) {
                    if (m_s_active_v(i)) {
                        assert(m_s_valence[v_min] <= m_s_valence[i]);
                    }
                }
            }
#endif

            // deactivate the vertex
            if (threadIdx.x == 0) {
                m_s_active_v.reset(v_min, true);

                // now we know the permutation of this vertex
                v_permute(VertexHandle(m_patch_info.patch_id, v_min)) =  // i;
                    num_active_v - i - 1;
            }

            // TODO
            //  if this is the not the last vertex to be numbered
            // if (i < num_active_v - 1) {
            //     // update graph
            //     update_mesh(block, v_min);
            // }

            block.sync();
        }

#ifndef NDEBUG
        for (int i = threadIdx.x; i < m_s_active_v.size(); i += blockThreads) {
            assert(!m_s_active_v(i));
        }
#endif
    }

    /**
     * @brief compute the vertex valence as a scatter operation (from edges to
     * vertices) by only accounting for active vertices. An edge is active if
     * both end of the edge are active vertices
     */
    __device__ __inline__ void compute_vertex_valence(
        cooperative_groups::thread_block& block)
    {
        fill_n<blockThreads>(m_s_valence, m_num_v, uint8_t(0));
        block.sync();

        for (uint16_t e = threadIdx.x; e < m_num_e; e += blockThreads) {
            if (m_s_active_e(e)) {
                const uint16_t v0 = m_s_ev[2 * e + 0];
                const uint16_t v1 = m_s_ev[2 * e + 1];

                if (m_s_active_v(v0) && m_s_active_v(v1)) {
                    atomicAdd(m_s_valence + v0, uint8_t(1));
                    atomicAdd(m_s_valence + v1, uint8_t(1));
                    assert(m_s_valence[v0] < 255);
                    assert(m_s_valence[v1] < 255);
                }
            }
        }
        block.sync();
    }


    /**
     * @brief return the vertex id with minimum valence
     */
    template <int item_per_thread = 4>
    __device__ __inline__ uint16_t min_valence_vertex(
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

                    if (m_s_active_v(i) && !m_s_active_v(j)) {
                        // if i is active but j is not
                        return i;
                    }
                    if (m_s_active_v(j) && !m_s_active_v(i)) {
                        // if j is active but i is not
                        return j;
                    }
                    if (m_s_active_v(j) && m_s_active_v(i)) {
                        // if both are active
                        return (m_s_valence[i] < m_s_valence[j]) ? i : j;
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

        const uint16_t mask_num_elements = DIVIDE_UP(m_s_active_v.size(), 32);

        const uint16_t rem_bits = m_s_active_v.size() & 31u;

        const uint32_t last_word_mask =
            (rem_bits == 0) ? 0xFFFFFFFFu : ((1u << rem_bits) - 1u);

        for (uint16_t i = threadIdx.x; i < mask_num_elements;
             i += blockThreads) {

            unsigned int x = m_s_active_v.m_bitmask[i];

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
     * @brief update EV by removing the vertex v_del and add new edges between
     * its 1-ring if there are not already there
     */
    __device__ __inline__ void update_mesh(
        cooperative_groups::thread_block& block,
        uint16_t                          v_del)
    {
        // TODO
    }


   private:
    PatchInfo m_patch_info;
    Bitmask   m_s_active_v;
    Bitmask   m_s_active_e;
    uint32_t  m_num_v, m_num_e;
    uint16_t* m_s_ev;
    uint8_t*  m_s_valence;
};
}  // namespace rxmesh