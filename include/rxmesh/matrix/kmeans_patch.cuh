#pragma once

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/rxmesh_queries.cuh"
#include "rxmesh/kernels/util.cuh"

#include "rxmesh/matrix/mgnd_permute.cuh"

namespace rxmesh {

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
        m_s_bipart       = Bitmask(m_num_v, shrd_alloc);


        m_s_active_v.reset(block);
        m_s_cur_active_v.reset(block);
        m_s_bipart.reset(block);


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
    VV        m_s_vv_cur;
    Bitmask   m_s_active_v;  // active vertices in the patch (minus not-owned)
    Bitmask   m_s_cur_active_v;
    Bitmask   m_s_bipart;  // the bipartition assignment
    uint32_t  m_num_v, m_num_e;
};
}  // namespace rxmesh