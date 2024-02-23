#pragma once
#include <stdint.h>

#include <cooperative_groups.h>

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/patch_info.h"

#include "rxmesh/attribute.h"

#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

#include "rxmesh/kernels/shmem_mutex.cuh"

namespace rxmesh {

// TODO: change the uniform shared memory allocation to per level allocation for
// less shared memory use

template <uint32_t blockThreads>
struct ALIGN(16) PartitionManager
{
    __device__ __inline__ PartitionManager()
        : m_s_ev(nullptr),
          m_s_vwgt(nullptr),
          m_s_vdegree(nullptr),
          m_s_vadjewgt(nullptr),
          m_s_mapping(nullptr),
          m_s_unmapping(nullptr),
          m_s_ewgt(nullptr),
          m_s_num_vertices(nullptr),
          m_s_num_edges(nullptr),
    {
    }

    __device__ __inline__ PartitionManager(
        cooperative_groups::thread_block& block,
        Context&                          context,
        ShmemAllocator&                   shrd_alloc,
        uint16_t                          req_level);


    __device__ __inline__ void PartitionManager<blockThreads>::matching(
        cooperative_groups::thread_block& block,
        rxmesh::ShmemAllocator&           shrd_alloc,
        const rxmesh::PatchInfo&          patch_info,
        // uint16_t*                s_ev,
        // rxmesh::Bitmask&         matched_edges,
        // rxmesh::Bitmask&         matched_vertices,
        // uint16_t num_vertices,
        // uint16_t num_edges,
        uint16_t curr_level);


    __device__ __inline__ uint16_t* num_vertices_at(uint16_t curr_level)
    {
        return m_num_vertices[curr_level];
    }

    __device__ __inline__ uint16_t* num_edges_at(uint16_t curr_level)
    {
        return m_num_edges[curr_level];
    }

    __device__ __inline__ uint16_t* get_ev(uint16_t curr_level)
    {
        return m_s_ev + (2 * m_num_edges * curr_level);
    }

    __device__ __inline__ uint16_t* get_mapping(uint16_t curr_level)
    {
        return m_s_mapping + (m_num_vertices * curr_level);
    }

    __device__ __inline__ Bitmask& get_matched_edges_bitmask(
        uint16_t curr_level)
    {
        return matched_bitmask_arr[curr_level];
    }

    __device__ __inline__ Bitmask& get_matched_vertices_bitmask(
        uint16_t curr_level)
    {
        return matched_bitmask_arr[curr_level];
    }

    __device__ __inline__ Bitmask& get_p0_vertices_bitmask(uint16_t curr_level)
    {
        return m_s_p0_vertices[curr_level];
    }

    __device__ __inline__ Bitmask& get_p1_vertices_bitmask(uint16_t curr_level)
    {
        return m_s_p1_vertices[curr_level];
    }

    __device__ __inline__ Bitmask& get_separator_vertices_bitmask(
        uint16_t curr_level)
    {
        return m_s_separator_vertices[curr_level];
    }

    __device__ __inline__ bool coarsen_owned(rxmesh::LocalEdgeT v_handle,
                                             uint16_t           level)
    {
        assert(level >= 0);

        if (level > 0) {
            return true;
        } else {
            return m_patch_info.is_owned(v_handle);
        }

        assert(1 == 0);
        return true;
    }

    __device__ __inline__ bool coarsen_owned(rxmesh::LocalVertexT e_handle,
                                             uint16_t             level)
    {
        assert(level >= 0);

        if (level > 0) {
            return true;
        } else {
            return m_patch_info.is_owned(e_handle);
        }

        assert(1 == 0);
        return true;
    }


   private:
    static constexpr int max_arr_size = 10;

    __device__ __inline__ std::array<Bitmask, max_arr_size> alloc_bm_arr(
        ShmemAllocator& shrd_alloc,
        uint16_t        req_arr_size,
        uint16_t        bm_size)
    {
        assert(max_arr_size >= req_arr_size);
        std::array<Bitmask, max_arr_size> bm_arr;

        for (int i = 0; i < req_arr_size; ++i) {
            bm_arr[i] = Bitmask(bm_size, shrd_alloc);
            bm_arr[i].reset(block);
        }

        return bm_arr;
    }

    // TODO: use public for variable temporary
   public:
    // The topology information: edge incident vertices and face incident
    // edges from the index in the original patchinfo ev
    uint16_t* m_s_ev;

    // Number of mesh elements in the patch in a array format
    // use 32 if the CUDA function requires
    uint16_t* m_s_num_vertices;
    uint16_t* m_s_num_edges;

    // mask indicate the ownership of the v & e for the base layer, may not need
    // this uint32_t *owned_mask_v, *owned_mask_e;

    // vertex related attribute
    uint16_t* m_s_vwgt;
    uint16_t* m_s_vdegree;
    uint16_t* m_s_vadjewgt;

    // edge related attribute
    uint16_t* m_s_ewgt;

    // array stores the mapping relationship
    uint16_t* m_s_mapping;

    // indicates the number of levels we have, notice right now the level size
    // is the max count
    uint16_t m_s_req_level;

    std::array<Bitmask, max_arr_size> m_s_matched_edges;
    std::array<Bitmask, max_arr_size> m_s_matched_vertices;

    // for partition
    std::array<Bitmask, max_arr_size> m_s_p0_vertices;
    std::array<Bitmask, max_arr_size> m_s_p1_vertices;
    std::array<Bitmask, max_arr_size> m_s_separator_vertices;

    uint16_t* m_s_tmp_ve_offset;
    uint16_t* m_s_tmp_ve_output;

    // basic rxmesh info
    PatchInfo m_patch_info;
    Context   m_context;
};

// TODO: destroyer

// TODO: check all the member variable are initialized
template <uint32_t blockThreads>
__device__ __inline__ PartitionManager<blockThreads>::PartitionManager(
    cooperative_groups::thread_block& block,
    Context&                          context,
    ShmemAllocator&                   shrd_alloc,
    uint16_t                          req_level)
{
    assert(req_level <= max_arr_size);
    m_s_req_level = req_level;

    m_context = context;

    uint32_t s_patch_id;

    __shared__ uint16_t level_count[1];
    m_s_level_count = level_count;

    s_patch_id = blockIdx.x;

    m_patch_info = m_context.m_patches_info[s_patch_id];

    m_s_num_vertices = shrd_alloc.alloc<uint16_t>(req_level);
    m_s_num_edges    = shrd_alloc.alloc<uint16_t>(req_level);

    m_s_num_vertices[0] = m_patch_info.num_vertices;
    m_s_num_edges[0]    = m_patch_info.num_edges;

    const uint16_t req_vertex_cap = m_s_num_vertices[0] * req_level;
    const uint16_t req_edge_cap   = m_s_num_edges[0] * req_level;

    // copy ev to shared memory
    m_s_ev = shrd_alloc.alloc<uint16_t>(2 * req_edge_cap);
    detail::load_async(block,
                       reinterpret_cast<uint16_t*>(m_patch_info.ev),
                       2 * m_s_num_edges[0],
                       m_s_ev,
                       false);

    // alloc shared memory for all vertex attributes
    m_s_vwgt     = shrd_alloc.alloc<uint16_t>(req_vertex_cap);
    m_s_vdegree  = shrd_alloc.alloc<uint16_t>(req_vertex_cap);
    m_s_vadjewgt = shrd_alloc.alloc<uint16_t>(req_vertex_cap);

    // alloc shared memory for edge attrobutes
    m_s_ewgt = shrd_alloc.alloc<uint16_t>(req_edge_cap);

    // edges chosen or vertex chosen
    m_s_matched_vertices =
        alloc_bm_arr(shrd_alloc, req_level, m_s_num_vertices[0]);
    m_s_matched_edges = alloc_bm_arr(shrd_alloc, req_level, m_s_num_edges[0]);

    // partition bitmask
    m_s_p0_vertices = alloc_bm_arr(shrd_alloc, req_level, m_s_num_vertices[0]);
    m_s_p1_vertices = alloc_bm_arr(shrd_alloc, req_level, m_s_num_vertices[0]);
    m_s_separator_vertices =
        alloc_bm_arr(shrd_alloc, req_level, m_s_num_vertices[0]);

    // alloc shared memory for mapping array
    m_s_mapping = shrd_alloc.alloc<uint16_t>(req_vertex_cap);

    // Tmp VE operation array which will be reused for multiple times
    m_s_tmp_ve_offset = shrd_alloc.alloc<uint16_t>(m_s_num_edges[0] * 2);
    m_s_tmp_ve_output = shrd_alloc.alloc<uint16_t>(m_s_num_edges[0] * 2);


    block.sync();
}

template <uint32_t blockThreads>
__device__ __inline__ void PartitionManager<blockThreads>::matching(
    cooperative_groups::thread_block& block,
    rxmesh::ShmemAllocator&           shrd_alloc,
    const rxmesh::PatchInfo&          patch_info,
    // uint16_t*                s_ev,
    // rxmesh::Bitmask&         matched_edges,
    // rxmesh::Bitmask&         matched_vertices,
    // uint16_t num_vertices,
    // uint16_t num_edges,
    uint16_t curr_level)
{
    // Get level by level parameter
    uint16_t  num_vertices     = m_s_num_vertices[curr_level];
    uint16_t  num_edges        = m_s_num_edges[curr_level];
    uint16_t* s_ev             = get_ev(curr_level);
    Bitmask&  matched_edges    = get_matched_edges_bitmask(curr_level);
    Bitmask&  matched_vertices = get_matched_edges_bitmask(curr_level);

    __shared__ uint16_t s_num_active_vertices[1];
    s_num_active_vertices[0] = num_vertices;

    // TODO: use edge priority to replace the id for selecting edges
    uint16_t* s_e_chosen_by_v = shrd_alloc.alloc<uint16_t>(num_vertices);

    rxmesh::Bitmask active_edges = Bitmask(num_edges, shrd_alloc);

    // Copy EV to offset array
    for (uint16_t i = threadIdx.x; i < num_edges * 2; i += blockThreads) {
        uint16_t m_s_tmp_ve_offset[i] = s_ev[i];
    }

    block.sync();

    // Get VE data here to avoid redundant computation
    v_e(num_vertices, num_edges, m_s_tmp_ve_offset, m_s_tmp_ve_output, nullptr);

    // TODO: add descriptions for every lambda
    while (float(s_num_active_vertices[0]) / float(num_vertices) > 0.25) {
        // reset the tmp array
        fill_n<blockThreads>(s_e_chosen_by_v, num_vertices, uint16_t(0));
        block.sync();

        // VE operation
        for (uint16_t v = threadIdx.x; v < num_vertices; v += blockthreads) {
            uint16_t start = m_s_tmp_ve_offset[v];
            uint16_t end   = m_s_tmp_ve_offset[v + 1];

            if (!coarsen_owned(LocalVertexT(v), curr_level, patch_info)) {
                continue;
            }

            uint16_t tgt_e_id = 0;

            // query for one ring edges
            for (uint16_t e = start; e < end; e++) {
                uint16_t edge = m_s_tmp_ve_output[e];

                if (!coarsen_owned(LocalEdgeT(e), curr_level, patch_info)) {
                    continue;
                }

                if (active_edges(e_local_id) && e_local_id > tgt_e_id) {
                    tgt_e_id = e_local_id;
                }
            }

            // TODO: assert memory access
            assert(v < num_vertices);
            s_e_chosen_by_v[v] = tgt_e_id;
        }

        block.sync();

        // EV operation
        for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
            uint16_t v0_local_id = s_ev[2 * e];
            uint16_t v1_local_id = s_ev[2 * e + 1];

            uint16_t v0_chosen_id = s_e_chosen_by_v[v0_local_id];
            uint16_t v1_chosen_id = s_e_chosen_by_v[v1_local_id];

            if (!coarsen_owned(LocalEdgeT(e), curr_level, patch_info)) {
                continue;
            }

            if (local_id == v1_chosen_id && local_id == v0_chosen_id) {
                m_s_matched_vertices.set(v0_local_id, true);
                m_s_matched_vertices.set(v1_local_id, true);
                m_s_matched_edges.set(e, true);
            }
        }

        block.sync();

        // VE operation
        for (uint16_t v = threadIdx.x; v < num_vertices; v += blockthreads) {
            uint16_t start = m_s_tmp_ve_offset[v];
            uint16_t end   = m_s_tmp_ve_offset[v + 1];

            if (!coarsen_owned(LocalEdgeT(v), curr_level, patch_info)) {
                continue;
            }

            if (m_s_matched_vertices(v)) {
                for (uint16_t e = start; e < end; e++) {
                    uint16_t e_local_id = m_s_tmp_ve_output[e];

                    if (coarsen_owned(
                            LocalEdgeT(e_local_id), curr_level, patch_info)) {
                        active_edges.set(e_local_id, false);
                    }
                }

                // count active vertices
                atomicAdd(&s_num_active_vertices[0], 1);
            }
        }

        block.sync();
    }

    //reset the tmp shared mem
    fill_n<blockThreads>(m_s_tmp_ve_offset num_edges * 2, uint16_t(0));
    fill_n<blockThreads>(m_s_tmp_ve_output, num_edges * 2, uint16_t(0));

    shrd_alloc.dealloc<uint16_t>(num_vertices);
    shrd_alloc.dealloc(active_edges.num_bytes());

    // 1. two hop implementation
    //    -> traditional MIS/MM
    // 2. admed implementation
    //    -> priority function pi
    //    -> CAS to resolve conflict
    //    ->
    // 3. the kamesh parallel HEM implementation
    //    ->
    //    ->
}

template <uint32_t blockThreads>
__device__ __inline__ void coarsening(cooperative_groups::thread_block& block,
                                      rxmesh::ShmemAllocator&  shrd_alloc,
                                      const rxmesh::PatchInfo& patch_info,
                                    //   uint16_t*                s_ev,
                                    //   uint16_t*                s_mapping,
                                    //   rxmesh::Bitmask&         matched_edges,
                                    //   rxmesh::Bitmask&         matched_vertices,
                                    //   uint16_t                 num_vertices,
                                    //   uint16_t                 num_edges,
                                      uint16_t                 curr_level)
{
    // Get level by level param
    uint16_t  num_vertices     = m_s_num_vertices[curr_level];
    uint16_t  num_edges        = m_s_num_edges[curr_level];
    uint16_t* s_ev             = get_ev(curr_level);
    uint16_t* s_mapping        = get_mapping(curr_level);
    Bitmask&  matched_edges    = get_matched_edges_bitmask(curr_level);
    Bitmask&  matched_vertices = get_matched_edges_bitmask(curr_level);

    // EV operation
    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];

        if (matched_edges(local_id)) {
            assert(matched_vertices(v0_local_id));
            assert(matched_vertices(v1_local_id));

            uint16_t coarse_id =
                v0_local_id < v1_local_id ? v0_local_id : v1_local_id;
            s_mapping[v0_local_id] = coarse_id;
            s_mapping[v1_local_id] = coarse_id;
        } else {
            if (!matched_vertices(v0_local_id)) {
                atomicCAS(&s_mapping[v0_local_id], INVALID16, v0_local_id);
            }

            if (!matched_vertices(v1_local_id)) {
                atomicCAS(&s_mapping[v1_local_id], INVALID16, v1_local_id);
            }
        }
    }

    block.sync();

    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];

        uint16_t v0_coarse_id =
            s_mapping[v0_local_id] < s_mapping[v1_local_id] ?
                s_mapping[v0_local_id] :
                s_mapping[v1_local_id];
        uint16_t v1_coarse_id =
            s_mapping[v0_local_id] < s_mapping[v1_local_id] ?
                s_mapping[v0_local_id] :
                s_mapping[v1_local_id];

        uint16_t tmp_coarse_edge_id =
            v0_coarse_id * num_vertices + v1_coarse_id;

        // TODO: needs extra mapping array for uncoarsening, may be solved just
        // using bitmask
        // TODO: sort and reduction for tmp_coarse_edge_id
        uint16_t coarse_edge_id = tmp_coarse_edge_id;

        atomicCAS(&s_ev[2 * coarse_edge_id + 0], 0, v0_coarse_id);
        atomicCAS(&s_ev[2 * coarse_edge_id + 1], 0, v1_coarse_id);
    }

    block.sync();
}

template <uint32_t blockThreads>
__device__ __inline__ void partition(cooperative_groups::thread_block& block,
                                     rxmesh::Context&                  context,
                                     rxmesh::ShmemAllocator& shrd_alloc)
{
    // TODO: use the active bitmask for non-continuous v_id
    // TODO: check the size indicating

    // bi_assignment_ggp(
    //     /*cooperative_groups::thread_block& */ block,
    //     /* const uint16_t                   */ num_vertices,
    //     /* const Bitmask&                   */ s_owned_v,
    //     /* const Bitmask&                   */ s_active_v,
    //     /* const uint16_t*                  */ m_s_vv_offset,
    //     /* const uint16_t*                  */ m_s_vv,
    //     /* Bitmask&                         */ s_assigned_v,
    //     /* Bitmask&                         */ s_current_frontier_v,
    //     /* Bitmask&                         */ s_next_frontier_v,
    //     /* Bitmask&                         */ s_partition_a_v,
    //     /* Bitmask&                         */ s_partition_b_v,
    //     /* int                              */ num_iter);
}

template <uint32_t blockThreads>
__device__ __inline__ void uncoarsening(cooperative_groups::thread_block& block,
                                        rxmesh::ShmemAllocator& shrd_alloc,
                                        uint16_t*               s_ev,
                                        uint16_t*               s_mapping,
                                        rxmesh::Bitmask&        matched_edges,
                                        rxmesh::Bitmask& matched_vertices,
                                        rxmesh::Bitmask& s_partition_a_v,
                                        rxmesh::Bitmask& s_partition_b_v,
                                        rxmesh::Bitmask& s_separator_v,
                                        rxmesh::Bitmask& s_coarse_partition_a_v,
                                        rxmesh::Bitmask& s_coarse_partition_b_v,
                                        rxmesh::Bitmask& s_coarse_separator_v,
                                        uint16_t         num_vertices,
                                        uint16_t         num_edges)
{
    // TODO: make this a coarsen manager
    // TODO: all the calculation for shared mem in one place
    uint16_t* s_ve_offset = shrd_alloc.alloc<uint16_t>(num_edges * 2);
    uint16_t* s_ve_output = shrd_alloc.alloc<uint16_t>(num_edges * 2);

    // Copy EV to offset array
    for (uint16_t i = threadIdx.x; i < num_edges * 2; i += blockThreads) {
        uint16_t s_ve_offset[i] = s_ev[i];
    }

    block.sync();

    // Get VE data here to avoid redundant computation
    v_e(num_vertices, num_edges, s_ve_offset, s_ve_output, nullptr);

    for (uint16_t v = threadIdx.x; v < num_vertices; v += blockthreads) {
        uint16_t start = s_ve_offset[v];
        uint16_t end   = s_ve_offset[v + 1];

        s_partition_a_v(v) = s_coarse_partition_a_v(s_mapping(v));
        s_partition_b_v(v) = s_coarse_partition_b_v(s_mapping(v));
        s_separator_v(v)   = s_coarse_separator_v(s_mapping(v));
    }

    block.sync();
}

}  // namespace rxmesh
