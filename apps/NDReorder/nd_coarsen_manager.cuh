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
struct ALIGN(16) CoarsePatchinfo
{
    __device__ __inline__ CoarsePatchinfo()
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

    __device__ __inline__ CoarsePatchinfo(
        cooperative_groups::thread_block& block,
        Context&                          context,
        ShmemAllocator&                   shrd_alloc,
        uint16_t                         req_level);

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

    __device__ __inline__ Bitmask& get_matched_edges_bitmask(uint16_t curr_level)
    {
        return matched_bitmask_arr[curr_level];
    }

    __device__ __inline__ Bitmask& get_matched_vertices_bitmask(uint16_t curr_level)
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

    __device__ __inline__ Bitmask& get_frontier_vertices_bitmask(uint16_t curr_level)
    {
        return m_s_frontier_vertices[curr_level];
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
    uint16_t* m_num_vertices;
    uint16_t* m_num_edges;

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
    std::array<Bitmask, max_arr_size> m_s_frontier_vertices;


    // basic rxmesh info
    PatchInfo m_patch_info;
    Context   m_context;
};

//TODO: destroyer

// TODO: check all the member variable are initialized
template <uint32_t blockThreads>
__device__ __inline__ CoarsePatchinfo<blockThreads>::CoarsePatchinfo(
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
    m_s_num_edges = shrd_alloc.alloc<uint16_t>(req_level);

    m_s_num_vertices[0] = m_patch_info.num_vertices;
    m_s_num_edges[0]    = m_patch_info.num_edges;

    const uint16_t req_vertex_cap = m_num_vertices * req_level;
    const uint16_t req_edge_cap   = m_num_edges * req_level;

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
    m_s_matched_edges    = alloc_bm_arr(shrd_alloc, req_level, m_num_edges);
    m_s_matched_vertices = alloc_bm_arr(shrd_alloc, req_level, m_num_vertices);

    // partition bitmask
    m_s_p0_vertices       = alloc_bm_arr(shrd_alloc, req_level, m_num_vertices);
    m_s_p1_vertices       = alloc_bm_arr(shrd_alloc, req_level, m_num_vertices);
    m_s_frontier_vertices = alloc_bm_arr(shrd_alloc, req_level, m_num_vertices);

    // alloc shared memory for mapping array
    m_s_mapping = shrd_alloc.alloc<uint16_t>(req_vertex_cap);


    // TODO: load owned_mask_v from patchinfo to sharedmem and cast it as a
    // sharedmem bitmask, use patch_info.is_owned() for now

    block.sync();
}

}  // namespace rxmesh
