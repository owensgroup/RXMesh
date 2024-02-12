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
          m_s_num_edges(nullptr)
    {
    }

    __device__ __inline__ CoarsePatchinfo(
        cooperative_groups::thread_block& block,
        Context&                          context,
        ShmemAllocator&                   shrd_alloc);

    __device__ __inline__ void init_array()
    {
    }

    // The topology information: edge incident vertices and face incident
    // edges from the index in the original patchinfo ev
    uint16_t* m_s_ev;

    // designed capacity at the beginning
    // uint16_t *m_s_vertices_capacity, *m_s_edges_capacity,
    // *m_s_faces_capacity;

    // Number of mesh elements in the patch in a array format
    // use 32 if the CUDA function requires
    uint16_t *m_s_num_vertices, *m_s_num_edges;

    // mask indicate the ownership of the v & e for the base layer, may not need
    // this uint32_t *owned_mask_v, *owned_mask_e;

    // vertex related attribute
    uint16_t* m_s_vwgt;
    uint16_t* m_s_vdegree;
    uint16_t* m_s_vadjewgt;

    // edge related attribute
    uint16_t* m_s_ewgt;

    // indicated the mapping relationship and unmapping relationship
    uint16_t* m_s_mapping;
    uint16_t* m_s_unmapping;

    // indicates the number of levels we have, notice right now the level size
    // is the max count
    uint16_t* m_s_level_count;

    // bit masks indicating whether the vertex or edges is marked as matched
    Bitmask m_s_active_edges;
    Bitmask m_s_candidate_edges;
    Bitmask m_s_matched_edges;
    Bitmask m_s_matched_vertices;

    // basic rxmesh info
    PatchInfo m_patch_info;
    Context   m_context;
};


template <uint32_t blockThreads>
__device__ __inline__ CoarsePatchinfo<blockThreads>::CoarsePatchinfo(
    cooperative_groups::thread_block& block,
    Context&                          context,
    ShmemAllocator&                   shrd_alloc)
{

    m_context = context;

    __shared__ uint32_t s_patch_id;

    // static shared memory for fixed size variables
    __shared__ uint16_t counts[2];
    m_s_num_vertices = counts + 0;
    m_s_num_edges    = counts + 1;


    __shared__ uint16_t level_count[1];
    m_s_level_count = level_count;

    s_patch_id = blockIdx.x;

    m_patch_info = m_context.m_patches_info[s_patch_id];

    // get the capacity for dynamic shared memory allocation
    // const uint16_t vert_cap = m_patch_info.vertices_capacity[0];
    // const uint16_t edge_cap = m_patch_info.edges_capacity[0];
    // const uint16_t face_cap = m_patch_info.faces_capacity[0];

    const uint16_t max_vertex_cap =
        static_cast<uint16_t>(m_context.m_max_num_vertices[0]);
    const uint16_t max_edge_cap =
        static_cast<uint16_t>(m_context.m_max_num_edges[0]);
    const uint16_t max_face_cap =
        static_cast<uint16_t>(m_context.m_max_num_faces[0]);

    // copy ev to shared memory
    m_s_ev = shrd_alloc.alloc<uint16_t>(2 * max_edge_cap);
    detail::load_async(block,
                       reinterpret_cast<uint16_t*>(m_patch_info.ev),
                       2 * m_s_num_edges[0],
                       m_s_ev,
                       false);

    // alloc shared memory for all vertex attributes
    m_s_vwgt     = shrd_alloc.alloc<uint16_t>(max_vertex_cap);
    m_s_vdegree  = shrd_alloc.alloc<uint16_t>(max_vertex_cap);
    m_s_vadjewgt = shrd_alloc.alloc<uint16_t>(max_vertex_cap);

    // alloc shared memory for edge attrobutes
    m_s_ewgt = shrd_alloc.alloc<uint16_t>(max_edge_cap);


    // edges chosen or vertex chosen
    m_s_active_edges     = Bitmask(max_edge_cap, shrd_alloc);
    m_s_candidate_edges  = Bitmask(max_edge_cap, shrd_alloc);
    m_s_matched_edges    = Bitmask(max_edge_cap, shrd_alloc);
    m_s_matched_vertices = Bitmask(max_vertex_cap, shrd_alloc);

    // alloc shared memory for mapping array
    m_s_mapping   = shrd_alloc.alloc<uint16_t>(max_vertex_cap);
    m_s_unmapping = shrd_alloc.alloc<uint16_t>(2 * max_edge_cap);

    block.sync();
}

}  // namespace rxmesh
