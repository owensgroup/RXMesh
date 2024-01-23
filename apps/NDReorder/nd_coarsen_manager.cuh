#pragma once
#include <stdint.h>

#include <cooperative_groups.h>

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/patch_info.h"

#include "rxmesh/attribute.h"

#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

#include "rxmesh/kernels/shmem_mutex.cuh"

namespace rxmesh {

// Structure of array design
// use CSR like structure to represent the level
template <uint32_t blockThreads>
struct ALIGN(16) CoarsePatchinfo
{
    __device__ __inline__ CoarsePatchinfo()
        : m_s_ev(nullptr),
          m_s_vwgt(nullptr),
          m_s_vdegree(nullptr),
          m_s_vadjewgt(nullptr),
          m_mapping(nullptr),
          m_unmapping(nullptr),
          m_s_ewgt(nullptr),
          m_s_num_vertices(nullptr),
          m_s_num_edges(nullptr)
    {
    }

    __device__ __inline__ CoarsePatchinfo(
        cooperative_groups::thread_block& block,
        Context&                          context,
        ShmemAllocator&                   shrd_alloc)
    {
        m_context = context;

        __shared__ uint32_t s_patch_id;

        // magic for s_patch_id
        // get a patch
        s_patch_id = m_context.m_patch_scheduler.pop();
#ifdef PROCESS_SINGLE_PATCH
        if (s_patch_id != current_p) {
            s_patch_id = INVALID32;
        }
#endif


        if (s_patch_id != INVALID32) {
            if (m_context.m_patches_info[s_patch_id].patch_id == INVALID32) {
                s_patch_id = INVALID32;
            }
        }

        // try to lock the patch
        if (s_patch_id != INVALID32) {
            bool locked =
                m_context.m_patches_info[s_patch_id].lock.acquire_lock(
                    blockIdx.x);

            if (!locked) {
                // if we can not, we add it again to the queue
                if (threadIdx.x == 0) {
                    bool ret = m_context.m_patch_scheduler.push(s_patch_id);
                    assert(ret);
                }

                // and signal other threads to also exit
                s_patch_id = INVALID32;
            }
        }

        if (s_patch_id != INVALID32) {
            m_s_num_vertices[0] =
                m_context.m_patches_info[s_patch_id].num_vertices[0];
            m_s_num_edges[0] =
                m_context.m_patches_info[s_patch_id].num_edges[0];
        }

        // block.sync();

        if (s_patch_id == INVALID32) {
            return;
        }


        m_patch_info = m_context.m_patches_info[s_patch_id];

        // static shared memory for fixed size variables
        __shared__ uint16_t counts[2];
        m_s_num_vertices = counts + 0;
        m_s_num_edges    = counts + 1;


        // get the capacity for dynamic shared memory allocation
        const uint16_t vert_cap = m_patch_info.vertices_capacity[0];
        const uint16_t edge_cap = m_patch_info.edges_capacity[0];
        const uint16_t face_cap = m_patch_info.faces_capacity[0];

        // copy ev to shared memory
        m_s_ev = shrd_alloc.alloc<uint16_t>(2 * edge_cap);
        detail::load_async(block,
                           reinterpret_cast<uint16_t*>(m_patch_info.ev),
                           2 * m_s_num_edges[0],
                           m_s_ev,
                           false);

        // alloc shared memory for all vertex attributes
        m_s_vwgt     = shrd_alloc.alloc<uint16_t>(vert_cap);
        m_s_vdegree  = shrd_alloc.alloc<uint16_t>(vert_cap);
        m_s_vadjewgt = shrd_alloc.alloc<uint16_t>(vert_cap);

        // alloc shared memory for edge attrobutes
        m_s_ewgt = shrd_alloc.alloc<uint16_t>(edge_cap);

        // block.sync();
    }

    // The topology information: edge incident vertices and face incident
    // edges from the index in the original patchinfo ev
    uint16_t* m_s_ev;

    // designed capacity at the beginning
    uint16_t *m_s_vertices_capacity, *m_s_edges_capacity, *m_s_faces_capacity;

    // Number of mesh elements in the patch in a array format
    // use 32 if the CUDA function requires
    uint16_t *m_s_num_vertices, *m_s_num_edges;

    // mask indicate the ownership of the v & e for the base layer, may not need
    // this uint32_t *owned_mask_v, *owned_mask_e;

    // vertex related attribute
    uint16_t* m_s_vwgt;
    uint16_t* m_s_vdegree;
    uint16_t* m_s_vadjewgt;

    // a private array owned by every vertex ?
    uint16_t* m_mapping;
    uint16_t* m_unmapping;

    // edge related attribute
    uint16_t* m_s_ewgt;

    // indicates the current level
    uint16_t* level_count;

    // basic rxmesh info
    PatchInfo m_patch_info;
    Context   m_context;
};

}  // namespace rxmesh
