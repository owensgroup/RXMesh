#pragma once

#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/util/meta.h"

namespace rxmesh {

template <uint32_t blockThreads, typename predicateT>
__device__ __inline__ void fake_delete_block_dispatcher(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc,
    PatchInfo&                        patch_info,
    uint16_t*                         s_fake_del_v,
    uint16_t*                         s_fake_del_e,
    uint16_t*                         s_fake_del_f,
    const predicateT                  predicate)
{
    // TODO fix the bank conflict
    for (uint16_t v = threadIdx.x; v < patch_info.num_vertices;
         v += blockThreads) {
        s_fake_del_v[v] = INVALID16;
    }

    for (uint16_t e = threadIdx.x; e < patch_info.num_edges;
         e += blockThreads) {
        s_fake_del_e[e] = INVALID16;
    }

    for (uint16_t f = threadIdx.x; f < patch_info.num_faces;
         f += blockThreads) {
        s_fake_del_f[f] = INVALID16;
    }


    for (uint16_t e = threadIdx.x; e < patch_info.num_edges;
         e += blockThreads) {
        if (predicate({patch_info.patch_id, e})) {
            if (::atomicCAS(s_fake_del_e + e, INVALID16, e) == INVALID16) {
            }
        }
    }
}

}  // namespace rxmesh