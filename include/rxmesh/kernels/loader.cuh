#pragma once

#include <assert.h>
#include <stdint.h>
#include "rxmesh/context.h"
#include "rxmesh/local.h"
#include "rxmesh/types.h"

namespace rxmesh {

/**
 * @brief load the patch FE
 * @param patch_info input patch info
 * @param patch_faces output FE
 * @return
 */
template <uint32_t blockThreads>
__device__ __forceinline__ void load_patch_FE(const PatchInfo& patch_info,
                                              LocalEdgeT*      fe)
{
    const uint32_t  size     = patch_info.num_faces * 3;
    const uint32_t  size32   = size / 2;
    const uint32_t  reminder = size % 2;
    const uint32_t* input_fe32 =
        reinterpret_cast<const uint32_t*>(patch_info.fe);
    uint32_t* output_fe32 = reinterpret_cast<uint32_t*>(fe);
    //#pragma unroll 3
    for (uint32_t i = threadIdx.x; i < size32; i += blockThreads) {
        uint32_t a     = input_fe32[i];
        output_fe32[i] = a;
    }

    if (reminder != 0) {
        if (threadIdx.x == 0) {
            fe[size - 1] = patch_info.fe[size - 1];
        }
    }
}

/**
 * @brief load the patch EV
 * @param patch_info input patch info
 * @param ev output EV
 * @return
 */
template <uint32_t blockThreads>
__device__ __forceinline__ void load_patch_EV(const PatchInfo& patch_info,
                                              LocalVertexT*    ev)
{
    const uint32_t  num_edges = patch_info.num_edges;
    const uint32_t* input_ev32 =
        reinterpret_cast<const uint32_t*>(patch_info.ev);
    uint32_t* output_ev32 = reinterpret_cast<uint32_t*>(ev);
#pragma unroll 2
    for (uint32_t i = threadIdx.x; i < num_edges; i += blockThreads) {
        uint32_t a     = input_ev32[i];
        output_ev32[i] = a;
    }
}

/**
 * @brief load the patch topology i.e., EV and FE
 * @param patch_info input patch info
 * @param load_ev input indicates if we should load EV
 * @param load_fe input indicates if we should load FE
 * @param s_ev where EV will be loaded
 * @param s_fe where FE will be loaded
 * @return
 */
template <uint32_t blockThreads>
__device__ __forceinline__ void load_mesh(const PatchInfo& patch_info,
                                          const bool       load_ev,
                                          const bool       load_fe,
                                          LocalVertexT*&   s_ev,
                                          LocalEdgeT*&     s_fe)
{

    if (load_ev) {
        load_patch_EV<blockThreads>(patch_info, s_ev);
    }
    // load patch faces
    if (load_fe) {
        if (load_ev) {
            // if we loaded the edges, then we need to move where
            // s_fe is pointing at to avoid overwrite
            s_fe =
                reinterpret_cast<LocalEdgeT*>(&s_ev[patch_info.num_edges * 2]);
        }
        load_patch_FE<blockThreads>(patch_info, s_fe);
    }
}

__device__ __forceinline__ void load_not_owned_local_id(
    const uint16_t  num_not_owned,
    uint16_t*       output_not_owned_local_id,
    const uint16_t* input_not_owned_local_id)
{
}

__device__ __forceinline__ void load_not_owned_patch(
    const uint16_t  num_not_owned,
    uint32_t*       output_not_owned_patch,
    const uint32_t* input_not_owned_patch)
{
    
}

/**
 * @brief Load local id and patch of the not-owned verteices, edges, or faces
 * based on query op.
 * @param patch_info input patch info
 * @param not_owned_local_id output local id
 * @param not_owned_patch output patch id
 * @param num_not_owned number of not-owned mesh elements
 */
template <Op op, uint32_t blockThreads>
__device__ __forceinline__ void load_not_owned(const PatchInfo& patch_info,
                                               uint16_t*& not_owned_local_id,
                                               uint32_t*& not_owned_patch,
                                               uint16_t&  num_not_owned)
{
    switch (op) {
        case Op::VV:
            num_not_owned =
                patch_info.num_vertices - patch_info.num_owned_vertices;
            not_owned_patch = not_owned_patch + 4 * patch_info.num_edges;
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_v);
            load_not_owned_local_id(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_v));
            break;
        case Op::VE:
            num_not_owned   = patch_info.num_edges - patch_info.num_owned_edges;
            not_owned_patch = not_owned_patch + 4 * patch_info.num_edges;
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_e);
            load_not_owned_local_id(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_e));
            break;
        case Op::VF:
            num_not_owned = patch_info.num_faces - patch_info.num_owned_faces;
            not_owned_patch =
                not_owned_patch + 3 * patch_info.num_faces +
                std::max(3 * patch_info.num_faces, 2 * patch_info.num_edges);
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_f);
            load_not_owned_local_id(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_f));
            break;
        case Op::FV:
            num_not_owned =
                patch_info.num_vertices - patch_info.num_owned_vertices;
            not_owned_patch = not_owned_patch + 3 * patch_info.num_faces;
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_v);
            load_not_owned_local_id(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_v));
            break;
        case Op::FE:
            num_not_owned   = patch_info.num_edges - patch_info.num_owned_edges;
            not_owned_patch = not_owned_patch + 3 * patch_info.num_faces;
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_e);
            load_not_owned_local_id(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_e));
            break;
        case Op::FF:
            break;
        case Op::EV:
            num_not_owned =
                patch_info.num_vertices - patch_info.num_owned_vertices;
            not_owned_patch = not_owned_patch + 2 * patch_info.num_edges;
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_v);
            load_not_owned_local_id(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_v));
            break;
        case Op::EF:
            num_not_owned   = patch_info.num_faces - patch_info.num_owned_faces;
            not_owned_patch = not_owned_patch + 6 * patch_info.num_faces;
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_f);
            load_not_owned_local_id(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_f));
            break;
        default:
            assert(1 != 1);
            break;
    }
}

}  // namespace rxmesh
