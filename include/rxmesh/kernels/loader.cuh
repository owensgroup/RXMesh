#pragma once

#include <assert.h>
#include <stdint.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>//for cuda::aligned_size_t

#include "rxmesh/context.h"
#include "rxmesh/local.h"
#include "rxmesh/types.h"
#include "rxmesh/util/util.h"

namespace rxmesh {
namespace detail {

template <typename T, typename SizeT>
__device__ __inline__ void load_async(const T*    in,
                                      const SizeT size,
                                      T*          out,
                                      bool        with_wait)
{
    namespace cg       = cooperative_groups;
    cg::thread_block g = cg::this_thread_block();

    cg::memcpy_async(
        block,
        out,
        in,
        cuda::aligned_size_t<128>(expand_to_align(sizeof(T) * size)));

    if (with_wait) {
        cg::wait(block);
    }
}

template <uint32_t blockThreads>
__device__ __forceinline__ void load_uint16(const uint16_t* in,
                                            const uint16_t  size,
                                            uint16_t*       out)
{
    const uint32_t  size32   = size / 2;
    const uint32_t  reminder = size % 2;
    const uint32_t* in32     = reinterpret_cast<const uint32_t*>(in);
    uint32_t*       out32    = reinterpret_cast<uint32_t*>(out);

    for (uint32_t i = threadIdx.x; i < size32; i += blockThreads) {
        uint32_t a = in32[i];
        out32[i]   = a;
    }

    if (reminder != 0) {
        if (threadIdx.x == 0) {
            out[size - 1] = in[size - 1];
        }
    }
}


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
    load_uint16<blockThreads>(reinterpret_cast<const uint16_t*>(patch_info.fe),
                              patch_info.num_faces * 3,
                              reinterpret_cast<uint16_t*>(fe));
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
    load_uint16<blockThreads>(reinterpret_cast<const uint16_t*>(patch_info.ev),
                              patch_info.num_edges * 2,
                              reinterpret_cast<uint16_t*>(ev));
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

template <Op op, uint32_t blockThreads>
__device__ __forceinline__ void load_mesh(const PatchInfo& patch_info,
                                          uint16_t*&       s_ev,
                                          uint16_t*&       s_fe,
                                          bool             with_wait)
{
    // assert(s_ev == s_fe);

    switch (op) {
        case Op::VV: {
            break;
        }
        case Op::VE: {
            break;
        }
        case Op::VF: {
            break;
        }
        case Op::FV: {
            break;
        }
        case Op::FE: {
            break;
        }
        case Op::FF: {
            break;
        }
        case Op::EV: {
            break;
        }
        case Op::EF: {
            break;
        }
        default: {
            assert(1 != 1);
            break;
        }
    }
}

template <uint32_t blockThreads>
__device__ __forceinline__ void load_not_owned_local_id(
    const uint16_t  num_not_owned,
    uint16_t*       output_not_owned_local_id,
    const uint16_t* input_not_owned_local_id)
{
    load_uint16<blockThreads>(
        input_not_owned_local_id, num_not_owned, output_not_owned_local_id);
}

template <uint32_t blockThreads>
__device__ __forceinline__ void load_not_owned_patch(
    const uint16_t  num_not_owned,
    uint32_t*       output_not_owned_patch,
    const uint32_t* input_not_owned_patch)
{
    for (uint32_t i = threadIdx.x; i < num_not_owned; i += blockThreads) {
        output_not_owned_patch[i] = input_not_owned_patch[i];
    }
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
                                               uint16_t&  num_owned)
{
    uint32_t num_not_owned = 0;
    switch (op) {
        case Op::VV: {
            num_owned     = patch_info.num_owned_vertices;
            num_not_owned = patch_info.num_vertices - num_owned;

            // should be 4*patch_info.num_edges but VV (offset and values) are
            // stored as uint16_t and not_owned_patch is uint32_t* so we need to
            // shift the pointer only by half this amount
            not_owned_patch = not_owned_patch + 2 * patch_info.num_edges;
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch<blockThreads>(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_v);
            load_not_owned_local_id<blockThreads>(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_v));
            break;
        }
        case Op::VE: {
            num_owned     = patch_info.num_owned_edges;
            num_not_owned = patch_info.num_edges - num_owned;

            // should be 4*patch_info.num_edges but VE (offset and values) are
            // stored as uint16_t and not_owned_patch is uint32_t* so we need to
            // shift the pointer only by half this amount
            not_owned_patch = not_owned_patch + 2 * patch_info.num_edges;
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch<blockThreads>(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_e);
            load_not_owned_local_id<blockThreads>(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_e));
            break;
        }
        case Op::VF: {
            num_owned     = patch_info.num_owned_faces;
            num_not_owned = patch_info.num_faces - num_owned;

            uint32_t shift = DIVIDE_UP(
                3 * patch_info.num_faces + std::max(3 * patch_info.num_faces,
                                                    2 * patch_info.num_edges),
                2);
            not_owned_patch = not_owned_patch + shift;
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch<blockThreads>(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_f);
            load_not_owned_local_id<blockThreads>(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_f));
            break;
        }
        case Op::FV: {
            num_owned     = patch_info.num_owned_vertices;
            num_not_owned = patch_info.num_vertices - num_owned;

            assert(2 * patch_info.num_edges >= (1 + 2) * num_not_owned);
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch<blockThreads>(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_v);
            load_not_owned_local_id<blockThreads>(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_v));
            break;
        }
        case Op::FE: {
            num_owned     = patch_info.num_owned_edges;
            num_not_owned = patch_info.num_edges - num_owned;

            // should be 3*patch_info.num_faces but FE is stored as uint16_t and
            // not_owned_patch is uint32_t* so we need to shift the pointer only
            // by half this amount
            not_owned_patch =
                not_owned_patch + DIVIDE_UP(3 * patch_info.num_faces, 2);
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch<blockThreads>(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_e);
            load_not_owned_local_id<blockThreads>(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_e));
            break;
        }
        case Op::FF: {
            num_owned     = patch_info.num_owned_faces;
            num_not_owned = patch_info.num_faces - num_owned;

            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch<blockThreads>(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_f);
            load_not_owned_local_id<blockThreads>(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_f));
            break;
        }
        case Op::EV: {
            num_owned     = patch_info.num_owned_vertices;
            num_not_owned = patch_info.num_vertices - num_owned;

            // should be 2*patch_info.num_edges but EV is stored as uint16_t and
            // not_owned_patch is uint32_t* so we need to shift the pointer only
            // by num_edges
            not_owned_patch = not_owned_patch + patch_info.num_edges;
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch<blockThreads>(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_v);
            load_not_owned_local_id<blockThreads>(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_v));
            break;
        }
        case Op::EF: {
            num_owned     = patch_info.num_owned_faces;
            num_not_owned = patch_info.num_faces - num_owned;

            // should be 6*patch_info.num_faces but EF (offset and values) are
            // stored as uint16_t and not_owned_patch is uint32_t* so we need to
            // shift the pointer only by half this amount
            not_owned_patch = not_owned_patch + 3 * patch_info.num_faces;
            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);
            load_not_owned_patch<blockThreads>(
                num_not_owned, not_owned_patch, patch_info.not_owned_patch_f);
            load_not_owned_local_id<blockThreads>(
                num_not_owned,
                not_owned_local_id,
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_f));
            break;
        }
        default: {
            assert(1 != 1);
            break;
        }
    }
}
}  // namespace detail
}  // namespace rxmesh
