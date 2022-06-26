#pragma once

#include <assert.h>
#include <stdint.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include "rxmesh/context.h"
#include "rxmesh/kernels/shmem_allocator.cuh"
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
    namespace cg           = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();

    cg::memcpy_async(block, out, in, sizeof(T) * size);

    if (with_wait) {
        cg::wait(block);
    }
}

/**
 * @brief store shared memory into global memory. Optimized for uint16_t but
 * also works okay with uint32_t
 */
template <uint32_t blockThreads, typename T, typename SizeT>
__device__ __forceinline__ void store(const T* in, const SizeT size, T* out)
{
    static_assert(std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t>,
                  "store() only works for uint16_t and uint32_t");

    if constexpr (std::is_same_v<T, uint16_t>) {
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

    if constexpr (std::is_same_v<T, uint32_t>) {
        for (uint32_t i = threadIdx.x; i < size; i += blockThreads) {
            uint32_t a = in[i];
            out[i]     = a;
        }
    }
}


/**
 * @brief load the patch topology based on the requirements of a query operation
 * @tparam op the query operation
 * @param patch_info input patch info
 * @param s_ev where EV will be loaded
 * @param s_fe where FE will be loaded
 * @param with_wait wither to add a sync at the end
 * @return
 */
template <Op op>
__device__ __forceinline__ void load_mesh_async(const PatchInfoV2& patch_info,
                                                ShmemAllocator&    shrd_alloc,
                                                uint16_t*&         s_ev,
                                                uint16_t*&         s_fe,
                                                bool               with_wait)
{
    switch (op) {
        case Op::VV: {
            s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges *
                                              sizeof(uint16_t));
            load_async(reinterpret_cast<uint16_t*>(patch_info.ev),
                       2 * patch_info.num_edges,
                       s_ev,
                       with_wait);
            break;
        }
        case Op::VE: {
            assert(2 * patch_info.num_edges > patch_info.num_vertices);
            s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges *
                                              sizeof(uint16_t));
            load_async(reinterpret_cast<uint16_t*>(patch_info.ev),
                       2 * patch_info.num_edges,
                       s_ev,
                       with_wait);
            break;
        }
        case Op::VF: {
            assert(3 * patch_info.num_faces > patch_info.num_vertices);
            s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces *
                                              sizeof(uint16_t));
            s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges *
                                              sizeof(uint16_t));
            load_async(reinterpret_cast<uint16_t*>(patch_info.fe),
                       3 * patch_info.num_faces,
                       s_fe,
                       false);
            load_async(reinterpret_cast<uint16_t*>(patch_info.ev),
                       2 * patch_info.num_edges,
                       s_ev,
                       with_wait);
            break;
        }
        case Op::FV: {
            // TODO need to revisit this
            s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges *
                                              sizeof(uint16_t));
            s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces *
                                              sizeof(uint16_t));
            load_async(reinterpret_cast<uint16_t*>(patch_info.ev),
                       2 * patch_info.num_edges,
                       s_ev,
                       false);

            load_async(reinterpret_cast<uint16_t*>(patch_info.fe),
                       3 * patch_info.num_faces,
                       s_fe,
                       with_wait);
            break;
        }
        case Op::FE: {
            s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces *
                                              sizeof(uint16_t));
            load_async(reinterpret_cast<uint16_t*>(patch_info.fe),
                       3 * patch_info.num_faces,
                       s_fe,
                       with_wait);
            break;
        }
        case Op::FF: {
            s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces *
                                              sizeof(uint16_t));
            load_async(reinterpret_cast<uint16_t*>(patch_info.fe),
                       3 * patch_info.num_faces,
                       s_fe,
                       with_wait);
            break;
        }
        case Op::EV: {
            s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges *
                                              sizeof(uint16_t));
            load_async(reinterpret_cast<uint16_t*>(patch_info.ev),
                       2 * patch_info.num_edges,
                       s_ev,
                       with_wait);
            break;
        }
        case Op::EF: {
            assert(3 * patch_info.num_faces > patch_info.num_edges);
            s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces *
                                              sizeof(uint16_t));
            load_async(reinterpret_cast<uint16_t*>(patch_info.fe),
                       3 * patch_info.num_faces,
                       s_fe,
                       with_wait);
            break;
        }
        default: {
            assert(1 != 1);
            break;
        }
    }
}

/**
 * @brief Load local id and patch of the not-owned vertices, edges, or faces
 * based on query op.
 * @tparam op the query operation
 * @param patch_info input patch info
 * @param not_owned_local_id output local id
 * @param not_owned_patch output patch id
 * @param num_owned number of owned mesh elements
 * @param with_wait to set a block sync after loading the memory
 */
template <Op op>
__device__ __forceinline__ void load_not_owned_async(
    const PatchInfo& patch_info,
    uint16_t*&       not_owned_local_id,
    uint32_t*&       not_owned_patch,
    uint16_t&        num_owned,
    bool             with_wait)
{
    uint16_t  num_not_owned        = 0;
    uint32_t* g_not_owned_patch    = nullptr;
    uint16_t* g_not_owned_local_id = nullptr;

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

            g_not_owned_patch = patch_info.not_owned_patch_v;
            g_not_owned_local_id =
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_v);
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

            g_not_owned_patch = patch_info.not_owned_patch_e;
            g_not_owned_local_id =
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_e);
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

            g_not_owned_patch = patch_info.not_owned_patch_f;
            g_not_owned_local_id =
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_f);
            break;
        }
        case Op::FV: {
            num_owned     = patch_info.num_owned_vertices;
            num_not_owned = patch_info.num_vertices - num_owned;

            assert(2 * patch_info.num_edges >= (1 + 2) * num_not_owned);

            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);

            g_not_owned_patch = patch_info.not_owned_patch_v;
            g_not_owned_local_id =
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_v);
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

            g_not_owned_patch = patch_info.not_owned_patch_e;
            g_not_owned_local_id =
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_e);
            break;
        }
        case Op::FF: {
            num_owned     = patch_info.num_owned_faces;
            num_not_owned = patch_info.num_faces - num_owned;

            not_owned_local_id =
                reinterpret_cast<uint16_t*>(not_owned_patch + num_not_owned);

            g_not_owned_patch = patch_info.not_owned_patch_f;
            g_not_owned_local_id =
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_f);
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

            g_not_owned_patch = patch_info.not_owned_patch_v;
            g_not_owned_local_id =
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_v);
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

            g_not_owned_patch = patch_info.not_owned_patch_f;
            g_not_owned_local_id =
                reinterpret_cast<uint16_t*>(patch_info.not_owned_id_f);
            break;
        }
        default: {
            assert(1 != 1);
            break;
        }
    }

    load_async(g_not_owned_patch, num_not_owned, not_owned_patch, false);
    load_async(
        g_not_owned_local_id, num_not_owned, not_owned_local_id, with_wait);
}
}  // namespace detail
}  // namespace rxmesh
