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
            s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges);
            load_async(reinterpret_cast<uint16_t*>(patch_info.ev),
                       2 * patch_info.num_edges,
                       s_ev,
                       with_wait);
            break;
        }
        case Op::VE: {
            assert(2 * patch_info.num_edges > patch_info.num_vertices);
            s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges);
            load_async(reinterpret_cast<uint16_t*>(patch_info.ev),
                       2 * patch_info.num_edges,
                       s_ev,
                       with_wait);
            break;
        }
        case Op::VF: {
            assert(3 * patch_info.num_faces > patch_info.num_vertices);
            s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces);
            s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges);
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
            s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges);
            s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces);
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
            s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces);
            load_async(reinterpret_cast<uint16_t*>(patch_info.fe),
                       3 * patch_info.num_faces,
                       s_fe,
                       with_wait);
            break;
        }
        case Op::FF: {
            s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces);
            load_async(reinterpret_cast<uint16_t*>(patch_info.fe),
                       3 * patch_info.num_faces,
                       s_fe,
                       with_wait);
            break;
        }
        case Op::EV: {
            s_ev = shrd_alloc.alloc<uint16_t>(2 * patch_info.num_edges);
            load_async(reinterpret_cast<uint16_t*>(patch_info.ev),
                       2 * patch_info.num_edges,
                       s_ev,
                       with_wait);
            break;
        }
        case Op::EF: {
            assert(3 * patch_info.num_faces > patch_info.num_edges);
            s_fe = shrd_alloc.alloc<uint16_t>(3 * patch_info.num_faces);
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

}  // namespace detail
}  // namespace rxmesh
