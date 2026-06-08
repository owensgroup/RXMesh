#pragma once

#include <assert.h>
#include <stdint.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include "rxmesh/kernels/shmem_allocator.cuh"
#include "rxmesh/local.h"
#include "rxmesh/types.h"
#include "rxmesh/util/util.h"

namespace rxmesh {
namespace detail {

#if defined(__HIP_PLATFORM_AMD__)
// HIP cooperative groups has no memcpy_async/wait. Fall back to a synchronous
// block-strided copy. RXMesh issues several load_async(..., with_wait=false)
// then drains them with one wait(block); wait_for_copy() maps that drain to a
// block barrier, so the copies (issued without a barrier here) are visible after
// it -- preserving the original ordering semantics.
template <typename T, typename SizeT>
__device__ __forceinline__ void block_strided_copy(
    cooperative_groups::thread_block& block,
    const T*                          in,
    const SizeT                       size,
    T*                                out)
{
    for (SizeT i = block.thread_rank(); i < size; i += block.size()) {
        out[i] = in[i];
    }
}
#endif

__device__ __forceinline__ void wait_for_copy(
    cooperative_groups::thread_block& block)
{
#if defined(__HIP_PLATFORM_AMD__)
    block.sync();
#else
    cooperative_groups::wait(block);
#endif
}

template <typename T, typename SizeT>
__device__ __forceinline__ void load_async(
    cooperative_groups::thread_block& block,
    const T*                          in,
    const SizeT                       size,
    T*                                out,
    bool                              with_wait)
{
#if defined(__HIP_PLATFORM_AMD__)
    block_strided_copy(block, in, size, out);
    if (with_wait) {
        block.sync();
    }
#else
    cooperative_groups::memcpy_async(block, out, in, sizeof(T) * size);

    if (with_wait) {
        cooperative_groups::wait(block);
    }
#endif
}


template <typename T, typename SizeT>
__device__ __forceinline__ void load_async(const T*    in,
                                           const SizeT size,
                                           T*          out,
                                           bool        with_wait)
{
    namespace cg           = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();

#if defined(__HIP_PLATFORM_AMD__)
    block_strided_copy(block, in, size, out);
    if (with_wait) {
        block.sync();
    }
#else
    cg::memcpy_async(block, out, in, sizeof(T) * size);

    if (with_wait) {
        cg::wait(block);
    }
#endif
}

/**
 * @brief store shared memory into global memory. Optimized for uint16_t but
 * also works okay for other types
 */
template <uint32_t blockThreads, typename T, typename SizeT>
__device__ __forceinline__ void store(const T* in, const SizeT size, T* out)
{
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
    } else {
        for (uint32_t i = threadIdx.x; i < size; i += blockThreads) {
            T a    = in[i];
            out[i] = a;
        }
    }
}


}  // namespace detail
}  // namespace rxmesh
