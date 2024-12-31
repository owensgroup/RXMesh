#pragma once

#include <assert.h>
#include <stdint.h>

#include "rxmesh/context.h"
#include "rxmesh/iterator.cuh"
#include "rxmesh/query.cuh"

namespace rxmesh {
namespace detail {
template <uint32_t blockThreads, Op op, typename LambdaT>
__global__ static void query_kernel(const Context context,
                                    const bool    oriented,
                                    LambdaT       user_lambda)
{
    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);

    ShmemAllocator shrd_alloc;

    query.dispatch<op>(block, shrd_alloc, user_lambda, oriented);
}
}  // namespace detail
}  // namespace rxmesh