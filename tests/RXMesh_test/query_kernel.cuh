#pragma once

#include <assert.h>
#include <stdint.h>

#include "rxmesh/attribute.h"
#include "rxmesh/context.h"

/**
 * @brief perform query of type of and store the output as well as the
 * corresponding input
 */
template <uint32_t   blockThreads,
          rxmesh::Op op,
          typename InputHandleT,
          typename OutputHandleT,
          typename InputAttributeT,
          typename OutputAttributeT>
__global__ static void query_kernel(
    const __grid_constant__ rxmesh::Context context,
    InputAttributeT                         input,
    OutputAttributeT                        output,
    const bool                              oriented = false)
{
    using namespace rxmesh;

    auto store_lambda = [&](const InputHandleT&            id,
                            const Iterator<OutputHandleT>& iter) {
        input(id) = id;

        for (uint32_t i = 0; i < iter.size(); ++i) {
            output(id, i) = iter[i];
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.template dispatch<op>(
        block,
        shrd_alloc,
        store_lambda,
        [](InputHandleT) { return true; },
        oriented);
}