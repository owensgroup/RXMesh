#pragma once

#include <assert.h>
#include <stdint.h>

#include "rxmesh/kernels/rxmesh_query_dispatcher.cuh"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_context.h"
#include "rxmesh/rxmesh_iterator.cuh"
#include "rxmesh/rxmesh_types.h"

/**
 * query()
 */
template <rxmesh::Op op, uint32_t blockThreads>
__launch_bounds__(blockThreads) __global__
    static void query(const rxmesh::RXMeshContext       context,
                      rxmesh::RXMeshAttribute<uint32_t> d_src,
                      rxmesh::RXMeshAttribute<uint32_t> output_container,
                      const bool                        oriented = false)
{
    using namespace rxmesh;

    static_assert(op != Op::EE, "Op::EE is not supported!");

    assert(output_container.is_device_allocated());

    auto store_lambda = [&](uint32_t id, RXMeshIterator& iter) {
        assert(iter.size() < output_container.get_num_attributes());

        d_src(id)               = id;
        output_container(id, 0) = iter.size();

        for (uint32_t i = 0; i < iter.size(); ++i) {
            output_container(id, i + 1) = iter[i];
        }
    };

    query_block_dispatcher<op, blockThreads>(context, store_lambda, oriented);
}

template <uint32_t   blockThreads,
          rxmesh::Op op,
          typename InputHandleT,
          typename OutputHandleT,
          typename InputAttributeT,
          typename OutputAttributeT>
__launch_bounds__(blockThreads) __global__
    static void query_kernel(const rxmesh::RXMeshContext context,
                             InputAttributeT             input,
                             OutputAttributeT            output,
                             const bool                  oriented = false)
{
    using namespace rxmesh;

    auto store_lambda = [&](InputHandleT&                    id,
                            RXMeshIteratorV1<OutputHandleT>& iter) {
        input(id) = id;

        for (uint32_t i = 0; i < iter.size(); ++i) {
            output(id, i) = iter[i];
        }
    };

    query_block_dispatcher_v1<op, blockThreads>(
        context, store_lambda, oriented);
}