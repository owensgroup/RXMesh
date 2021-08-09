#pragma once

#include <assert.h>
#include <stdint.h>

#include "rxmesh/kernels/rxmesh_iterator.cuh"
#include "rxmesh/kernels/rxmesh_query_dispatcher.cuh"
#include "rxmesh/rxmesh.h"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_context.h"

/**
 * query()
 */
template <RXMESH::Op op, uint32_t blockThreads>
__launch_bounds__(blockThreads) __global__
    static void query(const RXMESH::RXMeshContext       context,
                      RXMESH::RXMeshAttribute<uint32_t> d_src,
                      RXMESH::RXMeshAttribute<uint32_t> output_container,
                      const bool                        oriented = false)
{
    using namespace RXMESH;

    static_assert(op != Op::EE, "Op::EE is not supported!");

    assert(output_container.is_device_allocated());

    uint32_t block_offset = 0;
    if constexpr (op == Op::EV || op == Op::EF) {
        block_offset = context.get_edge_distribution()[blockIdx.x];
    } else if constexpr (op == Op::FV || op == Op::FE || op == Op::FF) {
        block_offset = context.get_face_distribution()[blockIdx.x];
    } else if constexpr (op == Op::VV || op == Op::VE || op == Op::VF) {
        block_offset = context.get_vertex_distribution()[blockIdx.x];
    }

    auto store_lambda = [&](uint32_t id, RXMeshIterator& iter) {
        assert(iter.size() < output_container.get_num_attribute_per_element());

        uint32_t id_offset = block_offset + iter.local_id();
        d_src(id_offset) = id;

        output_container(id_offset, 0) = iter.size();

        for (uint32_t i = 0; i < iter.size(); ++i) {
            output_container(id_offset, i + 1) = iter[i];
        }
    };

    query_block_dispatcher<op, blockThreads>(context, store_lambda, oriented);
}