#pragma once

#include <assert.h>
#include <stdint.h>

#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/iterator.cuh"
#include "rxmesh/kernels/rxmesh_query_dispatcher.cuh"


/**
 * @brief perform 2-ring VV query
 */
template <uint32_t blockThreads, rxmesh::Op op>
__global__ static void higher_query(
    const rxmesh::Context                         context,
    rxmesh::VertexAttribute<rxmesh::VertexHandle> input,
    rxmesh::VertexAttribute<rxmesh::VertexHandle> output)
{
    using namespace rxmesh;

    // the mesh element that this thread is assigned to
    VertexHandle thread_vertex;

    // number of vertices in the first ring
    uint32_t num_vv_1st_ring(0), num_vv(0);

    // computation done on the first ring/level
    // this is similar to the lambda function for query_block_dispatcher()
    auto first_ring_lambda = [&](VertexHandle            id,
                                 Iterator<VertexHandle>& iter) {
        assert(iter.size() < output.get_num_attributes());

        num_vv_1st_ring = iter.size();
        num_vv          = num_vv_1st_ring;

        // record the mesh element that this thread is assigned to
        thread_vertex        = id;
        input(thread_vertex) = thread_vertex;

        for (uint32_t i = 0; i < iter.size(); ++i) {
            output(thread_vertex, i) = iter[i];
        }
    };

    query_block_dispatcher<op, blockThreads>(context, first_ring_lambda);

    uint32_t next_id = 0;
    while (true) {
        VertexHandle next_vertex;

        if (thread_vertex.is_valid() && next_id < num_vv_1st_ring) {
            next_vertex = output(thread_vertex, next_id);
        }

        auto higher_rings_lambda = [&](const VertexHandle&   id,
                                       const VertexIterator& iter) {
            assert(id == next_vertex);

            for (uint32_t i = 0; i < iter.size(); ++i) {
                if (iter[i] != thread_vertex) {

                    // make sure that we don't store duplicate outputs
                    bool duplicate = false;
                    for (uint32_t j = 0; j < num_vv; ++j) {
                        if (iter[i] == output(thread_vertex, j)) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
                        output(thread_vertex, num_vv) = iter[i];
                        num_vv++;
                    }
                }
            }
        };

        higher_query_block_dispatcher<op, blockThreads>(
            context, next_vertex, higher_rings_lambda);

        bool is_done =
            (next_id >= num_vv_1st_ring) || !thread_vertex.is_valid();
        if (__syncthreads_and(is_done)) {
            break;
        }
        next_id++;
    }
}