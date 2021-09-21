#pragma once

#include <assert.h>
#include <stdint.h>

#include "rxmesh/kernels/rxmesh_iterator.cuh"
#include "rxmesh/kernels/rxmesh_query_dispatcher.cuh"
#include "rxmesh/rxmesh.h"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_context.h"

/**
 * higher_query()
 */
template <RXMESH::Op op, uint32_t blockThreads>
__launch_bounds__(blockThreads) __global__
    static void higher_query(const RXMESH::RXMeshContext       context,
                             RXMESH::RXMeshAttribute<uint32_t> d_src,
                             RXMESH::RXMeshAttribute<uint32_t> output_container,
                             const bool                        oriented = false)
{
    using namespace RXMESH;
    uint32_t block_offset = 0;
    if constexpr (op == Op::EV || op == Op::EF) {
        block_offset = context.get_edge_distribution()[blockIdx.x];
    } else if constexpr (op == Op::FV || op == Op::FE || op == Op::FF) {
        block_offset = context.get_face_distribution()[blockIdx.x];
    } else if constexpr (op == Op::VV || op == Op::VE || op == Op::VF) {
        block_offset = context.get_vertex_distribution()[blockIdx.x];
    }

    // the mesh element that this thread is assigned to
    uint32_t thread_element = INVALID32;

    // the location where thread_element will store its output
    uint32_t element_offset;

    // number of vertices in the first ring
    uint32_t num_vv_1st_ring(0), num_vv(0);

    // computation done on the first ring/level
    // this is similar to the lambda function for query_block_dispatcher()
    auto first_level_lambda = [&](uint32_t id, RXMeshIterator& iter) {
        assert(iter.size() < output_container.get_num_attribute_per_element());

        num_vv_1st_ring = iter.size();
        num_vv          = num_vv_1st_ring;

        // record the mesh element that this thread is assigned to
        thread_element = id;
        element_offset = block_offset + iter.local_id();

        d_src(element_offset) = id;

        output_container(element_offset, 0) = iter.size();
        for (uint32_t i = 0; i < iter.size(); ++i) {
            output_container(element_offset, i + 1) = iter[i];
        }
    };


    query_block_dispatcher<op, blockThreads>(
        context, first_level_lambda, oriented);

    uint32_t next_id = 1;
    while (true) {
        uint32_t next_vertex = INVALID32;

        if (thread_element != INVALID32 && next_id <= num_vv_1st_ring) {
            next_vertex = output_container(element_offset, next_id);
        }

        auto second_level_lambda = [&](uint32_t id, RXMeshIterator& iter) {
            assert(id == next_vertex);

            for (uint32_t i = 0; i < iter.size(); ++i) {
                if (iter[i] != thread_element) {

                    // make sure that we don't store duplicate outputs
                    bool duplicate = false;
                    for (uint32_t j = 1; j <= num_vv; ++j) {
                        if (iter[i] == output_container(element_offset, j)) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
                        num_vv++;
                        output_container(element_offset, num_vv) = iter[i];
                    }
                }
            }
        };

        query_block_dispatcher<op, blockThreads>(
            context, next_vertex, second_level_lambda);

        bool is_done =
            (next_id > num_vv_1st_ring) || (thread_element == INVALID32);
        if (__syncthreads_and(is_done)) {
            break;
        }
        next_id++;
    }

    if (thread_element != INVALID32) {
        output_container(element_offset, 0) = num_vv;
    }
}