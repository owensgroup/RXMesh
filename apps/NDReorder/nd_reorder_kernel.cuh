#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"
/**
 * initliaze the attribute for the degree related, could be mered with the
 * matching function
 */
template <typename T, uint32_t blockThreads>
__global__ static void init_attribute(const rxmesh::Context context,
                                      rxmesh::VertexAttribute<T> nedges,
                                      rxmesh::VertexAttribute<T> adjwgt,
                                      rxmesh::EdgeAttribute<T>   ewgt)
{
    using namespace rxmesh;

    auto nd_lambda = [&](VertexHandle vh, EdgeIterator& ve_iter) {
        T ewgt_sum = 0;
        for (uint32_t e = 0; v < ve_iter.size(); ++e) {
            ewgt_sum += ewgt(ve_iter[e]);
        }

        nedges(vh) = ve_iter.size();
        adjwgt(vh) = ewgt_sum;
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VE>(block, shrd_alloc, gc_lambda);
}