#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"

#include "rxmesh/bitmask.cuh"
#include "rxmesh/query.cuh"

namespace rxmesh {

namespace detail {
template <uint32_t blockThreads, typename T>
__global__ void identify_boundary_vertices(const Context      context,
                                           VertexAttribute<T> boundary_v)
{
    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);

    ShmemAllocator shrd_alloc;

    Bitmask bd_e(query.get_patch_info().num_edges[0], shrd_alloc);
    bd_e.reset(block);
    block.sync();

    auto boundary_edges = [&](EdgeHandle& e_id, const FaceIterator& iter) {
        if (iter.size() < 2) {
            bd_e.set(e_id.local_id(), true);
        }
    };

    query.dispatch<Op::EF>(block, shrd_alloc, boundary_edges);

    block.sync();


    auto boundary_vertices = [&](EdgeHandle& e_id, const VertexIterator& iter) {
        if (bd_e(e_id.local_id())) {
            boundary_v(iter[0], 0) = T(1);
            boundary_v(iter[1], 0) = T(1);
        }
    };

    query.dispatch<Op::EV>(block, shrd_alloc, boundary_vertices);
}
}  // namespace detail


}  // namespace rxmesh