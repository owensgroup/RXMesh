#pragma once
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"


template <typename T, uint32_t blockThreads>
__global__ static void populate_histogram(
    rxmesh::Context                  context,
    const rxmesh::VertexAttribute<T> coords)
{
    using namespace rxmesh;

    auto edge_len = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        const VertexHandle v0 = iter[0];
        const VertexHandle v1 = iter[1];

        const Vec3<T> p0(coords(v0, 0), coords(v0, 1), coords(v0, 2));
        const Vec3<T> p1(coords(v1, 0), coords(v1, 1), coords(v1, 2));

        T len2 = logf(glm::distance2(p0, p1));
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, edge_len);
}