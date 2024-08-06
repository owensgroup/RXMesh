#pragma once

#include "rxmesh/kernels/debug.cuh"
#include "rxmesh/query.cuh"

#include "link_condition.cuh"


using EdgeStatus = int8_t;
enum : EdgeStatus
{
    UNSEEN = 0,  // means we have not tested it before for e.g., split/flip/col
    SKIP   = 1,  // means we have tested it and it is okay to skip
    UPDATE = 2,  // means we should update it i.e., we have tested it before
    ADDED  = 3,  // means it has been added to during the split/flip/collapse
};


template <typename T, uint32_t blockThreads>
__global__ static void stats_kernel(const rxmesh::Context            context,
                                    const rxmesh::VertexAttribute<T> coords,
                                    rxmesh::EdgeAttribute<T>         edge_len,
                                    rxmesh::VertexAttribute<int> vertex_valence)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    auto compute_edge_len = [&](const EdgeHandle eh, const VertexIterator& ev) {
        const vec3<T> v0(coords(ev[0], 0), coords(ev[0], 1), coords(ev[0], 2));
        const vec3<T> v1(coords(ev[1], 0), coords(ev[1], 1), coords(ev[1], 2));

        T len = glm::distance(v0, v1);

        edge_len(eh) = len;
    };

    Query<blockThreads> query(context);
    query.compute_vertex_valence(block, shrd_alloc);
    query.dispatch<Op::EV>(block, shrd_alloc, compute_edge_len);

    for_each_vertex(query.get_patch_info(), [&](const VertexHandle vh) {
        vertex_valence(vh) = query.vertex_valence(vh);
    });
}

int is_done(const rxmesh::RXMeshDynamic&             rx,
            const rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
            int*                                     d_buffer)
{
    using namespace rxmesh;

    // if there is at least one edge that is UNSEEN, then we are not done yet
    CUDA_ERROR(cudaMemset(d_buffer, 0, sizeof(int)));

    rx.for_each_edge(
        DEVICE,
        [edge_status = *edge_status, d_buffer] __device__(const EdgeHandle eh) {
            if (edge_status(eh) == UNSEEN || edge_status(eh) == UPDATE) {
                ::atomicAdd(d_buffer, 1);
            }
        });

    CUDA_ERROR(cudaDeviceSynchronize());
    return d_buffer[0];
}
