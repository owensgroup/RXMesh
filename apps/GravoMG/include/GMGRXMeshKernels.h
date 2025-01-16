#pragma once
#include "rxmesh/query.cuh"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/util/bitmask_util.h"

using namespace rxmesh;

class VertexNeighbors;

template <typename T, uint32_t blockThreads>
__global__ static void findNumberOfCoarseNeighbors(
    const rxmesh::Context        context,
    rxmesh::VertexAttribute<int> clustered_vertices,
    int*                         number_of_neighbors,
    VertexNeighbors*             vns)
{

    auto cluster = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); i++) {

            if (clustered_vertices(v_id, 0) != clustered_vertices(vv[i], 0)) {
                int a = clustered_vertices(vv[i], 0);
                int b = clustered_vertices(v_id, 0);


                // atomicAdd(&number_of_neighbors[clustered_vertices(v_id,
                // 0)],1);

                vns[b].addNeighbor(a);


                // neighbor adding logic here where we say that b is a neighbor
                // of a
            }
        }
    };
    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, cluster);
}


template <typename T, uint32_t blockThreads>
__global__ static void cluster_points(
    const rxmesh::Context        context,
    rxmesh::VertexAttribute<T>   vertex_pos,
    rxmesh::VertexAttribute<T>   distance,
    rxmesh::VertexAttribute<int> clustered_vertices,
    int*                         flag)
{

    auto cluster = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); i++) {

            float dist =
                sqrtf(powf(vertex_pos(v_id, 0) - vertex_pos(vv[i], 0), 2) +
                      powf(vertex_pos(v_id, 1) - vertex_pos(vv[i], 1), 2) +
                      powf(vertex_pos(v_id, 2) - vertex_pos(vv[i], 2), 2)) +
                distance(vv[i], 0);


            if (dist < distance(v_id, 0) &&
                clustered_vertices(vv[i], 0) != -1) {
                distance(v_id, 0)           = dist;
                *flag                       = 15;
                clustered_vertices(v_id, 0) = clustered_vertices(vv[i], 0);
            }
        }
    };
    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, cluster);
}



template <typename T, uint32_t blockThreads>
__global__ static void sample_points(const rxmesh::Context      context,
                                     rxmesh::VertexAttribute<T> vertex_pos,
                                     rxmesh::VertexAttribute<T> distance,
                                     int*                       flag)
{

    auto sampler = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); i++) {

            float dist =
                sqrtf(powf(vertex_pos(v_id, 0) - vertex_pos(vv[i], 0), 2) +
                      powf(vertex_pos(v_id, 1) - vertex_pos(vv[i], 1), 2) +
                      powf(vertex_pos(v_id, 2) - vertex_pos(vv[i], 2), 2)) +
                distance(vv[i], 0);

            // printf("\nVertex: %u Distance : %f", context.linear_id(v_id),
            // dist);


            if (dist < distance(v_id, 0)) {
                distance(v_id, 0) = dist;
                *flag             = 15;
            }
        }
        // printf("\nFLAG : %d", *flag);
    };
    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, sampler);
}
