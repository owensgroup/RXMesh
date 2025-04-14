#pragma once

#include "rxmesh/context.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/query.cuh"

#include "NeighborHandling.h"

#include "hashtable.h"

namespace rxmesh {
namespace detail {

template <uint32_t blockThreads>
__global__ static void count_neighbors_1st_level(
    const Context          context,
    const DenseMatrix<int> vertex_cluster,
    GPUHashTable<Edge>     edge_hash_table)
{

    auto count = [&](EdgeHandle eh, VertexIterator& iter) {
        assert(iter.size() == 2);

        VertexHandle v0 = iter[0];
        VertexHandle v1 = iter[1];

        int v0_sample = vertex_cluster(v0);
        int v1_sample = vertex_cluster(v1);

        if (v0_sample != v1_sample) {
            
            int  min_vertex = std::min(v0_sample, v1_sample);
            int  max_vertex = std::max(v0_sample, v1_sample);
            Edge e(min_vertex, max_vertex);
        }
    };


    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, count);
}

template <uint32_t blockThreads>
__global__ static void count_neighbors_nth_level(
    const Context          context,
    const DenseMatrix<int> vertex_cluster,
    const DenseMatrix<int> prv_sample_neighbor_size_prefix,
    const DenseMatrix<int> prv_sample_neighbor,
    GPUHashTable<Edge>     edge_hash_table)
{

    auto count = [&](EdgeHandle eh, VertexIterator& iter) {
        assert(iter.size() == 2);

        VertexHandle v0 = iter[0];
        VertexHandle v1 = iter[1];

        int v0_sample = vertex_cluster(v0);
        int v1_sample = vertex_cluster(v1);

        if (v0_sample != v1_sample) {
            Edge e(v0_sample, v1_sample);
            edge_hash_table.insert(e);
        }
    };


    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, count);
}


__device__ __inline__ void compute_barycentric(const Eigen::Vector3f& v0,
                                               const Eigen::Vector3f& v1,
                                               const Eigen::Vector3f& v2,
                                               const Eigen::Vector3f& p,
                                               float&                 a,
                                               float&                 b,
                                               float&                 c)
{
    // Compute edges of the triangle
    Eigen::Vector3f edge1    = v1 - v0;
    Eigen::Vector3f edge2    = v2 - v0;
    Eigen::Vector3f pointVec = p - v0;

    // Compute normal of the triangle
    Eigen::Vector3f normal = edge1.cross(edge2);
    float area2 = normal.squaredNorm();  // Area of the triangle multiplied by 2

    // Compute barycentric coordinates
    float lambda0 = 0, lambda1 = 0, lambda2 = 0;

    lambda0 = (v1 - p).cross(v2 - p).dot(normal) / area2;
    lambda1 = (v2 - p).cross(v0 - p).dot(normal) / area2;
    lambda2 = (v0 - p).cross(v1 - p).dot(normal) / area2;

    a = lambda0;
    b = lambda1;
    c = lambda2;
}

__device__ __inline__ float projected_distance(const Eigen::Vector3f& v0,
                                               const Eigen::Vector3f& v1,
                                               const Eigen::Vector3f& v2,
                                               const Eigen::Vector3f& p)
{
    // Compute edges of the triangle
    Eigen::Vector3f edge1 = v1 - v0;
    Eigen::Vector3f edge2 = v2 - v0;

    // Compute the triangle normal
    Eigen::Vector3f normal        = edge1.cross(edge2);
    float           normal_length = normal.norm();

    if (normal_length < 1e-6f) {
        return -1.0f;  // Return -1 to indicate an error
    }

    // Normalize the normal
    normal.normalize();

    // Compute vector from point to the triangle vertex
    Eigen::Vector3f point_to_vertex = p - v0;

    // Project the vector onto the normal
    float distance = point_to_vertex.dot(normal);

    // Return the absolute distance
    return std::fabs(distance);
}


}  // namespace detail
}  // namespace rxmesh