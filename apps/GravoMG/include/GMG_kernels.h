#pragma once

#include "rxmesh/context.h"
#include "rxmesh/matrix/dense_matrix.cuh"
#include "rxmesh/query.cuh"

#include "NeighborHandling.h"

namespace rxmesh {
namespace detail {


__device__ __inline__ void mutex_lock(int* mutex)
{
    assert(mutex);
    __threadfence();
    while (::atomicCAS(mutex, 0, 1) != 0) {
        __threadfence();
    }
    __threadfence();
}

__device__ __inline__ void mutex_unlock(int* mutex)
{
    assert(mutex);
    __threadfence();
    ::atomicExch(mutex, 0);
    __threadfence();
}


template <uint32_t blockThreads>
__global__ static void count_num_neighbor_samples(
    const Context          context,
    const DenseMatrix<int> clustered_vertices,
    DenseMatrix<int>       sample_neighbor_size,
    DenseMatrix<int>       mutex)
{

    auto add_neighbour = [&](VertexHandle v_id, VertexIterator& vv) {
        int b = clustered_vertices(v_id, 0);

        for (int i = 0; i < vv.size(); i++) {

            int a = clustered_vertices(vv[i], 0);

            if (b != a) {

                mutex_lock(&mutex(b));

                // TODO this is not enough. because different v_id could add
                // the same a to b
                sample_neighbor_size(b)++;

                mutex_unlock(&mutex(b));
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, add_neighbour);
}


template <uint32_t blockThreads>
__global__ static void populate_neighbor_samples(
    const Context          context,
    const DenseMatrix<int> clustered_vertices,
    DenseMatrix<int>       sample_neighbor_size,
    const DenseMatrix<int> sample_neighbor_size_prefix,
    DenseMatrix<int>       sample_neighbor,
    DenseMatrix<int>       mutex)
{

    auto add_neighbour = [&](VertexHandle v_id, VertexIterator& vv) {
        int d     = 0;
        int start = sample_neighbor_size_prefix(v_id);

        int b = clustered_vertices(v_id);

        for (int i = 0; i < vv.size(); i++) {
            int a = clustered_vertices(vv[i]);


            if (b != a) {
                // TODO this is gonna create race condition
                sample_neighbor(start + d) = b;
                d++;
            }
        }

        assert(d == sample_neighbor_size(v_id));

        int v_idx = context.linear_id(v_id);

        assert(d == sample_neighbor_size_prefix(v_idx + 1) -
                        sample_neighbor_size_prefix(v_idx));
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, add_neighbour);
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