#pragma once

#include "rxmesh/context.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/query.cuh"

#include "rxmesh/matrix/gmg/hashtable.h"

namespace rxmesh {
namespace detail {

template <uint32_t blockThreads>
__global__ static void populate_edge_hashtable_1st_level(
    const Context          context,
    const DenseMatrix<int> vertex_cluster,
    GPUStorage<Edge>       edge_storage)
{

    auto count = [&](EdgeHandle eh, VertexIterator& iter) {
        assert(iter.size() == 2);

        VertexHandle v0 = iter[0];
        VertexHandle v1 = iter[1];

        int v0_sample = vertex_cluster(v0);
        int v1_sample = vertex_cluster(v1);

        if (v0_sample != v1_sample) {
            Edge e(v0_sample, v1_sample);
            bool inserted = edge_storage.insert(e);
            assert(inserted);
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

// Compute barycentric coordinates for closest point on triangle
__device__ __inline__ void compute_positive_barycentric_coords(
    const Eigen::Vector3f& p,
    const Eigen::Vector3f& a,
    const Eigen::Vector3f& b,
    const Eigen::Vector3f& c,
    Eigen::Vector3f&       barycentricCoords)
{
    // Edge vectors
    Eigen::Vector3f ab = b - a;
    Eigen::Vector3f ac = c - a;
    Eigen::Vector3f ap = p - a;

    float d1 = ab.dot(ap);
    float d2 = ac.dot(ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        barycentricCoords = Eigen::Vector3f(1, 0, 0);  // a
        return;
    }

    // Check if P in vertex region outside B
    Eigen::Vector3f bp = p - b;
    float           d3 = ab.dot(bp);
    float           d4 = ac.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) {
        barycentricCoords = Eigen::Vector3f(0, 1, 0);  // b
        return;
    }

    // Check if P in edge region of AB
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v           = d1 / (d1 - d3);
        barycentricCoords = Eigen::Vector3f(1 - v, v, 0);
        return;
    }

    // Check if P in vertex region outside C
    Eigen::Vector3f cp = p - c;
    float           d5 = ab.dot(cp);
    float           d6 = ac.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) {
        barycentricCoords = Eigen::Vector3f(0, 0, 1);  // c
        return;
    }

    // Check if P in edge region of AC
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w           = d2 / (d2 - d6);
        barycentricCoords = Eigen::Vector3f(1 - w, 0, w);
        return;
    }

    // Check if P in edge region of BC
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w           = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        barycentricCoords = Eigen::Vector3f(0, 1 - w, w);
        return;
    }

    // P inside face region � compute barycentrics via projection
    float denom = ab.dot(ab) * ac.dot(ac) - ab.dot(ac) * ab.dot(ac);
    float v     = (ap.dot(ab) * ac.dot(ac) - ap.dot(ac) * ab.dot(ac)) / denom;
    float w     = (ap.dot(ac) * ab.dot(ab) - ap.dot(ab) * ab.dot(ac)) / denom;
    float u     = 1.0f - v - w;
    barycentricCoords = Eigen::Vector3f(u, v, w);
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