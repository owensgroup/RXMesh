#pragma once
#include "rxmesh/util/vector.h"

/**
 * clamp_cot()
 */
template <typename T>
__host__ __device__ __forceinline__ void clamp_cot(T& v)
{
    // clamp cotangent values as if angles are in[1, 179]

    const T bound = 19.1;  // 3 degrees
    v             = (v < -bound) ? -bound : ((v > bound) ? bound : v);
}

/**
 * partial_voronoi_area()
 */
template <typename T>
__host__ __device__ __forceinline__ T
partial_voronoi_area(const rxmesh::Vector<3, T>& p,  // center
                     const rxmesh::Vector<3, T>& q,  // before center
                     const rxmesh::Vector<3, T>& r)  // after center

{
    // compute partial Voronoi area of the center vertex that is associated with
    // the triangle p->q->r (oriented ccw)
    using namespace rxmesh;

    // Edge vector p->q
    const Vector<3, T> pq = q - p;

    // Edge vector q->r
    const Vector<3, T> qr = r - q;

    // Edge vector p->r
    const Vector<3, T> pr = r - p;

    // compute and check triangle area
    T triangle_area = cross(pq, pr).norm();

    if (triangle_area <= std::numeric_limits<T>::min()) {
        return -1;
    }


    // dot products for each corner (of its two emanating edge vectors)
    T dotp = dot(pq, pr);
    T dotq = -dot(qr, pq);
    T dotr = dot(qr, pr);
    if (dotp < 0.0) {
        return 0.25 * triangle_area;
    }

    // angle at q or r obtuse
    else if (dotq < 0.0 || dotr < 0.0) {
        return 0.125 * triangle_area;
    }

    // no obtuse angles
    else {
        // cot(angle) = cos(angle)/sin(angle) = dot(A,B)/norm(cross(A,B))
        T cotq = dotq / triangle_area;
        T cotr = dotr / triangle_area;

        // clamp cot(angle) by clamping angle to [1,179]
        clamp_cot(cotq);
        clamp_cot(cotr);

        return 0.125 * (pr.norm2() * cotq + pq.norm2() * cotr);
    }

    return -1;
}

/**
 * edge_cotan_weight()
 */
template <typename T>
__host__ __device__ __forceinline__ T
edge_cotan_weight(const rxmesh::Vector<3, T>& p,
                  const rxmesh::Vector<3, T>& r,
                  const rxmesh::Vector<3, T>& q,
                  const rxmesh::Vector<3, T>& s)
{
    // Get the edge weight between the two vertices p-r where
    // q and s composes the diamond around p-r
    using namespace rxmesh;

    auto partial_weight = [&](const Vector<3, T>& v) -> T {
        const Vector<3, T> d0 = p - v;
        const Vector<3, T> d1 = r - v;

        T triangle_area = cross(d0, d1).norm();
        if (triangle_area > std::numeric_limits<T>::min()) {
            T cot = dot(d0, d1) / triangle_area;
            clamp_cot(cot);
            return cot;
        }
        return T(0.0);
    };

    T eweight = 0.0;
    eweight += partial_weight(q);
    eweight += partial_weight(s);

    assert(!isnan(eweight));
    assert(!isinf(eweight));

    return eweight;
}