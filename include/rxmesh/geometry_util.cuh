#pragma once
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "rxmesh/types.h"

namespace rxmesh {

/**
 * @brief Compute the signed volume of a tetrahedron.
 */
template <typename T>
__inline__ __device__ __host__ T signed_volume(const vec3<T>& x0,
                                               const vec3<T>& x1,
                                               const vec3<T>& x2,
                                               const vec3<T>& x3)
{
    // Equivalent to triple(x1-x0, x2-x0, x3-x0), six times the signed volume of
    // the tetrahedron. But, for robustness, we want the result (up to sign) to
    // be independent of the ordering. And want it as accurate as possible..
    // But all that stuff is hard, so let's just use the common assumption that
    // all coordinates are >0, and do something reasonably accurate in fp.

    // This formula does almost four times too much multiplication, but if the
    // coordinates are non-negative it suffers in a minimal way from
    // cancellation error.
    return (x0[0] * (x1[1] * x3[2] + x3[1] * x2[2] + x2[1] * x1[2]) +
            x1[0] * (x2[1] * x3[2] + x3[1] * x0[2] + x0[1] * x2[2]) +
            x2[0] * (x3[1] * x1[2] + x1[1] * x0[2] + x0[1] * x3[2]) +
            x3[0] * (x1[1] * x2[2] + x2[1] * x0[2] + x0[1] * x1[2]))

           - (x0[0] * (x2[1] * x3[2] + x3[1] * x1[2] + x1[1] * x2[2]) +
              x1[0] * (x3[1] * x2[2] + x2[1] * x0[2] + x0[1] * x3[2]) +
              x2[0] * (x1[1] * x3[2] + x3[1] * x0[2] + x0[1] * x1[2]) +
              x3[0] * (x2[1] * x1[2] + x1[1] * x0[2] + x0[1] * x2[2]));
}

template <typename T>
__inline__ __device__ __host__ vec3<T> tri_normal(const vec3<T>& p0,
                                                  const vec3<T>& p1,
                                                  const vec3<T>& p2)
{
    const vec3<T> u = p1 - p0;
    const vec3<T> v = p2 - p0;
    return glm::normalize(glm::cross(u, v));
};


template <typename T>
__inline__ __device__ __host__ T tri_area(const vec3<T>& p0,
                                          const vec3<T>& p1,
                                          const vec3<T>& p2)
{
    const vec3<T> u = p1 - p0;
    const vec3<T> v = p2 - p0;
    return T(0.5) * glm::length(glm::cross(u, v));
};

/**
 * @brief return the angle at c
 */
template <typename T>
__inline__ __device__ __host__ T tri_angle(const vec3<T>& l,
                                           const vec3<T>& c,
                                           const vec3<T>& r)
{
    vec3<T> ll = glm::normalize(l - c);
    vec3<T> rr = glm::normalize(r - c);
    return glm::acos(glm::dot(rr, ll));
};


template <typename T>
__inline__ __device__ __host__ void triangle_angles(const vec3<T>& a,
                                                    const vec3<T>& b,
                                                    const vec3<T>& c,
                                                    T&             angle_a,
                                                    T&             angle_b,
                                                    T&             angle_c)
{
    angle_a = tri_angle(b, a, c);
    angle_b = tri_angle(c, b, a);
    angle_c = tri_angle(a, c, b);
};


template <typename T>
__inline__ __device__ __host__ void triangle_min_max_angle(const vec3<T>& a,
                                                           const vec3<T>& b,
                                                           const vec3<T>& c,
                                                           T& min_angle,
                                                           T& max_angle)
{
    T angle_a, angle_b, angle_c;
    triangle_angles(a, b, c, angle_a, angle_b, angle_c);
    min_angle = std::min(angle_a, angle_b);
    min_angle = std::min(min_angle, angle_c);

    max_angle = std::max(angle_a, angle_b);
    max_angle = std::max(max_angle, angle_c);
};

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
 * compute partial Voronoi area of the center vertex that is associated with the
 * triangle p->q->r (oriented ccw)
 */
template <typename T>
__host__ __device__ __forceinline__ T
partial_voronoi_area(const vec3<T>& p,  // center
                     const vec3<T>& q,  // before center
                     const vec3<T>& r)  // after center

{
    // Edge vector p->q
    const vec3<T> pq = q - p;

    // Edge vector q->r
    const vec3<T> qr = r - q;

    // Edge vector p->r
    const vec3<T> pr = r - p;

    // compute and check triangle area
    T triangle_area = tri_area(p, q, r);

    if (triangle_area <= std::numeric_limits<T>::min()) {
        return -1;
    }


    // dot products for each corner (of its two emanating edge vectors)
    T dotp = glm::dot(pq, pr);
    T dotq = -glm::dot(qr, pq);
    T dotr = glm::dot(qr, pr);
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


        return 0.125 * (glm::length2(pr) * cotq + glm::length2(pq) * cotr);
    }

    return -1;
}

/**
 * Get the edge weight between the two vertices p-r where q and s composes the
 * diamond around p-r
 */
template <typename T>
__host__ __device__ __forceinline__ T edge_cotan_weight(const vec3<T>& p,
                                                        const vec3<T>& r,
                                                        const vec3<T>& q,
                                                        const vec3<T>& s)
{
    auto partial_weight = [&](const vec3<T>& v) -> T {
        const vec3<T> d0 = p - v;
        const vec3<T> d1 = r - v;

        T triangle_area = tri_area(p, r, v);

        if (triangle_area > std::numeric_limits<T>::min()) {
            T cot = glm::dot(d0, d1) / triangle_area;
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
}  // namespace rxmesh