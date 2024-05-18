#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

template <typename T>
using Vec3 = glm::vec<3, T, glm::defaultp>;


/**
 * @brief Compute the signed volume of a tetrahedron.
 */
template <typename T>
__inline__ __device__ T signed_volume(const Vec3<T>& x0,
                                      const Vec3<T>& x1,
                                      const Vec3<T>& x2,
                                      const Vec3<T>& x3)
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
__device__ __inline__ Vec3<T> tri_normal(const Vec3<T>& p0,
                                         const Vec3<T>& p1,
                                         const Vec3<T>& p2)
{
    const Vec3<T> u = p1 - p0;
    const Vec3<T> v = p2 - p0;
    return glm::normalize(glm::cross(u, v));
};


template <typename T>
__device__ __inline__ T tri_area(const Vec3<T>& p0,
                                 const Vec3<T>& p1,
                                 const Vec3<T>& p2)
{
    const Vec3<T> u = p1 - p0;
    const Vec3<T> v = p2 - p0;
    return T(0.5) * glm::length(glm::cross(u, v));
};


template <typename T>
__device__ __inline__ T tri_angle(const Vec3<T>& l,
                                  const Vec3<T>& c,
                                  const Vec3<T>& r)
{
    glm::vec3 ll = glm::normalize(l - c);
    glm::vec3 rr = glm::normalize(r - c);
    return glm::acos(glm::dot(rr, ll));
};


template <typename T>
__device__ __inline__ void triangle_angles(const Vec3<T>& a,
                                           const Vec3<T>& b,
                                           const Vec3<T>& c,
                                           T&             angle_a,
                                           T&             angle_b,
                                           T&             angle_c)
{
    angle_a = tri_angle(b, a, c);
    angle_b = tri_angle(c, b, a);
    angle_c = tri_angle(a, c, b);
};


template <typename T>
__device__ __inline__ void triangle_min_max_angle(const Vec3<T>& a,
                                                  const Vec3<T>& b,
                                                  const Vec3<T>& c,
                                                  T&             min_angle,
                                                  T&             max_angle)
{
    T angle_a, angle_b, angle_c;
    triangle_angles(a, b, c, angle_a, angle_b, angle_c);
    min_angle = std::min(angle_a, angle_b);
    min_angle = std::min(min_angle, angle_c);

    max_angle = std::max(angle_a, angle_b);
    max_angle = std::max(max_angle, angle_c);
};