#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

template <typename T>
using Vec3 = glm::vec<3, T, glm::defaultp>;


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