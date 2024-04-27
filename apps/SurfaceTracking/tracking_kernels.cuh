#pragma once
#include "rxmesh/rxmesh_dynamic.h"

template <typename T>
constexpr __inline__ __device__ T
noise_gen(const FlowNoise3<T>& noise, T x, T y, T z)
{
    return noise(z - 203.994, x + 169.47, y - 205.31);
}

template <typename T>
__inline__ __device__ Vec3<T> potential(const FlowNoise3<T>& noise,
                                        T                    x,
                                        T                    y,
                                        T                    z)
{


    constexpr T height_factor = 0.5;

    const Vec3<T> centre(0.0, 1.0, 0.0);
    const T       radius = 4.0;

    T sx = x / noise.noise_lengthscale;
    T sy = y / noise.noise_lengthscale;
    T sz = z / noise.noise_lengthscale;

    Vec3<T> psi_i(0.f, 0.f, noise_gen(noise, sx, sy, sz));

    T dist  = glm::length(Vec3<T>(x, y, z) - centre);
    T scale = std::max((radius - dist) / radius, T(0.0));
    psi_i *= scale;

    Vec3<T> psi(0, 0, 0);

    psi += height_factor * noise.noise_gain * psi_i;

    return psi;
}

template <typename T>
__inline__ __device__ void get_velocity(const FlowNoise3<T>& noise,
                                        const Vec3<T>&       x,
                                        Vec3<T>&             v)
{
    const T delta_x = noise.delta_x;

    v[0] = ((potential(noise, x[0], x[1] + delta_x, x[2])[2] -
             potential(noise, x[0], x[1] - delta_x, x[2])[2]) -
            (potential(noise, x[0], x[1], x[2] + delta_x)[1] -
             potential(noise, x[0], x[1], x[2] - delta_x)[1])) /
           (2 * delta_x);
    v[1] = ((potential(noise, x[0], x[1], x[2] + delta_x)[0] -
             potential(noise, x[0], x[1], x[2] - delta_x)[0]) -
            (potential(noise, x[0] + delta_x, x[1], x[2])[2] -
             potential(noise, x[0] - delta_x, x[1], x[2])[2])) /
           (2 * delta_x);
    v[2] = ((potential(noise, x[0] + delta_x, x[1], x[2])[1] -
             potential(noise, x[0] - delta_x, x[1], x[2])[1]) -
            (potential(noise, x[0], x[1] + delta_x, x[2])[0] -
             potential(noise, x[0], x[1] - delta_x, x[2])[0])) /
           (2 * delta_x);
}

template <typename T>
void curl_noise_predicate_new_position(rxmesh::RXMeshDynamic&      rx,
                                       const FlowNoise3<T>&        noise,
                                       rxmesh::VertexAttribute<T>& position,
                                       T                           current_t,
                                       T                           adaptive_dt)
{
    using namespace rxmesh;

    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle vh) {
        const Vec3<T> p(position(vh, 0), position(vh, 1), position(vh, 2));

        Vec3<T> v;

        get_velocity(noise, p, v);
        Vec3<T> k1 = adaptive_dt * v;

        get_velocity(noise, p + T(0.5) * k1, v);
        Vec3<T> k2 = adaptive_dt * v;

        get_velocity(noise, p + T(0.5) * k2, v);
        Vec3<T> k3 = adaptive_dt * v;


        get_velocity(noise, p + T(0.5) * k3, v);
        Vec3<T> k4 = adaptive_dt * v;

        const Vec3<T> new_p =
            p + T(1.0) / T(6.0) * (k1 + k4) + T(1.0) / T(3.0) * (k2 + k3);

        position(vh, 0) = new_p[0];
        position(vh, 1) = new_p[1];
        position(vh, 2) = new_p[2];
    });
}