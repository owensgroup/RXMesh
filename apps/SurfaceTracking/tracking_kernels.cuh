#pragma once
#include "rxmesh/rxmesh_dynamic.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
template <typename T>
using Vec3 = glm::vec<3, T, glm::defaultp>;

#include "rxmesh/query.cuh"

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


template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    avg_edge_length(const rxmesh::Context            context,
                    const rxmesh::VertexAttribute<T> position,
                    T*                               d_sum_edge_len)
{
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    auto len = [&](EdgeHandle eh, VertexIterator& iter) {
        const Vec3<T> v0(
            position(iter[0], 0), position(iter[0], 1), position(iter[0], 2));

        const Vec3<T> v1(
            position(iter[1], 0), position(iter[1], 1), position(iter[1], 2));

        ::atomicAdd(d_sum_edge_len, glm::distance(v0, v1));
    };

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, len);
}


template <uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    init_boundary_edges_vertices(const rxmesh::Context           context,
                                 rxmesh::EdgeAttribute<int8_t>   is_edge_bd,
                                 rxmesh::VertexAttribute<int8_t> is_vertex_bd)
{
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    auto bd_edges = [&](EdgeHandle eh, FaceIterator& iter) {
        assert(iter.size() == 1 || iter.size() == 2);

        if (iter.size() == 1) {
            is_edge_bd(eh) = 1;
        }
    };

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EF>(block, shrd_alloc, bd_edges);
    block.sync();

    auto bd_vertices = [&](EdgeHandle eh, VertexIterator& iter) {
        assert(iter.size() == 2);

        if (is_edge_bd(eh) == 1) {
            // possible race condition but it is okay since they all write the
            // same value
            is_vertex_bd(iter[0]) = 1;
            is_vertex_bd(iter[1]) = 1;
        }
    };
        
    query.dispatch<Op::EV>(block, shrd_alloc, bd_vertices);
}

template <typename T>
T compute_avg_edge_length(rxmesh::RXMeshDynamic&      rx,
                          rxmesh::VertexAttribute<T>& position)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 512;

    T* d_sum_edge_len;

    CUDA_ERROR(cudaMalloc((void**)&d_sum_edge_len, sizeof(T)));
    CUDA_ERROR(cudaMemset(d_sum_edge_len, 0, sizeof(T)));


    LaunchBox<blockThreads> launch_box;

    rx.update_launch_box(
        {Op::EV}, launch_box, (void*)avg_edge_length<T, blockThreads>, false);

    avg_edge_length<T, blockThreads><<<launch_box.blocks,
                                       launch_box.num_threads,
                                       launch_box.smem_bytes_dyn>>>(
        rx.get_context(), position, d_sum_edge_len);

    T h_sum_edge_len;
    CUDA_ERROR(cudaMemcpy(
        &h_sum_edge_len, d_sum_edge_len, sizeof(T), cudaMemcpyDeviceToHost));

    GPU_FREE(d_sum_edge_len);

    return h_sum_edge_len / rx.get_num_edges();
}


void init_boundary(rxmesh::RXMeshDynamic&           rx,
                   rxmesh::VertexAttribute<int8_t>& is_vertex_bd,
                   rxmesh::EdgeAttribute<int8_t>&   is_edge_bd)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 512;

    LaunchBox<blockThreads> launch_box;

    rx.update_launch_box({Op::EF, Op::EV},
                         launch_box,
                         (void*)init_boundary_edges_vertices<blockThreads>,
                         false);

    init_boundary_edges_vertices<blockThreads><<<launch_box.blocks,
                                                 launch_box.num_threads,
                                                 launch_box.smem_bytes_dyn>>>(
        rx.get_context(), is_edge_bd, is_vertex_bd);

    CUDA_ERROR(cudaDeviceSynchronize());
}