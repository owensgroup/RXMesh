#pragma once
#include <cuda_profiler_api.h>

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"

#include "util.cuh"


template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    vertex_smoothing(const rxmesh::Context            context,
                     const rxmesh::VertexAttribute<T> coords,
                     rxmesh::VertexAttribute<T>       new_coords,
                     rxmesh::VertexAttribute<bool>    v_boundary)
{
    // VV to compute vertex sum and normal
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    auto smooth = [&](VertexHandle v_id, VertexIterator& iter) {
        if (iter.size() == 0) {
            return;
        }

        if (v_boundary(v_id) || iter.size() == 0) {
            new_coords(v_id, 0) = coords(v_id, 0);
            new_coords(v_id, 1) = coords(v_id, 1);
            new_coords(v_id, 2) = coords(v_id, 2);
            return;
        }

        const vec3<T> v = coords.to_glm<3>(v_id);

        // compute both vertex normal and the new position
        // the new position is the average of the one-ring
        // while we iterate on the one ring to compute this new position, we
        // also compute the vertex normal
        // finally, we project the new position on the tangent plane of the
        // vertex (old position)

        // this is the last vertex in the one-ring (before r_id)
        VertexHandle q_id = iter.back();

        vec3<T> q = coords.to_glm<3>(q_id);

        vec3<T> new_v(0.0, 0.0, 0.0);
        vec3<T> v_normal(0.0, 0.0, 0.0);

        T w = 0.0;

        for (uint32_t i = 0; i < iter.size(); ++i) {
            // the current one ring vertex
            const VertexHandle r_id = iter[i];

            const vec3<T> r = coords.to_glm<3>(r_id);

            vec3<T> c = glm::cross(q - v, r - v);

            const T area = glm::length(c) / T(2.0);
            w += area;

            if (glm::length2(c) > 1e-6) {
                c = glm::normalize(c);
            }

            const vec3<T> n = c * area;

            v_normal += n;

            new_v += r;

            q_id = r_id;
            q    = r;
        }
        new_v /= T(iter.size());

        assert(w > 0);

        v_normal /= w;

        if (glm::length2(v_normal) < 1e-6) {
            new_v = v;
        } else {
            v_normal = glm::normalize(v_normal);

            new_v = new_v + (glm::dot(v_normal, (v - new_v)) * v_normal);
        }

        assert(!isnan(new_v[0]));
        assert(!isnan(new_v[1]));
        assert(!isnan(new_v[2]));

        new_coords(v_id, 0) = new_v[0];
        new_coords(v_id, 1) = new_v[1];
        new_coords(v_id, 2) = new_v[2];
    };

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, smooth, true);
}

template <typename T>
inline void tangential_relaxation(rxmesh::RXMeshDynamic&         rx,
                                  rxmesh::VertexAttribute<T>*    coords,
                                  rxmesh::VertexAttribute<T>*    new_coords,
                                  rxmesh::VertexAttribute<bool>* v_boundary,
                                  const int num_smooth_iters,
                                  rxmesh::Timers<rxmesh::GPUTimer>& timers)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 384;

    LaunchBox<blockThreads> launch_box;
    rx.update_launch_box({Op::VV},
                         launch_box,
                         (void*)vertex_smoothing<T, blockThreads>,
                         false,
                         true);

    timers.start("SmoothTotal");
    for (int i = 0; i < num_smooth_iters; ++i) {
        vertex_smoothing<T, blockThreads><<<launch_box.blocks,
                                            launch_box.num_threads,
                                            launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *new_coords, *v_boundary);
        std::swap(new_coords, coords);
    }
    timers.stop("SmoothTotal");

    RXMESH_INFO("Relax time {} (ms)", timers.elapsed_millis("SmoothTotal"));
}