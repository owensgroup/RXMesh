#pragma once

#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.h"


/**
 * update_step()
 */
template <typename T>
__device__ __forceinline__ T
update_step(const rxmesh::VertexHandle&       v0_id,
            const rxmesh::VertexHandle&       v1_id,
            const rxmesh::VertexHandle&       v2_id,
            const rxmesh::VertexAttribute<T>& geo_distance,
            const rxmesh::VertexAttribute<T>& coords,
            const T                           infinity_val)
{
    using namespace rxmesh;
    const vec3<T> v0 = coords.to_glm<3>(v0_id);
    const vec3<T> v1 = coords.to_glm<3>(v1_id);
    const vec3<T> v2 = coords.to_glm<3>(v2_id);

    const vec3<T> x0 = v1 - v0;
    const vec3<T> x1 = v2 - v0;

    T t0 = geo_distance(v1_id);
    T t1 = geo_distance(v2_id);

    T q00 = glm::dot(x0, x0);
    T q01 = glm::dot(x0, x1);
    T q10 = glm::dot(x1, x0);
    T q11 = glm::dot(x1, x1);


    T det = q00 * q11 - q01 * q10;

    T Q00 = q11 / det;
    T Q01 = -q01 / det;
    T Q10 = -q10 / det;
    T Q11 = q00 / det;

    T delta = t0 * (Q00 + Q10) + t1 * (Q01 + Q11);
    T dis   = delta * delta -
            (Q00 + Q01 + Q10 + Q11) *
                (t0 * t0 * Q00 + t0 * t1 * (Q10 + Q01) + t1 * t1 * Q11 - 1);
    T p = (delta + sqrt(dis)) / (Q00 + Q01 + Q10 + Q11);

    T tp0 = t0 - p;
    T tp1 = t1 - p;

    const vec3<T> n = (x0 * Q00 + x1 * Q10) * tp0 + (x0 * Q01 + x1 * Q11) * tp1;


    T cond0 = glm::dot(x0, n);
    T cond1 = glm::dot(x1, n);


    T c0 = cond0 * Q00 + cond1 * Q01;
    T c1 = cond0 * Q10 + cond1 * Q11;

    if (t0 == infinity_val || t1 == infinity_val || dis < 0 || c0 >= 0 ||
        c1 >= 0) {

        T dp0 = geo_distance(v1_id) + glm::length(x0);
        T dp1 = geo_distance(v2_id) + glm::length(x1);
        if (dp1 < dp0) {
            p = dp1;
        } else {
            p = dp0;
        }
    }
    return p;
}


template <typename T, uint32_t blockThreads>
__global__ static void relax_ptp_rxmesh(
    const rxmesh::Context              context,
    const rxmesh::VertexAttribute<T>   coords,
    rxmesh::VertexAttribute<T>         new_geo_dist,
    const rxmesh::VertexAttribute<T>   old_geo_dist,
    const rxmesh::VertexAttribute<int> toplesets,
    const int                          band_start,
    const int                          band_end,
    int*                               d_error,
    const T                            infinity_val,
    const T                            error_tol)
{
    using namespace rxmesh;

    auto in_active_set = [&](VertexHandle vi) {
        int my_band = toplesets(vi);
        return my_band >= band_start && my_band < band_end;
    };

    auto geo_lambda = [&](VertexHandle& vi, const VertexIterator& iter) {
        const int n = static_cast<int>(iter.size());
        if (n < 2) {
            return;
        }

        int my_band = toplesets(vi);
        if (my_band < band_start || my_band >= band_end) {
            return;
        }

        T current_dist = old_geo_dist(vi);
        T new_dist     = current_dist;

        for (int v = 0; v < n; ++v) {

            const VertexHandle vj = iter[v];
            const VertexHandle vk = iter[(v + 1) % n];

            T dist =
                update_step(vi, vk, vj, old_geo_dist, coords, infinity_val);

            if (dist < new_dist) {
                new_dist = dist;
            }
        }

        new_geo_dist(vi) = new_dist;
        // update our distance
        if (my_band == band_start) {
            T error = fabs(new_dist - current_dist) / current_dist;
            if (error < error_tol) {
                atomicAdd(d_error, 1);
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, geo_lambda, in_active_set, true);
}
