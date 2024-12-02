#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"

/**
 * gaussian_curvature()
 */
template <typename T, uint32_t blockThreads>
__global__ static void compute_gaussian_curvature(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<T> coords,
    rxmesh::VertexAttribute<T> gcs,
    rxmesh::VertexAttribute<T> amix)
{
    using namespace rxmesh;

    auto gc_lambda = [&](FaceHandle face_id, VertexIterator& fv) {
        // get the face's three vertices coordinates
        const vec3<T> c0 = coords.to_glm<3>(fv[0]);
        const vec3<T> c1 = coords.to_glm<3>(fv[1]);
        const vec3<T> c2 = coords.to_glm<3>(fv[2]);

        // the three edges length
        vec3<T> l(glm::distance2(c0, c1),
                  glm::distance2(c1, c2),
                  glm::distance2(c2, c0));

        T s = glm::length(glm::cross(c1 - c0, c2 - c0));

        vec3<T> c(glm::dot(c1 - c0, c2 - c0),
                  glm::dot(c2 - c1, c0 - c1),
                  glm::dot(c0 - c2, c1 - c2));

        vec3<T> rads(atan2(s, c[0]), atan2(s, c[1]), atan2(s, c[2]));

        bool is_ob = false;
        for (int i = 0; i < 3; ++i) {
            if (rads[i] > PI * 0.5)
                is_ob = true;
        }

        for (uint32_t v = 0; v < 3; ++v) {  // for every vertex in this face
            uint32_t v1 = (v + 1) % 3;
            uint32_t v2 = (v + 2) % 3;

            if (is_ob) {
                if (rads[v] > PI * 0.5) {
                    atomicAdd(&amix(fv[v]), 0.25 * s);
                } else {
                    atomicAdd(&amix(fv[v]), 0.125 * s);
                }
            } else {
                // veronoi region calculation
                atomicAdd(
                    &amix(fv[v]),
                    0.125 * ((l[v2]) * (c[v1] / s) + (l[v]) * (c[v2] / s)));
            }

            atomicAdd(&gcs(fv[v]), -rads[v]);
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, gc_lambda);
}