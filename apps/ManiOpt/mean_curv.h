#pragma once
#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename T>
void compute_mean_curv(RXMeshStatic& rx, VertexAttribute<T>& curv)
{
    auto x = *rx.get_input_vertex_coordinates();

    // mean-curvature normal accumulator
    auto v_Hn = *rx.add_vertex_attribute<T>("v_Hn", 3, DEVICE);

    // mixed area
    auto v_amix = *rx.add_vertex_attribute<T>("v_amix", 1, DEVICE);

    curv.reset(T(0), rxmesh::DEVICE);
    v_Hn.reset(T(0), rxmesh::DEVICE);
    v_amix.reset(T(0), rxmesh::DEVICE);


    rx.run_query_kernel<Op::FV, 256>(
        [=] __device__(FaceHandle face_id, VertexIterator & fv) {
            const vec3<T> p0 = x.to_glm<3>(fv[0]);
            const vec3<T> p1 = x.to_glm<3>(fv[1]);
            const vec3<T> p2 = x.to_glm<3>(fv[2]);


            const vec3<T> e0 = p1 - p0;
            const vec3<T> e1 = p2 - p1;
            const vec3<T> e2 = p0 - p2;

            const vec3<T> n = glm::cross(p1 - p0, p2 - p0);
            const T       s = glm::length(n);  // = 2*Area

            // squared edge lengths
            // l[0]=|p0-p1|^2, l[1]=|p1-p2|^2, l[2]=|p2-p0|^2
            vec3<T> l(glm::distance2(p0, p1),
                      glm::distance2(p1, p2),
                      glm::distance2(p2, p0));

            // c[v] = dot of the two edges incident to vertex v
            vec3<T> c(glm::dot(p1 - p0, p2 - p0),
                      glm::dot(p2 - p1, p0 - p1),
                      glm::dot(p0 - p2, p1 - p2));

            // angles at vertices (still needed for mixed area)
            vec3<T> rads(atan2(s, c[0]), atan2(s, c[1]), atan2(s, c[2]));

            bool is_ob = false;
            for (int i = 0; i < 3; ++i) {
                if (rads[i] > glm::pi<T>() * T(0.5))
                    is_ob = true;
            }

            // cotangents at vertices:
            // cot(theta_v) = cos/sin = c[v] / s   (since s = |cross| =
            // |a||b|sin) guard against degeneracy
            const T inv_s = (s > T(0)) ? (T(1) / s) : T(0);
            vec3<T> cotv(c[0] * inv_s, c[1] * inv_s, c[2] * inv_s);

            for (uint32_t v = 0; v < 3; ++v) {
                uint32_t v1 = (v + 1) % 3;
                uint32_t v2 = (v + 2) % 3;

                if (is_ob) {
                    if (rads[v] > glm::pi<T>() * T(0.5)) {
                        atomicAdd(&v_amix(fv[v]), T(0.25) * s);
                    } else {
                        atomicAdd(&v_amix(fv[v]), T(0.125) * s);
                    }
                } else {
                    atomicAdd(&v_amix(fv[v]),
                              T(0.125) * ((l[v2]) * (c[v1] * inv_s) +
                                          (l[v]) * (c[v2] * inv_s)));
                }
            }

            // Mean-curvature normal accumulation:
            // For edge (i,j), add 0.5 * cot(opposite angle) * (pi - pj) to both
            // endpoints In this triangle, opposite angles: edge (0,1) opposite
            // vertex 2 -> cotv[2] edge (1,2) opposite vertex 0 -> cotv[0] edge
            // (2,0) opposite vertex 1 -> cotv[1]
            //
            // We accumulate per-vertex vector Hn. Use atomicAdd per component.
            auto atomicAddVec3 = [&](VertexHandle vh, const vec3<T>& a) {
                atomicAdd(&v_Hn(vh, 0), a.x);
                atomicAdd(&v_Hn(vh, 1), a.y);
                atomicAdd(&v_Hn(vh, 2), a.z);
            };

            const T w01 = T(0.5) * cotv[2];
            const T w12 = T(0.5) * cotv[0];
            const T w20 = T(0.5) * cotv[1];

            // (p0 - p1) and (p1 - p0)
            atomicAddVec3(fv[0], w01 * (p0 - p1));
            atomicAddVec3(fv[1], w01 * (p1 - p0));

            atomicAddVec3(fv[1], w12 * (p1 - p2));
            atomicAddVec3(fv[2], w12 * (p2 - p1));

            atomicAddVec3(fv[2], w20 * (p2 - p0));
            atomicAddVec3(fv[0], w20 * (p0 - p2));
        });

    rx.for_each_vertex(DEVICE,
                       [curv, v_Hn, v_amix] __device__(const VertexHandle vh) {
                           const T ax = v_Hn(vh, 0);
                           const T ay = v_Hn(vh, 1);
                           const T az = v_Hn(vh, 2);

                           const T hn_norm = sqrt(ax * ax + ay * ay + az * az);
                           const T a       = v_amix(vh, 0);

                           curv(vh, 0) = (a > T(0)) ? (hn_norm / a) : T(0);
                       });
}