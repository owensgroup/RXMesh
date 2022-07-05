#pragma once

#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/kernels/query_dispatcher.cuh"
#include "rxmesh/util/vector.h"
/**
 * gaussian_curvature()
 */
template <typename T, uint32_t blockThreads>
__global__ static void compute_gaussian_curvature(const rxmesh::Context      context,
                                             rxmesh::VertexAttribute<T> coords,
                                             rxmesh::VertexAttribute<T> gcs, 
                                             rxmesh::VertexAttribute<T> amix)
{
    using namespace rxmesh;
    const double PI = 3.1415926535897932384626433832795028841971693993751058209;
    auto gc_lambda = [&](FaceHandle face_id, VertexIterator& fv) {
        // get the face's three vertices coordinates
        Vector<3, T> c0(coords(fv[0], 0), coords(fv[0], 1), coords(fv[0], 2));
        Vector<3, T> c1(coords(fv[1], 0), coords(fv[1], 1), coords(fv[1], 2));
        Vector<3, T> c2(coords(fv[2], 0), coords(fv[2], 1), coords(fv[2], 2));

        // the three edges length 
        Vector<3, T> l(dist2(c0, c1), dist2(c1, c2), dist2(c2, c0));
        printf("cuda lens: %f \n", dist2(c0, c1));
        T s = cross(c1 - c0, c2 - c0).norm();
        Vector<3, T> c(dot(c1 - c0, c2 - c0), 
                       dot(c2 - c1, c0 - c1), 
                       dot(c0 - c2, c1 - c2));
        Vector<3, T> rads(atan2(s, c[0]), 
                          atan2(s, c[1]), 
                          atan2(s, c[2]));

        bool is_ob = false;
        for (int i = 0; i < 3; ++i) {
            if (rads[i] > PI * 0.5) is_ob = true;
        }

        for (uint32_t v = 0; v < 3; ++v) {      // for every vertex in this face
            uint32_t v1    = (v + 1) % 3;
            uint32_t v2    = (v + 2) % 3;

            if (is_ob) {
                if (rads[v] > PI * 0.5) {
                    atomicAdd(&amix(fv[v]), 0.25 * s);
                } else {
                    atomicAdd(&amix(fv[v]), 0.125 * s);
                }
            } else {
                // veronoi region calculation
                atomicAdd(&amix(fv[v]), 0.125 * ( (l[v2]) * (c[v1] / s) 
                                                 + (l[v]) * (c[v2] / s)));
            }
            
            printf("cuda rads: %d \n", fv[v]);
            atomicAdd(&gcs(fv[v]), -rads[v]);
        }
    };

    query_block_dispatcher<Op::FV, blockThreads>(context, gc_lambda);
}