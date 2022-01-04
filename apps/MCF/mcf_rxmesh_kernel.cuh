#pragma once

#include "mcf_util.h"
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/kernels/query_dispatcher.cuh"
#include "rxmesh/util/vector.h"

/**
 * edge_cotan_weight()
 */
template <typename T>
__device__ __forceinline__ T
edge_cotan_weight(const rxmesh::VertexHandle&       p_id,
                  const rxmesh::VertexHandle&       r_id,
                  const rxmesh::VertexHandle&       q_id,
                  const rxmesh::VertexHandle&       s_id,
                  const rxmesh::VertexAttribute<T>& X)
{
    // Get the edge weight between the two vertices p-r where
    // q and s composes the diamond around p-r
    using namespace rxmesh;

    const Vector<3, T> p(X(p_id, 0), X(p_id, 1), X(p_id, 2));
    const Vector<3, T> r(X(r_id, 0), X(r_id, 1), X(r_id, 2));
    const Vector<3, T> q(X(q_id, 0), X(q_id, 1), X(q_id, 2));
    const Vector<3, T> s(X(s_id, 0), X(s_id, 1), X(s_id, 2));

    return edge_cotan_weight(p, r, q, s);
}

/**
 * partial_voronoi_area()
 */
template <typename T>
__device__ __forceinline__ T
partial_voronoi_area(const rxmesh::VertexHandle&       p_id,  // center
                     const rxmesh::VertexHandle&       q_id,  // before center
                     const rxmesh::VertexHandle&       r_id,  // after center
                     const rxmesh::VertexAttribute<T>& X)
{
    // compute partial Voronoi area of the center vertex that is associated with
    // the triangle p->q->r (oriented ccw)
    using namespace rxmesh;

    const Vector<3, T> p(X(p_id, 0), X(p_id, 1), X(p_id, 2));
    const Vector<3, T> q(X(q_id, 0), X(q_id, 1), X(q_id, 2));
    const Vector<3, T> r(X(r_id, 0), X(r_id, 1), X(r_id, 2));

    return partial_voronoi_area(p, q, r);
}

/**
 * init_B()
 */
template <typename T, uint32_t blockThreads>
__global__ static void init_B(const rxmesh::Context            context,
                              const rxmesh::VertexAttribute<T> X,
                              rxmesh::VertexAttribute<T>       B,
                              const bool use_uniform_laplace)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& p_id, const VertexIterator& iter) {
        if (use_uniform_laplace) {
            const T valence = static_cast<T>(iter.size());
            B(p_id, 0)      = X(p_id, 0) * valence;
            B(p_id, 1)      = X(p_id, 1) * valence;
            B(p_id, 2)      = X(p_id, 2) * valence;
        } else {

            // using Laplace weights
            T v_weight = 0;

            // this is the last vertex in the one-ring (before r_id)
            VertexHandle q_id = iter.back();

            for (uint32_t v = 0; v < iter.size(); ++v) {
                // the current one ring vertex
                VertexHandle r_id = iter[v];

                T tri_area = partial_voronoi_area(p_id, q_id, r_id, X);

                v_weight += (tri_area > 0) ? tri_area : 0.0;

                q_id = r_id;
            }
            v_weight = 0.5 / v_weight;

            B(p_id, 0) = X(p_id, 0) / v_weight;
            B(p_id, 1) = X(p_id, 1) / v_weight;
            B(p_id, 2) = X(p_id, 2) / v_weight;
        }
    };

    // With uniform Laplacian, we just need the valence, thus we
    // call query_block_dispatcher and set oriented to false
    query_block_dispatcher<Op::VV, blockThreads>(
        context, init_lambda, !use_uniform_laplace);
}

/**
 * mcf_matvec()
 */
template <typename T, uint32_t blockThreads>
__global__ static void rxmesh_matvec(const rxmesh::Context            context,
                                     const rxmesh::VertexAttribute<T> coords,
                                     const rxmesh::VertexAttribute<T> in,
                                     rxmesh::VertexAttribute<T>       out,
                                     const bool use_uniform_laplace,
                                     const T    time_step)
{

    // To compute the vertex cotan weight, we use the following configuration
    // where P is the center vertex we want to compute vertex weight for.
    // Looping over P's one ring should gives q->r->s.
    // P is the vertex that this thread is responsible of
    /*       r
          /  |  \
         /   |   \
        s    |    q
         \   |   /
           \ |  /
             p
    */
    using namespace rxmesh;

    auto matvec_lambda = [&](VertexHandle& p_id, const VertexIterator& iter) {
        T sum_e_weight(0);

        Vector<3, T> x(T(0));

        // vertex weight
        T v_weight(0);

        // this is the last vertex in the one-ring (before r_id)
        VertexHandle q_id = iter.back();

        for (uint32_t v = 0; v < iter.size(); ++v) {
            // the current one ring vertex
            VertexHandle r_id = iter[v];

            T e_weight = 0;
            if (use_uniform_laplace) {
                e_weight = 1;
            } else {
                // the second vertex in the one ring (after r_id)
                VertexHandle s_id =
                    (v == iter.size() - 1) ? iter[0] : iter[v + 1];

                e_weight = edge_cotan_weight(p_id, r_id, q_id, s_id, coords);

                // e_weight = max(0, e_weight) but without branch divergence
                e_weight = (static_cast<T>(e_weight >= 0.0)) * e_weight;
            }

            e_weight *= time_step;
            sum_e_weight += e_weight;

            x[0] -= e_weight * in(r_id, 0);
            x[1] -= e_weight * in(r_id, 1);
            x[2] -= e_weight * in(r_id, 2);


            // compute vertex weight
            if (use_uniform_laplace) {
                ++v_weight;
            } else {
                T tri_area = partial_voronoi_area(p_id, q_id, r_id, coords);
                v_weight += (tri_area > 0) ? tri_area : 0;
                q_id = r_id;
            }
        }

        // Diagonal entry
        if (use_uniform_laplace) {
            v_weight = 1.0 / v_weight;
        } else {
            v_weight = 0.5 / v_weight;
        }

        assert(!isnan(v_weight));
        assert(!isinf(v_weight));

        T diag       = ((1.0 / v_weight) + sum_e_weight);
        out(p_id, 0) = x[0] + diag * in(p_id, 0);
        out(p_id, 1) = x[1] + diag * in(p_id, 1);
        out(p_id, 2) = x[2] + diag * in(p_id, 2);
    };

    // With uniform Laplacian, we just need the valence, thus we
    // call query_block_dispatcher and set oriented to false
    query_block_dispatcher<Op::VV, blockThreads>(
        context, matvec_lambda, !use_uniform_laplace);
}