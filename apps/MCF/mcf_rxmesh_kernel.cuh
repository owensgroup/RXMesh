#pragma once

#include "mcf_util.h"
#include "rxmesh/kernels/rxmesh_query_dispatcher.cuh"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_context.h"
#include "rxmesh/util/math.h"
#include "rxmesh/util/vector.h"

/**
 * init_PR()
 */
template <typename T>
__global__ static void init_PR(const uint32_t                   num_vertices,
                               const RXMESH::RXMeshAttribute<T> B,
                               const RXMESH::RXMeshAttribute<T> S,
                               RXMESH::RXMeshAttribute<T>       R,
                               RXMESH::RXMeshAttribute<T>       P)
{
    // r = b-s = b - Ax
    // p= r
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_vertices) {
        R(idx, 0) = B(idx, 0) - S(idx, 0);
        R(idx, 1) = B(idx, 1) - S(idx, 1);
        R(idx, 2) = B(idx, 2) - S(idx, 2);

        P(idx, 0) = R(idx, 0);
        P(idx, 1) = R(idx, 1);
        P(idx, 2) = R(idx, 2);
    }
}

/**
 * edge_cotan_weight()
 */
template <typename T>
__device__ __forceinline__ T
edge_cotan_weight(const uint32_t                    p_id,
                  const uint32_t                    r_id,
                  const uint32_t                    q_id,
                  const uint32_t                    s_id,
                  const RXMESH::RXMeshAttribute<T>& X)
{
    // Get the edge weight between the two vertices p-r where
    // q and s composes the diamond around p-r
    using namespace RXMESH;

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
partial_voronoi_area(const uint32_t                    p_id,  // center
                     const uint32_t                    q_id,  // before center
                     const uint32_t                    r_id,  // after center
                     const RXMESH::RXMeshAttribute<T>& X)
{
    // compute partial Voronoi area of the center vertex that is associated with
    // the triangle p->q->r (oriented ccw)
    using namespace RXMESH;

    const Vector<3, T> p(X(p_id, 0), X(p_id, 1), X(p_id, 2));
    const Vector<3, T> q(X(q_id, 0), X(q_id, 1), X(q_id, 2));
    const Vector<3, T> r(X(r_id, 0), X(r_id, 1), X(r_id, 2));

    return partial_voronoi_area(p, q, r);
}

/**
 * init_B()
 */
template <typename T, uint32_t blockThreads>
__launch_bounds__(blockThreads) __global__
    static void init_B(const RXMESH::RXMeshContext      context,
                       const RXMESH::RXMeshAttribute<T> X,
                       RXMESH::RXMeshAttribute<T>       B,
                       const bool                       use_uniform_laplace)
{
    using namespace RXMESH;

    auto init_lambda = [&](uint32_t p_id, RXMeshIterator& iter) {
        if (use_uniform_laplace) {
            const T valence = static_cast<T>(iter.size());
            B(p_id, 0)      = X(p_id, 0) * valence;
            B(p_id, 1)      = X(p_id, 1) * valence;
            B(p_id, 2)      = X(p_id, 2) * valence;
        } else {

            // using Laplace weights
            T v_weight = 0;

            // this is the last vertex in the one-ring (before r_id)
            uint32_t q_id = iter.back();

            for (uint32_t v = 0; v < iter.size(); ++v) {
                // the current one ring vertex
                uint32_t r_id = iter[v];

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
__launch_bounds__(blockThreads) __global__
    static void mcf_matvec(const RXMESH::RXMeshContext      context,
                           const RXMESH::RXMeshAttribute<T> coords,
                           const RXMESH::RXMeshAttribute<T> in,
                           RXMESH::RXMeshAttribute<T>       out,
                           const bool                       use_uniform_laplace,
                           const T                          time_step)
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
    using namespace RXMESH;

    auto matvec_lambda = [&](uint32_t p_id, RXMeshIterator& iter) {
        T sum_e_weight(0);

        Vector<3, T> x(T(0));

        // vertex weight
        T v_weight(0);

        // this is the last vertex in the one-ring (before r_id)
        uint32_t q_id = iter.back();

        for (uint32_t v = 0; v < iter.size(); ++v) {
            // the current one ring vertex
            uint32_t r_id = iter[v];

            T e_weight = 0;
            if (use_uniform_laplace) {
                e_weight = 1;
            } else {
                // the second vertex in the one ring (after r_id)
                uint32_t s_id = (v == iter.size() - 1) ? iter[0] : iter[v + 1];

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