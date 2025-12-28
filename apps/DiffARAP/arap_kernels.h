#pragma once
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/svd3_cuda.h"

/**
 * @brief
 */
template <typename T, uint32_t blockThreads>
__global__ static void calc_edge_weights_mat(
    const rxmesh::Context            context,
    const rxmesh::VertexAttribute<T> coords,
    rxmesh::SparseMatrix<T>          edge_weights)
{

    using namespace rxmesh;

    auto calc_weights = [&](EdgeHandle edge_id, VertexIterator& vv) {
        // the edge goes from p-r while the q and s are the opposite vertices
        const VertexHandle p_id = vv[0];
        const VertexHandle r_id = vv[2];
        const VertexHandle q_id = vv[1];
        const VertexHandle s_id = vv[3];

        if (!p_id.is_valid() || !r_id.is_valid() || !q_id.is_valid() ||
            !s_id.is_valid()) {
            return;
        }

        const vec3<T> p = coords.to_glm<3>(p_id);
        const vec3<T> r = coords.to_glm<3>(r_id);
        const vec3<T> q = coords.to_glm<3>(q_id);
        const vec3<T> s = coords.to_glm<3>(s_id);

        // cotans[(v1, v2)] =np.dot(e1, e2) / np.linalg.norm(np.cross(e1, e2))

        T weight = 0;
        // if (q_id.is_valid())
        weight += dot((p - q), (r - q)) / length(cross(p - q, r - q));

        // if (s_id.is_valid())
        weight += dot((p - s), (r - s)) / length(cross(p - s, r - s));

        weight /= 2;
        weight = std::max(0.f, weight);

        edge_weights(p_id, r_id) = weight;
        edge_weights(r_id, p_id) = weight;
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, calc_weights);
}


/**
 * @brief
 */
template <typename T, uint32_t blockThreads>
__global__ static void calculate_rotation_matrix(
    const rxmesh::Context            context,
    const rxmesh::VertexAttribute<T> P,
    const rxmesh::VertexAttribute<T> P_prime,
    rxmesh::VertexAttribute<T>       rotations,
    const rxmesh::SparseMatrix<T>    weight_matrix)
{
    using namespace rxmesh;

    auto cal_rot = [&](VertexHandle v_id, VertexIterator& vv) {
        Eigen::Matrix3f S = Eigen::Matrix3f::Zero();

        for (int j = 0; j < vv.size(); j++) {

            float w = weight_matrix(v_id, vv[j]);

            Eigen::Vector<float, 3> pi_vector = {P(v_id, 0) - P(vv[j], 0),
                                                 P(v_id, 1) - P(vv[j], 1),
                                                 P(v_id, 2) - P(vv[j], 2)};


            Eigen::Vector<float, 3> pi_dash_vector = {
                P_prime(v_id, 0) - P_prime(vv[j], 0),
                P_prime(v_id, 1) - P_prime(vv[j], 1),
                P_prime(v_id, 2) - P_prime(vv[j], 2)};

            S = S + w * pi_vector * pi_dash_vector.transpose();
        }

        Eigen::Matrix3f U;         // left singular vectors
        Eigen::Matrix3f V;         // right singular vectors
        Eigen::Vector3f sing_val;  // singular values

        svd(S, U, sing_val, V);

        const float smallest_singular_value = sing_val.minCoeff();


        Eigen::Matrix3f R = V * U.transpose();

        if (R.determinant() < 0) {
            U.col(smallest_singular_value) =
                U.col(smallest_singular_value) * -1;
            R = V * U.transpose();
        }

        // Matrix R to vector attribute R

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                rotations(v_id, i * 3 + j) = R(i, j);
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, cal_rot);
}