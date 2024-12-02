#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"

#include "Eigen/Dense"

#include "rxmesh/util/svd3_cuda.h"

#include "polyscope/polyscope.h"

using namespace rxmesh;

template <typename T, uint32_t blockThreads>
__global__ static void calc_edge_weights_mat(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<T> coords,
    rxmesh::SparseMatrix<T>    edge_weights)
{

    auto calc_weights = [&](EdgeHandle edge_id, VertexIterator& vv) {
        // the edge goes from p-r while the q and s are the opposite vertices
        const rxmesh::VertexHandle p_id = vv[0];
        const rxmesh::VertexHandle r_id = vv[2];
        const rxmesh::VertexHandle q_id = vv[1];
        const rxmesh::VertexHandle s_id = vv[3];

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


template <typename T, uint32_t blockThreads>
__global__ static void calculate_rotation_matrix(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<T> ref_vertex_pos,
    rxmesh::VertexAttribute<T> deformed_vertex_pos,
    rxmesh::VertexAttribute<T> rotations,
    rxmesh::SparseMatrix<T>    weight_matrix)
{
    auto cal_rot = [&](VertexHandle v_id, VertexIterator& vv) {
        Eigen::Matrix3f S = Eigen::Matrix3f::Zero();

        for (int j = 0; j < vv.size(); j++) {

            float w = weight_matrix(v_id, vv[j]);

            Eigen::Vector<float, 3> pi_vector = {
                ref_vertex_pos(v_id, 0) - ref_vertex_pos(vv[j], 0),
                ref_vertex_pos(v_id, 1) - ref_vertex_pos(vv[j], 1),
                ref_vertex_pos(v_id, 2) - ref_vertex_pos(vv[j], 2)};


            Eigen::Vector<float, 3> pi_dash_vector = {
                deformed_vertex_pos(v_id, 0) - deformed_vertex_pos(vv[j], 0),
                deformed_vertex_pos(v_id, 1) - deformed_vertex_pos(vv[j], 1),
                deformed_vertex_pos(v_id, 2) - deformed_vertex_pos(vv[j], 2)};

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


template <typename T, uint32_t blockThreads>
__global__ static void calculate_b(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<T> ref_vertex_pos,
    rxmesh::VertexAttribute<T> deformed_vertex_pos,
    rxmesh::VertexAttribute<T> rotations,
    rxmesh::SparseMatrix<T>    weight_mat,
    rxmesh::DenseMatrix<T>     b_mat,
    rxmesh::VertexAttribute<T> constraints)
{
    auto init_lambda = [&](VertexHandle v_id, VertexIterator& vv) {
        // variable to store ith entry of b_mat
        Eigen::Vector3f bi(0.0f, 0.0f, 0.0f);

        // get rotation matrix for ith vertex
        Eigen::Matrix3f Ri = Eigen::Matrix3f::Zero(3, 3);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                Ri(i, j) = rotations(v_id, i * 3 + j);
        }

        for (int nei_index = 0; nei_index < vv.size(); nei_index++) {
            // get rotation matrix for neightbor j
            Eigen::Matrix3f Rj = Eigen::Matrix3f::Zero(3, 3);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Rj(i, j) = rotations(vv[nei_index], i * 3 + j);

            // find rotation addition
            Eigen::Matrix3f rot_add = Ri + Rj;
            // find coord difference
            Eigen::Vector3f vert_diff = {
                ref_vertex_pos(v_id, 0) - ref_vertex_pos(vv[nei_index], 0),
                ref_vertex_pos(v_id, 1) - ref_vertex_pos(vv[nei_index], 1),
                ref_vertex_pos(v_id, 2) - ref_vertex_pos(vv[nei_index], 2)};

            // update bi
            bi = bi +
                 0.5 * weight_mat(v_id, vv[nei_index]) * rot_add * vert_diff;
        }

        if (constraints(v_id, 0) == 0) {
            b_mat(v_id, 0) = bi[0];
            b_mat(v_id, 1) = bi[1];
            b_mat(v_id, 2) = bi[2];
        } else {
            b_mat(v_id, 0) = deformed_vertex_pos(v_id, 0);
            b_mat(v_id, 1) = deformed_vertex_pos(v_id, 1);
            b_mat(v_id, 2) = deformed_vertex_pos(v_id, 2);
        }
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, init_lambda);
}


template <typename T, uint32_t blockThreads>
__global__ static void calculate_system_matrix(
    const rxmesh::Context      context,
    rxmesh::SparseMatrix<T>    weight_matrix,
    rxmesh::SparseMatrix<T>    laplace_mat,
    rxmesh::VertexAttribute<T> constraints)

{
    auto calc_mat = [&](VertexHandle v_id, VertexIterator& vv) {
        if (constraints(v_id, 0) == 0) {
            for (int i = 0; i < vv.size(); i++) {
                laplace_mat(v_id, v_id) += weight_matrix(v_id, vv[i]);
                laplace_mat(v_id, vv[i]) -= weight_matrix(v_id, vv[i]);
            }
        } else {
            for (int i = 0; i < vv.size(); i++) {
                laplace_mat(v_id, vv[i]) = 0;
            }
            laplace_mat(v_id, v_id) = 1;
        }
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, calc_mat);
}


int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

    if (!rx.is_closed()) {
        RXMESH_ERROR("Input mesh should be closed without boundaries");
        return EXIT_FAILURE;
    }

    polyscope::view::upDir = polyscope::UpDir::ZUp;

    constexpr uint32_t CUDABlockSize = 256;

    // stays same across computation
    auto ref_vertex_pos = *rx.get_input_vertex_coordinates();

    // deformed vertex position that change every iteration
    auto deformed_vertex_pos = *rx.add_vertex_attribute<float>("deformedV", 3);
    deformed_vertex_pos.copy_from(ref_vertex_pos, DEVICE, DEVICE);

    // deformed vertex position as a matrix (used in the solver)
    std::shared_ptr<DenseMatrix<float>> deformed_vertex_pos_mat =
        deformed_vertex_pos.to_matrix();


    // vertex constraints where
    //  0 means free
    //  1 means user-displaced
    //  2 means fixed
    auto constraints = *rx.add_vertex_attribute<float>("FixedVertices", 1);

    // compute weights
    auto weights = rx.add_edge_attribute<float>("edgeWeights", 1);
    SparseMatrix<float> weight_matrix(rx);
    weight_matrix.reset(0.f, LOCATION_ALL);

    // system matrix
    SparseMatrix<float> laplace_mat(rx);
    laplace_mat.reset(0.f, LOCATION_ALL);

    // rotation matrix as a very attribute where every vertex has 3x3 matrix
    auto rotations = *rx.add_vertex_attribute<float>("RotationMatrix", 9);

    // b-matrix
    DenseMatrix<float> b_mat(rx, rx.get_num_vertices(), 3);
    b_mat.reset(0.f, LOCATION_ALL);

    // obtain cotangent weight matrix
    rxmesh::LaunchBox<CUDABlockSize> lb;
    rx.prepare_launch_box({rxmesh::Op::EVDiamond},
                          lb,
                          (void*)calc_edge_weights_mat<float, CUDABlockSize>);

    calc_edge_weights_mat<float, CUDABlockSize>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), ref_vertex_pos, weight_matrix);

    // set constraints
    const vec3<float> sphere_center(0.1818329, -0.99023, 0.325066);
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
        const vec3<float> p(deformed_vertex_pos(vh, 0),
                            deformed_vertex_pos(vh, 1),
                            deformed_vertex_pos(vh, 2));

        // fix the bottom
        if (p[2] < -0.63) {
            constraints(vh) = 2;
        }

        // move the jaw
        if (glm::distance(p, sphere_center) < 0.1) {
            constraints(vh) = 1;
        }
    });

    // move constraints to the host and add it to Polyscope
    constraints.move(DEVICE, HOST);

#if USE_POLYSCOPE
    rx.get_polyscope_mesh()->addVertexScalarQuantity("constraintsV",
                                                     constraints);
#endif

    // Calculate system matrix
    rx.prepare_launch_box({rxmesh::Op::VV},
                          lb,
                          (void*)calculate_system_matrix<float, CUDABlockSize>);

    calculate_system_matrix<float, CUDABlockSize>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), weight_matrix, laplace_mat, constraints);

    // pre_solve laplace_mat
    laplace_mat.pre_solve(rx, Solver::QR, PermuteMethod::NSTDIS);

    // launch box for rotation matrix calculation
    rxmesh::LaunchBox<CUDABlockSize> lb_rot;
    rx.prepare_launch_box(
        {rxmesh::Op::VV},
        lb_rot,
        (void*)calculate_rotation_matrix<float, CUDABlockSize>);


    // launch box for b matrix calculation
    rxmesh::LaunchBox<CUDABlockSize> lb_b_mat;
    rx.prepare_launch_box(
        {rxmesh::Op::VV}, lb_b_mat, (void*)calculate_b<float, CUDABlockSize>);


    // how many times will arap algorithm run?
    int iterations = 1;

    float       t    = 0;
    bool        flag = false;
    vec3<float> start(0.0f, 0.2f, 0.0f);
    vec3<float> end(0.0f, -0.2f, 0.0f);
    vec3<float> displacement(0.0f, 0.0f, 0.0f);

    auto polyscope_callback = [&]() mutable {
        t += flag ? -0.5f : 0.5f;

        flag = (t < 0 || t > 1.0f) ? !flag : flag;

        displacement = (1 - t) * start + (t)*end;

        // apply user deformation
        rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
            if (constraints(vh) == 1) {
                deformed_vertex_pos(vh, 0) += displacement[0];
                deformed_vertex_pos(vh, 1) += displacement[1];
                deformed_vertex_pos(vh, 2) += displacement[2];
            }
        });


        // process step
        for (int i = 0; i < iterations; i++) {
            // solver for rotation
            calculate_rotation_matrix<float, CUDABlockSize>
                <<<lb_rot.blocks, lb_rot.num_threads, lb_rot.smem_bytes_dyn>>>(
                    rx.get_context(),
                    ref_vertex_pos,
                    deformed_vertex_pos,
                    rotations,
                    weight_matrix);

            // solve for position
            calculate_b<float, CUDABlockSize>
                <<<lb_b_mat.blocks,
                   lb_b_mat.num_threads,
                   lb_b_mat.smem_bytes_dyn>>>(rx.get_context(),
                                              ref_vertex_pos,
                                              deformed_vertex_pos,
                                              rotations,
                                              weight_matrix,
                                              b_mat,
                                              constraints);

            laplace_mat.solve(b_mat, *deformed_vertex_pos_mat);
        }

        // move mat to the host
        deformed_vertex_pos_mat->move(DEVICE, HOST);
        deformed_vertex_pos.from_matrix(deformed_vertex_pos_mat.get());


#if USE_POLYSCOPE
        rx.get_polyscope_mesh()->updateVertexPositions(deformed_vertex_pos);
#endif
    };


#if USE_POLYSCOPE
    polyscope::state::userCallback = polyscope_callback;
    polyscope::show();
#endif

    deformed_vertex_pos_mat->release();
    weight_matrix.release();
    laplace_mat.release();
    b_mat.release();


    return 0;
}