#include "rxmesh/matrix/cholesky_solver.h"
#include "rxmesh/matrix/gmg_solver.h"
#include "rxmesh/matrix/sparse_matrix.h"
#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/svd3_cuda.h"


using namespace rxmesh;

using ConstraintsT = int;
enum : ConstraintsT
{
    Free   = 0,
    Handle = 1,
    Fixed  = 2,

};

template <typename T>
void calc_edge_weights_mat(const RXMeshStatic&       rx,
                           const VertexAttribute<T>& ref_vertex_pos,
                           SparseMatrix<T>&          weight_matrix)
{
    rx.run_query_kernel<Op::EVDiamond, 256>(
        [=] __device__(const EdgeHandle      edge_id,
                       const VertexIterator& vv) mutable {
            // the edge goes from p-r while the q and s are the opposite
            // vertices
            const VertexHandle p_id = vv[0];
            const VertexHandle r_id = vv[2];
            const VertexHandle q_id = vv[1];
            const VertexHandle s_id = vv[3];

            if (!p_id.is_valid() || !r_id.is_valid() || !q_id.is_valid() ||
                !s_id.is_valid()) {
                return;
            }

            const vec3<T> p = ref_vertex_pos.to_glm<3>(p_id);
            const vec3<T> r = ref_vertex_pos.to_glm<3>(r_id);
            const vec3<T> q = ref_vertex_pos.to_glm<3>(q_id);
            const vec3<T> s = ref_vertex_pos.to_glm<3>(s_id);

            // cotans[(v1, v2)] =np.dot(e1, e2) / np.linalg.norm(np.cross(e1,
            // e2))

            T weight = 0;
            // if (q_id.is_valid())
            weight += dot((p - q), (r - q)) / length(cross(p - q, r - q));

            // if (s_id.is_valid())
            weight += dot((p - s), (r - s)) / length(cross(p - s, r - s));

            weight /= 2;
            weight = std::max(0.f, weight);

            weight_matrix(p_id, r_id) = weight;
            weight_matrix(r_id, p_id) = weight;
        });
}


template <typename T>
void calculate_rotation_matrix(const RXMeshStatic&       rx,
                               const VertexAttribute<T>& ref_vertex_pos,
                               const VertexAttribute<T>& deformed_vertex_pos,
                               const SparseMatrix<T>&    weight_matrix,
                               VertexAttribute<T>&       rotations)
{
    rx.run_query_kernel<Op::VV, 256>([=] __device__(const VertexHandle    v_id,
                                                    const VertexIterator& vv) {
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
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rotations(v_id, i * 3 + j) = R(i, j);
            }
        }
    });
}

template <typename T>
void calculate_b(const RXMeshStatic&                  rx,
                 const VertexAttribute<T>&            ref_vertex_pos,
                 const VertexAttribute<T>&            deformed_vertex_pos,
                 const VertexAttribute<T>&            rotations,
                 const SparseMatrix<T>&               weight_matrix,
                 const VertexAttribute<ConstraintsT>& constraints,
                 DenseMatrix<T>&                      b_mat)
{
    rx.run_query_kernel<Op::VV, 256>([=] __device__(
                                         const VertexHandle    v_id,
                                         const VertexIterator& vv) mutable {
        if (constraints(v_id) == Free) {
            // variable to store ith entry of b_mat
            Eigen::Vector3f bi(0.0f, 0.0f, 0.0f);

            // get rotation matrix for ith vertex
            Eigen::Matrix3f Ri = Eigen::Matrix3f::Zero(3, 3);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++)
                    Ri(i, j) = rotations(v_id, i * 3 + j);
            }

            for (int i = 0; i < vv.size(); i++) {
                // get rotation matrix for neightbor j
                Eigen::Matrix3f Rj = Eigen::Matrix3f::Zero(3, 3);
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        Rj(i, j) = rotations(vv[i], i * 3 + j);

                // find rotation addition
                Eigen::Matrix3f rot_add = Ri + Rj;
                // find coord difference
                Eigen::Vector3f vert_diff = {
                    ref_vertex_pos(v_id, 0) - ref_vertex_pos(vv[i], 0),
                    ref_vertex_pos(v_id, 1) - ref_vertex_pos(vv[i], 1),
                    ref_vertex_pos(v_id, 2) - ref_vertex_pos(vv[i], 2)};

                // update bi
                bi =
                    bi + 0.5 * weight_matrix(v_id, vv[i]) * rot_add * vert_diff;
            }


            // fix the b due to eliminating the constrained vertices
            for (int i = 0; i < vv.size(); i++) {
                if (constraints(vv[i]) != Free) {
                    T w = -weight_matrix(v_id, vv[i]);
                    for (int j = 0; j < 3; ++j) {
                        bi[j] -= w * deformed_vertex_pos(vv[i], j);
                    }
                }
            }

            for (int j = 0; j < 3; ++j) {
                b_mat(v_id, j) = bi[j];
            }

        } else {
            for (int j = 0; j < 3; ++j) {
                b_mat(v_id, j) = deformed_vertex_pos(v_id, j);
            }
        }
    });
}


template <typename T>
void calculate_system_matrix(const RXMeshStatic&                  rx,
                             const SparseMatrix<T>&               weight_matrix,
                             const VertexAttribute<ConstraintsT>& constraints,
                             SparseMatrix<T>&                     laplace_mat)
{
    rx.run_query_kernel<Op::VV, 256>(
        [=] __device__(const VertexHandle    v_id,
                       const VertexIterator& vv) mutable {
            if (constraints(v_id) == Free) {
                for (int i = 0; i < vv.size(); i++) {
                    laplace_mat(v_id, v_id) += weight_matrix(v_id, vv[i]);
                    if (constraints(vv[i]) == Free) {
                        laplace_mat(v_id, vv[i]) -= weight_matrix(v_id, vv[i]);
                    }
                }
            } else {
                for (int i = 0; i < vv.size(); i++) {
                    laplace_mat(v_id, vv[i]) = 0;
                }
                laplace_mat(v_id, v_id) = 1;
            }
        });
}

template <typename T>
void set_contraints(const RXMeshStatic&            rx,
                    const VertexAttribute<T>&      vertex_pos,
                    VertexAttribute<ConstraintsT>& constraints)
{
    const vec3<float> sphere_center(0.1818329, -0.99023, 0.325066);
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
        const vec3<float> p(
            vertex_pos(vh, 0), vertex_pos(vh, 1), vertex_pos(vh, 2));
        // dragon
        // fix the bottom
        if (p[2] < -0.63) {
            constraints(vh) = Fixed;
        } else if (glm::distance(p, sphere_center) < 0.1) {
            // move the jaw
            constraints(vh) = Handle;
        } else {
            constraints(vh) = Free;
        }

        // cube
        // if (p[2] > 0 && p[1] > 0 && p[0] > 0) {
        //    constraints(vh) = Handle;
        //} else if (p[2] < 0 && p[1] < 0 && p[0] < 0) {
        //    constraints(vh) = Fixed;
        //} else {
        //    constraints(vh) = Free;
        //}
    });
}

int main(int argc, char** argv)
{
    rx_init(0);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

    if (!rx.is_closed()) {
        RXMESH_ERROR("Input mesh should be closed without boundaries");
        return EXIT_FAILURE;
    }

#if USE_POLYSCOPE
    polyscope::view::upDir = polyscope::UpDir::ZUp;
#endif

    constexpr uint32_t blockThreads = 256;

    // stays same across computation
    auto ref_vertex_pos = *rx.get_input_vertex_coordinates();

    // deformed vertex position that change every iteration
    auto deformed_vertex_pos = *rx.add_vertex_attribute<float>("deformedV", 3);
    deformed_vertex_pos.copy_from(ref_vertex_pos, DEVICE, DEVICE);

    // deformed vertex position as a matrix (used in the solver)
    DenseMatrix<float> deformed_vertex_pos_mat =
        *deformed_vertex_pos.to_matrix();


    // vertex constraints where
    auto constraints = *rx.add_vertex_attribute<ConstraintsT>("Constraints", 1);

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
    DenseMatrix<float> b_mat(rx, rx.get_num_vertices(), 3, LOCATION_ALL);
    b_mat.reset(0.f, LOCATION_ALL);

    // obtain cotangent weight matrix
    calc_edge_weights_mat(rx, ref_vertex_pos, weight_matrix);


    // set constraints
    set_contraints(rx, deformed_vertex_pos, constraints);


#if USE_POLYSCOPE
    // move constraints to the host and add it to Polyscope
    constraints.move(DEVICE, HOST);

    rx.get_polyscope_mesh()->addVertexScalarQuantity("constraintsV",
                                                     constraints);
#endif

    // Calculate system matrix
    calculate_system_matrix(rx, weight_matrix, constraints, laplace_mat);


    // pre_solve laplace_mat
    // GMGSolver solver(rx,
    //                 laplace_mat,
    //                 1000,
    //                 2,
    //                 2,
    //                 2,
    //                 CoarseSolver::Cholesky,
    //                 float(1e-6),
    //                 0.f);
    // solver.pre_solve(b_mat, deformed_vertex_pos_mat);

    CholeskySolver solver(&laplace_mat);

    CPUTimer timer;
    GPUTimer gtimer;

    timer.start();
    gtimer.start();

    solver.pre_solve(rx);

    timer.stop();
    gtimer.stop();

    RXMESH_INFO("solver.pre_solve took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());

    // how many times will arap algorithm run?
    int iterations = 1;

    float t    = 0;
    bool  flag = false;

    vec3<float> start(0.0f, 0.2f, 0.0f);
    vec3<float> end(0.0f, -0.2f, 0.0f);

    vec3<float> displacement(0.0f, 0.0f, 0.0f);

    bool is_running = false;

    auto take_step = [&]() mutable {
        t += flag ? -0.5f : 0.5f;

        flag = (t < 0 || t > 1.0f) ? !flag : flag;

        displacement = (1 - t) * start + (t)*end;

        // apply user deformation
        rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
            if (constraints(vh) == Handle) {
                deformed_vertex_pos(vh, 0) += displacement[0];
                deformed_vertex_pos(vh, 1) += displacement[1];
                deformed_vertex_pos(vh, 2) += displacement[2];
            }
        });


        // process step
        for (int i = 0; i < iterations; i++) {
            // solver for rotation
            calculate_rotation_matrix(rx,
                                      ref_vertex_pos,
                                      deformed_vertex_pos,
                                      weight_matrix,
                                      rotations);


            // solve for position
            calculate_b(rx,
                        ref_vertex_pos,
                        deformed_vertex_pos,
                        rotations,
                        weight_matrix,
                        constraints,
                        b_mat);

            solver.solve(b_mat, deformed_vertex_pos_mat);
        }

        // move mat to the host
        deformed_vertex_pos_mat.move(DEVICE, HOST);
        deformed_vertex_pos.from_matrix(&deformed_vertex_pos_mat);

#if USE_POLYSCOPE
        rx.get_polyscope_mesh()->updateVertexPositions(deformed_vertex_pos);
#endif
    };

    auto ps_callback = [&]() {
#if USE_POLYSCOPE

        ImGui::SameLine();
        if (ImGui::Button("Step")) {
            take_step();
        }

        if (ImGui::Button("Start")) {
            is_running = true;
        }

        ImGui::SameLine();
        if (ImGui::Button("Pause")) {
            is_running = false;
        }

        if (is_running) {
            take_step();
        }
#endif
    };


#if USE_POLYSCOPE
    polyscope::state::userCallback = ps_callback;
    polyscope::show();
#endif

    deformed_vertex_pos_mat.release();
    weight_matrix.release();
    laplace_mat.release();
    b_mat.release();


    return 0;
}