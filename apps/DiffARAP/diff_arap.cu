#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"

#include "arap_kernels.h"

using namespace rxmesh;

template <typename T>
void arap(RXMeshStatic& rx)
{
    constexpr int VariableDim = 3;

    constexpr uint32_t blockThreads = 256;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    using HessMatT = typename ProblemT::HessMatT;

    ProblemT problem(rx, true);

    CholeskySolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(
        problem.hess.get());

    NetwtonSolver newton_solver(problem, &solver);

    auto P = *rx.get_input_vertex_coordinates();

    // deformed vertex position that change every iteration
    auto& P_prime = *problem.objective;
    P_prime.copy_from(P, DEVICE, DEVICE);

    // vertex constraints where
    //  0 means free
    //  1 means user-displaced
    //  2 means fixed
    auto constraints = *rx.add_vertex_attribute<int>("Constraints", 1);
    auto bc          = *rx.add_vertex_attribute<int>("bc", 1);
    constraints.reset(0, LOCATION_ALL);
    bc.reset(0, LOCATION_ALL);

    // rotation matrix as a very attribute where every vertex has 3x3 matrix
    auto rotations = *rx.add_vertex_attribute<T>("RotationMatrix", 9);

    // weights matrix
    SparseMatrix<float> weight_matrix(rx);
    weight_matrix.reset(0.f, LOCATION_ALL);

    // set constraints
    const vec3<float> sphere_center(0.1818329, -0.99023, 0.325066);
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
        const vec3<float> p(P(vh, 0), P(vh, 1), P(vh, 2));

        // fix the bottom
        if (p[2] < -0.63) {
            constraints(vh) = 2;
            bc(vh)          = 1;
        }

        // move the jaw
        if (glm::distance(p, sphere_center) < 0.1) {
            constraints(vh) = 1;
            bc(vh)          = 1;
        }
    });

#if USE_POLYSCOPE
    // move constraints to the host and add it to Polyscope
    constraints.move(DEVICE, HOST);
    bc.move(DEVICE, HOST);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("constraintsV",
                                                     constraints);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("bc", bc);
#endif


    // obtain cotangent weight matrix
    rx.run_kernel<blockThreads>({Op::EVDiamond},
                                calc_edge_weights_mat<float, blockThreads>,
                                P,
                                weight_matrix);


    // energy term
    problem.template add_term<Op::EV, true>(
        [=] __device__(const auto& eh, const auto& iter, auto& obj) {
            using ActiveT = ACTIVE_TYPE(eh);

            auto vi = iter[0];
            auto vj = iter[1];

            Eigen::Vector3<ActiveT> pi_prime =
                iter_val<ActiveT, 3>(eh, iter, obj, 0);

            Eigen::Vector3<ActiveT> pj_prime =
                iter_val<ActiveT, 3>(eh, iter, obj, 1);

            Eigen::Vector3<T> pi = P.to_eigen<3>(vi);

            Eigen::Vector3<T> pj = P.to_eigen<3>(vj);

            // r
            Eigen::Matrix<T, 3, 3> ri;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ri(i, j) = rotations(vi, i * 3 + j);
                }
            }

            Eigen::Vector3<ActiveT> e = (pi_prime - pj_prime) - ri * (pi - pj);

            ActiveT E = weight_matrix(vi, vj) * e.squaredNorm();

            return E;
        });


    T   convergence_eps = 1e-2;
    int newton_max_iter = 150;

    float       t    = 0;
    bool        flag = false;
    vec3<float> start(0.0f, 0.2f, 0.0f);
    vec3<float> end(0.0f, -0.2f, 0.0f);
    vec3<float> displacement(0.0f, 0.0f, 0.0f);


    int  num_steps         = 0;
    int  totla_newton_iter = 0;
    bool is_running        = false;

    Timers<GPUTimer> timer;
    timer.add("Step");
    timer.add("LineSearch");
    timer.add("LinearSolver");
    timer.add("Diff");
    timer.add("NetwonSolve");

    auto polyscope_callback = [&]() mutable {
        bool step_once = false;

        if (ImGui::Button("Start")) {
            is_running = true;
        }

        ImGui::SameLine();
        if (ImGui::Button("Step")) {
            step_once = true;
        }

        if (step_once || is_running) {

            t += flag ? -0.5f : 0.5f;

            flag = (t < 0 || t > 1.0f) ? !flag : flag;

            displacement = (1 - t) * start + (t)*end;

            // apply user deformation
            rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
                if (constraints(vh) == 1) {
                    P_prime(vh, 0) += displacement[0];
                    P_prime(vh, 1) += displacement[1];
                    P_prime(vh, 2) += displacement[2];
                }
            });


            // process step
            num_steps++;

            RXMESH_INFO("Step {}", num_steps);

            timer.start("Step");
            // solver for rotation
            rx.run_kernel<blockThreads>(
                {Op::VV},
                calculate_rotation_matrix<float, blockThreads>,
                P,
                P_prime,
                rotations,
                weight_matrix);

            // solve for position via Newton
            timer.start("NetwonSolve");
            for (int iter = 0; iter < newton_max_iter; ++iter) {
                totla_newton_iter++;
                // evaluate energy terms
                timer.start("Diff");
                problem.eval_terms();
                timer.stop("Diff");

                // get the current value of the loss function
                // T f = problem.get_current_loss();
                // RXMESH_INFO("Iter {} =, Newton Iter= {}: Energy = {}",
                //            num_steps,
                //            iter,
                //            f);

                // apply bc
                newton_solver.apply_bc(bc);

                // direction newton
                timer.start("LinearSolver");
                newton_solver.compute_direction();
                timer.stop("LinearSolver");


                // newton decrement
                if (0.5f * problem.grad.dot(newton_solver.dir) <
                    convergence_eps) {
                    break;
                }

                // line search
                timer.start("LineSearch");
                newton_solver.line_search();
                timer.stop("LineSearch");
            }
            timer.stop("Step");
            timer.stop("NetwonSolve");
        }

        // move mat to the host
        P_prime.move(DEVICE, HOST);

        if (ImGui::Button("Export")) {
            rx.export_obj(std::to_string(num_steps) + ".obj", P_prime);
        }

        ImGui::SameLine();
        if (ImGui::Button("Pause")) {
            is_running = false;
        }

#if USE_POLYSCOPE
        rx.get_polyscope_mesh()->updateVertexPositions(P_prime);
#endif
    };

#ifdef USE_POLYSCOPE
    polyscope::options::groundPlaneMode =
        polyscope::GroundPlaneMode::ShadowOnly;
    polyscope::view::upDir         = polyscope::UpDir::ZUp;
    polyscope::state::userCallback = polyscope_callback;
    rx.get_polyscope_mesh()->setSurfaceColor(glm::vec3(0.941, 0.901, 0.549));
    polyscope::show();

#endif

    weight_matrix.release();

    RXMESH_INFO(
        "DiffArap: #V= {}, #Steps ={}, time= {} (ms), time/step= {} ms/iter",
        rx.get_num_vertices(),
        num_steps,
        timer.elapsed_millis("Step"),
        timer.elapsed_millis("Step") / float(num_steps));

    RXMESH_INFO("LinearSolver {} (ms), Diff {} (ms), LineSearch {} (ms)",
                timer.elapsed_millis("LinearSolver"),
                timer.elapsed_millis("Diff"),
                timer.elapsed_millis("LineSearch"));

    RXMESH_INFO(
        "LinearSolver/step {} (ms), Diff/step {} (ms), LineSearch/step {} (ms)",
        timer.elapsed_millis("LinearSolver") / float(num_steps),
        timer.elapsed_millis("Diff") / float(num_steps),
        timer.elapsed_millis("LineSearch") / float(num_steps));

    RXMESH_INFO(
        "#Newton_iter = {}, NewtonSolve/Newton_iter = {}, "
        "LinearSolver/Newton_iter {} (ms), Diff/Newton_iter "
        "{} (ms), LineSearch/Newton_iter {} (ms)",
        totla_newton_iter,
        timer.elapsed_millis("NetwonSolve") / float(totla_newton_iter),
        timer.elapsed_millis("LinearSolver") / float(totla_newton_iter),
        timer.elapsed_millis("Diff") / float(totla_newton_iter),
        timer.elapsed_millis("LineSearch") / float(totla_newton_iter));
}

int main(int argc, char** argv)
{
    rx_init(0);

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

    if (!rx.is_closed()) {
        RXMESH_ERROR("Input mesh should be closed without boundaries");
        return EXIT_FAILURE;
    }


    arap<float>(rx);


    return 0;
}