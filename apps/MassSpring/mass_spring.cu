
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/geometry_factory.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"

using namespace rxmesh;


template <typename T>
void mass_spring(RXMeshStatic& rx)
{
    constexpr int VariableDim = 3;

    constexpr uint32_t blockThreads = 256;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    using HessMatT = typename ProblemT::HessMatT;

    const T rho             = 1000;  // density
    const T k               = 1e4;   // stiffness
    const T initial_stretch = 1.3;
    const T tol             = 0.01;
    const T h               = 0.02;  // time step
    const T inv_h           = T(1) / h;

    glm::vec3 bb_lower(0), bb_upper(0);
    rx.bounding_box(bb_lower, bb_upper);
    glm::vec3 bb = bb_upper - bb_lower;

    // mass per vertex = rho * volume /num_vertices
    T mass = rho * bb[0] * bb[1] / rx.get_num_vertices();

    ProblemT problem(rx);

    CholeskySolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(&problem.hess);

    NetwtonSolver newton_solver(problem, &solver);

    auto rest_l = *rx.add_edge_attribute<T>("RestLen", 1);

    auto velocity = *rx.add_vertex_attribute<T>("Velocity", 3);
    velocity.reset(0, DEVICE);

    auto is_bc = *rx.add_vertex_attribute<int8_t>("isBC", 1);
    is_bc.reset(0, DEVICE);

    auto x = *rx.get_input_vertex_coordinates();

    auto& x_tilde = *problem.objective;

    // set boundary conditions
    rx.for_each_vertex(
        DEVICE,
        [bb_upper, bb_lower, is_bc, x] __device__(const VertexHandle& vh) {
            if (x(vh, 0) < std::numeric_limits<T>::min()) {
                is_bc(vh) = 1;
            }
        });

#if USE_POLYSCOPE
    // add BC to polyscope
    is_bc.move(DEVICE, HOST);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("BC", is_bc);
#endif

    // calc rest length
    rx.run_query_kernel<Op::EV, blockThreads>(
        [=] __device__(const EdgeHandle& eh, const VertexIterator& iter) {
            Eigen::Vector3<T> a = x.to_eigen<3>(iter[0]);
            Eigen::Vector3<T> b = x.to_eigen<3>(iter[1]);

            rest_l(eh) = (a - b).squaredNorm();
        });

    // apply initial stretch along the y direction
    // rx.for_each_vertex(
    //    DEVICE,
    //    [initial_stretch, x, x_tilde] __device__(const VertexHandle& vh) {
    //        x(vh, 1) = x(vh, 1) * initial_stretch;
    //    });


    // add inertial energy term
    T half_mass = T(0.5) * mass;
    problem.template add_term<Op::V, true>(
        [x, half_mass] __device__(const auto& vh, auto& obj) mutable {
            using ActiveT = ACTIVE_TYPE(vh);

            Eigen::Vector3<ActiveT> x_tilda = iter_val<ActiveT, 3>(vh, obj);

            Eigen::Vector3<T> xx = x.to_eigen<3>(vh);

            Eigen::Vector3<ActiveT> l = xx - x_tilda;

            ActiveT E = half_mass * l.squaredNorm();

            return E;
        });

    // add elastic potential energy term
    T half_k_times_h_sq = T(0.5) * k * h * h;
    problem.template add_term<Op::EV, true>(
        [rest_l, half_k_times_h_sq] __device__(
            const auto& eh, const auto& iter, auto& obj) mutable {
            assert(iter[0].is_valid() && iter[1].is_valid());

            assert(iter.size() == 2);

            using ActiveT = ACTIVE_TYPE(eh);

            const Eigen::Vector3<ActiveT> a =
                iter_val<ActiveT, 3>(eh, iter, obj, 0);
            const Eigen::Vector3<ActiveT> b =
                iter_val<ActiveT, 3>(eh, iter, obj, 1);

            const T r = rest_l(eh);

            const Eigen::Vector3<ActiveT> diff = a - b;

            const ActiveT ratio = diff.squaredNorm() / r;

            const ActiveT s = (ratio - T(1.0));

            const ActiveT E = half_k_times_h_sq * r * s * s;

            return E;
        });


    // add gravity energy
    Eigen::Vector3<T> g(0.0, -9.81, 0.0);
    const T           neg_mass_times_h_sq = -mass * h * h;
    problem.template add_term<Op::V, true>(
        [x, neg_mass_times_h_sq, g] __device__(const auto& vh,
                                               auto&       obj) mutable {
            using ActiveT = ACTIVE_TYPE(vh);

            Eigen::Vector3<ActiveT> x_tilda = iter_val<ActiveT, 3>(vh, obj);

            ActiveT E = neg_mass_times_h_sq * x_tilda.dot(g);

            return E;
        });

    int time_step = 0;

    Timers<GPUTimer> timer;
    timer.add("Step");
    timer.add("LineSearch");
    timer.add("LinearSolver");
    timer.add("Diff");


    auto step_forward = [&]() {
        // update x_tilde
        timer.start("Step");
        rx.for_each_vertex(
            DEVICE,
            [x, x_tilde, velocity, h] __device__(VertexHandle vh) mutable {
                for (int i = 0; i < 3; ++i) {
                    x_tilde(vh, i) = x(vh, i) + h * velocity(vh, i);
                }
            });

        // evaluate energy
        timer.start("Diff");
        problem.eval_terms();
        timer.stop("Diff");

        // T f = problem.get_current_loss();
        // RXMESH_INFO("Time step: {}, Energy: {}", time_step, f);

        // apply bc
        newton_solver.apply_bc(is_bc);

        // get newton direction
        timer.start("LinearSolver");
        newton_solver.newton_direction();
        timer.stop("LinearSolver");


        // residual is abs_max(newton_dir)/ h
        T residual = newton_solver.dir.abs_max() / h;

        int iter = 0;
        if (residual > tol) {
            timer.start("LineSearch");
            newton_solver.line_search();
            timer.stop("LineSearch");

            // evaluate energy
            timer.start("Diff");
            problem.eval_terms();
            timer.stop("Diff");

            // apply bc
            newton_solver.apply_bc(is_bc);

            // get newton direction
            timer.start("LinearSolver");
            newton_solver.newton_direction();
            timer.stop("LinearSolver");

            // residual is abs_max(newton_dir)/ h
            residual = newton_solver.dir.abs_max() / h;

            iter++;
        }

        //  update velocity
        rx.for_each_vertex(
            DEVICE,
            [x, x_tilde, velocity, inv_h] __device__(VertexHandle vh) mutable {
                for (int i = 0; i < 3; ++i) {
                    velocity(vh, i) = inv_h * (x_tilde(vh, i) - x(vh, i));

                    x(vh, i) = x_tilde(vh, i);
                }
            });

        time_step++;
        timer.stop("Step");
    };

#if USE_POLYSCOPE
    polyscope::options::groundPlaneHeightFactor = 0.8;
    // polyscope::options::groundPlaneMode =
    //     polyscope::GroundPlaneMode::ShadowOnly;

    bool is_running = false;

    auto ps_callback = [&]() mutable {
        auto step_and_update = [&]() {
            step_forward();
            x_tilde.move(DEVICE, HOST);
            velocity.move(DEVICE, HOST);
            auto vel = rx.get_polyscope_mesh()->addVertexVectorQuantity(
                "Velocity", velocity);
            rx.get_polyscope_mesh()->updateVertexPositions(x_tilde);
        };
        if (ImGui::Button("Step")) {
            step_and_update();
        }


        ImGui::SameLine();
        if (ImGui::Button("Start")) {
            is_running = true;
        }

        ImGui::SameLine();
        if (ImGui::Button("Pause")) {
            is_running = false;
        }

        ImGui::SameLine();
        if (ImGui::Button("Export")) {
            rx.export_obj("MS_" + std::to_string(time_step) + ".obj", x_tilde);
        }

        if (is_running) {
            step_and_update();
        }
    };

    polyscope::state::userCallback = ps_callback;
    polyscope::show();
#else
    while (time_step < 5) {
        step_forward();
    }
#endif


    RXMESH_INFO(
        "Mass-spring: #time_step ={}, time= {} (ms), "
        "timer/iteration= {} ms/iter",
        time_step,
        timer.elapsed_millis("Step"),
        timer.elapsed_millis("Step") / float(time_step));
    RXMESH_INFO("LinearSolver {} (ms), Diff {} (ms), LineSearch {} (ms)",
                timer.elapsed_millis("LinearSolver"),
                timer.elapsed_millis("Diff"),
                timer.elapsed_millis("LineSearch"));

    RXMESH_INFO(
        "LinearSolver/iter {} (ms), Diff/iter {} (ms), LineSearch/iter {} (ms)",
        timer.elapsed_millis("LinearSolver") / float(time_step),
        timer.elapsed_millis("Diff") / float(time_step),
        timer.elapsed_millis("LineSearch") / float(time_step));
}

int main(int argc, char** argv)
{
    Log::init(spdlog::level::info);

    using T = float;

    std::vector<std::vector<T>>        verts;
    std::vector<std::vector<uint32_t>> fv;

    int n = 16;

    if (argc == 2) {
        n = atoi(argv[1]);
    }

    T dx = 1 / T(n - 1);

    create_plane(verts, fv, n, n, 2, dx, true); 

    RXMeshStatic rx(fv);
    rx.add_vertex_coordinates(verts, "Coords");

    RXMESH_INFO(
        "#Faces: {}, #Vertices: {}", rx.get_num_faces(), rx.get_num_vertices());


    mass_spring<T>(rx);
}