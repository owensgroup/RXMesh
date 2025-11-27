
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/geometry_factory.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"

#include "barrier_energy.h"
#include "boundary_condition.h"
#include "draw.h"
#include "gravity_energy.h"
#include "inertial_energy.h"
#include "mass_spring_energy.h"

using namespace rxmesh;

enum class Scenario
{
    Flag    = 0,
    DropBox = 1,

};

template <typename T>
void mass_spring(RXMeshStatic& rx, T dx, Scenario scenario)
{
    constexpr int VariableDim = 3;

    constexpr uint32_t blockThreads = 256;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    using HessMatT = typename ProblemT::HessMatT;

    const T rho             = 100;  // density
    const T k               = 4e4;  // stiffness
    const T initial_stretch = 1.3;
    const T tol             = 0.01;
    const T h               = 0.01;  // time step
    const T inv_h           = T(1) / h;
    const T y_ground        = T(-1.0);

    glm::vec3 bb_lower(0), bb_upper(0);
    rx.bounding_box(bb_lower, bb_upper);
    glm::vec3 bb = bb_upper - bb_lower;

    // mass per vertex = rho * volume /num_vertices
    T mass = rho * bb[0] * bb[1] / rx.get_num_vertices();

    ProblemT problem(rx, true);

    CholeskySolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(
        problem.hess.get());

    NetwtonSolver newton_solver(problem, &solver);

    auto rest_l = *rx.add_edge_attribute<T>("RestLen", 1);

    auto velocity = *rx.add_vertex_attribute<T>("Velocity", 3);
    velocity.reset(0, DEVICE);

    auto is_bc = *rx.add_vertex_attribute<int8_t>("isBC", 1);
    is_bc.reset(0, DEVICE);

    auto contact_area = *rx.add_vertex_attribute<T>("ContactArea", 1);
    contact_area.reset(dx, DEVICE);

    auto x = *rx.get_input_vertex_coordinates();

    auto& x_tilde = *problem.objective;
    x_tilde.copy_from(x, DEVICE, DEVICE);

    typename ProblemT::DenseMatT alpha(rx, rx.get_num_vertices(), DEVICE);

    // set boundary conditions and scenario specific energies
    switch (scenario) {
        case Scenario::Flag:
            flag_bc(rx, is_bc, x, bb_lower, bb_upper);
            break;
        case Scenario::DropBox:
            barrier_energy(problem, x, contact_area, h, y_ground);
            break;
        default:
            break;
    };


    // apply initial stretch along the y direction
    // apply_init_stretch(rx, x, initial_stretch);

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


    // add inertial energy term
    inertial_energy(problem, x, mass);


    // add elastic potential energy term
    mass_spring_energy(problem, rest_l, h, k);

    // add gravity energy
    gravity_energy(problem, x, h, mass);

    int time_step = 0;

    Timers<GPUTimer> timer;
    timer.add("Step");
    timer.add("LineSearch");
    timer.add("LinearSolver");
    timer.add("Diff");


    auto step_forward = [&]() {
        // evaluate energy
        timer.start("Diff");
        problem.eval_terms();
        timer.stop("Diff");


        // update x_tilde
        timer.start("Step");
        rx.for_each_vertex(
            DEVICE,
            [x, x_tilde, velocity, h] __device__(VertexHandle vh) mutable {
                for (int i = 0; i < 3; ++i) {
                    x_tilde(vh, i) = x(vh, i) + h * velocity(vh, i);
                }
            });

        // apply bc
        newton_solver.apply_bc(is_bc);

        // get newton direction
        timer.start("LinearSolver");
        newton_solver.compute_direction();
        timer.stop("LinearSolver");

        // residual is abs_max(newton_dir)/ h
        T residual = newton_solver.dir.abs_max() / h;


        T f = problem.get_current_loss();


        int iter = 0;
        while (residual > tol) {

            RXMESH_INFO("Time step: {}, Energy: {}, Residual: {}",
                        time_step,
                        f,
                        residual);

            timer.start("LineSearch");
            T line_search_init_step = 1;
            if (scenario == Scenario::DropBox) {
                line_search_init_step =
                    init_step_size(rx, newton_solver.dir, alpha, x, y_ground);
            }
            newton_solver.line_search(line_search_init_step, 0.5);
            timer.stop("LineSearch");

            // evaluate energy
            timer.start("Diff");
            problem.eval_terms();
            timer.stop("Diff");

            // apply bc
            newton_solver.apply_bc(is_bc);

            // get newton direction
            timer.start("LinearSolver");
            newton_solver.compute_direction();
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
    draw(rx, x_tilde, velocity, step_forward, time_step);
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

    create_plane(verts, fv, n, n, 2, dx, false, vec3<float>(-0.5, -0.5, 0));

    RXMeshStatic rx(fv);
    rx.add_vertex_coordinates(verts, "Coords");

    RXMESH_INFO(
        "#Faces: {}, #Vertices: {}", rx.get_num_faces(), rx.get_num_vertices());


    mass_spring<T>(rx, dx, Scenario::DropBox);
}