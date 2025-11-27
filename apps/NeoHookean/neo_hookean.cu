#include "rxmesh/rxmesh_static.h"

#include "rxmesh/geometry_factory.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"

#include "barrier_energy.h"
#include "boundary_condition.h"
#include "draw.h"
#include "friction_energy.h"
#include "gravity_energy.h"
#include "inertial_energy.h"
#include "init.h"
#include "neo_hookean_energy.h"
#include "spring_energy.h"

#include <Eigen/Core>

using namespace rxmesh;

using T = float;

constexpr uint32_t num_dbc_vertices = 3;
VertexHandle       v_dbc[num_dbc_vertices];  // Dirichlet node index

void neo_hookean(RXMeshStatic& rx, T dx)
{
    constexpr int VariableDim = 3;

    constexpr uint32_t blockThreads = 256;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    using HessMatT = typename ProblemT::HessMatT;

    // Problem parameters
    const int max_vv_candidate_pairs = 50;

    const T        density        = 1000;  // rho
    const T        young_mod      = 1e5;   // E
    const T        poisson_ratio  = 0.4;   // nu
    const T        time_step      = 0.01;  // h
    const T        fricition_coef = 0.11;  // mu
    const T        stiffness_coef = 4e4;
    const T        tol            = 0.01;
    const T        inv_time_step  = T(1) / time_step;
    DenseMatrix<T> dbc_stiff(1, 1);
    dbc_stiff(0) = 1000;
    dbc_stiff.move(HOST, DEVICE);
    const T dhat  = 0.01;
    const T kappa = 1e5;

    // TODO the limits and velocity should be different for different Dirichlet
    // nodes
    const vec3<T> v_dbc_vel(0, -0.5, 0);        // Dirichlet node velocity
    const vec3<T> v_dbc_limit(0, -0.7, 0);      // Dirichlet node limit position
    const vec3<T> ground_o(0.0f, -1.0f, 0.0f);  // a point on the slope
    const vec3<T> ground_n =
        glm::normalize(vec3<T>(0.0f, 1.0f, 0.0f));  // normal of the slope


    // Derived parameters
    const T mu_lame = 0.5 * young_mod / (1 + poisson_ratio);
    const T lam     = young_mod * poisson_ratio /
                  ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));

    glm::vec3 bb_lower(0), bb_upper(0);
    rx.bounding_box(bb_lower, bb_upper);
    glm::vec3 bb = bb_upper - bb_lower;

    // mass per vertex = rho * volume /num_vertices
    T mass = density * bb[0] * bb[0] /
             (rx.get_num_vertices() - num_dbc_vertices);  // m

    // Attributes
    auto velocity = *rx.add_vertex_attribute<T>("Velocity", 3);  // v
    velocity.reset(0, DEVICE);

    auto volume = *rx.add_face_attribute<T>("Volume", 1);  // vol
    volume.reset(0, DEVICE);

    auto is_dbc_satisfied = *rx.add_vertex_attribute<int>("DBCSatisfied", 1);
    is_dbc_satisfied.reset(0, DEVICE);

    VertexReduceHandle<int> rh(is_dbc_satisfied);

    auto mu_lambda = *rx.add_vertex_attribute<T>("mu_lambda", 1);  // mu_lambda
    mu_lambda.reset(0, DEVICE);

    auto inv_b =
        *rx.add_face_attribute<Eigen::Matrix<T, 3, 3>>("InvB", 1);  // IB

    auto is_dbc = *rx.add_vertex_attribute<int8_t>("isBC", 1);


    auto dbc_target = *rx.add_vertex_attribute<T>("DBCTarget", 3);
    dbc_target.reset(0, DEVICE);

    auto contact_area = *rx.add_vertex_attribute<T>("ContactArea", 1);
    contact_area.reset(dx, DEVICE);  // perimeter split to each vertex

    // Diff problem and solvers
    ProblemT problem(rx, true, max_vv_candidate_pairs);

    CholeskySolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(
        problem.hess.get());

    NetwtonSolver newton_solver(problem, &solver);

    auto& x = *problem.objective;
    x.copy_from(*rx.get_input_vertex_coordinates(), DEVICE, DEVICE);

    auto x_n     = *rx.add_vertex_attribute_like("x_n", x);
    auto x_tilde = *rx.add_vertex_attribute_like("x_tilde", x);


    // Initializations
    init_volume_inverse_b(rx, x, volume, inv_b);

    rx.for_each_vertex(HOST, [&](VertexHandle vh) mutable {
        // doing it on the host since v_dbc is an array on the host
        is_dbc(vh) = 0;
        for (int i = 0; i < num_dbc_vertices; ++i) {
            if (vh == v_dbc[i]) {
                is_dbc(vh) = 1;
            }
        }
    });
    is_dbc.move(HOST, DEVICE);


    typename ProblemT::DenseMatT alpha(
        rx, std::max(rx.get_num_vertices(), rx.get_num_faces()), 1, DEVICE);

#if USE_POLYSCOPE
    // add BC to polyscope
    rx.get_polyscope_mesh()->addVertexScalarQuantity("DBC", is_dbc);

    // add volume to polyscope
    volume.move(DEVICE, HOST);
    rx.get_polyscope_mesh()->addFaceScalarQuantity("Volume", volume);
#endif

    // add inertial energy term OK
    inertial_energy(problem, x, x_tilde, is_dbc, mass);

    // add spring energy term
    spring_energy(problem, x, dbc_target, is_dbc, mass, dbc_stiff);


    // add gravity energy OK
    gravity_energy(problem, x, is_dbc, time_step, mass);

    // add barrier energy
    floor_barrier_energy(problem,
                         contact_area,
                         x,
                         time_step,
                         is_dbc,
                         ground_n,
                         ground_o,
                         dhat,
                         kappa);

    ceiling_barrier_energy(
        problem, contact_area, x, time_step, ground_n, ground_o, dhat, kappa);


    T line_search_init_step = 0;

    // add friction energy
    // TODO alpha should change during different runs (e.g., in the line search)
    // friction_energy(problem,
    //                x,
    //                x_n,
    //                newton_solver.dir,
    //                line_search_init_step,
    //                mu_lambda,
    //                time_step,
    //                ground_n);

    // add neo hooken energy
    neo_hookean_energy(
        problem, x, is_dbc, volume, inv_b, mu_lame, time_step, lam);


    int steps = 0;

    Timers<GPUTimer> timer;
    timer.add("Step");
    timer.add("LineSearch");
    timer.add("LinearSolver");
    timer.add("Diff");

    auto step_forward = [&]() {
        // x_tilde = x + v*h
        timer.start("Step");
        rx.for_each_vertex(DEVICE, [=] __device__(VertexHandle vh) mutable {
            for (int i = 0; i < 3; ++i) {
                x_tilde(vh, i) = x(vh, i) + time_step * velocity(vh, i);
            }
        });

        // copy current position
        x_n.copy_from(x, DEVICE, DEVICE);

        // compute mu * lambda for each node using x_n
        /*compute_mu_lambda(rx,
                          fricition_coef,
                          dhat,
                          kappa,
                          ground_n,
                          ground_o,
                          x,
                          contact_area,
                          mu_lambda);*/

        // target position for each DBC in the current time step
        update_dbc(
            rx, is_dbc, x, v_dbc_vel, v_dbc_limit, time_step, dbc_target);

        // evaluate energy
        add_contact(rx, problem.vv_pairs, v_dbc[0], is_dbc, x, dhat);
        problem.update_hessian();
        problem.eval_terms();

        // DBC satisfied
        check_dbc_satisfied(
            rx, is_dbc_satisfied, x, is_dbc, dbc_target, time_step, tol);

        // how many DBC are satisfied
        int num_satisfied = rh.reduce(is_dbc_satisfied, cub::Sum(), 0);

        // satisfied DBC are eliminated from the system which is the same
        // as adding boundary conditions where we zero out their gradients
        // and hessian (except the diagonal entries)
        newton_solver.apply_bc(is_dbc_satisfied);

        // get newton direction
        newton_solver.compute_direction();

        // residual is abs_max(newton_dir)/ h
        T residual = newton_solver.dir.abs_max() / time_step;

        T f = problem.get_current_loss();
        RXMESH_INFO("Step: {}, Energy: {}, Residual: {}", steps, f, residual);

        int iter = 0;
        while (residual > tol || num_satisfied != num_dbc_vertices) {

            if (residual <= tol && num_satisfied != num_dbc_vertices) {
                dbc_stiff.multiply(T(2));
            }

            T nh_step = neo_hookean_step_size(rx, x, newton_solver.dir, alpha);

            T bar_step = barrier_step_size(rx,
                                           newton_solver.dir,
                                           alpha,
                                           v_dbc[0],
                                           x,
                                           is_dbc,
                                           ground_n,
                                           ground_o);

            line_search_init_step = std::min(nh_step, bar_step);

            // TODO: line search should pass the step to the friction energy
            newton_solver.line_search(line_search_init_step, 0.5);

            // evaluate energy
            add_contact(rx, problem.vv_pairs, v_dbc[0], is_dbc, x, dhat);
            problem.update_hessian();
            problem.eval_terms();

            // T f = problem.get_current_loss();
            // RXMESH_INFO("Subsetp, Energy: {}", f);

            // DBC satisfied
            check_dbc_satisfied(
                rx, is_dbc_satisfied, x, is_dbc, dbc_target, time_step, tol);

            // how many DBC are satisfied
            num_satisfied = rh.reduce(is_dbc_satisfied, cub::Sum(), 0);

            // apply bc
            newton_solver.apply_bc(is_dbc_satisfied);

            // get newton direction
            newton_solver.compute_direction();

            // residual is abs_max(newton_dir)/ h
            residual = newton_solver.dir.abs_max() / time_step;

            iter++;
        }

        //  update velocity
        rx.for_each_vertex(
            DEVICE,
            [x, x_n, velocity, inv_time_step = 1.0 / time_step] __device__(
                VertexHandle vh) mutable {
                for (int i = 0; i < 3; ++i) {
                    velocity(vh, i) = inv_time_step * (x(vh, i) - x_n(vh, i));

                    // x(vh, i) = x_tilde(vh, i);
                }
            });

        steps++;
        timer.stop("Step");
    };

#if USE_POLYSCOPE
    draw(rx, x, velocity, step_forward, steps);
#else
    while (steps < 5) {
        step_forward();
    }
#endif


    RXMESH_INFO(
        "NeoHookean: #step ={}, time= {} (ms), timer/iteration= {} ms/iter",
        steps,
        timer.elapsed_millis("Step"),
        timer.elapsed_millis("Step") / float(steps));

    // RXMESH_INFO("LinearSolver {} (ms), Diff {} (ms), LineSearch {} (ms)",
    //             timer.elapsed_millis("LinearSolver"),
    //             timer.elapsed_millis("Diff"),
    //             timer.elapsed_millis("LineSearch"));
    //
    // RXMESH_INFO(
    //     "LinearSolver/iter {} (ms), Diff/iter {} (ms), LineSearch/iter {}(ms)
    //     ", timer.elapsed_millis(" LinearSolver ") / float(steps),
    //     timer.elapsed_millis("Diff") / float(steps),
    //     timer.elapsed_millis("LineSearch") / float(steps));
}

int main(int argc, char** argv)
{
    rx_init(0, spdlog::level::info);


    std::vector<std::vector<T>>        verts;
    std::vector<std::vector<uint32_t>> fv;

    int n = 5;

    if (argc == 2) {
        n = atoi(argv[1]);
    }

    T dx = 1 / T(n - 1);

    create_plane(verts, fv, n, n, 2, dx, false, vec3<float>(-0.5, -0.5, 0));

    uint32_t t = verts.size();

    // dirichlet nodes
    const vec3<T> dbc[3] = {
        {0.5f, 0.6f, 0.0f}, {-0.5f, 0.6f, 0.1f}, {-0.5f, 0.6f, -0.1f}};

    assert(sizeof(dbc) / sizeof(vec3<T>) == num_dbc_vertices);

    verts.push_back({dbc[0][0], dbc[0][1], dbc[0][2]});
    verts.push_back({dbc[1][0], dbc[1][1], dbc[1][2]});
    verts.push_back({dbc[2][0], dbc[2][1], dbc[2][2]});

    fv.push_back({t, t + 1, t + 2});

    RXMeshStatic rx(fv);
    rx.add_vertex_coordinates(verts, "Coords");

    RXMESH_INFO(
        "#Faces: {}, #Vertices: {}", rx.get_num_faces(), rx.get_num_vertices());

    auto x = *rx.get_input_vertex_coordinates();

    rx.for_each_vertex(
        HOST,
        [&](VertexHandle vh) {
            const vec3<T> p = x.to_glm<3>(vh);
            for (int i = 0; i < 3; ++i) {
                if (glm::distance2(p, dbc[i]) < 0.0001) {
                    for (int j = 0; j < 3; ++j) {
                        if (!v_dbc[j].is_valid()) {
                            v_dbc[j] = vh;
                            break;
                        }
                    }
                }
            }
        },
        NULL,
        false);


    neo_hookean(rx, dx);
}