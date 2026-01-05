#include "rxmesh/rxmesh_static.h"

#include "rxmesh/geometry_factory.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"

#include "barrier_energy.h"
#include "bending_energy.h"
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

struct PhysicsParams {
    T   density        = 1000;   // rho
    T   young_mod      = 1e5;    // E
    T   poisson_ratio  = 0.4;    // nu
    T   time_step      = 0.01;   // h
    T   fricition_coef = 0.11;   // mu
    T   stiffness_coef = 4e4;
    T   tol            = 0.01;
    T   dhat           = 0.1;
    T   kappa          = 1e5;
    T   bending_stiff  = 1e3;    // k_b
    int num_steps      = 5;      // Number of simulation steps
};

void neo_hookean(RXMeshStatic& rx, T dx, const PhysicsParams& params)
{
    // printf("neo_hookean: Starting function\n");

    constexpr int VariableDim = 3;

    constexpr uint32_t blockThreads = 256;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    using HessMatT = typename ProblemT::HessMatT;

    // Problem parameters
    const int max_vv_candidate_pairs = 500;

    const T        density        = params.density;
    const T        young_mod      = params.young_mod;
    const T        poisson_ratio  = params.poisson_ratio;
    const T        time_step      = params.time_step;
    const T        fricition_coef = params.fricition_coef;
    const T        stiffness_coef = params.stiffness_coef;
    const T        tol            = params.tol;
    const T        inv_time_step  = T(1) / time_step;
    const T dhat          = params.dhat;
    const T kappa         = params.kappa;
    const T bending_stiff = params.bending_stiff;

    // TODO the limits and velocity should be different for different Dirichlet
    // nodes
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
    T total_volume = bb[0] * bb[1] * bb[2];
    T mass = density * total_volume /
             (rx.get_num_vertices());  // m

    // printf("neo_hookean: Setting up attributes\n");

    // Attributes
    auto velocity = *rx.add_vertex_attribute<T>("Velocity", 3);  // v
    velocity.reset(0, DEVICE);

    auto volume = *rx.add_face_attribute<T>("Volume", 1);  // vol
    volume.reset(0, DEVICE);

    auto mu_lambda = *rx.add_vertex_attribute<T>("mu_lambda", 1);  // mu_lambda
    mu_lambda.reset(0, DEVICE);

    auto inv_b =
        *rx.add_face_attribute<Eigen::Matrix<T, 3, 3>>("InvB", 1);  // IB

    auto contact_area = *rx.add_vertex_attribute<T>("ContactArea", 1);
    contact_area.reset(dx, DEVICE);  // perimeter split to each vertex

    // Bending energy attributes
    auto rest_angle = *rx.add_edge_attribute<T>("RestAngle", 1);
    rest_angle.reset(0, DEVICE);

    auto edge_area = *rx.add_edge_attribute<T>("EdgeArea", 1);
    edge_area.reset(0, DEVICE);

    // Get region labels for multiple meshes
    auto region_label = *rx.get_vertex_region_label();

    // Diff problem and solvers
    ProblemT problem(rx, true, max_vv_candidate_pairs);

    // Pre-allocate BVH bounding boxes buffer for contact detection
    BVHBuffers<T> bvh_buffers(rx.get_num_vertices());

    CGSolver<T, ProblemT::DenseMatT::OrderT> solver(*problem.hess, 1, 1000);

    // CholeskySolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(
    //     problem.hess.get());

    NetwtonSolver newton_solver(problem, &solver);

    auto& x = *problem.objective;
    x.copy_from(*rx.get_input_vertex_coordinates(), DEVICE, DEVICE);

    auto x_n     = *rx.add_vertex_attribute_like("x_n", x);
    auto x_tilde = *rx.add_vertex_attribute_like("x_tilde", x);


    // printf("neo_hookean: Running initializations\n");

    // Initializations
    init_volume_inverse_b(rx, x, volume, inv_b);
    // printf("neo_hookean: Finished init_volume_inverse_b\n");

    init_bending(rx, x, rest_angle, edge_area);
    // printf("neo_hookean: Finished init_bending\n");

    // // Debug: print bending initialization stats
    // rest_angle.move(DEVICE, HOST);
    // edge_area.move(DEVICE, HOST);

    // T min_angle = std::numeric_limits<T>::max();
    // T max_angle = std::numeric_limits<T>::lowest();
    // T min_area = std::numeric_limits<T>::max();
    // T max_area = std::numeric_limits<T>::lowest();
    // int num_internal_edges = 0;

    // rx.for_each_edge(HOST, [&](EdgeHandle eh) {
    //     T angle = rest_angle(eh);
    //     T area = edge_area(eh);
    //     if (area > 0) {  // internal edge
    //         num_internal_edges++;
    //         min_angle = std::min(min_angle, angle);
    //         max_angle = std::max(max_angle, angle);
    //         min_area = std::min(min_area, area);
    //         max_area = std::max(max_area, area);
    //     }
    // });

    // RXMESH_INFO("Bending initialization:");
    // RXMESH_INFO("  Internal edges: {}", num_internal_edges);
    // RXMESH_INFO("  Rest angle range: [{}, {}] radians", min_angle, max_angle);
    // RXMESH_INFO("  Edge area range: [{}, {}]", min_area, max_area);

    // rest_angle.move(HOST, DEVICE);
    // edge_area.move(HOST, DEVICE);


    typename ProblemT::DenseMatT alpha(
        rx, std::max(rx.get_num_vertices(), rx.get_num_faces()), 1, DEVICE);

#if USE_POLYSCOPE
    // add volume to polyscope
    volume.move(DEVICE, HOST);
    rx.get_polyscope_mesh()->addFaceScalarQuantity("Volume", volume);
#endif

    // printf("neo_hookean: Adding energy terms\n");

    // add inertial energy term
    inertial_energy(problem, x_tilde, mass);
    // printf("neo_hookean: Added inertial energy\n");

    // add gravity energy
    gravity_energy(problem, time_step, mass);
    // printf("neo_hookean: Added gravity energy\n");

    // add barrier energy
    floor_barrier_energy(problem,
                         contact_area,
                         time_step,
                         ground_n,
                         ground_o,
                         dhat,
                         kappa);

    // printf("neo_hookean: Added floor barrier energy\n");

    vv_contact_energy(problem, contact_area, time_step, dhat, kappa);
    // printf("neo_hookean: Added vv contact energy\n");

    DenseMatrix<T, Eigen::RowMajor> dir(
        rx, problem.grad.rows(), problem.grad.cols(), LOCATION_ALL);

    DenseMatrix<T, Eigen::RowMajor> grad(
        rx, problem.grad.rows(), problem.grad.cols(), LOCATION_ALL);

    T line_search_init_step = 0;

    // add friction energy
    // TODO alpha should change during different runs (e.g., in the line search)
    // friction_energy(problem,
    //                x_n,
    //                newton_solver.dir,
    //                line_search_init_step,
    //                mu_lambda,
    //                time_step,
    //                ground_n);

    // add neo hooken energy
    neo_hookean_energy(problem, volume, inv_b, mu_lame, time_step, lam);
    // printf("neo_hookean: Added neo-hookean energy\n");

    // add bending energy
    bending_energy(problem, rest_angle, edge_area, bending_stiff, time_step);
    // printf("neo_hookean: Added bending energy\n");


    int steps = 0;

    Timers<GPUTimer> timer;
    timer.add("Step");
    timer.add("ContactDetection");
    timer.add("EnergyEval");
    timer.add("LinearSolver");
    timer.add("LineSearch");
    timer.add("StepSize");

    auto step_forward = [&]() {
        // printf("neo_hookean: step_forward() - Starting step %d\n", steps);

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


        // evaluate energy
        // printf("neo_hookean: step_forward() - Adding contact\n");
        timer.start("ContactDetection");
        add_contact(problem,
                    rx,
                    problem.vv_pairs,
                    bvh_buffers,
                    x,
                    contact_area,
                    time_step,
                    dhat,
                    kappa,
                    region_label);
        timer.stop("ContactDetection");

        // printf("neo_hookean: step_forward() - Updating hessian\n");
        timer.start("EnergyEval");
        problem.update_hessian();
        // printf("neo_hookean: step_forward() - Evaluating terms\n");
        problem.eval_terms();
        timer.stop("EnergyEval");
        // printf("neo_hookean: step_forward() - Finished evaluating terms\n");

        grad.copy_from(problem.grad, DEVICE, DEVICE);

        // get newton direction
        // printf("neo_hookean: step_forward() - Computing newton direction\n");
        timer.start("LinearSolver");
        newton_solver.compute_direction();
        timer.stop("LinearSolver");
        // printf("neo_hookean: step_forward() - Finished computing newton direction\n");

        dir.copy_from(newton_solver.dir, DEVICE, DEVICE);
        // residual is abs_max(newton_dir)/ h
        T residual = newton_solver.dir.abs_max() / time_step;
        // printf("neo_hookean: step_forward() - Initial residual: %f\n", residual);

        T f = problem.get_current_loss();
        RXMESH_INFO(
            "*******Step: {}, Energy: {}, Residual: {}",
            steps,
            f,
            residual);

        int iter = 0;
        // printf("neo_hookean: step_forward() - Entering iteration loop\n");
        while (residual > tol) {
            // printf("neo_hookean: step_forward() - Iteration %d, residual: %f, num_satisfied: %d\n", iter, residual, num_satisfied);
            // printf("neo_hookean: step_forward() - Iteration %d, residual: %f \n", iter, residual);

            // printf("neo_hookean: step_forward() - Computing neo_hookean_step_size\n");
            timer.start("StepSize");
            T nh_step = neo_hookean_step_size(rx, x, newton_solver.dir, alpha);
            // printf("neo_hookean: step_forward() - nh_step: %f\n", nh_step);

            // printf("neo_hookean: step_forward() - Computing barrier_step_size\n");
            T bar_step = barrier_step_size(rx,
                                           newton_solver.dir,
                                           alpha,
                                           x,
                                           ground_n,
                                           ground_o);
            // printf("neo_hookean: step_forward() - bar_step: %f\n", bar_step);

            line_search_init_step = std::min(nh_step, bar_step);
            timer.stop("StepSize");
            // printf("neo_hookean: step_forward() - line_search_init_step: %f\n", line_search_init_step);

            // TODO: line search should pass the step to the friction energy
            // printf("neo_hookean: step_forward() - Starting line search\n");
            timer.start("LineSearch");
            bool ls_success = newton_solver.line_search(
                line_search_init_step, 0.5, 64, 0.0, [&](auto temp_x) {
                    add_contact(problem,
                                rx,
                                problem.vv_pairs,
                                bvh_buffers,
                                temp_x,
                                contact_area,
                                time_step,
                                dhat,
                                kappa,
                                region_label);
                });
            timer.stop("LineSearch");
            // printf("neo_hookean: step_forward() - Finished line search, success: %d\n", ls_success);

            if (!ls_success) {
                RXMESH_WARN("Line search failed!");
            }

            // evaluate energy
            // printf("neo_hookean: step_forward() - Re-evaluating energy after line search\n");
            timer.start("ContactDetection");
            add_contact(problem,
                        rx,
                        problem.vv_pairs,
                        bvh_buffers,
                        x,
                        contact_area,
                        time_step,
                        dhat,
                        kappa,
                        region_label);
            timer.stop("ContactDetection");

            // printf("neo_hookean: step_forward() - Updating hessian after line search\n");
            timer.start("EnergyEval");
            problem.update_hessian();
            // printf("neo_hookean: step_forward() - Evaluating terms after line search\n");
            problem.eval_terms();
            timer.stop("EnergyEval");

            T f = problem.get_current_loss();
            // printf("neo_hookean: step_forward() - Current loss: %f\n", f);

            // get newton direction
            // printf("neo_hookean: step_forward() - Computing newton direction (iteration %d)\n", iter);
            timer.start("LinearSolver");
            newton_solver.compute_direction();
            timer.stop("LinearSolver");

            // residual is abs_max(newton_dir)/ h
            residual = newton_solver.dir.abs_max() / time_step;
            // printf("neo_hookean: step_forward() - New residual: %f\n", residual);

            RXMESH_INFO(
                "  Subsetp: {}, F: {}, R: {}, line_search_init_step={}, ",
                iter,
                f,
                residual,
                line_search_init_step);

            iter++;

            if (iter > 10) {
                break;
            }
        }

        RXMESH_INFO("===================");

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

    // printf("declared everything. starting simulation.\n");
#if USE_POLYSCOPE
    draw(rx, x, velocity, step_forward, dir, grad, steps);
#else
    while (steps < params.num_steps) {
        step_forward();
    }
#endif


    // Print comprehensive timing summary
    RXMESH_INFO("=== TIMING SUMMARY ===");
    RXMESH_INFO("Number of steps: {}", steps);
    RXMESH_INFO("Total Step Time:        {:.2f} ms ({:.2f} ms/iter)",
                timer.elapsed_millis("Step"),
                timer.elapsed_millis("Step") / float(steps));
    RXMESH_INFO("  Contact Detection:    {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("ContactDetection"),
                100.0 * timer.elapsed_millis("ContactDetection") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("ContactDetection") / float(steps));
    RXMESH_INFO("  Energy Evaluation:    {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("EnergyEval"),
                100.0 * timer.elapsed_millis("EnergyEval") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("EnergyEval") / float(steps));
    RXMESH_INFO("  Linear Solver:        {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("LinearSolver"),
                100.0 * timer.elapsed_millis("LinearSolver") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("LinearSolver") / float(steps));
    RXMESH_INFO("  Line Search:          {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("LineSearch"),
                100.0 * timer.elapsed_millis("LineSearch") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("LineSearch") / float(steps));
    RXMESH_INFO("  Step Size Compute:    {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("StepSize"),
                100.0 * timer.elapsed_millis("StepSize") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("StepSize") / float(steps));
    RXMESH_INFO("======================");
}

int main(int argc, char** argv)
{
    rx_init(0, spdlog::level::info);

    // Parse command line arguments for physics parameters
    PhysicsParams params;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--density" && i + 1 < argc) {
            params.density = std::atof(argv[++i]);
        } else if (arg == "--young" && i + 1 < argc) {
            params.young_mod = std::atof(argv[++i]);
        } else if (arg == "--poisson" && i + 1 < argc) {
            params.poisson_ratio = std::atof(argv[++i]);
        } else if (arg == "--timestep" && i + 1 < argc) {
            params.time_step = std::atof(argv[++i]);
        } else if (arg == "--friction" && i + 1 < argc) {
            params.fricition_coef = std::atof(argv[++i]);
        } else if (arg == "--stiffness" && i + 1 < argc) {
            params.stiffness_coef = std::atof(argv[++i]);
        } else if (arg == "--tol" && i + 1 < argc) {
            params.tol = std::atof(argv[++i]);
        } else if (arg == "--dhat" && i + 1 < argc) {
            params.dhat = std::atof(argv[++i]);
        } else if (arg == "--kappa" && i + 1 < argc) {
            params.kappa = std::atof(argv[++i]);
        } else if (arg == "--bending" && i + 1 < argc) {
            params.bending_stiff = std::atof(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            params.num_steps = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --density <val>    Density (default: 1000)\n");
            printf("  --young <val>      Young's modulus (default: 1e5)\n");
            printf("  --poisson <val>    Poisson ratio (default: 0.4)\n");
            printf("  --timestep <val>   Time step (default: 0.01)\n");
            printf("  --friction <val>   Friction coefficient (default: 0.11)\n");
            printf("  --stiffness <val>  Stiffness coefficient (default: 4e4)\n");
            printf("  --tol <val>        Tolerance (default: 0.01)\n");
            printf("  --dhat <val>       Contact distance threshold (default: 0.1)\n");
            printf("  --kappa <val>      Contact stiffness (default: 1e5)\n");
            printf("  --bending <val>    Bending stiffness (default: 1e3)\n");
            printf("  --steps <val>      Number of simulation steps (default: 5)\n");
            printf("  --help, -h         Show this help message\n");
            return 0;
        }
    }

    RXMESH_INFO("Physics Parameters:");
    RXMESH_INFO("  Density: {}", params.density);
    RXMESH_INFO("  Young's modulus: {}", params.young_mod);
    RXMESH_INFO("  Poisson ratio: {}", params.poisson_ratio);
    RXMESH_INFO("  Time step: {}", params.time_step);
    RXMESH_INFO("  Friction coefficient: {}", params.fricition_coef);
    RXMESH_INFO("  Stiffness coefficient: {}", params.stiffness_coef);
    RXMESH_INFO("  Tolerance: {}", params.tol);
    RXMESH_INFO("  dhat: {}", params.dhat);
    RXMESH_INFO("  kappa: {}", params.kappa);
    RXMESH_INFO("  Bending stiffness: {}", params.bending_stiff);
    RXMESH_INFO("  Number of steps: {}", params.num_steps);

    // Load multiple meshes using RXMeshStatic's multiple mesh constructor
    std::vector<std::string> inputs = {"input/el_topo_sphere_1280.obj",
                                       "input/el_topo_sphere_1280.obj"};

    RXMeshStatic rx(inputs);

    RXMESH_INFO(
        "#Faces: {}, #Vertices: {}", rx.get_num_faces(), rx.get_num_vertices());

    T dx = 0.1f;  // mesh spacing for contact area
    auto x = *rx.get_input_vertex_coordinates();
    auto region_label = *rx.get_vertex_region_label();

    T translate_y = 2.5f;
    rx.for_each_vertex(
        DEVICE,
        [=] __device__ (VertexHandle vh) mutable {
          if (region_label(vh) == 1) {
            x(vh, 1) += translate_y;
          }
        },
        NULL,
        false
    );
    x.move(DEVICE, HOST);
#if USE_POLYSCOPE
    rx.get_polyscope_mesh()->updateVertexPositions(x);
#endif

    neo_hookean(rx, dx, params);
}
