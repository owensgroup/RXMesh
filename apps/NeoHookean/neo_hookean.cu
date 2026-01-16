#include <Eigen/Core>
#include <unordered_set>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/geometry_factory.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"

#include "barrier_energy.h"
#include "bending_energy.h"
#include "draw.h"
#include "gravity_energy.h"
#include "inertial_energy.h"
#include "init.h"
#include "neo_hookean_energy.h"
#include "scene_config.h"

#include "naive_hessian_update.h"

#include <nvtx3/nvToolsExt.h>

using namespace rxmesh;

using T = float;


struct PhysicsParams
{
    T                density        = 1000;  // rho
    T                young_mod      = 1e5;   // E
    T                poisson_ratio  = 0.4;   // nu
    T                time_step      = 0.01;  // h
    T                stiffness_coef = 4e4;
    T                tol            = 0.01;
    T                dhat           = 0.2;
    T                kappa          = 1e5;
    T                bending_stiff  = 1e8;  // k_b
    std::vector<int> export_steps;          // List of step IDs to export as OBJ
    int              num_steps       = 5;   // Number of simulation steps
    int              cg_max_iter     = 1000;
    int              newton_max_iter = 10;

    // Box boundary (5 walls - no ceiling)
    bool use_box   = false;
    T    box_min_x = -5.0;
    T    box_max_x = 5.0;
    T    box_min_y = -1.0;
    T    box_min_z = -5.0;
    T    box_max_z = 5.0;
};

void neo_hookean(RXMeshStatic& rx, T dx, const PhysicsParams& params)
{
    constexpr int VariableDim = 3;

    constexpr uint32_t blockThreads = 256;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    using HessMatT = typename ProblemT::HessMatT;

    // Problem parameters
    const int max_vv_candidate_pairs = rx.get_num_vertices();
    const int max_vf_candidate_pairs = rx.get_num_faces();

    const T density        = params.density;
    const T young_mod      = params.young_mod;
    const T poisson_ratio  = params.poisson_ratio;
    const T time_step      = params.time_step;
    const T stiffness_coef = params.stiffness_coef;
    const T tol            = params.tol;
    const T inv_time_step  = T(1) / time_step;
    const T dhat           = params.dhat;
    const T kappa          = params.kappa;
    const T bending_stiff  = params.bending_stiff;

    // Box boundary (if enabled) - 5 walls, no ceiling
    const T    box_min_x = params.box_min_x;
    const T    box_max_x = params.box_max_x;
    const T    box_min_y = params.box_min_y;
    const T    box_min_z = params.box_min_z;
    const T    box_max_z = params.box_max_z;
    const bool use_box   = params.use_box;

    // Derived parameters
    const T mu_lame = 0.5 * young_mod / (1 + poisson_ratio);
    const T lam     = young_mod * poisson_ratio /
                  ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));

    glm::vec3 bb_lower(0), bb_upper(0);
    rx.bounding_box(bb_lower, bb_upper);
    glm::vec3 bb = bb_upper - bb_lower;

    // mass per vertex = rho * volume /num_vertices
    T total_volume = bb[0] * bb[1] * bb[2];
    T mass         = density * total_volume / (rx.get_num_vertices());  // m


    // Attributes
    auto velocity = *rx.add_vertex_attribute<T>("Velocity", 3);  // v
    velocity.reset(0, DEVICE);

    auto volume = *rx.add_face_attribute<T>("Volume", 1);  // vol
    volume.reset(0, DEVICE);


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
    auto vertex_region_label = *rx.get_vertex_region_label();
    auto face_region_label   = *rx.get_face_region_label();

    // Store vertex handles for each face (3 vertices per face)
    auto face_vertices = *rx.add_face_attribute<uint64_t>("FaceVertices", 3);

    // Diff problem and solvers
    ProblemT problem(rx, true, max_vv_candidate_pairs, max_vf_candidate_pairs);

    // Pre-allocate BVH bounding boxes buffer for contact detection
    BVHBuffers<T> vertex_bvh_buffers(rx.get_num_vertices());
    BVHBuffers<T> face_bvh_buffers(rx.get_num_faces());

    // CGSolver<T, ProblemT::DenseMatT::OrderT> solver(*problem.hess, 1,
    // params.cg_max_iter);
    PCGSolver<T, ProblemT::DenseMatT::OrderT> solver(
        *problem.hess, 1, params.cg_max_iter);


    NetwtonSolver newton_solver(problem, &solver);

    auto& x = *problem.objective;
    x.copy_from(*rx.get_input_vertex_coordinates(), DEVICE, DEVICE);

    auto x_n     = *rx.add_vertex_attribute_like("x_n", x);
    auto x_tilde = *rx.add_vertex_attribute_like("x_tilde", x);

    // Initializations
    init_volume_inverse_b(rx, x, volume, inv_b);
    init_bending(rx, x, rest_angle, edge_area);
    init_face_vertices(rx, face_vertices);


    typename ProblemT::DenseMatT alpha(
        rx, std::max(rx.get_num_vertices(), rx.get_num_faces()), 1, DEVICE);

#if USE_POLYSCOPE
    volume.move(DEVICE, HOST);
    rx.get_polyscope_mesh()->addFaceScalarQuantity("Volume", volume);
#endif

    // add inertial energy term
    inertial_energy(problem, x_tilde, mass);

    // add gravity energy
    gravity_energy(problem, time_step, mass);

    // add barrier energy
    box_barrier_energy(problem,
                       contact_area,
                       time_step,
                       box_min_x,
                       box_max_x,
                       box_min_y,
                       box_min_z,
                       box_max_z,
                       dhat,
                       kappa);

    // add contact energy
    vv_contact_energy(problem, contact_area, time_step, dhat, kappa);
    vf_contact_energy(problem, contact_area, time_step, dhat, kappa);

    T line_search_init_step = 0;

    // add neo hooken energy
    neo_hookean_energy(problem, volume, inv_b, mu_lame, time_step, lam);


    // add bending energy
    bending_energy(problem, rest_angle, edge_area, bending_stiff, time_step);

    int max_vv_contact(0), max_vf_contact(0);

    int steps = 0;

    Timers<GPUTimer> timer;
    timer.add("Step");
    timer.add("ContactDetection_Explicit");
    timer.add("ContactDetection_LineSearch");
    timer.add("ContactDetection_PostLineSearch");
    timer.add("ContactDetection");
    timer.add("EnergyEval");
    timer.add("LinearSolver");
    timer.add("LineSearch");
    timer.add("StepSize");
    timer.add("UpdateHessian");
    timer.add("UpdateHessian_CuSPARSE");

    auto step_forward = [&]() {
        // x_tilde = x + v*h
        // x_n = x (copy current position)
        timer.start("Step");

        nvtxRangePushA("Step");        
        rx.for_each_vertex(DEVICE, [=] __device__(VertexHandle vh) mutable {
            for (int i = 0; i < 3; ++i) {
                x_tilde(vh, i) = x(vh, i) + time_step * velocity(vh, i);
                x_n(vh, i)     = x(vh, i);
            }
        });

        // evaluate energy
        timer.start("ContactDetection_Explicit");
        nvtxRangePushA("ContactDetection_Explicit");
        add_contact(problem,
                    rx,
                    problem.vv_pairs,
                    problem.vf_pairs,
                    vertex_bvh_buffers,
                    face_bvh_buffers,
                    x,
                    contact_area,
                    time_step,
                    dhat,
                    kappa,
                    vertex_region_label,
                    face_region_label,
                    face_vertices);
        max_vv_contact = std::max(max_vv_contact, problem.vv_pairs.num_pairs());
        max_vf_contact = std::max(max_vf_contact, problem.vf_pairs.num_pairs());

        nvtxRangePop();
        timer.stop("ContactDetection_Explicit");

        timer.start("UpdateHessian");
        nvtxRangePushA("UpdateHessian");
        problem.update_hessian();
        nvtxRangePop();
        timer.stop("UpdateHessian");

        timer.start("EnergyEval");
        nvtxRangePushA("EnergyEval");
        problem.eval_terms();
        nvtxRangePop();
        timer.stop("EnergyEval");

        // get newton direction
        timer.start("LinearSolver");
        nvtxRangePushA("LinearSolver");
        newton_solver.compute_direction();
        nvtxRangePop();
        timer.stop("LinearSolver");


        // residual is abs_max(newton_dir)/ h
        T residual = newton_solver.dir.abs_max() / time_step;

        T f = problem.get_current_loss();
        RXMESH_INFO(
            "*****Step: {}, Energy: {}, Residual: {}", steps, f, residual);

        int iter = 0;

        while (residual > tol) {

            timer.start("StepSize");
            T nh_step = neo_hookean_step_size(rx, x, newton_solver.dir, alpha);


            T bar_step = box_barrier_step_size(rx,
                                               newton_solver.dir,
                                               alpha,
                                               x,
                                               box_min_x,
                                               box_max_x,
                                               box_min_y,
                                               box_min_z,
                                               box_max_z);

            line_search_init_step = std::min(nh_step, bar_step);
            timer.stop("StepSize");

            timer.start("LineSearch");
            nvtxRangePushA("LineSearch");
            bool ls_success = newton_solver.line_search(
                line_search_init_step, 0.5, 64, 0.0, [&](auto temp_x) {
                    timer.start("ContactDetection_LineSearch");
                    nvtxRangePushA("LineSearch");
                    add_contact(problem,
                                rx,
                                problem.vv_pairs,
                                problem.vf_pairs,
                                vertex_bvh_buffers,
                                face_bvh_buffers,
                                temp_x,
                                contact_area,
                                time_step,
                                dhat,
                                kappa,
                                vertex_region_label,
                                face_region_label,
                                face_vertices);
                    max_vv_contact =
                        std::max(max_vv_contact, problem.vv_pairs.num_pairs());
                    max_vf_contact =
                        std::max(max_vf_contact, problem.vf_pairs.num_pairs());
                    nvtxRangePop();
                    timer.stop("ContactDetection_LineSearch");
                });
            nvtxRangePop();
            timer.stop("LineSearch");


            if (!ls_success) {
                RXMESH_WARN("Line search failed!");
            }

            // evaluate energy
            timer.start("ContactDetection_PostLineSearch");
            nvtxRangePushA("ContactDetection_PostLineSearch");
            add_contact(problem,
                        rx,
                        problem.vv_pairs,
                        problem.vf_pairs,
                        vertex_bvh_buffers,
                        face_bvh_buffers,
                        x,
                        contact_area,
                        time_step,
                        dhat,
                        kappa,
                        vertex_region_label,
                        face_region_label,
                        face_vertices);
            max_vv_contact =
                std::max(max_vv_contact, problem.vv_pairs.num_pairs());
            max_vf_contact =
                std::max(max_vf_contact, problem.vf_pairs.num_pairs());
            nvtxRangePop();
            timer.stop("ContactDetection_PostLineSearch");


            timer.start("UpdateHessian");
            nvtxRangePushA("UpdateHessian");
            problem.update_hessian();
            nvtxRangePop();
            timer.stop("UpdateHessian");


            timer.start("EnergyEval");
            nvtxRangePushA("EnergyEval");
            problem.eval_terms();
            nvtxRangePop();
            timer.stop("EnergyEval");


            // get newton direction
            timer.start("LinearSolver");
            nvtxRangePushA("LinearSolver");
            newton_solver.compute_direction();
            nvtxRangePop();
            timer.stop("LinearSolver");

            // residual is abs_max(newton_dir)/ h
            residual = newton_solver.dir.abs_max() / time_step;


            T f = problem.get_current_loss();
            RXMESH_INFO(
                "  Subsetp: {}, F: {}, R: {}, line_search_init_step={}, ",
                iter,
                f,
                residual,
                line_search_init_step);

            iter++;

            if (iter > params.newton_max_iter) {
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
                }
            });

        steps++;
        timer.stop("Step");
        nvtxRangePop();
    };


#if USE_POLYSCOPE
    draw(rx, problem, x, velocity, step_forward, steps);
#else
    // Convert export_steps vector to set for O(1) lookup
    std::unordered_set<int> export_set(params.export_steps.begin(),
                                       params.export_steps.end());

    while (steps < params.num_steps) {
        step_forward();

        // Check if we should export at this step
        // steps is already incremented in step_forward()
        if (export_set.count(steps - 1) > 0) {
            // Move vertex positions to HOST
            x.move(DEVICE, HOST);

            // Create filename with step number
            std::string mesh_filename = STRINGIFY(OUTPUT_DIR) +
                                        std::string("scene_step_") +
                                        std::to_string(steps - 1) + ".obj";

            RXMESH_INFO(
                "Exporting mesh at step {} to {}", steps - 1, mesh_filename);
            rx.export_obj(mesh_filename, x);

            // Export Hessian matrix
            problem.hess->move(DEVICE, HOST);
            std::string hess_filename = STRINGIFY(OUTPUT_DIR) +
                                        std::string("hessian_step_") +
                                        std::to_string(steps - 1) + ".txt";
            problem.hess->to_file(hess_filename);
            RXMESH_INFO(
                "Exported Hessian at step {} to {} (dim: {}x{}, nnz: {})",
                steps - 1,
                hess_filename,
                problem.hess->rows(),
                problem.hess->cols(),
                problem.hess->non_zeros());
        }
    }
#endif


    // Print comprehensive timing summary
    RXMESH_INFO("=== TIMING SUMMARY ===");
    RXMESH_INFO("Number of steps: {}", steps);
    RXMESH_INFO("Max VV contact pair: {}, max VF contact pairs: {}",
                max_vv_contact,
                max_vf_contact);
    RXMESH_INFO("Total Step Time:        {:.2f} ms ({:.2f} ms/iter)",
                timer.elapsed_millis("Step"),
                timer.elapsed_millis("Step") / float(steps));

    // clang-format off
    // Contact Detection broken down by context
    T total_contact = timer.elapsed_millis("ContactDetection_Explicit") +
                      timer.elapsed_millis("ContactDetection_LineSearch") +
                      timer.elapsed_millis("ContactDetection_PostLineSearch");
    RXMESH_INFO("  Contact Detection (Total): {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                total_contact,
                100.0 * total_contact / timer.elapsed_millis("Step"),
                total_contact / float(steps));
    RXMESH_INFO("    - Explicit:          {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("ContactDetection_Explicit"),
                100.0 * timer.elapsed_millis("ContactDetection_Explicit") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("ContactDetection_Explicit") / float(steps));
    RXMESH_INFO("    - LineSearch:        {:.2f} ms  ({:.1f}%)",
                timer.elapsed_millis("ContactDetection_LineSearch"),
                100.0 * timer.elapsed_millis("ContactDetection_LineSearch") / timer.elapsed_millis("Step"));
    RXMESH_INFO("    - PostLineSearch:    {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("ContactDetection_PostLineSearch"),
                100.0 * timer.elapsed_millis("ContactDetection_PostLineSearch") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("ContactDetection_PostLineSearch") / float(steps));

    RXMESH_INFO("  Energy Evaluation:    {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("EnergyEval"),
                100.0 * timer.elapsed_millis("EnergyEval") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("EnergyEval") / float(steps));
    RXMESH_INFO("  Linear Solver:        {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("LinearSolver"),
                100.0 * timer.elapsed_millis("LinearSolver") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("LinearSolver") / float(steps));
    //subtracting the collision detection done in line search to get the actual time for line search 
    RXMESH_INFO("  Line Search:          {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("LineSearch") - timer.elapsed_millis("ContactDetection_LineSearch"),
                100.0 * (timer.elapsed_millis("LineSearch") - timer.elapsed_millis("ContactDetection_LineSearch")) / timer.elapsed_millis("Step"),
                (timer.elapsed_millis("LineSearch") - timer.elapsed_millis("ContactDetection_LineSearch")) / float(steps));
    RXMESH_INFO("  Step Size Compute:    {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("StepSize"),
                100.0 * timer.elapsed_millis("StepSize") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("StepSize") / float(steps));
    RXMESH_INFO("  Update Hessian Compute:    {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("UpdateHessian"),
                100.0 * timer.elapsed_millis("UpdateHessian") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("UpdateHessian") / float(steps));
    // clang-format on

    RXMESH_INFO("======================");
}

/**
 * Struct to store per-instance transformation data
 */
template <typename T>
struct InstanceTransform
{
    T tx, ty, tz;  // Translation
    T scale;       // Scale
    T vx, vy, vz;  // Initial velocity
};

/**
 * Generate mesh instances from scene configuration
 * Returns: vector of mesh filepaths (with duplicates for instances)
 *          vector of transforms (one per instance)
 */
template <typename T>
void generate_instances(const SceneConfig<T>&              config,
                        std::vector<std::string>&          mesh_paths,
                        std::vector<InstanceTransform<T>>& transforms)
{
    mesh_paths.clear();
    transforms.clear();

    for (const auto& mesh_cfg : config.meshes) {
        int total_instances = mesh_cfg.num_instances_x *
                              mesh_cfg.num_instances_y *
                              mesh_cfg.num_instances_z;

        RXMESH_INFO("Generating {} instances of {}",
                    total_instances,
                    mesh_cfg.filepath);

        // Generate instances in 3D grid
        for (int iz = 0; iz < mesh_cfg.num_instances_z; ++iz) {
            for (int iy = 0; iy < mesh_cfg.num_instances_y; ++iy) {
                for (int ix = 0; ix < mesh_cfg.num_instances_x; ++ix) {
                    // Add mesh filepath
                    mesh_paths.push_back(mesh_cfg.filepath);

                    // Compute instance transform
                    InstanceTransform<T> transform;
                    transform.tx = mesh_cfg.offset_x + ix * mesh_cfg.spacing_x;
                    transform.ty = mesh_cfg.offset_y + iy * mesh_cfg.spacing_y;
                    transform.tz = mesh_cfg.offset_z + iz * mesh_cfg.spacing_z;
                    transform.scale = mesh_cfg.scale;
                    transform.vx    = mesh_cfg.velocity_x;
                    transform.vy    = mesh_cfg.velocity_y;
                    transform.vz    = mesh_cfg.velocity_z;

                    transforms.push_back(transform);
                }
            }
        }
    }

    RXMESH_INFO("Total instances generated: {}", mesh_paths.size());
}

int main(int argc, char** argv)
{
    rx_init(0, spdlog::level::info);

    // Check for config file argument
    std::string config_file;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
            break;
        }
    }

    // Parse command line arguments for physics parameters
    PhysicsParams params;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config") {
            i++;  // Skip the config file path
            continue;
        } else if (arg == "--density" && i + 1 < argc) {
            params.density = std::atof(argv[++i]);
        } else if (arg == "--cg_max_iter" && i + 1 < argc) {
            params.cg_max_iter = std::atoi(argv[++i]);
        } else if (arg == "--newton_max_iter" && i + 1 < argc) {
            params.newton_max_iter = std::atoi(argv[++i]);
        } else if (arg == "--young" && i + 1 < argc) {
            params.young_mod = std::atof(argv[++i]);
        } else if (arg == "--poisson" && i + 1 < argc) {
            params.poisson_ratio = std::atof(argv[++i]);
        } else if (arg == "--timestep" && i + 1 < argc) {
            params.time_step = std::atof(argv[++i]);
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
            // clang-format off
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --config <file>         Load scene from configuration file\n");
            printf("  --density <val>         Density (default: 1000)\n");
            printf("  --cg_max_iter <val>     CG max num of iterations\n");
            printf("  --newton_max_iter <val> Newton max num of iterations\n");
            printf("  --young <val>           Young's modulus (default: 1e5)\n");
            printf("  --poisson <val>         Poisson ratio (default: 0.4)\n");
            printf("  --timestep <val>        Time step (default: 0.01)\n");            
            printf("  --stiffness <val>       Stiffness coefficient (default: 4e4)\n");
            printf("  --tol <val>             Tolerance (default: 0.01)\n");
            printf("  --dhat <val>            Contact distance threshold (default: 0.1)\n");
            printf("  --kappa <val>           Contact stiffness (default: 1e5)\n");
            printf("  --bending <val>         Bending stiffness (default: 1e3)\n");
            printf("  --steps <val>           Number of simulation steps (default: 5)\n");
            printf("  --help, -h              Show this help message\n");
            // clang-format on            
            return 0;
        }
    }

    // Load meshes and apply transforms
    std::vector<std::string>          inputs;
    std::vector<InstanceTransform<T>> transforms;

    if (!config_file.empty()) {
        // Parse config file
        RXMESH_INFO("Loading scene from config file: {}", config_file);
        SceneConfig<T> scene_config;
        if (!SceneConfigParser<T>::parse(config_file, scene_config)) {
            RXMESH_ERROR("Failed to parse config file");
            return 1;
        }

        // Override params with simulation config
        params.time_step      = scene_config.simulation.time_step;
        params.stiffness_coef = scene_config.simulation.stiffness_coef;
        
        params.tol              = scene_config.simulation.tol;
        params.dhat             = scene_config.simulation.dhat;
        params.kappa            = scene_config.simulation.kappa;
        params.num_steps        = scene_config.simulation.num_steps;
        params.cg_max_iter      = scene_config.simulation.cg_max_iter;
        params.newton_max_iter  = scene_config.simulation.newton_max_iter;        
        params.export_steps     = scene_config.simulation.export_steps;

        // Box parameters
        params.use_box   = scene_config.simulation.use_box;
        params.box_min_x = scene_config.simulation.box_min_x;
        params.box_max_x = scene_config.simulation.box_max_x;
        params.box_min_y = scene_config.simulation.box_min_y;
        params.box_min_z = scene_config.simulation.box_min_z;
        params.box_max_z = scene_config.simulation.box_max_z;

        // Box parameters
        params.use_box   = scene_config.simulation.use_box;
        params.box_min_x = scene_config.simulation.box_min_x;
        params.box_max_x = scene_config.simulation.box_max_x;
        params.box_min_y = scene_config.simulation.box_min_y;
        params.box_min_z = scene_config.simulation.box_min_z;
        params.box_max_z = scene_config.simulation.box_max_z;

        // Generate instances
        generate_instances(scene_config, inputs, transforms);
    } else {
        // Default: load two spheres (backward compatible)
        inputs = {STRINGIFY(INPUT_DIR) "el_topo_sphere_1280.obj",
                  STRINGIFY(INPUT_DIR) "el_topo_sphere_1280.obj"};

        // Create default transforms
        InstanceTransform<T> t1{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
        InstanceTransform<T> t2{0.0, 2.5, 0.0, 1.0, 0.0, 0.0, 0.0};
        transforms = {t1, t2};
    }

    RXMESH_INFO("Physics Parameters:");
    RXMESH_INFO("  Density: {}", params.density);
    RXMESH_INFO("  Young's modulus: {}", params.young_mod);
    RXMESH_INFO("  Poisson ratio: {}", params.poisson_ratio);
    RXMESH_INFO("  Time step: {}", params.time_step);
    RXMESH_INFO("  Stiffness coefficient: {}", params.stiffness_coef);
    RXMESH_INFO("  Tolerance: {}", params.tol);
    RXMESH_INFO("  dhat: {}", params.dhat);
    RXMESH_INFO("  kappa: {}", params.kappa);
    RXMESH_INFO("  Bending stiffness: {}", params.bending_stiff);
    RXMESH_INFO("  Number of steps: {}", params.num_steps);
    RXMESH_INFO("  CG max iter: {}", params.cg_max_iter);
    RXMESH_INFO("  Newton max iter: {}", params.newton_max_iter);
    if (params.use_box) {
        RXMESH_INFO("  Box enabled: [{}, {}] x [{}, open] x [{}, {}]",
                    params.box_min_x,
                    params.box_max_x,
                    params.box_min_y,
                    params.box_min_z,
                    params.box_max_z);
    }

    // Load meshes
    RXMeshStatic rx(inputs);
    RXMESH_INFO(
        "#Faces: {}, #Vertices: {}", rx.get_num_faces(), rx.get_num_vertices());

    T    dx       = 0.1f;  // mesh spacing for contact area
    auto x        = *rx.get_input_vertex_coordinates();
    

    // Apply transformations per instance
    auto vertex_region_label = *rx.get_vertex_region_label();
    
    rx.for_each_vertex(
        HOST,
        [=] __host__(VertexHandle vh) mutable {
            int region = vertex_region_label(vh);
            if (region >= 0 && region < static_cast<int>(transforms.size())) {
                const auto& t = transforms[region];

                // Apply scale
                if (t.scale != T(1.0)) {
                    x(vh, 0) *= t.scale;
                    x(vh, 1) *= t.scale;
                    x(vh, 2) *= t.scale;
                }

                // Apply translation
                x(vh, 0) += t.tx;
                x(vh, 1) += t.ty;
                x(vh, 2) += t.tz;                
            }
        },
        NULL,
        false);

    x.move(HOST, DEVICE);
#if USE_POLYSCOPE
    rx.get_polyscope_mesh()->updateVertexPositions(x);
#endif

    neo_hookean(rx, dx, params);
}
