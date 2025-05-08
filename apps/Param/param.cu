// Reference Implementation
// https://github.com/patr-schm/TinyAD-Examples/blob/main/apps/parametrization_openmesh.cc
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/algo/tutte_embedding.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"


struct arg
{
    std::string obj_file_name   = STRINGIFY(INPUT_DIR) "bunnyhead.obj";
    std::string output_folder   = STRINGIFY(OUTPUT_DIR);
    std::string uv_file_name    = "";
    std::string solver          = "chol";
    uint32_t    device_id       = 0;
    float       cg_abs_tol      = 1e-6;
    float       cg_rel_tol      = 0.0;
    uint32_t    cg_max_iter     = 10;
    uint32_t    newton_max_iter = 100;
    char**      argv;
    int         argc;
} Arg;


using namespace rxmesh;


template <typename T>
void add_mesh_to_polyscope(RXMeshStatic&       rx,
                           VertexAttribute<T>& v,
                           std::string         name)
{
#if USE_POLYSCOPE
    if (v.get_num_attributes() == 3) {
        polyscope::registerSurfaceMesh(name, v, rx.get_polyscope_mesh()->faces);
    } else {
        auto v3 = *rx.add_vertex_attribute<T>(name, 3);

        rx.for_each_vertex(HOST, [&](const VertexHandle h) {
            v3(h, 0) = v(h, 0);
            v3(h, 1) = v(h, 1);
            v3(h, 2) = 0;
        });

        polyscope::registerSurfaceMesh(
            name, v3, rx.get_polyscope_mesh()->faces);

        rx.remove_attribute(name);
    }
#endif
}

template <typename T, typename ProblemT, typename SolverT>
void parameterize(RXMeshStatic& rx, ProblemT& problem, SolverT& solver)
{
    NetwtonSolver newton_solver(problem, &solver);

    auto coordinates = *rx.get_input_vertex_coordinates();

    // TODO this is a AoS and should be converted into SoA
    auto rest_shape =
        *rx.add_face_attribute<Eigen::Matrix<T, 2, 2>>("fRestShape", 1);

    if (Arg.uv_file_name.empty()) {
        tutte_embedding(rx, coordinates, *problem.objective);
    } else {
        std::vector<std::vector<uint32_t>> fv;
        std::vector<std::vector<float>>    uv;
        import_obj(Arg.uv_file_name, uv, fv);
        if (uv.size() != rx.get_num_vertices()) {
            RXMESH_ERROR(
                "Number of vertices in the the input UV file {} does not match "
                "the number of vertices in the mesh {}.",
                uv.size(),
                rx.get_num_vertices());
        }
        rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
            uint32_t id = rx.map_to_global(vh);

            (*problem.objective)(vh, 0) = uv[id][0];
            (*problem.objective)(vh, 1) = uv[id][1];
        });

        problem.objective->move(HOST, DEVICE);
    }

#if USE_POLYSCOPE
    rx.get_polyscope_mesh()->addVertexParameterizationQuantity(
        "uv_tutte", *problem.objective);

    auto bnd = *rx.add_vertex_attribute<bool>("Bnd", 1);
    rx.get_boundary_vertices(bnd);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("Boundary", bnd);

    add_mesh_to_polyscope(rx, *problem.objective, "tutte_mesh");
#endif

    constexpr uint32_t blockThreads = 256;

    // 1) compute rest shape
    rx.run_query_kernel<Op::FV, blockThreads>(
        [=] __device__(const FaceHandle& fh, const VertexIterator& iter) {
            const VertexHandle v0 = iter[0];
            const VertexHandle v1 = iter[1];
            const VertexHandle v2 = iter[2];

            assert(v0.is_valid() && v1.is_valid() && v2.is_valid());

            // 3d position
            Eigen::Vector3<T> ar_3d = coordinates.to_eigen<3>(v0);
            Eigen::Vector3<T> br_3d = coordinates.to_eigen<3>(v1);
            Eigen::Vector3<T> cr_3d = coordinates.to_eigen<3>(v2);

            // Local 2D coordinate system
            Eigen::Vector3<T> n  = (br_3d - ar_3d).cross(cr_3d - ar_3d);
            Eigen::Vector3<T> b1 = (br_3d - ar_3d).normalized();
            Eigen::Vector3<T> b2 = n.cross(b1).normalized();

            // Express a, b, c in local 2D coordinates system
            Eigen::Vector2<T> ar_2d(T(0.0), T(0.0));
            Eigen::Vector2<T> br_2d((br_3d - ar_3d).dot(b1), T(0.0));
            Eigen::Vector2<T> cr_2d((cr_3d - ar_3d).dot(b1),
                                    (cr_3d - ar_3d).dot(b2));

            // Save 2-by-2 matrix with edge vectors as columns
            Eigen::Matrix<T, 2, 2> fout = col_mat(br_2d - ar_2d, cr_2d - ar_2d);

            rest_shape(fh) = fout;
        });


    // add energy term
    problem.template add_term<Op::FV, true>(
        [=] __device__(const auto& fh, const auto& iter, auto& objective) {
            // fh is a face handle
            // iter is an iterator over fh's vertices
            // objective is the uv coordinates

            assert(iter[0].is_valid() && iter[1].is_valid() &&
                   iter[2].is_valid());

            assert(iter.size() == 3);

            using ActiveT = ACTIVE_TYPE(fh);

            // uv
            Eigen::Vector2<ActiveT> a =
                iter_val<ActiveT, 2>(fh, iter, objective, 0);
            Eigen::Vector2<ActiveT> b =
                iter_val<ActiveT, 2>(fh, iter, objective, 1);
            Eigen::Vector2<ActiveT> c =
                iter_val<ActiveT, 2>(fh, iter, objective, 2);


            // Triangle flipped?
            Eigen::Matrix<ActiveT, 2, 2> M = col_mat(b - a, c - a);

            if (M.determinant() <= 0.0) {
                using PassiveT = PassiveType<ActiveT>;
                return ActiveT(std::numeric_limits<PassiveT>::max());
            }

            // Get constant 2D rest shape and area of triangle t
            const Eigen::Matrix<T, 2, 2> Mr = rest_shape(fh);

            const T A = T(0.5) * Mr.determinant();

            // Compute symmetric Dirichlet energy
            Eigen::Matrix<ActiveT, 2, 2> J = M * Mr.inverse();

            ActiveT res = A * (J.squaredNorm() + J.inverse().squaredNorm());


            return res;
        });


    T convergence_eps = 1e-2;

    int iter;

    Timers<GPUTimer> timer;
    timer.add("Total");
    timer.add("DiffCG");

    timer.start("Total");


    for (iter = 0; iter < Arg.newton_max_iter; ++iter) {

        timer.start("DiffCG");

        if (Arg.solver == "cg_mat_free") {
            // calc gradient only if we are using matrix free solver
            problem.eval_terms_grad_only();
        } else {
            // calc loss function
            problem.eval_terms();
        }


        // get the current value of the loss function
        /*T f = problem.get_current_loss();
        RXMESH_INFO("Iteration= {}: Energy = {}", iter, f);*/

        // direction newton
        newton_solver.newton_direction();
        timer.stop("DiffCG");

        // newton decrement
        /*if (0.5f * problem.grad.dot(newton_solver.dir) < convergence_eps) {
            break;
        }*/

        // line search
        newton_solver.line_search();
    }

    timer.stop("Total");


    RXMESH_INFO(
        "Parametrization: iterations ={}, time= {} (ms), "
        "timer/iteration= {} ms/iter, diff_cg_time/iter = {} solver_time = {} "
        "(ms)",
        iter,
        timer.elapsed_millis("Total"),
        timer.elapsed_millis("Total") / float(iter),
        timer.elapsed_millis("DiffCG") / float(iter),
        newton_solver.solve_time);


    problem.objective->move(DEVICE, HOST);

#if USE_POLYSCOPE
    rx.get_polyscope_mesh()->addVertexParameterizationQuantity(
        "uv_opt", *problem.objective);

    add_mesh_to_polyscope(rx, *problem.objective, "opt_mesh");

    polyscope::show();
#endif
}

int main(int argc, char** argv)
{
    Log::init(spdlog::level::info);

    using T = float;

    if (argc > 1) {
        if (cmd_option_exists(argv, argc + argv, "-h")) {
            // clang-format off
            RXMESH_INFO("\nUsage: Param.exe < -option X>\n"
                        " -h:                 Display this massage and exit\n"
                        " -input:             Input OBJ mesh file. Default is {} \n"                  
                        " -uv:                Input UV OBJ file. If empty, will compyte tutte embedding. Default is {} \n"                        
                        " -o:                 JSON file output folder. Default is {} \n"
                        " -solver:            Solver to use. Options are cg_mat_free, cg, pcg, chol, or lu. Default is {}\n"
                        " -abs_eps:           Iterative solvers absolute tolerance. Default is {}\n"
                        " -rel_eps:           Iterative solvers relative tolerance. Default is {}\n"
                        " -cg_max_iter:       Maximum number of iterations for iterative solvers. Default is {}\n"
                        " -newton_max_iter:   Maximum number of iterations for Newton solver. Default is {}\n"
                        " -device_id:         GPU device ID. Default is {}",
            Arg.obj_file_name,Arg.uv_file_name, Arg.output_folder,  Arg.solver, Arg.cg_abs_tol, Arg.cg_rel_tol, Arg.cg_max_iter, Arg.newton_max_iter, Arg.device_id);
            // clang-format on
            exit(EXIT_SUCCESS);
        }

        if (cmd_option_exists(argv, argc + argv, "-input")) {
            Arg.obj_file_name =
                std::string(get_cmd_option(argv, argv + argc, "-input"));
        }
        if (cmd_option_exists(argv, argc + argv, "-uv")) {
            Arg.uv_file_name =
                std::string(get_cmd_option(argv, argv + argc, "-uv"));
        }
        if (cmd_option_exists(argv, argc + argv, "-o")) {
            Arg.output_folder =
                std::string(get_cmd_option(argv, argv + argc, "-o"));
        }

        if (cmd_option_exists(argv, argc + argv, "-cg_max_iter")) {
            Arg.cg_max_iter =
                std::atoi(get_cmd_option(argv, argv + argc, "-cg_max_iter"));
        }

        if (cmd_option_exists(argv, argc + argv, "-newton_max_iter")) {
            Arg.newton_max_iter = std::atoi(
                get_cmd_option(argv, argv + argc, "-newton_max_iter"));
        }

        if (cmd_option_exists(argv, argc + argv, "-abs_eps")) {
            Arg.cg_abs_tol =
                std::atof(get_cmd_option(argv, argv + argc, "-abs_eps"));
        }

        if (cmd_option_exists(argv, argc + argv, "-rel_eps")) {
            Arg.cg_rel_tol =
                std::atof(get_cmd_option(argv, argv + argc, "-rel_eps"));
        }

        if (cmd_option_exists(argv, argc + argv, "-device_id")) {
            Arg.device_id =
                atoi(get_cmd_option(argv, argv + argc, "-device_id"));
        }

        if (cmd_option_exists(argv, argc + argv, "-solver")) {
            Arg.solver =
                std::string(get_cmd_option(argv, argv + argc, "-solver"));
        }
    }

    RXMESH_INFO("input= {}", Arg.obj_file_name);
    RXMESH_INFO("output_folder= {}", Arg.output_folder);
    RXMESH_INFO("solver= {}", Arg.solver);
    RXMESH_INFO("cg_max_iter= {}", Arg.cg_max_iter);
    RXMESH_INFO("newton_max_iter= {}", Arg.newton_max_iter);
    RXMESH_INFO("abs_eps= {0:f}", Arg.cg_abs_tol);
    RXMESH_INFO("rel_eps= {0:f}", Arg.cg_rel_tol);
    RXMESH_INFO("device_id= {}", Arg.device_id);


    cuda_query(Arg.device_id);

    RXMeshStatic rx(Arg.obj_file_name);

    if (rx.is_closed()) {
        RXMESH_ERROR(
            "The input mesh is closed. The input mesh should have boundaries.");
        exit(EXIT_FAILURE);
    }

    constexpr int VariableDim = 2;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    bool assmble_hessian = Arg.solver != "cg_mat_free";

    ProblemT problem(rx, assmble_hessian);

    using HessMatT      = typename ProblemT::HessMatT;
    constexpr int Order = ProblemT::DenseMatT::OrderT;

    if (Arg.solver == "chol") {
        CholeskySolver<HessMatT, Order> solver(&problem.hess);
        parameterize<T>(rx, problem, solver);
    } else if (Arg.solver == "lu") {
        CholeskySolver<HessMatT, Order> solver(&problem.hess);
        parameterize<T>(rx, problem, solver);
    } else if (Arg.solver == "cg") {
        CGSolver<T, Order> solver(
            problem.hess, 1, Arg.cg_max_iter, Arg.cg_abs_tol, Arg.cg_rel_tol);
        parameterize<T>(rx, problem, solver);
    } else if (Arg.solver == "pcg") {
        PCGSolver<T, Order> solver(
            problem.hess, 1, Arg.cg_max_iter, Arg.cg_abs_tol, Arg.cg_rel_tol);
        parameterize<T>(rx, problem, solver);
    } else if (Arg.solver == "cg_mat_free") {
        int num_rows = VariableDim * rx.get_num_vertices();

        CGMatFreeSolver<T, Order> solver(
            num_rows, 1, Arg.cg_max_iter, Arg.cg_abs_tol, Arg.cg_rel_tol);
        parameterize<T>(rx, problem, solver);
    }
}