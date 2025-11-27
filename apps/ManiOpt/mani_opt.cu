// Reference Implementation
// https://github.com/patr-schm/TinyAD-Examples/blob/main/apps/manifold_optimization.cc
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/lbfgs_solver.h"
#include "rxmesh/diff/newton_solver.h"

struct arg
{
    std::string obj_file_name   = STRINGIFY(INPUT_DIR) "giraffe.obj";
    std::string embed_file_name = STRINGIFY(INPUT_DIR) "giraffe_embedding.obj";
    std::string output_folder   = STRINGIFY(OUTPUT_DIR);
    uint32_t    device_id       = 0;
    std::string solver          = "newton";
    int         history         = 5;
    uint32_t    max_iter        = 100;
    char**      argv;
    int         argc;
} Arg;


using namespace rxmesh;

enum class Direction
{
    Default   = 0,
    Equator   = 1,
    NorthPole = 2,
};

std::string direction_name(Direction dir)
{
    switch (dir) {
        case Direction::Default:
            return "default";
            break;
        case Direction::Equator:
            return "equator";
            break;
        case Direction::NorthPole:
            return "northpole";
            break;
        default:
            return "unknown";
            break;
    }
}

template <typename T>
void add_mesh_to_polyscope(RXMeshStatic&       rx,
                           VertexAttribute<T>& v,
                           std::string         name)
{
#ifdef USE_POLYSCOPE
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

template <typename T>
__host__ __device__ Eigen::Vector3<T> any_tangent(const Eigen::Vector3<T>& _p)
{
    // Compute an arbitrary tangent vector of the sphere at position p.
    //
    // Find coordinate axis spanning the largest angle with _p.
    // Return cross product that of axis with _p
    Eigen::Vector3<T> tang;

    T min_dot2 = std::numeric_limits<T>::max();

    Eigen::Vector3<T> list[3] = {Eigen::Vector3<T>(1.0, 0.0, 0.0),
                                 Eigen::Vector3<T>(0.0, 1.0, 0.0),
                                 Eigen::Vector3<T>(0.0, 0.0, 1.0)};

    for (const Eigen::Vector3<T>& ax : list) {
        T dot2 = _p.dot(ax);
        dot2 *= dot2;
        if (dot2 < min_dot2) {
            min_dot2 = dot2;
            tang     = ax.cross(_p).normalized();
        }
    }

    return tang;
}

template <typename T>
void compute_local_bases(const RXMeshStatic&       rx,
                         const VertexAttribute<T>& S,
                         VertexAttribute<T>&       B1,
                         VertexAttribute<T>&       B2)
{
    // Compute an orthonormal tangent space basis of the sphere at each vertex.

    rx.for_each_vertex(DEVICE,
                       [S, B1, B2] __device__(const VertexHandle vh) mutable {
                           Eigen::Vector3<T> s = S.template to_eigen<3>(vh);

                           Eigen::Vector3<T> b1 = any_tangent(s);

                           Eigen::Vector3<T> b2 = s.cross(b1);

                           B1.from_eigen(vh, b1);
                           B2.from_eigen(vh, b2);
                       });
}

template <typename U, typename T>
__host__ __device__ Eigen::Vector3<U> retract(Eigen::Vector2<U>&        v_tang,
                                              const VertexHandle&       vh,
                                              const VertexAttribute<T>& S,
                                              const VertexAttribute<T>& B1,
                                              const VertexAttribute<T>& B2)
{
    // Retraction operator: map from a local tangent space to the
    // sphere.

    // Evaluate target point in 3D ambient space and project to
    // sphere via normalization.

    const Eigen::Vector3<T> s  = S.template to_eigen<3>(vh);
    const Eigen::Vector3<T> b1 = B1.template to_eigen<3>(vh);
    const Eigen::Vector3<T> b2 = B2.template to_eigen<3>(vh);

    const Eigen::Vector3<U> ret =
        (s + v_tang[0] * b1 + v_tang[1] * b2).normalized().eval();

    return ret;
}

template <typename T, typename ProblemT, typename SolverT>
void manifold_optimization(RXMeshStatic&                          rx,
                           ProblemT&                              problem,
                           SolverT&                               solver,
                           const std::vector<std::vector<float>>& init_s,
                           const Direction                        dir)
{


    auto S  = *rx.add_vertex_attribute<T>(init_s, "S");
    auto B1 = *rx.add_vertex_attribute<T>("B1", 3);
    auto B2 = *rx.add_vertex_attribute<T>("B2", 3);

    // auto fcolor = *rx.add_face_attribute<T>("fColor", 1);

    compute_local_bases(rx, S, B1, B2);


    // add energy term
    problem.template add_term<Op::FV, true>([=] __device__(const auto& fh,
                                                           const auto& iter,
                                                           auto&       obj) {
        // fh is a face handle
        // iter is an iterator over fh's vertices


        assert(iter[0].is_valid() && iter[1].is_valid() && iter[2].is_valid());

        assert(iter.size() == 3);

        using ActiveT = ACTIVE_TYPE(fh);

        // tangent vectors at the triangle three vertices (a,b,c)
        Eigen::Vector2<ActiveT> a_tang = iter_val<ActiveT, 2>(fh, iter, obj, 0);
        Eigen::Vector2<ActiveT> b_tang = iter_val<ActiveT, 2>(fh, iter, obj, 1);
        Eigen::Vector2<ActiveT> c_tang = iter_val<ActiveT, 2>(fh, iter, obj, 2);


        // Retract 2D tangent vectors to 3D points on the sphere.
        Eigen::Vector3<ActiveT> a_mani = retract(a_tang, iter[0], S, B1, B2);
        Eigen::Vector3<ActiveT> b_mani = retract(b_tang, iter[1], S, B1, B2);
        Eigen::Vector3<ActiveT> c_mani = retract(c_tang, iter[2], S, B1, B2);


        // Objective: injectivity barrier + Dirichlet energy
        ActiveT volume =
            (1.0 / 6.0) * col_mat(a_mani, b_mani, c_mani).determinant();

        if (volume <= 0.0) {
            using PassiveT = PassiveType<ActiveT>;
            return ActiveT(std::numeric_limits<PassiveT>::max());
        }

        ActiveT E = -0.1 * log(volume);
        E += (a_mani - b_mani).squaredNorm() + (b_mani - c_mani).squaredNorm() +
             (c_mani - a_mani).squaredNorm();

        if (dir == Direction::Equator) {
            E += sqr(a_mani.y()) + sqr(b_mani.y()) + sqr(c_mani.y());
        } else if (dir == Direction::NorthPole) {
            E += sqr(1.0 - a_mani.y()) + sqr(1.0 - b_mani.y()) +
                 sqr(1.0 - c_mani.y());
        }

        //(void)fcolor;
        // if constexpr (is_scalar_v<ActiveT>) {
        //    fcolor(fh) = E.val();
        //}
        return E;
    });


    T convergence_eps = 1e-1;

    int iter;

    Timers<GPUTimer> timer;
    timer.add("Total");
    timer.add("Diff");

    timer.start("Total");

    if (Arg.solver == "lbfgs") {
        timer.start("Diff");
        problem.eval_terms();
        timer.stop("Diff");
    }

    for (iter = 0; iter < Arg.max_iter; ++iter) {


        problem.objective->reset(0, DEVICE);

        timer.start("Diff");

        if (Arg.solver == "newton") {
            problem.eval_terms();
            timer.stop("Diff");
        }


        /*T f = problem.get_current_loss();
        RXMESH_INFO("Iteration= {}: Energy = {}", iter, f);*/


        solver.compute_direction();

        // RXMESH_INFO(
        //     "   grad.norm2()= {}, dir.norm2() = {},"
        //     " grad.dot(dir)={} ",
        //     problem.grad.norm2(),
        //     solver.dir.norm2(),
        //     problem.grad.dot(solver.dir));


        // if (0.5f * problem.grad.dot(solver.dir) < convergence_eps) {
        //     break;
        // }

        if (problem.grad.norm2() < convergence_eps) {
            break;
        }

        if (Arg.solver == "newton") {
            solver.line_search();
        } else {
            solver.line_search(1.0, 0.8, 200);
            timer.stop("Diff");
        }


        // Re-center local bases
        rx.for_each_vertex(DEVICE,
                           [S, B1, B2, obj = *problem.objective] __device__(
                               const VertexHandle h) mutable {
                               Eigen::Vector2<T> v =
                                   obj.template to_eigen<2>(h);
                               Eigen::Vector3<T> s = retract(v, h, S, B1, B2);
                               S.from_eigen(h, s);
                           });

        compute_local_bases(rx, S, B1, B2);
    }
    timer.stop("Total");

    RXMESH_INFO(
        "Manifold Optimization: iterations ={}, time= {} (ms), diff_time= {} "
        "(ms), final objective= {}",
        iter,
        timer.elapsed_millis("Total"),
        timer.elapsed_millis("Diff"),
        problem.get_current_loss());

#ifdef USE_POLYSCOPE
    S.move(DEVICE, HOST);

    add_mesh_to_polyscope(rx, S, direction_name(dir));

    // auto ps = polyscope::registerSurfaceMesh(
    //     direction_name(dir) + std::to_string(iter),
    //     S,
    //     rx.get_polyscope_mesh()->faces);
    //
    // fcolor.move(DEVICE, HOST);
    // auto ps_q = ps->addFaceScalarQuantity("Energy", fcolor);
    // ps_q->setEnabled(true);
    //  ps_q->setMapRange({6.466e-01, 5.621e+00});

    // ps->setEnabled(false);
    // rx.export_obj("mani_" + std::to_string(iter) + ".obj", S);


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
                        " -embed:             Input initial embedding mesh (OBJ file). Default is {} \n"
                        " -o:                 JSON file output folder. Default is {} \n"
                        " -solver:            Solver to use. Options are newton and lbfgs. Default is {}\n",
                        " -max_iter:          Maximum number of iterations for Newton solver. Default is {}\n"
                        " -history:           History size in LBFGS. Default is {}\n"
                        " -device_id:         GPU device ID. Default is {}",                        
            Arg.obj_file_name, Arg.embed_file_name, Arg.output_folder, Arg.solver, Arg.max_iter, Arg.history, Arg.device_id);
            // clang-format on
            exit(EXIT_SUCCESS);
        }

        if (cmd_option_exists(argv, argc + argv, "-input")) {
            Arg.obj_file_name =
                std::string(get_cmd_option(argv, argv + argc, "-input"));
        }
        if (cmd_option_exists(argv, argc + argv, "-embed")) {
            Arg.embed_file_name =
                std::string(get_cmd_option(argv, argv + argc, "-embed"));
        }

        if (cmd_option_exists(argv, argc + argv, "-solver")) {
            Arg.solver =
                std::string(get_cmd_option(argv, argv + argc, "-solver"));
        }

        if (cmd_option_exists(argv, argc + argv, "-o")) {
            Arg.output_folder =
                std::string(get_cmd_option(argv, argv + argc, "-o"));
        }

        if (cmd_option_exists(argv, argc + argv, "-max_iter")) {
            Arg.max_iter =
                std::atoi(get_cmd_option(argv, argv + argc, "-max_iter"));
        }

        if (cmd_option_exists(argv, argc + argv, "-history")) {
            Arg.history =
                std::atoi(get_cmd_option(argv, argv + argc, "-history"));
        }

        if (cmd_option_exists(argv, argc + argv, "-device_id")) {
            Arg.device_id =
                atoi(get_cmd_option(argv, argv + argc, "-device_id"));
        }
    }

    RXMESH_INFO("input= {}", Arg.obj_file_name);
    RXMESH_INFO("embed= {}", Arg.embed_file_name);
    RXMESH_INFO("output_folder= {}", Arg.output_folder);
    RXMESH_INFO("solver= {}", Arg.solver);
    RXMESH_INFO("max_iter= {}", Arg.max_iter);
    RXMESH_INFO("device_id= {}", Arg.device_id);


    RXMeshStatic rx(Arg.obj_file_name);

    std::vector<std::vector<uint32_t>> fv;
    std::vector<std::vector<float>>    init_s;
    import_obj(Arg.embed_file_name, init_s, fv);

    if (rx.get_num_faces() != fv.size()) {
        RXMESH_ERROR(
            "The input mesh and initial embedding have different number of "
            "faces");
    }

    if (rx.get_num_vertices() != init_s.size()) {
        RXMESH_ERROR(
            "The input mesh and initial embedding have different number of "
            "vertices");
    }

    if (Arg.solver == "newton") {

        using ProblemT = DiffScalarProblem<T, 2, VertexHandle, true>;
        ProblemT problem(rx, true);
        using HessMatT = typename ProblemT::HessMatT;
        LUSolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(
            problem.hess.get());
        NetwtonSolver newton_solver(problem, &solver);

        manifold_optimization<T>(
            rx, problem, newton_solver, init_s, Direction::Default);
    } else if (Arg.solver == "lbfgs") {
        using ProblemT = DiffScalarProblem<T, 2, VertexHandle, false>;
        ProblemT problem(rx);

        LBFGSSolver lbfgs_solver(problem, Arg.history);
        manifold_optimization<T>(
            rx, problem, lbfgs_solver, init_s, Direction::Default);
    }
}