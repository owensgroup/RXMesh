#include <CLI/CLI.hpp>
#include <cstdlib>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/gradient_descent.h"

#include "rxmesh/util/report.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "bunnyhead.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    int         device_id     = 0;
    bool        area          = false;
    double      learning_rate = 0.01;
    int         num_iter      = 100;
    char**      argv;
    int         argc;
} Arg;


using namespace rxmesh;

#include "manual.h"

template <typename T>
void smoothing(RXMeshStatic& rx)
{
    Report report("SmoothingRXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rx, "Input");
    report.add_member("num_faces", rx.get_num_faces());

    constexpr int VariableDim = 3;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, false>;

    ProblemT problem(rx, false);

    auto v_input_pos = *rx.get_input_vertex_coordinates();

    problem.objective->copy_from(v_input_pos, DEVICE, DEVICE);

    if (Arg.area) {
        problem.template add_term<Op::FV>(
            [=] __device__(const auto& fh, const auto& iter, auto& objective) {
                assert(iter.size() == 3);

                using ActiveT = ACTIVE_TYPE(fh);

                Eigen::Vector3<ActiveT> x0 =
                    iter_val<ActiveT, 3>(fh, iter, objective, 0);
                Eigen::Vector3<ActiveT> x1 =
                    iter_val<ActiveT, 3>(fh, iter, objective, 1);
                Eigen::Vector3<ActiveT> x2 =
                    iter_val<ActiveT, 3>(fh, iter, objective, 2);

                Eigen::Vector3<ActiveT> d0 = (x1 - x0);
                Eigen::Vector3<ActiveT> d1 = (x2 - x0);
                Eigen::Vector3<ActiveT> N  = d0.cross(d1);

                return T(0.5) * N.norm();
            });
    } else {
        problem.template add_term<Op::EV>(
            [=] __device__(const auto& eh, const auto& iter, auto& objective) {
                assert(iter.size() == 2);

                using ActiveT = ACTIVE_TYPE(eh);

                // pos
                Eigen::Vector3<ActiveT> d0 =
                    iter_val<ActiveT, 3>(eh, iter, objective, 0);
                Eigen::Vector3<ActiveT> d1 =
                    iter_val<ActiveT, 3>(eh, iter, objective, 1);
                Eigen::Vector3<ActiveT> dist = (d0 - d1);

                ActiveT dist_sq = dist.squaredNorm();

                return dist_sq;
            });
    }

    GradientDescent gd(problem, Arg.learning_rate);

    GPUTimer timer;
    timer.start();

    for (int iter = 0; iter < Arg.num_iter; ++iter) {

        problem.eval_terms();

        // TODO comment out this part for benchmarking
        // float energy = problem.get_current_loss();
        // if (iter % 10 == 0) {
        //    RXMESH_INFO("Iteration = {}: Energy = {}", iter, energy);
        //}

        gd.take_step();
    }
    timer.stop();

    CUDA_ERROR(cudaDeviceSynchronize());

    RXMESH_INFO("Smoothing GD took {} (ms), ms/iter = {} ",
                timer.elapsed_millis(),
                timer.elapsed_millis() / float(Arg.num_iter));

    report.add_member("method", std::string("Ours"));
    report.add_member("total_time_ms", timer.elapsed_millis());
    report.add_member("num_iter", Arg.num_iter);

    report.write(Arg.output_folder + "/rxmesh_smoothing",
                 "Smoothing_RXMesh_" + extract_file_name(Arg.obj_file_name));

#if USE_POLYSCOPE
    problem.objective->move(DEVICE, HOST);
    rx.get_polyscope_mesh()->updateVertexPositions(*problem.objective);
#endif
}

int main(int argc, char** argv)
{
    using T = float;

    CLI::App app{"Smoothing - Gradient descent mesh smoothing"};

    app.add_option("-i,--input", Arg.obj_file_name, "Input OBJ mesh file")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "bunnyhead.obj"));

    app.add_flag("--area", Arg.area, "Use area-based gradients instead of edge-based");

    app.add_option("-n,--iter", Arg.num_iter, "Number of iterations")
        ->default_val(100);

    app.add_option("--lr", Arg.learning_rate, "Gradient descent learning rate")
        ->default_val(0.01);

    app.add_option("-o,--output", Arg.output_folder, "JSON file output folder")
        ->default_val(std::string(STRINGIFY(OUTPUT_DIR)));

    app.add_option("-d,--device_id", Arg.device_id, "GPU device ID")
        ->default_val(0);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    rx_init(Arg.device_id);

    Arg.argv = argv;
    Arg.argc = argc;

    RXMESH_INFO("input= {}", Arg.obj_file_name);
    RXMESH_INFO("area= {}", (Arg.area ? "true" : "false"));
    RXMESH_INFO("iter= {}", Arg.num_iter);
    RXMESH_INFO("lr= {}", Arg.learning_rate);
    RXMESH_INFO("output_folder= {}", Arg.output_folder);
    RXMESH_INFO("device_id= {}", Arg.device_id);

    RXMeshStatic rx(Arg.obj_file_name);

    smoothing<T>(rx);

    manual<T>(rx);

#if USE_POLYSCOPE
    polyscope::show();
#endif
}