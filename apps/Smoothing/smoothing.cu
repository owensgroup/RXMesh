#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/gradient_descent.h"

#include "rxmesh/util/report.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "bunnyhead.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    uint32_t    device_id     = 0;
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

    ProblemT problem(rx);

    auto v_input_pos = *rx.get_input_vertex_coordinates();

    problem.objective->copy_from(v_input_pos, DEVICE, DEVICE);


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
    Log::init(spdlog::level::info);

    using T = float;

    Arg.argv = argv;
    Arg.argc = argc;


    if (argc > 1) {
        if (cmd_option_exists(argv, argc + argv, "-h")) {
            // clang-format off
            RXMESH_INFO("\nUsage: Param.exe < -option X>\n"
                        " -h:                 Display this massage and exit\n"
                        " -input:      Input OBJ mesh file. Default is {} \n"
                        " -iter:              Number of iterations. Default is {} \n"
                        " -lr:                Gradient descent learning rate. Default is {} \n"
                        " -o:                 JSON file output folder. Default is {} \n"
                        " -device_id:         GPU device ID. Default is {}",
            Arg.obj_file_name, Arg.num_iter, Arg.learning_rate, Arg.output_folder, Arg.device_id);
            // clang-format on
            exit(EXIT_SUCCESS);
        }

        if (cmd_option_exists(argv, argc + argv, "-input")) {
            Arg.obj_file_name =
                std::string(get_cmd_option(argv, argv + argc, "-input"));
        }
        if (cmd_option_exists(argv, argc + argv, "-lr")) {
            Arg.learning_rate = atof(get_cmd_option(argv, argv + argc, "-lr"));
        }
        if (cmd_option_exists(argv, argc + argv, "-o")) {
            Arg.output_folder =
                std::string(get_cmd_option(argv, argv + argc, "-o"));
        }

        if (cmd_option_exists(argv, argc + argv, "-device_id")) {
            Arg.device_id =
                atoi(get_cmd_option(argv, argv + argc, "-device_id"));
        }
        if (cmd_option_exists(argv, argc + argv, "-iter")) {
            Arg.num_iter = atoi(get_cmd_option(argv, argv + argc, "-iter"));
        }
    }

    RXMESH_INFO("input= {}", Arg.obj_file_name);
    RXMESH_INFO("iter= {}", Arg.num_iter);
    RXMESH_INFO("lr= {}", Arg.learning_rate);
    RXMESH_INFO("output_folder= {}", Arg.output_folder);
    RXMESH_INFO("device_id= {}", Arg.device_id);


    cuda_query(Arg.device_id);

    RXMeshStatic rx(Arg.obj_file_name);


    smoothing<T>(rx);

    manual<T>(rx);

    // #if USE_POLYSCOPE
    //     polyscope::show();
    // #endif
}