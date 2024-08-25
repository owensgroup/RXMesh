// Parallel version of
// Desbrun, Mathieu, et al "Implicit Fairing of Irregular Meshes using Diffusion
// and Curvature Flow." SIGGRAPH 1999

#include <omp.h>

#include "gtest/gtest.h"
#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/export_tools.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/log.h"

struct arg
{
    std::string obj_file_name       = STRINGIFY(INPUT_DIR) "dragon.obj";
    std::string output_folder       = STRINGIFY(OUTPUT_DIR);
    uint32_t    device_id           = 0;
    float       time_step           = 0.001;
    float       cg_tolerance        = 1e-6;
    uint32_t    max_num_cg_iter     = 1000;
    bool        use_uniform_laplace = false;
    uint32_t    nd_level            = 1;
    char**      argv;
    int         argc;
} Arg;

#include "mcf_cg.h"
#include "mcf_cusolver_chol.cuh"


TEST(App, MCF)
{
    using namespace rxmesh;
    using dataT = float;

    // Select device
    cuda_query(Arg.device_id);

    RXMeshStatic rx(Arg.obj_file_name);

    // RXMesh Impl
    mcf_cg<dataT>(rx);

    // RXMesh cusolver Impl
    mcf_cusolver_chol<dataT>(rx, PermuteMethod::CUSTOM);
}

int main(int argc, char** argv)
{
    using namespace rxmesh;
    Log::init();

    ::testing::InitGoogleTest(&argc, argv);
    Arg.argv = argv;
    Arg.argc = argc;
    if (argc > 1) {
        if (cmd_option_exists(argv, argc + argv, "-h")) {
            // clang-format off
            RXMESH_INFO("\nUsage: MCF.exe < -option X>\n"
                        " -h:                 Display this massage and exit\n"
                        " -input:             Input file. Input file should be under the input/ subdirectory\n"
                        "                     Default is {} \n"
                        "                     Hint: Only accept OBJ files\n"
                        " -o:                 JSON file output folder. Default is {} \n"
                        " -uniform_laplace:   Use uniform Laplace weights. Default is {} \n"
                        " -dt:                Time step (delta t). Default is {} \n"
                        "                     Hint: should be between (0.001, 1) for cotan Laplace or between (1, 100) for uniform Laplace\n"
                        " -eps:               Conjugate gradient tolerance. Default is {}\n"
                        " -max_cg_iter:       Conjugate gradient maximum number of iterations. Default is {}\n"
                        " -nd_level:          ND level. Default is {}\n"                        
                        " -device_id:         GPU device ID. Default is {}",
            Arg.obj_file_name, Arg.output_folder,  (Arg.use_uniform_laplace? "true" : "false"), Arg.time_step, Arg.cg_tolerance, Arg.max_num_cg_iter, Arg.device_id);
            // clang-format on
            exit(EXIT_SUCCESS);
        }

        if (cmd_option_exists(argv, argc + argv, "-input")) {
            Arg.obj_file_name =
                std::string(get_cmd_option(argv, argv + argc, "-input"));
        }
        if (cmd_option_exists(argv, argc + argv, "-o")) {
            Arg.output_folder =
                std::string(get_cmd_option(argv, argv + argc, "-o"));
        }
        if (cmd_option_exists(argv, argc + argv, "-dt")) {
            Arg.time_step = std::atof(get_cmd_option(argv, argv + argc, "-dt"));
        }
        if (cmd_option_exists(argv, argc + argv, "-max_cg_iter")) {
            Arg.max_num_cg_iter =
                std::atoi(get_cmd_option(argv, argv + argc, "-max_cg_iter"));
        }

        if (cmd_option_exists(argv, argc + argv, "-eps")) {
            Arg.cg_tolerance =
                std::atof(get_cmd_option(argv, argv + argc, "-eps"));
        }
        if (cmd_option_exists(argv, argc + argv, "-uniform_laplace")) {
            Arg.use_uniform_laplace = true;
        }
        if (cmd_option_exists(argv, argc + argv, "-device_id")) {
            Arg.device_id =
                atoi(get_cmd_option(argv, argv + argc, "-device_id"));
        }
        if (cmd_option_exists(argv, argc + argv, "-nd_level")) {
            Arg.nd_level = atoi(get_cmd_option(argv, argv + argc, "-nd_level"));
        }
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("max_num_cg_iter= {}", Arg.max_num_cg_iter);
    RXMESH_TRACE("cg_tolerance= {0:f}", Arg.cg_tolerance);
    RXMESH_TRACE("use_uniform_laplace= {}", Arg.use_uniform_laplace);
    RXMESH_TRACE("time_step= {0:f}", Arg.time_step);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    return RUN_ALL_TESTS();
}