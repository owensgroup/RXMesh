// Parallel version of
// Desbrun, Mathieu, et al "Implicit Fairing of Irregular Meshes using Diffusion
// and Curvature Flow." SIGGRAPH 1999

#include "gtest/gtest.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/cuda_query.h"

struct arg
{
    std::string obj_file_name       = STRINGIFY(INPUT_DIR) "rocker-arm.obj";
    std::string output_folder       = STRINGIFY(OUTPUT_DIR);
    std::string perm_method         = "nstdis";
    std::string solver              = "gmg";
    uint32_t    device_id           = 0;
    float       time_step           = 10;
    float       cg_tolerance        = 1e-6;
    float       gmg_tolerance_abs   = 1e-6;
    float       gmg_tolerance_rel   = 1e-6;
    uint32_t    max_num_iter        = 100;
    bool        use_uniform_laplace = true;
    int         levels              = 3;
    char**      argv;
    int         argc;
} Arg;

#include "mcf_cg.h"
#include "mcf_cg_mat_free.h"
#include "mcf_chol.h"
#include "mcf_gmg.h"


TEST(App, MCF)
{
    using namespace rxmesh;
    using dataT = float;

    // Select device
    cuda_query(Arg.device_id);

    RXMeshStatic rx(Arg.obj_file_name);

    ASSERT_TRUE(rx.is_edge_manifold());
    if (Arg.solver == "cg") {
        mcf_cg<dataT>(rx);
    } else if (Arg.solver == "pcg") {
        mcf_pcg<dataT>(rx);
    } else if (Arg.solver == "cg_mat_free") {
        mcf_cg_mat_free<dataT>(rx);
    } else if (Arg.solver == "pcg_mat_free") {
        mcf_pcg_mat_free<dataT>(rx);
    } else if (Arg.solver == "gmg") {
        mcf_gmg<dataT>(rx);
    } else if (Arg.solver == "chol") {
        mcf_cusolver_chol<dataT>(rx, string_to_permute_method(Arg.perm_method));
    } else {
        RXMESH_ERROR("Unrecognized input solver type: {}", Arg.solver);
    }
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
                        " -solver:            Solver to use. Options are cg_mat_free, pcg_mat_free, cg, pcg, chol, or gmg. Default is {}\n" 
                        " -eps:               Conjugate gradient tolerance. Default is {}\n"
                        " -perm:              Permutation method for Cholesky factorization. Default is {}\n"
                        " -max_iter:          Maximum number of iterations for iterative solvers. Default is {}\n"                                            
                        " -levels:            Number of levels in the hierarchy, inlcudes the finest level(only for GMG): {} ",
                        " -tol_abs:           Absolute tolerance for GMG solver: {}",
                        " -tol_rel:           Relative tolerance for GMG solver: {}",
                        " -device_id:         GPU device ID. Default is {}",
            Arg.obj_file_name, Arg.output_folder,  (Arg.use_uniform_laplace? "true" : "false"), Arg.time_step, Arg.solver, Arg.cg_tolerance, Arg.perm_method, Arg.max_num_iter,Arg.gmg_tolerance_abs,Arg.gmg_tolerance_rel, Arg.device_id);
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
        if (cmd_option_exists(argv, argc + argv, "-max_iter")) {
            Arg.max_num_iter =
                std::atoi(get_cmd_option(argv, argv + argc, "-max_iter"));
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
        if (cmd_option_exists(argv, argc + argv, "-perm")) {
            Arg.perm_method =
                std::string(get_cmd_option(argv, argv + argc, "-perm"));
        }
        if (cmd_option_exists(argv, argc + argv, "-solver")) {
            Arg.solver =
                std::string(get_cmd_option(argv, argv + argc, "-solver"));
        }
        if (cmd_option_exists(argv, argc + argv, "-levels")) {
            Arg.levels =
                std::atoi(get_cmd_option(argv, argv + argc, "-levels"));
        }
        if (cmd_option_exists(argv, argc + argv, "-tol_abs")) {
            Arg.gmg_tolerance_abs =
                std::atof(get_cmd_option(argv, argv + argc, "-tol_abs"));
        }
        if (cmd_option_exists(argv, argc + argv, "-tol_rel")) {
            Arg.gmg_tolerance_rel =
                std::atof(get_cmd_option(argv, argv + argc, "-tol_rel"));
        }
    }

    RXMESH_INFO("input= {}", Arg.obj_file_name);
    RXMESH_INFO("output_folder= {}", Arg.output_folder);
    RXMESH_INFO("solver= {}", Arg.solver);
    RXMESH_INFO("perm= {}", Arg.perm_method);
    RXMESH_INFO("max_num_iter= {}", Arg.max_num_iter);
    RXMESH_INFO("cg_tolerance= {0:f}", Arg.cg_tolerance);
    RXMESH_INFO("use_uniform_laplace= {}", Arg.use_uniform_laplace);
    RXMESH_INFO("time_step= {0:f}", Arg.time_step);
    RXMESH_INFO("levels= {}", Arg.levels);
    RXMESH_INFO("gmg_tolerance_rel= {}", Arg.gmg_tolerance_rel);
    RXMESH_INFO("gmg_tolerance_abs= {}", Arg.gmg_tolerance_abs);
    RXMESH_INFO("device_id= {}", Arg.device_id);

    return RUN_ALL_TESTS();
}