// Parallel version of
// Desbrun, Mathieu, et al "Implicit Fairing of Irregular Meshes using Diffusion
// and Curvature Flow." SIGGRAPH 1999

#include "gtest/gtest.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/cuda_query.h"

struct arg
{
    std::string obj_file_name       = STRINGIFY(INPUT_DIR) "dragon.obj";
    std::string output_folder       = STRINGIFY(OUTPUT_DIR);
    std::string perm_method         = "nstdis";
    std::string solver              = "gmg";
    std::string coarse_solver       = "jacobi";
    uint32_t    device_id           = 0;
    float       time_step           = 10;
    float       tol_abs             = 1e-6;
    float       tol_rel             = 0.0;
    uint32_t    max_num_iter        = 100;
    bool        use_uniform_laplace = true;
    int         levels              = 5;
    int         threshold           = 1000;
    bool        render_hierarchy    = false;
    bool        create_mat          = false;
    bool        use_new_ptap        = true;  //
    bool        ptap_verify         = true;  //
    char**      argv;
    int         argc;
} Arg;

#include "mcf_cg.h"
#include "mcf_cg_mat_free.h"
#include "mcf_chol.h"
#include "mcf_gmg.h"

void creat_matrices(rxmesh::RXMeshStatic& rx)
{
    using namespace rxmesh;

    uint32_t num_vertices = rx.get_num_vertices();

    auto coords = rx.get_input_vertex_coordinates();

    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  B_mat(rx, num_vertices, 3);

    rx.run_kernel<256>({Op::VV},
                       mcf_B_setup<float, 256>,
                       *coords,
                       B_mat,
                       Arg.use_uniform_laplace);

    rx.run_kernel<256>({Op::VV},
                       mcf_A_setup<float, 256>,
                       *coords,
                       A_mat,
                       Arg.use_uniform_laplace,
                       Arg.time_step);


    A_mat.move(DEVICE, HOST);
    B_mat.move(DEVICE, HOST);
    std::string output_dir = Arg.output_folder + "MCF_matrices";

    auto A_mat_copy = A_mat.to_eigen_copy();
    auto B_mat_copy = B_mat.to_eigen();

    // std::cout << B_mat_copy << "\n";
    // std::cout << "\n*******\n*******\n";
    // std::cout << A_mat_copy << "\n";

    std::filesystem::create_directories(output_dir);

    Eigen::saveMarketDense(
        B_mat_copy,
        output_dir + "/" + extract_file_name(Arg.obj_file_name) + "_B.mtx");
    Eigen::saveMarket(
        A_mat_copy,
        output_dir + "/" + extract_file_name(Arg.obj_file_name) + "_A.mtx");

    rx.export_obj(
        output_dir + "/" + extract_file_name(Arg.obj_file_name) + ".obj",
        *coords);

    RXMESH_INFO("Wrote A and b .mtx files and mesh obj to {}/", output_dir);
}

TEST(App, MCF)
{
    using namespace rxmesh;
    using dataT = float;

    // Select device
    cuda_query(Arg.device_id);

    RXMeshStatic rx(Arg.obj_file_name, "", 256);

    ASSERT_TRUE(rx.is_edge_manifold());
    if (Arg.create_mat) {
        creat_matrices(rx);
    } else if (Arg.solver == "cg") {
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
    Log::init(spdlog::level::info);

    ::testing::InitGoogleTest(&argc, argv);
    Arg.argv = argv;
    Arg.argc = argc;
    if (argc > 1) {
        if (cmd_option_exists(argv, argc + argv, "-h")) {
            // clang-format off
            RXMESH_INFO("\nUsage: MCF.exe < -option X>\n"
                        " -h:                 Display this massage and exit\n"
                        " -input:             Input file. Input file should be under the input/ subdirectory. Default is {}\n"                        
                        " -o:                 JSON file output folder. Default is {} \n"
                        " -uniform_laplace:   Use uniform Laplace weights. Default is {} \n"
                        " -dt:                Time step (delta t). Default is {} \n"
                        "                     Hint: should be between (0.001, 1) for cotan Laplace or between (1, 100) for uniform Laplace\n"
                        " -solver:            Solver to use. Options are cg_mat_free, pcg_mat_free, cg, pcg, chol, or gmg. Default is {}\n"                         
                        " -perm:              Permutation method for Cholesky factorization (symrcm, symamd, nstdis, gpumgnd, gpund). Default is {}\n"
                        " -max_iter:          Maximum number of iterations for iterative solvers. Default is {}\n"                                            
                        " -tol_abs:           Iterative solver absolute tolerance. Default is {}\n"
                        " -tol_rel:           Iterative solver relative tolerance. Default is {}\n"
                        " -create_mat:        Export the linear system matrices (.mtx) and mesh obj to files and exit. Default is {}\n"
                        " -levels:            GMG number of levels in the hierarchy, includes the finest level. Default is {}\n"
                        " -csolver:           GMG coarse solver. Default is {}\n"
                        " -threshold:         GMG threshold for the coarsest level in the hierarchy, i.e., number of vertices in the coarsest level. Default is {}\n"
                        " -rh:                GMG render hierarchy. Default is {}\n"
                        " -device_id:         GPU device ID. Default is {}\n",
            Arg.obj_file_name, Arg.output_folder,  
            (Arg.use_uniform_laplace? "true" : "false"), 
            Arg.time_step, 
            Arg.solver,             
            Arg.perm_method, 
            Arg.max_num_iter,            
            Arg.tol_abs,
            Arg.tol_rel, 
            Arg.create_mat,
            Arg.levels,
            Arg.coarse_solver,
            Arg.threshold,
            (Arg.render_hierarchy? "true" : "false"),
            Arg.device_id);
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
            Arg.tol_abs =
                std::atof(get_cmd_option(argv, argv + argc, "-tol_abs"));
        }
        if (cmd_option_exists(argv, argc + argv, "-tol_rel")) {
            Arg.tol_rel =
                std::atof(get_cmd_option(argv, argv + argc, "-tol_rel"));
        }
        if (cmd_option_exists(argv, argc + argv, "-threshold")) {
            Arg.threshold =
                std::atoi(get_cmd_option(argv, argv + argc, "-threshold"));
        }
        if (cmd_option_exists(argv, argc + argv, "-csolver")) {
            Arg.coarse_solver =
                std::string(get_cmd_option(argv, argv + argc, "-csolver"));
        }
        if (cmd_option_exists(argv, argc + argv, "-rh")) {
            Arg.render_hierarchy = true;
        }

        if (cmd_option_exists(argv, argc + argv, "new_ptap")) {
            Arg.use_new_ptap = true;
        }

        if (cmd_option_exists(argv, argc + argv, "-create_mat")) {
            Arg.create_mat = true;
        }
        if (cmd_option_exists(argv, argc + argv, "-v")) {
            Arg.ptap_verify = true;
        }
    }

    RXMESH_INFO("input= {}", Arg.obj_file_name);
    RXMESH_INFO("output_folder= {}", Arg.output_folder);
    RXMESH_INFO("solver= {}", Arg.solver);
    RXMESH_INFO("perm= {}", Arg.perm_method);
    RXMESH_INFO("max_num_iter= {}", Arg.max_num_iter);
    RXMESH_INFO("use_uniform_laplace= {}", Arg.use_uniform_laplace);
    RXMESH_INFO("time_step= {0:f}", Arg.time_step);
    RXMESH_INFO("levels= {}", Arg.levels);
    RXMESH_INFO("tol_abs= {}", Arg.tol_abs);
    RXMESH_INFO("tol_rel= {}", Arg.tol_rel);
    RXMESH_INFO("device_id= {}", Arg.device_id);
    RXMESH_INFO("GMG Coarse solver= {}", Arg.coarse_solver);
    RXMESH_INFO("Render GMG hierarchy= {}", Arg.render_hierarchy);

    return RUN_ALL_TESTS();
}