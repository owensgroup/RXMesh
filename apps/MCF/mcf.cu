// Parallel version of
// Desbrun, Mathieu, et al "Implicit Fairing of Irregular Meshes using Diffusion
// and Curvature Flow." SIGGRAPH 1999

#include <CLI/CLI.hpp>
#include <cstdlib>

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/log.h"

struct arg
{
    std::string obj_file_name        = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder        = STRINGIFY(OUTPUT_DIR);
    std::string perm_method          = "nstdis";
    std::string solver               = "chol";
    uint32_t    device_id            = 0;
    float       time_step            = 10;
    float       tol_abs              = 1e-6;
    float       tol_rel              = 0.0;
    uint32_t    max_num_iter         = 100;
    bool        use_uniform_laplace  = true;
    std::string gmg_csolver          = "cholesky";
    std::string gmg_sampling         = "random";
    int         gmg_levels           = 5;
    int         gmg_threshold        = 1000;
    bool        gmg_render_hierarchy = false;
    bool        create_mat           = false;
    bool        gmg_pruned_ptap      = false;
    bool        gmg_verify_ptap      = false;
    char**      argv;
    int         argc;
} Arg;

#include "mcf_cg.h"
#include "mcf_cg_mat_free.h"
#include "mcf_chol.h"
#include "mcf_gmg.h"

#ifdef USE_CUDSS
#include "mcf_cudss.h"
#endif

void creat_matrices(rxmesh::RXMeshStatic& rx)
{
    using namespace rxmesh;

    uint32_t num_vertices = rx.get_num_vertices();

    auto coords = rx.get_input_vertex_coordinates();

    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  B_mat(rx, num_vertices, 3, LOCATION_ALL);

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

int mcf()
{
    using namespace rxmesh;
    using dataT = float;

    // Select device
    cuda_query(Arg.device_id);

    RXMeshStatic rx(Arg.obj_file_name, "", 256);

    if (!rx.is_edge_manifold()) {
        RXMESH_ERROR("MCF requires an edge-manifold mesh");
        return EXIT_FAILURE;
    }

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
        mcf_gmg<dataT>(rx,
                       string_to_coarse_solver(Arg.gmg_csolver),
                       string_to_sampling(Arg.gmg_sampling));
    } else if (Arg.solver == "chol") {
        mcf_cusolver_chol<dataT>(rx, string_to_permute_method(Arg.perm_method));
#ifdef USE_CUDSS
    } else if (Arg.solver == "cudss_chol") {
        mcf_cudss_chol<dataT>(rx, string_to_permute_method(Arg.perm_method));
#endif
    } else {
        RXMESH_ERROR("Unrecognized input solver type: {}", Arg.solver);
        return EXIT_FAILURE;
    }

    return 0;
}

int main(int argc, char** argv)
{
    using namespace rxmesh;

    CLI::App app{
        "MCF - Implicit Fairing of Irregular Meshes using Diffusion and "
        "Curvature Flow"};

    app.add_option("-i,--input", Arg.obj_file_name, "Input OBJ mesh file")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "dragon.obj"));

    app.add_option("-o,--output", Arg.output_folder, "JSON file output folder")
        ->default_val(std::string(STRINGIFY(OUTPUT_DIR)));

    app.add_flag("--uniform_laplace",
                 Arg.use_uniform_laplace,
                 "Use uniform Laplace weights")
        ->default_val(true);

    app.add_option("--dt", Arg.time_step, "Time step (delta t)")
        ->default_val(10.0f);

    app.add_option("-s,--solver", Arg.solver, "Solver to use")
        ->default_val(std::string("chol"))
        ->check(CLI::IsMember({"cg",
                               "pcg",
                               "cg_mat_free",
                               "pcg_mat_free",
                               "chol",
                               "cudss_chol",
                               "gmg"}));

    app.add_option("--perm",
                   Arg.perm_method,
                   "Permutation method for Cholesky factorization")
        ->default_val(std::string("nstdis"))
        ->check(
            CLI::IsMember({"symrcm", "symamd", "nstdis", "gpumgnd", "gpund"}));

    app.add_option("--max_iter",
                   Arg.max_num_iter,
                   "Maximum number of iterations for iterative solvers")
        ->default_val(100u);

    app.add_option(
           "--tol_abs", Arg.tol_abs, "Iterative solver absolute tolerance")
        ->default_val(1e-6f);

    app.add_option(
           "--tol_rel", Arg.tol_rel, "Iterative solver relative tolerance")
        ->default_val(0.0f);

    app.add_flag("--create_mat",
                 Arg.create_mat,
                 "Export the linear system matrices (.mtx) and mesh obj to "
                 "files and exit");

    app.add_option("--gmg_levels",
                   Arg.gmg_levels,
                   "GMG number of levels in the hierarchy")
        ->default_val(5);

    app.add_option("--gmg_csolver", Arg.gmg_csolver, "GMG coarse solver")
        ->default_val(std::string("cholesky"))
        ->check(CLI::IsMember({"jacobi", "cholesky", "cudsscholesky"}));

    app.add_option("--gmg_sampling",
                   Arg.gmg_sampling,
                   "GMG sampling method to create the hierarchy")
        ->default_val(std::string("random"))
        ->check(CLI::IsMember({"random", "fps", "kmeans"}));

    app.add_option("--gmg_threshold",
                   Arg.gmg_threshold,
                   "GMG threshold for the coarsest level")
        ->default_val(1000);

    app.add_flag("--gmg_pruned_ptap",
                 Arg.gmg_pruned_ptap,
                 "GMG toggle using pruned PtAP for fast construction");

    app.add_flag("--gmg_verify_ptap",
                 Arg.gmg_verify_ptap,
                 "GMG toggle verifying the construction of PtAP");

    app.add_flag("--gmg_rh",
                 Arg.gmg_render_hierarchy,
                 "GMG toggle rendering the hierarchy");

    app.add_option("-d,--device_id", Arg.device_id, "GPU device ID")
        ->default_val(0u);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    rx_init(Arg.device_id);

    Arg.argv = argv;
    Arg.argc = argc;

    RXMESH_INFO("input= {}", Arg.obj_file_name);
    RXMESH_INFO("output_folder= {}", Arg.output_folder);
    RXMESH_INFO("use_uniform_laplace= {}", Arg.use_uniform_laplace);
    RXMESH_INFO("time_step= {0:f}", Arg.time_step);
    RXMESH_INFO("solver= {}", Arg.solver);
    RXMESH_INFO("perm= {}", Arg.perm_method);
    RXMESH_INFO("max_num_iter= {}", Arg.max_num_iter);
    RXMESH_INFO("tol_abs= {}", Arg.tol_abs);
    RXMESH_INFO("tol_rel= {}", Arg.tol_rel);
    RXMESH_INFO("create_mat= {}", Arg.create_mat);
    RXMESH_INFO("gmg_levels= {}", Arg.gmg_levels);
    RXMESH_INFO("gmg_csolver= {}", Arg.gmg_csolver);
    RXMESH_INFO("gmg_sampling= {}", Arg.gmg_sampling);
    RXMESH_INFO("gmg_threshold= {}", Arg.gmg_threshold);
    RXMESH_INFO("gmg_pruned_ptap= {}", Arg.gmg_pruned_ptap);
    RXMESH_INFO("gmg_verify_ptap= {}", Arg.gmg_verify_ptap);
    RXMESH_INFO("gmg_render_hierarchy= {}", Arg.gmg_render_hierarchy);
    RXMESH_INFO("device_id= {}", Arg.device_id);

    return mcf();
}