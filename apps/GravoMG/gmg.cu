#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "include/GPUGMG.h"

#include "include/interactive.h"

#include "rxmesh/geometry_factory.h"

#include "include/GMG.h"
#include "include/v_cycle.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "torus.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    uint32_t    device_id     = 0;
    bool        offline       = false;
    char**      argv;
    int         argc;
} Arg;


TEST(Apps, GMGRefactor)
{
    using namespace rxmesh;

    using T = float;

    cuda_query(Arg.device_id);

    RXMeshStatic rx(Arg.obj_file_name);

    /*
    std::vector<std::vector<float>>    planeVerts;
    std::vector<std::vector<uint32_t>> planeFaces;
    uint32_t                           nx = 10;
    uint32_t                           ny = 10;
    create_plane(planeVerts, planeFaces, nx, ny);
    RXMeshStatic rx(planeFaces);
    rx.add_vertex_coordinates(planeVerts, "plane");
    */

    ASSERT_TRUE(rx.is_edge_manifold());

    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  B_mat(rx, rx.get_num_vertices(), 3);
    DenseMatrix<float>  X_mat(rx, rx.get_num_vertices(), 3);
    setupMCF(rx, A_mat, B_mat);

    GMG<T> gmg(rx);

    VCycle<T> v_cyc(gmg, rx, A_mat, B_mat);
    v_cyc.solve(gmg, A_mat, B_mat, X_mat, 2);

    // auto samples = *rx.add_vertex_attribute<int>("s", 1);
    // gmg.m_sample_id.move(DEVICE, HOST);
    // samples.from_matrix(&gmg.m_sample_id);
    //
    // auto ps_mesh = rx.get_polyscope_mesh();
    // ps_mesh->addVertexScalarQuantity("samples", samples);
    // polyscope::show();
}

TEST(Apps, GMG)
{
    using namespace rxmesh;

    using T = float;

    cuda_query(Arg.device_id);

    RXMeshStatic rx(Arg.obj_file_name);

    /*
    std::vector<std::vector<float>>    planeVerts;
    std::vector<std::vector<uint32_t>> planeFaces;
    uint32_t                           nx = 10;
    uint32_t                           ny = 10;
    create_plane(planeVerts, planeFaces, nx, ny);
    RXMeshStatic rx(planeFaces);
    rx.add_vertex_coordinates(planeVerts, "plane");
    */

    ASSERT_TRUE(rx.is_edge_manifold());

    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  B_mat(rx, rx.get_num_vertices(), 3);
    setupMCF(rx, A_mat, B_mat);
    B_mat.move(DEVICE, HOST);

    VectorCSR3D RHS(rx.get_num_vertices());
    for (int i = 0; i < B_mat.rows(); i++) {
        RHS.vector[i * 3]     = B_mat(i, 0);
        RHS.vector[i * 3 + 1] = B_mat(i, 1);
        RHS.vector[i * 3 + 2] = B_mat(i, 2);
    }
    VectorCSR3D X(rx.get_num_vertices());
    X.reset();

    GPUGMG g(rx);
    g.ConstructOperators(rx);

    GMGVCycle gmg(g.N);

    gmg.prolongationOperators           = g.prolongationOperatorCSR;
    gmg.prolongationOperatorsTransposed = g.prolongationOperatorCSRTranspose;
    gmg.LHS                             = g.equationsPerLevel;
    gmg.RHS                             = g.B_v;
    gmg.max_number_of_levels            = 0;
    gmg.post_relax_iterations           = 5;
    gmg.pre_relax_iterations            = 5;
    gmg.ratio                           = g.ratio;


#ifdef USE_POLYSCOPE
    if (!Arg.offline) {
        interactive_menu(gmg, rx);
    } else {
#endif
        Timers<GPUTimer> timers;
        timers.add("solve");

        timers.start("solve");
        gmg.solve();
        timers.stop("solve");

        RXMESH_TRACE("Solving time: {}", timers.elapsed_millis("solve"));
#ifdef USE_POLYSCOPE
    }
#endif
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
            RXMESH_INFO(
                "\nUsage: GravoMG.exe < -option X>\n"
                " -h:          Display this massage and exit\n"
                " -input:      Input obj file. Default is {} \n"
                " -o:          JSON file output folder. Default is {} \n"
                " -f:          Use off-line mode. Only considered if "
                "compiled with USE_POLYSCOPE. Default is {} \n"
                " -device_id:  GPU device ID. Default is {}",
                Arg.obj_file_name,
                Arg.output_folder,
                Arg.offline ? "true" : "false",
                Arg.device_id);
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
        if (cmd_option_exists(argv, argc + argv, "-device_id")) {
            Arg.device_id =
                atoi(get_cmd_option(argv, argv + argc, "-device_id"));
        }
        if (cmd_option_exists(argv, argc + argv, "-f")) {
            Arg.offline = true;
        }
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("device_id= {}", Arg.device_id);


    return RUN_ALL_TESTS();
}