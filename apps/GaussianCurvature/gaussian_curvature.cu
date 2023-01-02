// Compute the gaussian curvature according to
// Mark Meyer, Mathieu Desbrun, Peter Schroder, and Alan H. Barr. "Discrete
// Differential-Geometry Operators for Triangulated 2-Manifolds"
// International Workshop on Visualization and Mathematics

#include "gtest/gtest.h"
#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"

constexpr double PI = 3.1415926535897932384626433832795028841971693993751058209;
#include "gaussian_curvature_kernel.cuh"
#include "gaussian_curvature_ref.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "bumpy-cube.obj";
    uint32_t    device_id     = 0;
} Arg;

template <typename T>
void gaussian_curvature_rxmesh(const std::vector<T>& gaussian_curvature_gold)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    RXMeshStatic rx(Arg.obj_file_name, false);

    // input coordinates
    auto coords = *rx.get_input_vertex_coordinates();

    // gaussian curvatures
    auto v_gc = *rx.add_vertex_attribute<T>("v_gc", 1, rxmesh::LOCATION_ALL);

    // mixed area for integration
    auto v_amix =
        *rx.add_vertex_attribute<T>("v_amix", 1, rxmesh::LOCATION_ALL);

    // launch box
    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({rxmesh::Op::FV},
                          launch_box,
                          (void*)compute_gaussian_curvature<T, blockThreads>);

    // initialization
    v_gc.reset(2 * PI, rxmesh::DEVICE);
    v_amix.reset(0, rxmesh::DEVICE);

    compute_gaussian_curvature<T, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), coords, v_gc, v_amix);

    rx.for_each_vertex(DEVICE,
                       [v_gc, v_amix] __device__(const VertexHandle vh) {
                           v_gc(vh, 0) = v_gc(vh, 0) / v_amix(vh, 0);
                       });

    CUDA_ERROR(cudaDeviceSynchronize());


    // Verify
    v_gc.move(rxmesh::DEVICE, rxmesh::HOST);

    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = rx.map_to_global(vh);
        EXPECT_NEAR(std::abs(gaussian_curvature_gold[v_id]),
                    std::abs(v_gc(vh, 0)),
                    0.001);
    });

#if USE_POLYSCOPE
    // visualize
    polyscope::view::upDir = polyscope::UpDir::ZUp;
    auto polyscope_mesh    = rx.get_polyscope_mesh();
    polyscope_mesh->setEdgeWidth(1.0);
    polyscope_mesh->addVertexScalarQuantity("vGaussianCurv", v_gc);
    polyscope::show();
#endif
}

TEST(Apps, GaussianCurvature)
{
    using namespace rxmesh;

    // Select device
    cuda_query(Arg.device_id);

    // Load mesh
    std::vector<std::vector<float>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    ASSERT_TRUE(import_obj(Arg.obj_file_name, Verts, Faces));

    // Serial reference
    std::vector<float> gaussian_curvature_gold(Verts.size());
    gaussian_curvature_ref(Faces, Verts, gaussian_curvature_gold);

    // RXMesh Impl
    gaussian_curvature_rxmesh(gaussian_curvature_gold);
}

int main(int argc, char** argv)
{
    using namespace rxmesh;
    Log::init();

    ::testing::InitGoogleTest(&argc, argv);

    if (argc > 1) {
        if (cmd_option_exists(argv, argc + argv, "-h")) {
            // clang-format off
            RXMESH_INFO("\nUsage: GaussianCurvature.exe < -option X>\n"
                        " -h:          Display this massage and exits\n"
                        " -input:      Input file. Input file should under the input/ subdirectory\n"
                        "              Default is {} \n"
                        "              Hint: Only accepts OBJ files\n"                                              
                        " -device_id:  GPU device ID. Default is {}",
            Arg.obj_file_name,  Arg.device_id);
            // clang-format on
            exit(EXIT_SUCCESS);
        }

        if (cmd_option_exists(argv, argc + argv, "-input")) {
            Arg.obj_file_name =
                std::string(get_cmd_option(argv, argv + argc, "-input"));
        }

        if (cmd_option_exists(argv, argc + argv, "-device_id")) {
            Arg.device_id =
                atoi(get_cmd_option(argv, argv + argc, "-device_id"));
        }
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    return RUN_ALL_TESTS();
}