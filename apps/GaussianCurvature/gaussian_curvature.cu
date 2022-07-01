// Compute the vertex normal according to
// Max, Nelson. "Weights for computing vertex normals from facet normals."
// Journal of Graphics Tools 4, no. 2 (1999): 1-6.

#include <cuda_profiler_api.h>
#include "gtest/gtest.h"
#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"
#include "gaussian_curvature_kernel.cuh"
#include "gaussian_curvature_ref.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    uint32_t    num_run       = 1;
    uint32_t    device_id     = 0;
    char**      argv;
    int         argc;
} Arg;

template <typename T>
void gaussian_curvature_rxmesh(rxmesh::RXMeshStatic&         rxmesh,
                          const std::vector<std::vector<T>>& Verts,
                          const std::vector<T>&              gaussian_curvature_gold)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    // Report
    Report report("VertexNormal_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rxmesh);
    report.add_member("method", std::string("RXMesh"));
    report.add_member("blockThreads", blockThreads);

    auto coords = rxmesh.add_vertex_attribute<T>(Verts, "coordinates");

    // gaussian curvatures
    auto v_gc =
        rxmesh.add_vertex_attribute<T>("v_gc", 1, rxmesh::LOCATION_ALL);

    // mixed area for integration
    auto v_amix =
        rxmesh.add_vertex_attribute<T>("v_amix", 1, rxmesh::LOCATION_ALL);

    // launch box
    LaunchBox<blockThreads> launch_box;
    rxmesh.prepare_launch_box({rxmesh::Op::FV},
                              launch_box,
                              (void*)compute_gaussian_curvature<T, blockThreads>);


    TestData td;
    td.test_name   = "GaussianCurvature";
    td.num_threads = launch_box.num_threads;
    td.num_blocks  = launch_box.blocks;
    td.dyn_smem    = launch_box.smem_bytes_dyn;
    td.static_smem = launch_box.smem_bytes_static;
    td.num_reg     = launch_box.num_registers_per_thread;

    float vn_time = 0;
    for (uint32_t itr = 0; itr < Arg.num_run; ++itr) {
        v_gc->reset(2 * PI, rxmesh::DEVICE);
        v_amix->reset(0, rxmesh::DEVICE);
        GPUTimer timer;
        timer.start();

        compute_gaussian_curvature<T, blockThreads><<<launch_box.blocks,
                                                 launch_box.num_threads,
                                                 launch_box.smem_bytes_dyn>>>(
            rxmesh.get_context(), *coords, *v_gc, *v_amix);

        auto v_gc_val = *v_gc;
        auto v_amix_val = *v_amix;
        
        rxmesh.for_each_vertex(
            DEVICE,
            [v_gc_val, v_amix_val] __device__(const VertexHandle vh) {
                v_gc_val(vh, 0) = v_gc_val(vh, 0), v_amix_val(vh, 0);
            });

        timer.stop();
        CUDA_ERROR(cudaDeviceSynchronize());
        CUDA_ERROR(cudaGetLastError());
        CUDA_ERROR(cudaProfilerStop());
        td.time_ms.push_back(timer.elapsed_millis());
        vn_time += timer.elapsed_millis();
    }

    RXMESH_TRACE("gaussian_curvature_rxmesh() vertex normal kernel took {} (ms)",
                 vn_time / Arg.num_run);

    // Verify
    v_gc->move(rxmesh::DEVICE, rxmesh::HOST);

    rxmesh.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = rxmesh.map_to_global(vh);

        for (uint32_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(std::abs(gaussian_curvature_gold[v_id * 3 + i]),
                        std::abs((*v_gc)(vh, i)),
                        0.0001);
        }
    });

    // Finalize report
    report.add_test(td);
    report.write(Arg.output_folder + "/rxmesh",
                 "VertexNormal_RXMesh_" + extract_file_name(Arg.obj_file_name));
}

TEST(Apps, VertexNormal)
{
    using namespace rxmesh;
    using dataT = float;

    // Select device
    cuda_query(Arg.device_id);

    // Load mesh
    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    ASSERT_TRUE(import_obj(Arg.obj_file_name, Verts, Faces));


    RXMeshStatic rxmesh(Faces, false); // FV initialize

    // Serial reference
    std::vector<dataT> gaussian_curvature_gold(Verts.size());
    gaussian_curvature_ref(Faces, Verts, gaussian_curvature_gold);

    // RXMesh Impl
    gaussian_curvature_rxmesh(rxmesh, Verts, gaussian_curvature_gold);
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
            RXMESH_INFO("\nUsage: VertexNormal.exe < -option X>\n"
                        " -h:          Display this massage and exits\n"
                        " -input:      Input file. Input file should under the input/ subdirectory\n"
                        "              Default is {} \n"
                        "              Hint: Only accepts OBJ files\n"
                        " -o:          JSON file output folder. Default is {} \n"
                        " -num_run:    Number of iterations for performance testing. Default is {} \n"                        
                        " -device_id:  GPU device ID. Default is {}",
            Arg.obj_file_name, Arg.output_folder, Arg.num_run, Arg.device_id);
            // clang-format on
            exit(EXIT_SUCCESS);
        }

        if (cmd_option_exists(argv, argc + argv, "-num_run")) {
            Arg.num_run = atoi(get_cmd_option(argv, argv + argc, "-num_run"));
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
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("num_run= {}", Arg.num_run);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    return RUN_ALL_TESTS();
}