// Compute the vertex normal according to
// Max, Nelson. "Weights for computing vertex normals from facet normals."
// Journal of Graphics Tools 4, no. 2 (1999): 1-6.

#include <cuda_profiler_api.h>
#include <CLI/CLI.hpp>
#include <cstdlib>

#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"
#include "vertex_normal_kernel.cuh"
#include "vertex_normal_ref.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    uint32_t    num_run       = 1;
    uint32_t    device_id     = 0;
    char**      argv;
    int         argc;
} Arg;

#include "vertex_normal_hardwired.cuh"

template <typename T>
void vertex_normal_rxmesh(rxmesh::RXMeshStatic&              rx,
                          const std::vector<std::vector<T>>& Verts,
                          const std::vector<T>&              vertex_normal_gold)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    // Report
    Report report("VertexNormal_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rx);
    report.add_member("method", std::string("RXMesh"));
    report.add_member("blockThreads", blockThreads);

    auto coords = rx.add_vertex_attribute<T>(Verts, "coordinates");


    // normals
    auto v_normals =
        rx.add_vertex_attribute<T>("v_normals", 3, rxmesh::LOCATION_ALL);

    // launch box
    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({rxmesh::Op::FV},
                          launch_box,
                          (void*)compute_vertex_normal<T, blockThreads>);


    TestData td;
    td.test_name   = "VertexNormal";
    td.num_threads = launch_box.num_threads;
    td.num_blocks  = launch_box.blocks;
    td.dyn_smem    = launch_box.smem_bytes_dyn;
    td.static_smem = launch_box.smem_bytes_static;
    td.num_reg     = launch_box.num_registers_per_thread;

    float vn_time = 0;
    for (uint32_t itr = 0; itr < Arg.num_run; ++itr) {
        v_normals->reset(0, rxmesh::DEVICE);
        GPUTimer timer;
        timer.start();

        compute_vertex_normal<T, blockThreads><<<launch_box.blocks,
                                                 launch_box.num_threads,
                                                 launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *v_normals);

        timer.stop();
        CUDA_ERROR(cudaDeviceSynchronize());
        CUDA_ERROR(cudaGetLastError());
        CUDA_ERROR(cudaProfilerStop());
        td.time_ms.push_back(timer.elapsed_millis());
        vn_time += timer.elapsed_millis();
    }

    RXMESH_TRACE("vertex_normal_rxmesh() vertex normal kernel took {} (ms)",
                 vn_time / Arg.num_run);

    // Verify
    v_normals->move(rxmesh::DEVICE, rxmesh::HOST);

    bool passed = true;
    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = rx.map_to_global(vh);

        for (uint32_t i = 0; i < 3; ++i) {
            float ref = std::abs(vertex_normal_gold[v_id * 3 + i]);
            float val = std::abs((*v_normals)(vh, i));
            if (std::abs(ref - val) > 0.0001f) {
                passed = false;
            }
        }
    });

    if (!passed) {
        RXMESH_ERROR("VertexNormal RXMesh validation failed");
    }

    // Finalize report
    report.add_test(td);
    report.write(Arg.output_folder + "/rxmesh",
                 "VertexNormal_RXMesh_" + extract_file_name(Arg.obj_file_name));
}

int vertex_normal_main()
{
    using namespace rxmesh;
    using dataT = float;

    // Load mesh
    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    if (!import_obj(Arg.obj_file_name, Verts, Faces)) {
        RXMESH_ERROR("Failed to import OBJ file: {}", Arg.obj_file_name);
        return EXIT_FAILURE;
    }

    RXMeshStatic rx(Faces);

    // Serial reference
    std::vector<dataT> vertex_normal_gold(3 * Verts.size());
    vertex_normal_ref(Faces, Verts, vertex_normal_gold);

    // RXMesh Impl
    vertex_normal_rxmesh(rx, Verts, vertex_normal_gold);

    // Hardwired Impl
    vertex_normal_hardwired(Faces, Verts, vertex_normal_gold);

    return 0;
}

int main(int argc, char** argv)
{
    using namespace rxmesh;

    CLI::App app{"VertexNormal - Vertex-normal computation benchmark"};

    app.add_option("-i,--input", Arg.obj_file_name, "Input OBJ mesh file")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "sphere3.obj"));

    app.add_option("-o,--output", Arg.output_folder, "JSON file output folder")
        ->default_val(std::string(STRINGIFY(OUTPUT_DIR)));

    app.add_option("--num_run",
                   Arg.num_run,
                   "Number of iterations for performance testing")
        ->default_val(1u);

    app.add_option("-d,--device_id", Arg.device_id, "GPU device ID")
        ->default_val(0u);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    rx_init(static_cast<int>(Arg.device_id));

    Arg.argv = argv;
    Arg.argc = argc;

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("num_run= {}", Arg.num_run);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    return vertex_normal_main();
}