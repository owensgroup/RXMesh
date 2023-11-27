#include "gtest/gtest.h"
#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"

#include "nd_reorder_kernel.cuh"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "bumpy-cube.obj";
    uint32_t    device_id     = 0;
} Arg;

template <typename T>
void nd_reorder()
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    RXMeshStatic rx(Arg.obj_file_name);

    // input coordinates
    auto coords = *rx.get_input_vertex_coordinates();

    // vertex weight
    auto vwgt = *rx.add_vertex_attribute<T>("vwgt", 1, rxmesh::LOCATION_ALL);

    // vertex look up index in the adj list(TO BE DISCARDED and use dynamic
    // RXMesh)
    auto iedges =
        *rx.add_vertex_attribute<T>("iedges", 1, rxmesh::LOCATION_ALL);

    // vertex degree (num of edges around a vertex)
    auto nedges =
        *rx.add_vertex_attribute<T>("nedges", 1, rxmesh::LOCATION_ALL);

    // the sum of the edge weight that is around a vertex
    auto adjwgt =
        *rx.add_vertex_attribute<T>("adjwgt", 1, rxmesh::LOCATION_ALL);

    // the pairing VertexHandle
    auto vpair =
        *rx.add_vertex_attribute<VertexHandle>("vpair", 1, rxmesh::LOCATION_ALL);

    // the sum of edge weight that has been contracted into the vertex (HCM)
    auto cewgt = *rx.add_vertex_attribute<T>("cewgt", 1, rxmesh::LOCATION_ALL);

    // edge weight
    auto ewgt = *rx.add_edge_attribute<T>("ewgt", 1, rxmesh::LOCATION_ALL);

    // initialization
    vwgt.reset(1, rxmesh::DEVICE);
    // iedges: discarded
    // nedges: compute later
    // adjwgt: compute later
    cewgt.reset(0, rxmesh::DEVICE);
    vpair.reset(1<<10, rxmesh::DEVICE); // for a VertexHandle doesn't exist
    ewgt.reset(1, rxmesh::DEVICE);

    // launch box
    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box(
        {rxmesh::Op::VE}, launch_box, (void*)init_attribute<T, blockThreads>);

    init_attribute<T, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), nedges, adjwgt, ewgt);


    rx.prepare_launch_box(
        {rxmesh::Op::VE}, launch_box, (void*)heavy_edge_matching<T, blockThreads>);

    heavy_edge_matching<T, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), vpair, ewgt);

    CUDA_ERROR(cudaDeviceSynchronize());
}

TEST(Apps, NDReorder)
{
    using namespace rxmesh;

    // Select device
    cuda_query(Arg.device_id);

    // nd reorder implementation
    nd_reorder<int>();
}

int main(int argc, char** argv)
{
    using namespace rxmesh;
    Log::init();

    ::testing::InitGoogleTest(&argc, argv);

    if (argc > 1) {
        if (cmd_option_exists(argv, argc + argv, "-h")) {
            // clang-format off
            RXMESH_INFO("\nUsage: NDReorder.exe < -option X>\n"
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


// batch info file