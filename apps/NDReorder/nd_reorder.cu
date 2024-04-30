#include "gtest/gtest.h"
#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"

#include "nd_cross_patch_oedering.cuh"
#include "nd_reorder_kernel.cuh"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    uint32_t    device_id     = 0;
} Arg;

template <uint32_t blockThreads>
void cross_patch_ordering(rxmesh::RXMeshStatic&             rx,
                          rxmesh::VertexAttribute<uint16_t> v_ordering,
                          uint32_t smem_bytes_dyn)
{
    using namespace rxmesh;

    bool is_coarsen_flag = true;

    while (is_coarsen_flag) {
        bool     is_matching_flag = true;
        uint16_t level            = 0;

        RXMESH_TRACE("Matching");
        match_patches_init<blockThreads>
            <<<rx.get_num_patches(), blockThreads, smem_bytes_dyn>>>(rx.get_context(), level);
        CUDA_ERROR(cudaDeviceSynchronize());

        while (is_matching_flag) {


            // match_patches_confirm<blockThreads><<<rx.get_num_patches(),
            // blockThreads>>>(
            //     rx.get_context(), level);
            // CUDA_ERROR(cudaDeviceSynchronize());

            // update the is_matching_flag here
            is_matching_flag = false;
        }

        // match_patches_result_update_level<blockThreads><<<rx.get_num_patches(),
        // blockThreads>>>(
        //         rx.get_context(), level);
        //     CUDA_ERROR(cudaDeviceSynchronize());

        // update level counter
        ++level;

        // update the is_coarsen_flag here
        is_coarsen_flag = false;
    }
}

void nd_reorder()
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    RXMeshStatic rx(Arg.obj_file_name);

    // rx.save(STRINGIFY(OUTPUT_DIR) + extract_file_name(Arg.obj_file_name) +
    //         "_nd_patches");

    // RXMeshDynamic rx(Arg.obj_file_name,
    //                  STRINGIFY(OUTPUT_DIR) +
    //                      extract_file_name(Arg.obj_file_name) +
    //                      "_nd_patches");

    // vertex color attribute
    auto attr_matched_v =
        rx.add_vertex_attribute<uint16_t>("attr_matched_v", 1);
    auto attr_active_e = rx.add_edge_attribute<uint16_t>("attr_active_e", 1);

    uint16_t req_levels     = 5;
    uint32_t blocks         = rx.get_num_patches();
    uint32_t threads        = blockThreads;
    size_t   smem_bytes_dyn = 0;

    smem_bytes_dyn += (1 + 1 * req_levels) * rx.max_bitmask_size<LocalEdgeT>();
    smem_bytes_dyn +=
        (6 + 4 * req_levels) * rx.max_bitmask_size<LocalVertexT>();
    smem_bytes_dyn +=
        (4 + 5 * req_levels) * rx.get_per_patch_max_edges() * sizeof(uint16_t);
    smem_bytes_dyn += (1 + 4 * req_levels) * rx.get_per_patch_max_vertices() *
                      sizeof(uint16_t);
    smem_bytes_dyn +=
        (11 + 11 * req_levels) * ShmemAllocator::default_alignment;

    RXMESH_TRACE("blocks: {}, threads: {}, smem_bytes: {}",
                 blocks,
                 threads,
                 smem_bytes_dyn);

    // vertex ordering attribute to store the result
    auto v_reorder =
        rx.add_vertex_attribute<uint16_t>("v_reorder", 1, rxmesh::LOCATION_ALL);

    // Phase: cross patch reordering
    cross_patch_ordering<blockThreads>(rx, *v_reorder, smem_bytes_dyn);

    // Phase: single patch reordering
    // nd_single_patch_main<blockThreads><<<blocks, threads, smem_bytes_dyn>>>(
    //     rx.get_context(), *v_reorder, *attr_matched_v, *attr_active_e,
    //     req_levels);

    CUDA_ERROR(cudaDeviceSynchronize());

#if USE_POLYSCOPE
    // Tests using coloring
    // Move vertex color to the host
    attr_matched_v->move(rxmesh::DEVICE, rxmesh::HOST);
    attr_active_e->move(rxmesh::DEVICE, rxmesh::HOST);

    // polyscope instance associated with rx
    auto polyscope_mesh = rx.get_polyscope_mesh();

    // pass vertex color to polyscope
    polyscope_mesh->addVertexScalarQuantity("attr_matched_v", *attr_matched_v);
    polyscope_mesh->addEdgeScalarQuantity("attr_active_e", *attr_active_e);

    // render
    polyscope::show();
#endif

    RXMESH_TRACE("DONE!!!!!!!!!!!!!!");
}

TEST(Apps, NDReorder)
{
    using namespace rxmesh;

    // Select device
    cuda_query(Arg.device_id);

    // nd reorder implementation
    nd_reorder();
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