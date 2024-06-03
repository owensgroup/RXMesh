#include "gtest/gtest.h"
#include "rxmesh/util/import_obj.h"

#include "nd_cross_patch_ordering.cuh"
#include "nd_single_patch_ordering.cuh"

#include <vector>
#include "cusparse.h"
#include "rxmesh/context.h"
#include "rxmesh/types.h"

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/rxmesh_dynamic.h"

#include "rxmesh/matrix/sparse_matrix.cuh"

#include "thrust/sort.h"

#include "check_nnz.h"

#include "nd_mgnd_implementation.cuh"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    uint32_t    device_id     = 0;
} Arg;

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
    auto v_ordering = rx.add_vertex_attribute<uint16_t>(
        "v_ordering", 1, rxmesh::LOCATION_ALL);
    v_ordering->reset(INVALID16, rxmesh::DEVICE);

    uint32_t* reorder_array;
    CUDA_ERROR(cudaMallocManaged(&reorder_array,
                                 sizeof(uint32_t) * rx.get_num_vertices()));


    // Phase: cross patch reordering
    // GPUTimer timer;
    // timer.start();

    // cross_patch_ordering<blockThreads>(
    //     rx, *v_ordering, reorder_array, smem_bytes_dyn);

    // timer.stop();
    // float total_time = timer.elapsed_millis();

    // RXMESH_INFO("Cross patch ordering time: {} ms", total_time);


    // Phase: single patch reordering
    nd_single_patch_main<blockThreads><<<blocks, threads, smem_bytes_dyn>>>(
        rx.get_context(), *v_ordering, *attr_matched_v, *attr_active_e,
        req_levels);
    CUDA_ERROR(cudaDeviceSynchronize());
    RXMESH_TRACE("single patch ordering done");


    // correctness check
    uint16_t* v_count;
    CUDA_ERROR(cudaMallocManaged(&v_count, sizeof(uint16_t) * 5));
    cudaMemset(v_count, 0, sizeof(uint16_t) * 5);
    v_count[0] = rx.get_num_vertices();
    nd_single_patch_test_v_count<blockThreads><<<blocks, threads>>>(
        rx.get_context(), v_count);

    CUDA_ERROR(cudaDeviceSynchronize());

    RXMESH_INFO("v_count[0] = {}, v_count[1] = {}, v_count[2] = {}",
                v_count[0],
                v_count[1],
                v_count[2]);

    // for timing purposes: measure the time for cuda metis
    // rxmesh::SparseMatrix<float> A_mat(rx);
    // A_mat.spmat_chol_reorder(Reorder::NSTDIS);

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
    
    // for get the nnz data
    // std::vector<uint32_t> reorder_vector(reorder_array, reorder_array + rx.get_num_vertices()); 

    // processmesh_ordering(Arg.obj_file_name, reorder_vector);

    // processmesh_original(Arg.obj_file_name);

    // processmesh_metis(Arg.obj_file_name);
}

TEST(Apps, NDReorder)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    RXMeshStatic rx(Arg.obj_file_name);

    // Select device
    cuda_query(Arg.device_id);

    // allocate result array
    uint32_t* reorder_array;
    CUDA_ERROR(cudaMallocManaged(&reorder_array,
                                 sizeof(uint32_t) * rx.get_num_vertices()));
    CUDA_ERROR(cudaMemset(reorder_array, 0, sizeof(uint32_t) * rx.get_num_vertices()));

    // nd reorder implementation
    // nd_reorder();

    // mgnd reorder implementation
    mgnd_reorder(rx, reorder_array);

    reorder_array_correctness_check(reorder_array, rx.get_num_vertices());

     // for get the nnz data
    std::vector<uint32_t> reorder_vector(reorder_array, reorder_array + rx.get_num_vertices()); 
    processmesh_ordering(Arg.obj_file_name, (reorder_vector));
    processmesh_original(Arg.obj_file_name);
    processmesh_metis(Arg.obj_file_name);   
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