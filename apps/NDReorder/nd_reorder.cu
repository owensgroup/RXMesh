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
#include "rxmesh/matrix/test_spmat_reordering.cuh"

#include "thrust/sort.h"

#include "check_nnz.h"

#include "nd_mgnd_implementation.cuh"
#include "nd_cross_patch_nd_implementation.cuh"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    uint16_t nd_level        = 4;
    uint32_t    device_id     = 0;
} Arg;

TEST(Apps, NDReorder)
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

    // Select device
    cuda_query(Arg.device_id);

    // allocate result array
    uint32_t* reorder_array;
    CUDA_ERROR(cudaMallocManaged(&reorder_array,
                                 sizeof(uint32_t) * rx.get_num_vertices()));
    CUDA_ERROR(cudaMemset(reorder_array, 0, sizeof(uint32_t) * rx.get_num_vertices()));

    GPUTimer timer;
    timer.start();

    cuda_nd_reorder(rx, reorder_array, Arg.nd_level, true);

    timer.stop();
    float total_time = timer.elapsed_millis();

    RXMESH_INFO("ND overall Reordering time: {} ms", total_time);

    reorder_array_correctness_check(reorder_array, rx.get_num_vertices());

    //  // for get the nnz data
    std::vector<uint32_t> reorder_vector(reorder_array, reorder_array + rx.get_num_vertices()); 
    processmesh_ordering(Arg.obj_file_name, (reorder_vector));
    processmesh_metis(Arg.obj_file_name);   
    processmesh_original(Arg.obj_file_name);
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

        if (cmd_option_exists(argv, argc + argv, "-nd_level")) {
            Arg.nd_level =
                atoi(get_cmd_option(argv, argv + argc, "-nd_level"));
        }
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    return RUN_ALL_TESTS();
}
