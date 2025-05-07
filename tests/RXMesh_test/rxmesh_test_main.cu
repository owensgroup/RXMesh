#include <vector>
#include "gtest/gtest.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/report.h"

struct RXMeshTestArg
{
    uint32_t    num_run       = 1;
    uint32_t    device_id     = 0;
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    int         argc          = argc;
    char**      argv          = argv;
} rxmesh_args;

// clang-format off
#include "test_queries.h"
#include "test_patch_scheduler.cuh"
#include "test_patch_lock.cuh"
#include "test_wasted_work.cuh"
#include "test_grad.h"
// clang-format on

int main(int argc, char** argv)
{
    using namespace rxmesh;
    Log::init();

    ::testing::InitGoogleTest(&argc, argv);
    rxmesh_args.argc = argc;
    rxmesh_args.argv = argv;
    if (argc > 1) {
        if (cmd_option_exists(argv, argc + argv, "-h")) {
            // clang-format off
            RXMESH_INFO("\nUsage: RXMesh_test.exe < -option X>\n"
                        " -h:          Display this massage and exit\n"
                        " -input:      Input OBJ mesh file. Default is {} \n"
                        " -o:          JSON file output folder. Default is {} \n"
                        " -num_run:    Number of iterations for performance testing. Default is {} \n"
                        " -device_id:  GPU device ID. Default is {}",
            rxmesh_args.obj_file_name, rxmesh_args.output_folder ,rxmesh_args.num_run,rxmesh_args.device_id);
            // clang-format on
            exit(EXIT_SUCCESS);
        }


        if (cmd_option_exists(argv, argc + argv, "-num_run")) {
            rxmesh_args.num_run =
                atoi(get_cmd_option(argv, argv + argc, "-num_run"));
        }

        if (cmd_option_exists(argv, argc + argv, "-input")) {
            rxmesh_args.obj_file_name =
                std::string(get_cmd_option(argv, argv + argc, "-input"));
        }
        if (cmd_option_exists(argv, argc + argv, "-o")) {
            rxmesh_args.output_folder =
                std::string(get_cmd_option(argv, argv + argc, "-o"));
        }
        if (cmd_option_exists(argv, argc + argv, "-device_id")) {
            rxmesh_args.device_id =
                atoi(get_cmd_option(argv, argv + argc, "-device_id"));
        }
    }


    RXMESH_INFO("input= {}", rxmesh_args.obj_file_name);
    RXMESH_INFO("output_folder= {}", rxmesh_args.output_folder);
    RXMESH_INFO("num_run= {}", rxmesh_args.num_run);
    RXMESH_INFO("device_id= {}", rxmesh_args.device_id);

    cuda_query(rxmesh_args.device_id);

    return RUN_ALL_TESTS();
}
