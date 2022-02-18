#include <vector>
#include "gtest/gtest.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/vector.h"

using dataT = float;

struct RXMeshTestArg
{
    uint32_t    num_run       = 1;
    uint32_t    device_id     = 0;
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    bool        quite         = false;
    int         argc          = argc;
    char**      argv          = argv;
} rxmesh_args;

// clang-format off
#include "test_higher_queries.h"
#include "test_queries.h"
#include "test_attribute.cuh"
#include "test_for_each.h"
#include "test_edge_flip.cuh"
#include "test_validate.h"
#include "test_deletion.cuh"
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
                        " -h:          Display this massage and exits\n"
                        " -input:      Input file. Input file should under the input/ subdirectory\n"
                        "              Default is {} \n"
                        "              Hint: Only accepts OBJ files\n"
                        " -o:          JSON file output folder. Default is {} \n"
                        " -num_run:    Number of iterations for performance testing. Default is {} \n"                        
                        " -q:          Run in quite mode. Default is false\n"
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
        if (cmd_option_exists(argv, argc + argv, "-q")) {
            rxmesh_args.quite = true;
        }
    }

    if (!rxmesh_args.quite) {
        RXMESH_TRACE("input= {}", rxmesh_args.obj_file_name);
        RXMESH_TRACE("output_folder= {}", rxmesh_args.output_folder);
        RXMESH_TRACE("num_run= {}", rxmesh_args.num_run);
        RXMESH_TRACE("device_id= {}", rxmesh_args.device_id);
    }
    return RUN_ALL_TESTS();
}
