#include "gtest/gtest.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "Fennec_Fox.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    uint32_t    target        = 0;
    uint32_t    device_id     = 0;
    char**      argv;
    int         argc;
} Arg;

#include "simplification_rxmesh.cuh"

TEST(Apps, Simplification)
{
    using namespace rxmesh;

    // Select device
    cuda_query(Arg.device_id);

    RXMeshDynamic rx(Arg.obj_file_name, "", true);
    // rx.save(STRINGIFY(OUTPUT_DIR) + extract_file_name(Arg.obj_file_name) +
    //        "_patches");

    // RXMeshDynamic rx(Arg.obj_file_name,
    //                  STRINGIFY(OUTPUT_DIR) +
    //                      extract_file_name(Arg.obj_file_name) + "_patches",
    //                  true);

    ASSERT_TRUE(rx.is_edge_manifold());

    simplification_rxmesh(rx, Arg.target);
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
            RXMESH_INFO("\nUsage: Simplification.exe < -option X>\n"
                        " -h:          Display this massage and exit\n"
                        " -input:      Input file. Input file should be under the input/ subdirectory\n"
                        "              Default is {} \n"
                        "              Hint: Only accept OBJ files\n"
                        " -target:     The final/target number of faces in the output mesh\n"
                        " -o:          JSON file output folder. Default is {} \n"
                        " -device_id:  GPU device ID. Default is {}",
            Arg.obj_file_name, Arg.output_folder, Arg.device_id);
            // clang-format on
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
        if (cmd_option_exists(argv, argc + argv, "-target")) {
            Arg.target = false;
        }
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("device_id= {}", Arg.device_id);
    RXMESH_TRACE("target= {}", Arg.target);

    return RUN_ALL_TESTS();
}