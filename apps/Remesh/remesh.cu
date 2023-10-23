#include "gtest/gtest.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    float       relative_len  = 0.9;
    uint32_t    num_iter      = 3;
    uint32_t    device_id     = 0;
    char**      argv;
    int         argc;
} Arg;

#include "remesh_rxmesh.cuh"

TEST(Apps, Remesh)
{
    using namespace rxmesh;

    // Select device
    cuda_query(Arg.device_id);

    // RXMeshDynamic rx(Arg.obj_file_name);
    // rx.save(STRINGIFY(OUTPUT_DIR) + extract_file_name(Arg.obj_file_name) +
    //        "_patches");

    RXMeshDynamic rx(Arg.obj_file_name,
                     STRINGIFY(OUTPUT_DIR) +
                         extract_file_name(Arg.obj_file_name) + "_patches");

    ASSERT_TRUE(rx.is_closed());

    remesh_rxmesh(rx);
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
            RXMESH_INFO("\nUsage: Remesh.exe < -option X>\n"
                        " -h:              Display this massage and exit\n"
                        " -input:          Input file. Input file should be under the input/ subdirectory\n"
                        "                  Default is {} \n"
                        "                  Hint: Only accept OBJ files\n"
                        " -num_iter:       Number of remeshing iterations. Default is {}\n"
                        " -relative_len:   Target edge length as a ratio of the input mesh average edge length. Default is {}\n"
                        "                  Hint: should be slightly less than the average edge length of the input mesh\n"
                        " -o:              JSON file output folder. Default is {} \n"
                        " -device_id:      GPU device ID. Default is {}",
            Arg.obj_file_name, Arg.num_iter,Arg.relative_len, Arg.output_folder, Arg.device_id);
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
        if (cmd_option_exists(argv, argc + argv, "-num_iter")) {
            Arg.num_iter = atoi(get_cmd_option(argv, argv + argc, "-num_iter"));
        }
        if (cmd_option_exists(argv, argc + argv, "-relative_len")) {
            Arg.relative_len =
                atoi(get_cmd_option(argv, argv + argc, "-relative_len"));
        }
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("device_id= {}", Arg.device_id);
    RXMESH_TRACE("num_iter= {}", Arg.num_iter);
    RXMESH_TRACE("relative_len= {}", Arg.relative_len);


    return RUN_ALL_TESTS();
}