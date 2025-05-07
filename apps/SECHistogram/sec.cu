#include "gtest/gtest.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

#include <filesystem>

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    float       target        = 0.1;
    float       reduce_ratio  = 0.1;
    uint32_t    device_id     = 0;
    char**      argv;
    int         argc;
} Arg;

#include "sec_rxmesh.cuh"

TEST(Apps, SEC)
{
    using namespace rxmesh;

    // Select device
    cuda_query(Arg.device_id);

    RXMeshDynamic rx(Arg.obj_file_name, "", 256, 3.5, 1.5);

    // const std::string p_file = STRINGIFY(OUTPUT_DIR) +
    //                            extract_file_name(Arg.obj_file_name) +
    //                            "_patches";
    // RXMeshDynamic rx(Arg.obj_file_name, p_file);
    // if (!std::filesystem::exists(p_file)) {
    //     rx.save(p_file);
    // }

    ASSERT_TRUE(rx.is_edge_manifold());

    ASSERT_TRUE(rx.is_closed());

    uint32_t final_num_vertices = Arg.target * rx.get_num_vertices();

    sec_rxmesh(rx, final_num_vertices);
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
            RXMESH_INFO("\nUsage: ShortestEdgeCollapse.exe < -option X>\n"
                        " -h:          Display this massage and exit\n"
                        " -input:      Input OBJ mesh file. Default is {} \n"
                        " -target:     The fraction of output #vertices from the input. Default is {}\n"
                        " -r:          Reduction ratio. Default is {}\n"
                        " -o:          JSON file output folder. Default is {} \n"
                        " -device_id:  GPU device ID. Default is {}",
            Arg.obj_file_name,Arg.target, Arg.reduce_ratio, Arg.output_folder, Arg.device_id);
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
            Arg.target = atof(get_cmd_option(argv, argv + argc, "-target"));
        }
        if (cmd_option_exists(argv, argc + argv, "-r")) {
            Arg.reduce_ratio = atof(get_cmd_option(argv, argv + argc, "-r"));
        }
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("device_id= {}", Arg.device_id);
    RXMESH_TRACE("target= {}", Arg.target);
    RXMESH_TRACE("reduce_ratio= {}", Arg.reduce_ratio);

    return RUN_ALL_TESTS();
}