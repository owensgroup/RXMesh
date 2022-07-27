#include <cuda_profiler_api.h>
#include "gtest/gtest.h"
#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"
#include "sparse_matrix.cuh"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    uint32_t    num_run       = 1;
    uint32_t    device_id     = 0;
    char**      argv;
    int         argc;
} Arg;

TEST(Apps, SparseMatrix)
{
    using namespace rxmesh;
    using dataT = float;

    // Select device
    cuda_query(Arg.device_id);

    // Load mesh
    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    ASSERT_TRUE(import_obj(Arg.obj_file_name, Verts, Faces));


    RXMeshStatic rxmesh(Faces, false);

    // TODO: fillin the spmat test

    SparseMatInfo<int> spmat(rxmesh);
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
            RXMESH_INFO("\nUsage: SparseMatrix.exe < -option X>\n"
                        " -h:          Display this massage and exit\n"
                        " -input:      Input file. Input file should be under the input/ subdirectory\n"
                        "              Default is {} \n"
                        "              Hint: Only accept OBJ files\n"
                        " -o:          JSON file output folder. Default is {} \n"
                        " -num_run:    Number of iterations for performance testing. Default is {} \n"                        
                        " -device_id:  GPU device ID. Default is {}",
            Arg.obj_file_name, Arg.output_folder, Arg.num_run, Arg.device_id);
            // clang-format on
            exit(EXIT_SUCCESS);
        }

        if (cmd_option_exists(argv, argc + argv, "-num_run")) {
            Arg.num_run = atoi(get_cmd_option(argv, argv + argc, "-num_run"));
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
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("num_run= {}", Arg.num_run);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    return RUN_ALL_TESTS();
}
