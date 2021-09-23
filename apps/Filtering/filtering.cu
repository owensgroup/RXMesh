// Parallel version of
// Fleishman, Shachar, Iddo Drori, and Daniel Cohen-Or.
//"Bilateral mesh denoising." ACM SIGGRAPH 2003 Papers.2003. 950-953.

#include <omp.h>

#include "../common/openmesh_trimesh.h"
#include "gtest/gtest.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/export_tools.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/log.h"

struct arg
{
    std::string obj_file_name   = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder   = STRINGIFY(OUTPUT_DIR);
    uint32_t    device_id       = 0;
    uint32_t    num_filter_iter = 5;
    char**      argv;
    int         argc;
} Arg;

#include "filtering_openmesh.h"
#include "filtering_rxmesh.cuh"

TEST(App, Filtering)
{
    using namespace rxmesh;
    using dataT = float;

    // Select device
    cuda_query(Arg.device_id);


    // Load mesh
    std::vector<std::vector<uint32_t>> Faces;
    std::vector<std::vector<dataT>>    Verts;
    ASSERT_TRUE(import_obj(Arg.obj_file_name, Verts, Faces));

    RXMeshStatic rxmesh_static(Faces, false);

    TriMesh input_mesh;
    ASSERT_TRUE(OpenMesh::IO::read_mesh(input_mesh, Arg.obj_file_name));


    // OpenMesh Impl
    rxmesh::RXMeshAttribute<dataT> ground_truth;
    size_t                         max_neighbour_size = 0;
    filtering_openmesh(
        omp_get_max_threads(), input_mesh, ground_truth, max_neighbour_size);


    // RXMesh Impl
    filtering_rxmesh(rxmesh_static, Verts, ground_truth, max_neighbour_size);

    // Release allocation
    ground_truth.release();
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
            RXMESH_INFO("\nUsage: Filtering.exe < -option X>\n"
                        " -h:                Display this massage and exits\n"
                        " -input:            Input file. Input file should under the input/ subdirectory\n"
                        "                    Default is {} \n"
                        "                    Hint: Only accepts OBJ files\n"
                        " -o:                JSON file output folder. Default is {} \n"
                        " -num_filter_iter:  Iteration count. Default is {} \n"
                        " -device_id:        GPU device ID. Default is {}",
             Arg.obj_file_name, Arg.output_folder ,Arg.num_filter_iter ,Arg.device_id);
            // clang-format on
            exit(EXIT_SUCCESS);
        }

        if (cmd_option_exists(argv, argc + argv, "-num_filter_iter")) {
            Arg.num_filter_iter =
                atoi(get_cmd_option(argv, argv + argc, "-num_filter_iter"));
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
    RXMESH_TRACE("num_filter_iter= {}", Arg.num_filter_iter);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    return RUN_ALL_TESTS();
}