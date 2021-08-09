// Parallel version of
// Desbrun, Mathieu, et al "Implicit Fairing of Irregular Meshes using Diffusion
// and Curvature Flow." SIGGRAPH 1999

#include <omp.h>

#include "../common/openmesh_trimesh.h"
#include "gtest/gtest.h"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/export_tools.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/log.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    uint32_t    device_id = 0;
    float       time_step = 0.001;
    float       cg_tolerance = 1e-6;
    uint32_t    max_num_cg_iter = 1000;
    bool        use_uniform_laplace = false;
    char**      argv;
    int         argc;
    bool        shuffle = false;
    bool        sort = false;

} Arg;

#include "mcf_openmesh.h"
#include "mcf_rxmesh.h"


TEST(App, MCF)
{
    using namespace RXMESH;
    using dataT = float;

    if (Arg.shuffle) {
        ASSERT_FALSE(Arg.sort) << " cannot shuffle and sort at the same time!";
    }
    if (Arg.sort) {
        ASSERT_FALSE(Arg.shuffle)
            << " cannot shuffle and sort at the same time!";
    }

    // Select device
    cuda_query(Arg.device_id);


    // Load mesh
    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    if (!import_obj(Arg.obj_file_name, Verts, Faces)) {
        exit(EXIT_FAILURE);
    }

    if (Arg.shuffle) {
        shuffle_obj(Faces, Verts);
    }

    // Create RXMeshStatic instance. If Arg.sort is true, Faces and Verts will
    // be sorted based on the patching happening inside RXMesh
    RXMeshStatic rxmesh_static(Faces, Verts, Arg.sort, false);


    // Since OpenMesh only accepts input as obj files, if the input mesh is
    // shuffled or sorted, we have to write it to a temp file so that OpenMesh
    // can pick it up
    TriMesh input_mesh;
    if (Arg.sort || Arg.shuffle) {
        export_obj(Faces, Verts, "temp.obj", false);
        ASSERT_TRUE(OpenMesh::IO::read_mesh(input_mesh, "temp.obj"));
    } else {
        ASSERT_TRUE(OpenMesh::IO::read_mesh(input_mesh, Arg.obj_file_name));
    }

    //*** OpenMesh Impl
    RXMESH::RXMeshAttribute<dataT> ground_truth;
    mcf_openmesh(omp_get_max_threads(), input_mesh, ground_truth);

    //*** RXMesh Impl
    mcf_rxmesh(rxmesh_static, Verts, ground_truth);


    // Release allocation
    ground_truth.release();
}

int main(int argc, char** argv)
{
    using namespace RXMESH;
    Log::init();

    ::testing::InitGoogleTest(&argc, argv);
    Arg.argv = argv;
    Arg.argc = argc;
    if (argc > 1) {
        if (cmd_option_exists(argv, argc + argv, "-h")) {
            // clang-format off
            RXMESH_INFO("\nUsage: MCF.exe < -option X>\n"
                        " -h:                 Display this massage and exits\n"
                        " -input:             Input file. Input file should under the input/ subdirectory\n"
                        "                     Default is {} \n"
                        "                     Hint: Only accepts OBJ files\n"
                        " -o:                 JSON file output folder. Default is {} \n"
                        " -uniform_laplace:   Use uniform Laplace weights. Default is {} \n"
                        " -dt:                Time step (delta t). Default is {} \n"
                        "                     Hint: should be between (0.001, 1) for cotan Laplace or between (1, 100) for uniform Laplace\n"
                        " -eps:               Conjugate gradient tolerance. Default is {}\n"
                        " -max_cg_iter:       Conjugate gradient maximum number of iterations. Default is {}\n"
                        " -s:                 Shuffle input. Default is false.\n"
                        " -p:                 Sort input using patching output. Default is false\n"
                        " -device_id:         GPU device ID. Default is {}",
            Arg.obj_file_name, Arg.output_folder,  (Arg.use_uniform_laplace? "true" : "false"), Arg.time_step, Arg.cg_tolerance, Arg.max_num_cg_iter, Arg.device_id);
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
        if (cmd_option_exists(argv, argc + argv, "-dt")) {
            Arg.time_step = std::atof(get_cmd_option(argv, argv + argc, "-dt"));
        }
        if (cmd_option_exists(argv, argc + argv, "-max_cg_iter")) {
            Arg.max_num_cg_iter =
                std::atoi(get_cmd_option(argv, argv + argc, "-max_cg_iter"));
        }

        if (cmd_option_exists(argv, argc + argv, "-eps")) {
            Arg.cg_tolerance =
                std::atof(get_cmd_option(argv, argv + argc, "-eps"));
        }
        if (cmd_option_exists(argv, argc + argv, "-uniform_laplace")) {
            Arg.use_uniform_laplace = true;
        }
        if (cmd_option_exists(argv, argc + argv, "-s")) {
            Arg.shuffle = true;
        }
        if (cmd_option_exists(argv, argc + argv, "-p")) {
            Arg.sort = true;
        }
        if (cmd_option_exists(argv, argc + argv, "-device_id")) {
            Arg.device_id =
                atoi(get_cmd_option(argv, argv + argc, "-device_id"));
        }
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("max_num_cg_iter= {}", Arg.max_num_cg_iter);
    RXMESH_TRACE("cg_tolerance= {0:f}", Arg.cg_tolerance);
    RXMESH_TRACE("use_uniform_laplace= {}", Arg.use_uniform_laplace);
    RXMESH_TRACE("time_step= {0:f}", Arg.time_step);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    return RUN_ALL_TESTS();
}