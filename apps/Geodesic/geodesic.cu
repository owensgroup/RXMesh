// Compute geodesic distance according to
// Calla, Luciano A. Romero, Lizeth J. Fuentes Perez, and Anselmo A. Montenegro.
// "A minimalistic approach for fast computation of geodesic distances on
// triangular meshes." Computers & Graphics 84 (2019): 77-92

#include <cuda_profiler_api.h>
#include <random>

#include "../common/openmesh_trimesh.h"
#include "gtest/gtest.h"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/export_tools.h"
#include "rxmesh/util/import_obj.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    uint32_t    device_id     = 0;
    char**      argv;
    int         argc;
    bool        shuffle   = false;
    bool        sort      = false;
    uint32_t    num_seeds = 1;

} Arg;

#include "geodesic_ptp_openmesh.h"
#include "geodesic_ptp_rxmesh.h"

TEST(App, GEODESIC)
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

    ASSERT_TRUE(import_obj(Arg.obj_file_name, Verts, Faces));

    if (Arg.shuffle) {
        shuffle_obj(Faces, Verts);
    }

    // Create RXMeshStatic instance. If Arg.sort is true, Faces and Verts will
    // be sorted based on the patching happening inside RXMesh
    RXMeshStatic<PATCH_SIZE> rxmesh_static(Faces, Verts, Arg.sort, false);
    ASSERT_TRUE(rxmesh_static.is_closed())
        << "Geodesic only works on watertight/closed manifold mesh without "
           "boundaries";
    ASSERT_TRUE(rxmesh_static.is_edge_manifold())
        << "Geodesic only works on watertight/closed manifold mesh without "
           "boundaries";

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

    // Generate Seeds
    std::vector<uint32_t> h_seeds(Arg.num_seeds);
    std::random_device    dev;
    std::mt19937          rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, rxmesh_static.get_num_vertices());
    for (auto& s : h_seeds) {
        s = dist(rng);
        // s = 0;
    }


    //*** OpenMesh Impl
    RXMeshAttribute<dataT> ground_truth;

    // Save a map from vertex id to topleset (number of hops from
    // (closest?) source). It's used by OpenMesh to help construct
    // sorted_index and limit. We keep it for RXMesh because it is
    // used to quickly determine whether or not a vertex is within
    // the "update band".
    RXMeshAttribute<uint32_t> toplesets("toplesets");
    toplesets.init(Verts.size(),
                   1u,
                   RXMESH::HOST);  // will move() to DEVICE later


    std::vector<uint32_t> sorted_index;
    std::vector<uint32_t> limits;
    geodesic_ptp_openmesh(
        input_mesh, h_seeds, ground_truth, sorted_index, limits, toplesets);

    // export_attribute_VTK("geo_openmesh.vtk", Faces, Verts, false,
    //                     ground_truth.operator->(),
    //                     ground_truth.operator->());

    // Now that OpenMesh has calculated the toplesets,
    // move to DEVICE -- it's needed by RXMesh version
    toplesets.move(RXMESH::HOST, RXMESH::DEVICE);


    //*** RXMesh Impl
    EXPECT_TRUE(geodesic_rxmesh(rxmesh_static,
                                Faces,
                                Verts,
                                h_seeds,
                                ground_truth,
                                sorted_index,
                                limits,
                                toplesets))
        << "RXMesh failed!!";


    // Release allocation
    ground_truth.release();
    toplesets.release();
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
            RXMESH_INFO("\nUsage: Geodesic.exe < -option X>\n"
                        " -h:          Display this massage and exits\n"
                        " -input:      Input file. Input file should under the input/ subdirectory\n"
                        "              Default is {} \n"
                        "              Hint: Only accepts OBJ files\n"
                        " -o:          JSON file output folder. Default is {} \n"
                       // "-num_seeds:   Number of input seeds. Default is {}\n"                        
                        " -s:          Shuffle input. Default is false.\n"
                        " -p:          Sort input using patching output. Default is false.\n"
                        " -device_id:  GPU device ID. Default is {}",
            Arg.obj_file_name, Arg.output_folder ,Arg.num_seeds, Arg.device_id);
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
        // if (cmd_option_exists(argv, argc + argv, "-num_seeds")) {
        //    Arg.num_seeds =
        //        atoi(get_cmd_option(argv, argv + argc, "-num_seeds"));
        //}
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("num_seeds= {}", Arg.num_seeds);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    return RUN_ALL_TESTS();
}