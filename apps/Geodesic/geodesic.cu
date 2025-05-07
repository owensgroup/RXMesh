// Compute geodesic distance according to
// Calla, Luciano A. Romero, Lizeth J. Fuentes Perez, and Anselmo A. Montenegro.
// "A minimalistic approach for fast computation of geodesic distances on
// triangular meshes." Computers & Graphics 84 (2019): 77-92

#include <cuda_profiler_api.h>
#include <random>

#include "gtest/gtest.h"

#include "../common/openmesh_trimesh.h"

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/import_obj.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    uint32_t    device_id     = 0;
    char**      argv;
    int         argc;
    uint32_t    num_seeds = 1;

} Arg;

#include "geodesic_ptp_openmesh.h"
#include "geodesic_ptp_rxmesh.h"

TEST(App, Geodesic)
{
    using namespace rxmesh;
    using dataT = float;

    // Select device
    cuda_query(Arg.device_id);

    RXMeshStatic rx(Arg.obj_file_name);
    ASSERT_TRUE(rx.is_closed())
        << "Geodesic only works on watertight/closed manifold mesh without "
           "boundaries";
    ASSERT_TRUE(rx.is_edge_manifold())
        << "Geodesic only works on watertight/closed manifold mesh without "
           "boundaries";


    // Generate Seeds
    std::vector<uint32_t> h_seeds(Arg.num_seeds);
    std::random_device    dev;
    std::mt19937          rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, rx.get_num_vertices());
    for (auto& s : h_seeds) {
        s = dist(rng);
        // s = 0;
    }


    // Save a map from vertex id to topleset (number of hops from
    // (closest?) source). It's used by OpenMesh to help construct
    // sorted_index and limit. We keep it for RXMesh because it is
    // used to quickly determine whether or not a vertex is within
    // the "update band".
    std::vector<uint32_t> toplesets(rx.get_num_vertices(), 1u);
    std::vector<uint32_t> sorted_index;
    std::vector<uint32_t> limits;
    geodesic_ptp_openmesh<dataT>(h_seeds, sorted_index, limits, toplesets);

    // RXMesh Impl
    geodesic_rxmesh<dataT>(rx, h_seeds, sorted_index, limits, toplesets);
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
            RXMESH_INFO("\nUsage: Geodesic.exe < -option X>\n"
                        " -h:          Display this massage and exit\n"
                        " -input:      Input OBJ mesh file. Default is {} \n"
                        " -o:          JSON file output folder. Default is {} \n"
                       // "-num_seeds:   Number of input seeds. Default is {}\n"
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