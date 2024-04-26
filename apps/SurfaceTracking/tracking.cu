#include "gtest/gtest.h"
#include "rxmesh/geometry_factory.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

struct arg
{
    std::string output_folder              = STRINGIFY(OUTPUT_DIR);
    uint32_t    device_id                  = 0;
    float       frame_dt                   = 0.05;
    float       sim_dt                     = 0.05;
    float       end_sim_t                  = 20.0;
    float       m_max_volume_change        = 0.0005;
    float       m_min_edge_length          = 0.5;
    float       m_max_edge_length          = 1.5;
    float       m_min_curvature_multiplier = 1.0;
    float       m_max_curvature_multiplier = 1.0;
    float       m_friction_coefficient     = 0.0;
    char**      argv;
    int         argc;
} Arg;

#include "tracking_rxmesh.cuh"

TEST(Apps, SurfaceTracking)
{
    using namespace rxmesh;

    // Select device
    cuda_query(Arg.device_id);

    std::vector<std::vector<float>> verts;

    std::vector<std::vector<uint32_t>> fv;

    const Vector<3, float> lower_corner(-3.0, 0.0, -3.0);

    create_plane(verts, fv, 60, 60, 0.1f, lower_corner);

    RXMeshDynamic rx(fv);
    //  rx.save(STRINGIFY(OUTPUT_DIR) + std::string("plane_patches"));

    // RXMeshDynamic rx(fv, STRINGIFY(OUTPUT_DIR) +
    // std::string("plane_patches"));

    rx.add_vertex_coordinates(verts, "plane");

    tracking_rxmesh(rx);
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
                        " -o:          JSON file output folder. Default is {} \n"
                        " -device_id:  GPU device ID. Default is {}",
            Arg.output_folder, Arg.device_id);
            // clang-format on
            exit(EXIT_SUCCESS);
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

    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    return RUN_ALL_TESTS();
}