#include "gtest/gtest.h"
#include "rxmesh/geometry_factory.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

inline float deg2rad(float deg)
{
    // r = PI*d / 180
    return deg * M_PI / 180.f;
}


struct arg
{
    std::string output_folder               = STRINGIFY(OUTPUT_DIR);
    std::string plane_name                  = "";
    int         n                           = 60;  // grid point along x and y
    uint32_t    device_id                   = 0;
    float       frame_dt                    = 0.05;
    float       sim_dt                      = 0.05;
    float       end_sim_t                   = 5.0;
    float       max_volume_change           = 0.0005;
    float       min_edge_length             = 0.5;
    float       collapser_min_edge_length   = 0;
    float       max_edge_length             = 1.5;
    float       splitter_max_edge_length    = 0.0;
    float       edge_flip_min_length_change = 1e-8;
    float       min_triangle_area           = 1e-7;
    float       min_triangle_angle          = deg2rad(0.f);
    float       max_triangle_angle          = deg2rad(180.f);
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

    const vec3<float> lower_corner(-3.0, 0.0, -3.0);

    Arg.plane_name =
        "plane" + std::to_string(Arg.n) + "x" + std::to_string(Arg.n);

    float spacing = 6.f / float(Arg.n);

    create_plane(verts, fv, Arg.n, Arg.n, 1, spacing, lower_corner);

    RXMeshDynamic rx(fv);

    // RXMeshDynamic rx(fv, STRINGIFY(OUTPUT_DIR) + Arg.plane_name +
    // "_patches"); rx.save(STRINGIFY(OUTPUT_DIR) + Arg.plane_name +
    // "_patches");

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
                        " -n:          Number of point along x(or y) direction. Default is {} \n"
                        " -o:          JSON file output folder. Default is {} \n"
                        " -device_id:  GPU device ID. Default is {}",
            Arg.n, Arg.output_folder, Arg.device_id);
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
        if (cmd_option_exists(argv, argc + argv, "-n")) {
            Arg.n = atoi(get_cmd_option(argv, argv + argc, "-n"));
        }
    }

    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("device_id= {}", Arg.device_id);
    RXMESH_TRACE("n= {}", Arg.n);

    return RUN_ALL_TESTS();
}