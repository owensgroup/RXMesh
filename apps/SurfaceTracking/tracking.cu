#include <CLI/CLI.hpp>
#include <cstdlib>

#include "rxmesh/geometry_factory.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

using namespace rxmesh;

inline float deg2rad(float deg)
{
    // r = PI*d / 180
    return deg * M_PI / 180.f;
}


struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "el_topo_sphere_1280.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    std::string plane_name    = "";
    int         n             = 60;  // grid point along x and y
    uint32_t    device_id     = 0;
    float       frame_dt      = 0.05;
    float       sim_dt        = 0.05;
    float       end_sim_t     = 5.0;
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

int main(int argc, char** argv)
{
    using namespace rxmesh;

    CLI::App app{"SurfaceTracking - Surface tracking simulation"};

    app.add_option("-i,--input", Arg.obj_file_name, "Input OBJ mesh file")
        ->default_val(
            std::string(STRINGIFY(INPUT_DIR) "el_topo_sphere_1280.obj"));

    app.add_option("-o,--output", Arg.output_folder, "JSON file output folder")
        ->default_val(std::string(STRINGIFY(OUTPUT_DIR)));

    app.add_option("-n,--grid_n",
                   Arg.n,
                   "Grid resolution used for the optional plane setup")
        ->default_val(60);

    app.add_option("--frame_dt", Arg.frame_dt, "Frame step size")
        ->default_val(0.05f);

    app.add_option("--sim_dt", Arg.sim_dt, "Simulation time step")
        ->default_val(0.05f);

    app.add_option("--end_sim_t", Arg.end_sim_t, "Simulation duration")
        ->default_val(5.0f);

    app.add_option("-d,--device_id", Arg.device_id, "GPU device ID")
        ->default_val(0u);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    rx_init(static_cast<int>(Arg.device_id));

    Arg.argv = argv;
    Arg.argc = argc;

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("device_id= {}", Arg.device_id);
    RXMESH_TRACE("n= {}", Arg.n);
    RXMESH_TRACE("frame_dt= {}", Arg.frame_dt);
    RXMESH_TRACE("sim_dt= {}", Arg.sim_dt);
    RXMESH_TRACE("end_sim_t= {}", Arg.end_sim_t);


    RXMeshDynamic rx(Arg.obj_file_name, "", 256, 3.5, 5);

    tracking_rxmesh(rx);

    return 0;
}