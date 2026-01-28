#include <CLI/CLI.hpp>
#include <cstdlib>

#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

#include "rxmesh/geometry_factory.h"
struct arg
{
    std::string obj_file_name    = STRINGIFY(INPUT_DIR) "cloth.obj";
    std::string output_folder    = STRINGIFY(OUTPUT_DIR);
    uint32_t    nx               = 66;
    uint32_t    ny               = 66;
    float       relative_len     = 1.0;
    int         num_smooth_iters = 5;
    uint32_t    num_iter         = 3;
    uint32_t    device_id        = 0;
    char**      argv;
    int         argc;
} Arg;

#include "remesh_rxmesh.cuh"

int remesh()
{
    using namespace rxmesh;

    // std::vector<std::vector<float>>    verts;
    // std::vector<std::vector<uint32_t>> fv;
    // create_plane(verts, fv, Arg.nx, Arg.ny);
    // RXMeshDynamic rx(fv);
    // rx.add_vertex_coordinates(verts, "Coords");


    RXMeshDynamic rx(Arg.obj_file_name, "", 512, 2.0, 2);
    // rx.save(STRINGIFY(OUTPUT_DIR) + extract_file_name(Arg.obj_file_name) +
    //         "_patches");

    // RXMeshDynamic rx(Arg.obj_file_name,
    //                  STRINGIFY(OUTPUT_DIR) +
    //                      extract_file_name(Arg.obj_file_name) + "_patches");
    //

    if (!rx.is_edge_manifold()) {
        RXMESH_ERROR("Remesh requires an edge-manifold mesh");
        return EXIT_FAILURE;
    }

    // rx.export_obj("grid_" + std::to_string(Arg.nx) + "_" +
    //                   std::to_string(Arg.ny) + ".obj",
    //               *rx.get_input_vertex_coordinates());

    remesh_rxmesh(rx);

    return 0;
}


int main(int argc, char** argv)
{
    using namespace rxmesh;

    CLI::App app{"Remesh - Mesh remeshing application"};

    app.add_option("-i,--input", Arg.obj_file_name, "Input OBJ mesh file")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "cloth.obj"));

    app.add_option("-o,--output", Arg.output_folder, "JSON file output folder")
        ->default_val(std::string(STRINGIFY(OUTPUT_DIR)));

    app.add_option(
           "-n,--num_iter", Arg.num_iter, "Number of remeshing iterations")
        ->default_val(3u);

    app.add_option("--relative_len",
                   Arg.relative_len,
                   "Target edge length as a ratio of the input mesh average "
                   "edge length")
        ->default_val(1.0f);

    app.add_option(
           "--nx", Arg.nx, "Grid size in x direction (for plane generation)")
        ->default_val(66u);

    app.add_option(
           "--ny", Arg.ny, "Grid size in y direction (for plane generation)")
        ->default_val(66u);

    app.add_option("-d,--device_id", Arg.device_id, "GPU device ID")
        ->default_val(0u);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    rx_init(Arg.device_id);

    Arg.argv = argv;
    Arg.argc = argc;

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("device_id= {}", Arg.device_id);
    RXMESH_TRACE("num_iter= {}", Arg.num_iter);
    RXMESH_TRACE("relative_len= {}", Arg.relative_len);
    RXMESH_TRACE("nx= {}", Arg.nx);
    RXMESH_TRACE("ny= {}", Arg.ny);

    return remesh();
}