#include <CLI/CLI.hpp>
#include <cstdlib>

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
    int         device_id     = 0;
    char**      argv;
    int         argc;
} Arg;

#include "sec_rxmesh.cuh"

int sec_histogram()
{
    using namespace rxmesh;

    RXMeshDynamic rx(Arg.obj_file_name, "", 256, 3.5, 1.5);

    // const std::string p_file = STRINGIFY(OUTPUT_DIR) +
    //                            extract_file_name(Arg.obj_file_name) +
    //                            "_patches";
    // RXMeshDynamic rx(Arg.obj_file_name, p_file);
    // if (!std::filesystem::exists(p_file)) {
    //     rx.save(p_file);
    // }

    if (!rx.is_edge_manifold()) {
        RXMESH_ERROR("SECHistogram requires an edge-manifold mesh");
        return EXIT_FAILURE;
    }

    if (!rx.is_closed()) {
        RXMESH_ERROR("SECHistogram requires a closed mesh");
        return EXIT_FAILURE;
    }

    uint32_t final_num_vertices = Arg.target * rx.get_num_vertices();

    sec_rxmesh(rx, final_num_vertices);

    return 0;
}


int main(int argc, char** argv)
{
    using namespace rxmesh;

    CLI::App app{
        "SECHistogram - Shortest edge collapse with histogram-based selection"};

    app.add_option("-i,--input", Arg.obj_file_name, "Input OBJ mesh file")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "sphere3.obj"));

    app.add_option("-o,--output", Arg.output_folder, "JSON file output folder")
        ->default_val(std::string(STRINGIFY(OUTPUT_DIR)));

    app.add_option("-t,--target",
                   Arg.target,
                   "Fraction of output #vertices relative to input")
        ->default_val(0.1f);

    app.add_option(
           "-r,--reduce_ratio", Arg.reduce_ratio, "Reduction ratio per pass")
        ->default_val(0.1f);

    app.add_option("-d,--device_id", Arg.device_id, "GPU device ID")
        ->default_val(0);

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
    RXMESH_TRACE("target= {}", Arg.target);
    RXMESH_TRACE("reduce_ratio= {}", Arg.reduce_ratio);

    return sec_histogram();
}