#include <CLI/CLI.hpp>
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

using namespace rxmesh;

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "torus.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    bool        verify        = true;
    bool        skip_mcf      = false;
    uint32_t    device_id     = 0;
    char**      argv;
    int         argc;
} Arg;

#include "delaunay_rxmesh.cuh"

int main(int argc, char** argv)
{
    CLI::App app{"Delaunay edge flip application"};
    
    app.add_option("-i,--input", Arg.obj_file_name, "Input OBJ mesh file")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "torus.obj"));
    
    app.add_option("-o,--output", Arg.output_folder, "JSON file output folder")
        ->default_val(std::string(STRINGIFY(OUTPUT_DIR)));
    
    app.add_option("-d,--device_id", Arg.device_id, "GPU device ID")
        ->default_val(0u);
    
    bool no_verify = false;
    app.add_flag("--no_verify", no_verify, "Do not verify the output using OpenMesh");
    
    app.add_flag("--skip_mcf", Arg.skip_mcf, "Skip running MCF before and after Delaunay edge flip");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }
    
    // Handle --no_verify flag: if set, verify becomes false
    if (no_verify) {
        Arg.verify = false;
    }
    
    Arg.argv = argv;
    Arg.argc = argc;

    rx_init(Arg.device_id);

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("device_id= {}", Arg.device_id);
    RXMESH_TRACE("verify= {}", Arg.verify);
    RXMESH_TRACE("skip_mcf= {}", Arg.skip_mcf);

    // Select device
    cuda_query(Arg.device_id);

    RXMeshDynamic rx(Arg.obj_file_name, "", 512, 2.0, 2.0);
    // rx.save(STRINGIFY(OUTPUT_DIR) + extract_file_name(Arg.obj_file_name) +
    //        "_patches");

    // RXMeshDynamic rx(Arg.obj_file_name,
    //                 STRINGIFY(OUTPUT_DIR) +
    //                     extract_file_name(Arg.obj_file_name) + "_patches");

    if (!rx.is_edge_manifold()) {
        RXMESH_ERROR("Mesh is not edge manifold");
        return 1;
    }

    if (!Arg.skip_mcf) {
        if (!rx.is_closed()) {
            RXMESH_ERROR("mcf_rxmesh only takes watertight/closed mesh without boundaries");
            return 1;
        }
    }

    delaunay_rxmesh(rx, Arg.verify, Arg.skip_mcf);

    return 0;
}