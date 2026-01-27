// Compute geodesic distance according to
// Calla, Luciano A. Romero, Lizeth J. Fuentes Perez, and Anselmo A. Montenegro.
// "A minimalistic approach for fast computation of geodesic distances on
// triangular meshes." Computers & Graphics 84 (2019): 77-92

#include <cuda_profiler_api.h>
#include <CLI/CLI.hpp>
#include <cstdlib>
#include <random>

#include "../common/openmesh_trimesh.h"

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/log.h"

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

void geodesic()
{
    using namespace rxmesh;
    using dataT = float;

    RXMeshStatic rx(Arg.obj_file_name);
    if (!rx.is_closed()) {
        RXMESH_ERROR(
            "Geodesic only works on watertight/closed manifold mesh without "
            "boundaries");
        return;
    }
    if (!rx.is_edge_manifold()) {
        RXMESH_ERROR(
            "Geodesic only works on watertight/closed manifold mesh without "
            "boundaries");
        return;
    }


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


    CLI::App app{"Geodesic - Compute geodesic distances on triangular meshes"};

    app.add_option("-i,--input", Arg.obj_file_name, "Input OBJ mesh file")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "sphere3.obj"));

    app.add_option("-o,--output", Arg.output_folder, "JSON file output folder")
        ->default_val(std::string(STRINGIFY(OUTPUT_DIR)));

    app.add_option("-d,--device_id", Arg.device_id, "GPU device ID")
        ->default_val(0u);

    // num_seeds is commented out in the original, keeping it for now but not
    // exposing it app.add_option("--num_seeds", Arg.num_seeds, "Number of input
    // seeds")
    //     ->default_val(1u);

    rx_init(Arg.device_id);
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    Arg.argv = argv;
    Arg.argc = argc;

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("num_seeds= {}", Arg.num_seeds);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    geodesic();
}