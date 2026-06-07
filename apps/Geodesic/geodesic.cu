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
    int         num_seeds = 1;
    int         seed_id   = -1;

} Arg;

#include "geodesic_ptp_openmesh.h"
#include "geodesic_ptp_rxmesh.h"
#include "geodesic_ptp_rxmesh_graph.h"


void geodesic()
{
    using namespace rxmesh;

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
    DenseMatrix<int> h_seeds(Arg.num_seeds, 1, HOST);
    if (Arg.seed_id < 0) {
        std::random_device                                       dev;
        std::mt19937                                             rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(
            0, rx.get_num_vertices());
        for (int i = 0; i < Arg.num_seeds; ++i) {
            h_seeds(i, 0) = dist(rng);
        }
    } else {
        assert(Arg.num_seeds == 1);
        h_seeds(0, 0) = Arg.seed_id;
    }


    // Build the per-vertex topleset (BFS level from the seed set) and
    // the matching band-offset `limits` array. The PTP kernel uses
    // topleset as the active-set predicate and limits as its (i, j)
    // band window.
    std::shared_ptr<VertexAttribute<int>> d_toplesets;
    DenseMatrix<int>                      limits;
    int                                   limits_size = 0;


    std::vector<int> h_toplesets(rx.get_num_vertices(), 1);
    limits = DenseMatrix<int>(rx.get_num_vertices() + 2, 1, HOST);
    geodesic_ptp_openmesh<rx_coord_t>(
        h_seeds, limits, limits_size, h_toplesets);

    d_toplesets = rx.add_vertex_attribute(h_toplesets, "topleset");


    geodesic_rxmesh<rx_coord_t>(rx, h_seeds, limits, limits_size, *d_toplesets);
}

int main(int argc, char** argv)
{
    using namespace rxmesh;


    CLI::App app{"Geodesic - Compute geodesic distances on triangular meshes"};

    app.add_option("-i,--input", Arg.obj_file_name, "Input OBJ mesh file")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "sphere3.obj"));

    app.add_option("-s,--source",
                   Arg.seed_id,
                   "Source vertex ID for computing geodesic distance")
        ->default_val(-1);

    app.add_option("-o,--output", Arg.output_folder, "JSON file output folder")
        ->default_val(std::string(STRINGIFY(OUTPUT_DIR)));

    app.add_option("-d,--device_id", Arg.device_id, "GPU device ID")
        ->default_val(0u);

    // num_seeds is commented out in the original, keeping it for now but not
    // exposing it app.add_option("--num_seeds", Arg.num_seeds, "Number of input
    // seeds")
    //     ->default_val(1u);

    rx_init(Arg.device_id, spdlog::level::trace);
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
    RXMESH_TRACE("num_seeds= {}", Arg.num_seeds);
    RXMESH_TRACE("source = {}", Arg.seed_id);

    geodesic();
}