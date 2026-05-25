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
    std::vector<uint32_t> h_seeds(Arg.num_seeds);
    if (Arg.seed_id < 0) {
        std::random_device                                       dev;
        std::mt19937                                             rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(
            0, rx.get_num_vertices());
        for (auto& s : h_seeds) {
            s = dist(rng);
            // s = 0;
        }
    } else {
        assert(Arg.num_seeds == 1);
        h_seeds[0] = Arg.seed_id;
    }


    // Save a map from vertex id to topleset (number of hops from
    // (closest?) source). It's used by OpenMesh to help construct
    // sorted_index and limit. We keep it for RXMesh because it is
    // used to quickly determine whether or not a vertex is within
    // the "update band".
    std::vector<uint32_t> toplesets(rx.get_num_vertices(), 1u);
    std::vector<uint32_t> sorted_index;
    std::vector<uint32_t> limits;
    geodesic_ptp_openmesh<rx_coord_t>(h_seeds, sorted_index, limits, toplesets);


    // RXMesh Impl
    //geodesic_rxmesh<rx_coord_t>(rx, h_seeds, sorted_index, limits, toplesets);

    if (Arg.topleset_backend == "gpu") {
        auto seed_mask = *rx.add_vertex_attribute<uint8_t>("seed_mask", 1);
        seed_mask.reset(uint8_t(0), HOST);
        rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
            const uint32_t v_id = rx.map_to_global(vh);
            for (int k = 0; k < h_seeds.rows(); ++k) {
                if (h_seeds(k, 0) == static_cast<int>(v_id)) {
                    seed_mask(vh, 0) = uint8_t(1);
                    break;
                }
            }
        });
        seed_mask.move(HOST, DEVICE);

        d_toplesets =
            rx.add_vertex_attribute<int>("topleset", 1, LOCATION_ALL);

        uint32_t    num_levels = 0;
        const float bfs_ms     = compute_toplesets_device<int>(
            rx, seed_mask, *d_toplesets, num_levels);
        RXMESH_INFO("GPU topleset BFS took {} (ms), {} levels",
                    bfs_ms,
                    num_levels);

        d_toplesets->move(DEVICE, HOST);

        limits = DenseMatrix<int>(num_levels + 2, 1, HOST);
        build_limits_from_toplesets<int>(
            rx, *d_toplesets, num_levels, limits, limits_size);
    } else if (Arg.topleset_backend == "cpu") {
        std::vector<int> h_toplesets(rx.get_num_vertices(), 1);
        limits = DenseMatrix<int>(rx.get_num_vertices() + 2, 1, HOST);
        geodesic_ptp_openmesh<rx_coord_t>(
            h_seeds, limits, limits_size, h_toplesets);

        d_toplesets = rx.add_vertex_attribute(h_toplesets, "topleset");
    } else {
        RXMESH_ERROR(
            "Unknown --topleset_backend '{}'. Expected 'gpu' or 'cpu'.",
            Arg.topleset_backend);
        return;
    }

    geodesic_rxmesh_graph<rx_coord_t>(
        rx, h_seeds, limits, limits_size, *d_toplesets);

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
    RXMESH_TRACE("num_seeds= {}", Arg.num_seeds);
    RXMESH_TRACE("source = {}", Arg.seed_id);

    geodesic();
}