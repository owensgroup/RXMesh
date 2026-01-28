#include <vector>

#include <CLI/CLI.hpp>
#include <cstdlib>

#include "gtest/gtest.h"

#include "rxmesh/util/report.h"

struct RXMeshTestArg
{
    int         device_id     = 0;
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    int         argc          = argc;
    char**      argv          = argv;
} rxmesh_args;

// clang-format off
#include "test_queries.h"
#include "test_patch_scheduler.cuh"
#include "test_patch_lock.cuh"
#include "test_wasted_work.cuh"
#include "test_grad.h"
// clang-format on

int main(int argc, char** argv)
{
    using namespace rxmesh;

    ::testing::InitGoogleTest(&argc, argv);

    CLI::App app{"RXMeshTest"};

    app.add_option(
           "-i,--input", rxmesh_args.obj_file_name, "Input OBJ mesh file")
        ->default_val(rxmesh_args.obj_file_name);

    app.add_option("-d,--device_id", rxmesh_args.device_id, "GPU device ID")
        ->default_val(rxmesh_args.device_id);

    app.add_option(
           "-o,--output", rxmesh_args.output_folder, "JSON file output folder")
        ->default_val(rxmesh_args.output_folder);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    rx_init(rxmesh_args.device_id);

    rxmesh_args.argc = argc;
    rxmesh_args.argv = argv;

    RXMESH_INFO("input= {}", rxmesh_args.obj_file_name);
    RXMESH_INFO("output_folder= {}", rxmesh_args.output_folder);
    RXMESH_INFO("device_id= {}", rxmesh_args.device_id);

    return RUN_ALL_TESTS();
}
