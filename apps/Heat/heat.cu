#include "rxmesh/rxmesh_static.h"
#include <CLI/CLI.hpp>

using namespace rxmesh;


int main(int argc, char** argv)
{
    CLI::App app{"Heat"};
    
    std::string mesh_path = STRINGIFY(INPUT_DIR) "sphere3.obj";
    uint32_t device_id = 0;
    
    app.add_option("-i,--input", mesh_path, "Input OBJ mesh file")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "sphere3.obj"));
    
    app.add_option("-d,--device_id", device_id, "GPU device ID")
        ->default_val(0u);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    rx_init(device_id);

    RXMeshStatic rx(mesh_path);

#if USE_POLYSCOPE
    polyscope::show();
#endif
}