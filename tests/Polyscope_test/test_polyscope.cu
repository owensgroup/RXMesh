#include <vector>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/log.h"

int main(int argc, char** argv)
{
    rxmesh::Log::init();
    rxmesh::cuda_query(0);

    polyscope::init();

    rxmesh::RXMeshStatic rxmesh(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto psMesh =
        polyscope::registerSurfaceMesh("RXMesh",
                                       *rxmesh.get_input_vertex_coordinates(),
                                       *rxmesh.get_input_face_indices());

    polyscope::show();

    return 0;
}