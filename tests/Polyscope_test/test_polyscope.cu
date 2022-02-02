#include <vector>
#define POLYSCOPE_NO_STANDARDIZE_FALLTHROUGH
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

    std::vector<std::vector<float>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    import_obj(STRINGIFY(INPUT_DIR) "sphere3.obj", Verts, Faces);

    rxmesh::RXMeshStatic rxmesh(Faces);

    auto coords = rxmesh.add_vertex_attribute<float>(Verts, "coordinates");
    auto indices = rxmesh.add_face_attribute<uint32_t>(Faces, "indices");

    auto psMesh = polyscope::registerSurfaceMesh("RXMesh", *coords, *indices);

    
    polyscope::show();

    return 0;
}