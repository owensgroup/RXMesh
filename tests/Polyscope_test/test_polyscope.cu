#include <vector>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/log.h"

int main(int argc, char** argv)
{
    rxmesh::Log::init();
    rxmesh::cuda_query(0);

    polyscope::init();

    rxmesh::RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

    std::string p_name = "RXMesh";

    auto polyscope_mesh =
        polyscope::registerSurfaceMesh(p_name,
                                       *rx.get_input_vertex_coordinates(),
                                       *rx.get_input_face_indices());

    auto vertex_pos   = *rx.get_input_vertex_coordinates();
    auto vertex_color = *rx.add_vertex_attribute<float>("vColor", 3);

    rx.for_each_vertex(
        rxmesh::DEVICE,
        [vertex_color, vertex_pos] __device__(const rxmesh::VertexHandle vh) {
            vertex_color(vh, 0) = 0.9;
            vertex_color(vh, 1) = vertex_pos(vh, 1);
            vertex_color(vh, 2) = 0.9;
        });

    vertex_color.move(rxmesh::DEVICE, rxmesh::HOST);


    polyscope_mesh->addVertexColorQuantity("vColor", vertex_color);

    polyscope::show();

    return 0;
}