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

    polyscope::view::upDir = polyscope::UpDir::ZUp;

    polyscope::init();

    rxmesh::RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

    auto polyscope_mesh = rx.get_polyscope_mesh();

    polyscope_mesh->setEdgeWidth(1.0);

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

    rx.polyscope_render_face_patch();

    rx.polyscope_render_vertex_patch();

    rx.polyscope_render_edge_patch();

    polyscope::show();

    return 0;
}
