#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

int main(int argc, char** argv)
{
    rx_init(0);

    polyscope::view::upDir = polyscope::UpDir::ZUp;


    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");


    // Vertex Color
    auto vertex_pos   = *rx.get_input_vertex_coordinates();
    auto vertex_color = *rx.add_vertex_attribute<float>("vColor", 3);
    rx.for_each_vertex(
        DEVICE, [vertex_color, vertex_pos] __device__(const VertexHandle vh) {
            vertex_color(vh, 0) = 0.9;
            vertex_color(vh, 1) = vertex_pos(vh, 1);
            vertex_color(vh, 2) = 0.9;
        });

    vertex_color.move(DEVICE, HOST);

    // Vertex Normal
    auto vertex_normals = *rx.add_vertex_attribute<float>("vNormals", 3);
    vertex_normals.reset(0, LOCATION_ALL);


    rx.run_query_kernel<Op::FV, 256>([=] __device__(FaceHandle face_id,
                                                    VertexIterator & fv) {
        // get the face's three vertices coordinates
        glm::fvec3 c0(
            vertex_pos(fv[0], 0), vertex_pos(fv[0], 1), vertex_pos(fv[0], 2));
        glm::fvec3 c1(
            vertex_pos(fv[1], 0), vertex_pos(fv[1], 1), vertex_pos(fv[1], 2));
        glm::fvec3 c2(
            vertex_pos(fv[2], 0), vertex_pos(fv[2], 1), vertex_pos(fv[2], 2));

        // compute the face normal
        glm::fvec3 n = cross(c1 - c0, c2 - c0);

        // the three edges length
        glm::fvec3 l(glm::distance2(c0, c1),
                     glm::distance2(c1, c2),
                     glm::distance2(c2, c0));

        // add the face's normal to its vertices
        for (uint32_t v = 0; v < 3; ++v) {      // for every vertex in this face
            for (uint32_t i = 0; i < 3; ++i) {  // for the vertex 3 coordinates
                atomicAdd(&vertex_normals(fv[v], i),
                          n[i] / (l[v] + l[(v + 2) % 3]));
            }
        }
    });

    vertex_normals.move(DEVICE, HOST);

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->setEdgeWidth(1.0);
    ps_mesh->addVertexColorQuantity("vColor", vertex_color);
    ps_mesh->addVertexVectorQuantity("vNormal", vertex_normals);

    polyscope::show();

    return 0;
}