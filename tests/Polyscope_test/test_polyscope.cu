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

    // Face Normal
    auto face_normals = *rx.add_face_attribute<float>("fNormals", 3);
    face_normals.reset(0, LOCATION_ALL);


    rx.run_query_kernel<Op::FV, 256>(
        [=] __device__(FaceHandle face_id, VertexIterator & fv) mutable {
            // get the face's three vertices coordinates
            const vec3<float> c0 = vertex_pos.to_glm<3>(fv[0]);
            const vec3<float> c1 = vertex_pos.to_glm<3>(fv[1]);
            const vec3<float> c2 = vertex_pos.to_glm<3>(fv[2]);


            // compute the face normal
            glm::fvec3 n = cross(c1 - c0, c2 - c0);

            n = glm::normalize(n);

            // store the normals
            face_normals.from_glm(face_id, n);
        });

    face_normals.move(DEVICE, HOST);

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->setEdgeWidth(1.0);
    ps_mesh->addVertexColorQuantity("vColor", vertex_color);
    ps_mesh->addFaceVectorQuantity("fNormal", face_normals);

    polyscope::show();

    return 0;
}