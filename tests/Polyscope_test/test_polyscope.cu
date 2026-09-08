#include <cmath>

#include "rxmesh/geometry_util.cuh"
#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

int main(int argc, char** argv)
{
    rx_init(0);

    if (argc != 2 ||
        ((argv[1][0] != '0' && argv[1][0] != '1') || argv[1][1] != 0)) {
        RXMESH_ERROR("Usage: Polyscope_test <0: triangle | 1: tet>");
        return 1;
    }

    const bool is_tet = argv[1][0] == '1';

    polyscope::view::upDir = polyscope::UpDir::ZUp;

    RXMeshStatic rx(is_tet ? STRINGIFY(INPUT_DIR) "car.msh" :
                             STRINGIFY(INPUT_DIR) "dragon.obj");


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

    if (!is_tet) {
        // Face Normal
        auto face_normals = *rx.add_face_attribute<float>("fNormals", 3);
        face_normals.reset(0, LOCATION_ALL);

        rx.for_each<Op::FV, 256>(
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

        // Edge attibute
        auto edge_dist = *rx.add_edge_attribute<float>("edge_dist", 1);
        edge_dist.reset(-1.0f, HOST);

        rx.for_each<Op::EV, 256>(
            [=] __device__(EdgeHandle eh, VertexIterator & ev) mutable {
                const vec3<float> c0 = vertex_pos.to_glm<3>(ev[0]);
                const vec3<float> c1 = vertex_pos.to_glm<3>(ev[1]);

                vec3<float> d = (c0 + c1) / 2.f;

                edge_dist(eh) = glm::l2Norm(d);
            });

        edge_dist.move(DEVICE, HOST);

        auto ps_mesh = rx.get_polyscope_mesh();
        ps_mesh->setEdgeWidth(1.0);
        ps_mesh->addVertexColorQuantity("vColor", vertex_color);
        ps_mesh->addFaceVectorQuantity("fNormal", face_normals);
        ps_mesh->addEdgeScalarQuantity("edge_dist", edge_dist);

        rx.render_patch(1);
    } else {
        auto tet_volume = *rx.add_tet_attribute<float>("tVolume", 1);

        rx.for_each<Op::TV, 256>([=] __device__(TetHandle th,
                                                VertexIterator & tv) mutable {
            const vec3<float> c0 = vertex_pos.to_glm<3>(tv[0]);
            const vec3<float> c1 = vertex_pos.to_glm<3>(tv[1]);
            const vec3<float> c2 = vertex_pos.to_glm<3>(tv[2]);
            const vec3<float> c3 = vertex_pos.to_glm<3>(tv[3]);

            tet_volume(th) = std::fabs(signed_volume(c0, c1, c2, c3)) / 6.0f;
        });

        tet_volume.move(DEVICE, HOST);

        auto ps_mesh = rx.get_polyscope_volume_mesh();
        ps_mesh->addVertexColorQuantity("vColor", vertex_color);
        ps_mesh->addCellScalarQuantity("tVolume", tet_volume);

        rx.render_patch_volume(1);
    }

    polyscope::show();

    return 0;
}
