#include "gtest/gtest.h"

#include <cstdio>
#include <string>
#include <vector>

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/msh_io.h"

namespace {

void check_msh_load_save(rxmesh::RXMeshStatic& mesh,
                         const std::string&    filename)
{
    using namespace rxmesh;

    const auto coords = mesh.get_input_vertex_coordinates();

    std::vector<std::vector<rx_coord_t>> expected_vertices(
        mesh.get_num_vertices(), std::vector<rx_coord_t>(3));
    mesh.for_each_vertex(HOST, [&](const VertexHandle vh) {
        const uint32_t v_id        = mesh.linear_id(vh);
        expected_vertices[v_id][0] = (*coords)(vh, 0);
        expected_vertices[v_id][1] = (*coords)(vh, 1);
        expected_vertices[v_id][2] = (*coords)(vh, 2);
    });

    const bool     is_tet_mesh          = mesh.get_num_tets() != 0;
    const uint32_t vertices_per_simplex = is_tet_mesh ? 4 : 3;
    const uint32_t num_simplices =
        is_tet_mesh ? mesh.get_num_tets() : mesh.get_num_faces();

    std::vector<uint32_t> expected_connectivity(vertices_per_simplex *
                                                num_simplices);
    if (is_tet_mesh) {
        mesh.create_tet_list(expected_connectivity.data());
    } else {
        mesh.create_face_list(expected_connectivity.data());
    }

    mesh.export_msh(filename, *coords);

    std::vector<std::vector<rx_coord_t>> actual_vertices;
    std::vector<std::vector<uint32_t>>   actual_simplices;
    const MeshKind kind = load_msh(filename, actual_vertices, actual_simplices);

    EXPECT_EQ(kind, is_tet_mesh ? MeshKind::Tet : MeshKind::Triangle);
    ASSERT_EQ(actual_vertices.size(), expected_vertices.size());
    ASSERT_EQ(actual_simplices.size(), num_simplices);

    for (uint32_t v = 0; v < actual_vertices.size(); ++v) {
        ASSERT_EQ(actual_vertices[v].size(), 3);
        for (uint32_t c = 0; c < 3; ++c) {
            EXPECT_EQ(actual_vertices[v][c], expected_vertices[v][c]);
        }
    }

    for (uint32_t s = 0; s < actual_simplices.size(); ++s) {
        ASSERT_EQ(actual_simplices[s].size(), vertices_per_simplex);
        for (uint32_t v = 0; v < vertices_per_simplex; ++v) {
            EXPECT_EQ(actual_simplices[s][v],
                      expected_connectivity[vertices_per_simplex * s + v]);
        }
    }
}

}  // namespace

TEST(RXMeshStatic, Export)
{
    using namespace rxmesh;

    CUDA_ERROR(cudaDeviceReset());

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto v_attr_scalar = *rx.add_vertex_attribute<float>("vScalar", 1);
    auto v_attr_vec2   = *rx.add_vertex_attribute<float>("vVector2", 2);
    auto v_attr_vec3   = *rx.add_vertex_attribute<float>("vVector3", 3);

    auto f_attr_scalar = *rx.add_face_attribute<float>("fScalar", 1);
    auto f_attr_vec2   = *rx.add_face_attribute<float>("fVector2", 2);
    auto f_attr_vec3   = *rx.add_face_attribute<float>("fVector3", 3);

    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        v_attr_scalar(vh, 0) = rand() % 100;

        for (uint32_t i = 0; i < v_attr_vec2.get_num_attributes(); ++i) {
            v_attr_vec2(vh, i) = rand() % 100;
        }

        for (uint32_t i = 0; i < v_attr_vec3.get_num_attributes(); ++i) {
            v_attr_vec3(vh, i) = rand() % 100;
        }
    });


    rx.for_each_face(HOST, [&](const FaceHandle& fh) {
        f_attr_scalar(fh, 0) = rand() % 100;

        for (uint32_t i = 0; i < f_attr_vec2.get_num_attributes(); ++i) {
            f_attr_vec2(fh, i) = rand() % 100;
        }

        for (uint32_t i = 0; i < f_attr_vec3.get_num_attributes(); ++i) {
            f_attr_vec3(fh, i) = rand() % 100;
        }
    });


    rx.export_obj("sphere3.obj", *rx.get_input_vertex_coordinates());

    rx.export_vtk("sphere3.vtk",
                  *rx.get_input_vertex_coordinates(),
                  v_attr_scalar,
                  v_attr_vec2,
                  v_attr_vec3,
                  f_attr_scalar,
                  f_attr_vec2,
                  f_attr_vec3);

    const std::string triangle_msh =
        std::string(STRINGIFY(OUTPUT_DIR)) + "sphere3_roundtrip.msh";
    check_msh_load_save(rx, triangle_msh);
    std::remove(triangle_msh.c_str());

    RXMeshStatic      tet_rx(STRINGIFY(INPUT_DIR) "car.msh");
    const std::string tet_msh =
        std::string(STRINGIFY(OUTPUT_DIR)) + "car_roundtrip.msh";
    check_msh_load_save(tet_rx, tet_msh);
    std::remove(tet_msh.c_str());

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}
