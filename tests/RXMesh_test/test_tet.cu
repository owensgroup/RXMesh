#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <set>
#include <vector>

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/MshLoader.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

TEST(RXMeshStatic, TetVerticesOrientation)
{
    std::vector<std::vector<uint32_t>> permuted_tets = {
        {0, 1, 2, 3}, {6, 4, 7, 5}, {9, 11, 8, 10}};
    rxmesh::RXMeshStatic rx(permuted_tets, std::string(), 1);

    std::vector<uint32_t> reconstructed_tets(4 * permuted_tets.size());
    rx.create_tet_list(reconstructed_tets.data(), true);
    for (uint32_t t = 0; t < permuted_tets.size(); ++t) {
        for (uint32_t v = 0; v < 4; ++v) {
            EXPECT_EQ(reconstructed_tets[4 * t + v], permuted_tets[t][v]);
        }
    }
}

TEST(RXMeshStatic, Tet)
{
    std::vector<std::vector<rx_coord_t>> vertices;
    std::vector<std::vector<uint32_t>>   tets;

    EXPECT_EQ(rxmesh::load_msh(STRINGIFY(INPUT_DIR) "car.msh", vertices, tets),
              rxmesh::MeshKind::Tet);

    rxmesh::RXMeshStatic mesh(std::string(STRINGIFY(INPUT_DIR) "car.msh"),
                              std::string(),
                              512,
                              1.0f,
                              1.0f,
                              0.8f);

    EXPECT_EQ(mesh.get_num_tets(), tets.size());
    EXPECT_EQ(mesh.get_num_tets(true), tets.size());
    EXPECT_EQ(mesh.get_num_elements<rxmesh::TetHandle>(), tets.size());

    std::vector<uint32_t> global_tets(4 * mesh.get_num_tets());
    mesh.create_tet_list(global_tets.data(), true);
    for (uint32_t t = 0; t < tets.size(); ++t) {
        for (uint32_t v = 0; v < 4; ++v) {
            EXPECT_EQ(global_tets[4 * t + v], tets[t][v]);
        }
    }

    std::set<std::array<uint32_t, 3>> input_faces;

    const auto tet_face_indices = rxmesh::tet_faces();
    for (const auto& tet : tets) {
        for (const auto& indices : tet_face_indices) {
            std::array<uint32_t, 3> face = {
                tet[indices[0]], tet[indices[1]], tet[indices[2]]};
            std::sort(face.begin(), face.end());
            input_faces.insert(face);
        }
    }
    ASSERT_EQ(input_faces.size(), mesh.get_num_faces());

    std::vector<uint32_t> global_faces(3 * mesh.get_num_faces());
    mesh.create_face_list(global_faces.data(), true);

    std::set<std::array<uint32_t, 3>> output_faces;
    for (uint32_t f = 0; f < mesh.get_num_faces(); ++f) {
        std::array<uint32_t, 3> face = {global_faces[3 * f],
                                        global_faces[3 * f + 1],
                                        global_faces[3 * f + 2]};
        EXPECT_LT(face[0], face[1]);
        EXPECT_LT(face[1], face[2]);
        std::sort(face.begin(), face.end());
        EXPECT_EQ(input_faces.count(face), 1);
        EXPECT_TRUE(output_faces.insert(face).second);
    }
    EXPECT_EQ(output_faces.size(), mesh.get_num_faces());

    std::vector<glm::uvec4> compact_tets;
    mesh.create_tet_list(compact_tets);
    ASSERT_EQ(compact_tets.size(), mesh.get_num_tets());

    std::vector<uint32_t> compact_tets_raw(4 * mesh.get_num_tets());
    mesh.create_tet_list(compact_tets_raw.data());

    bool has_nonidentity_tet_mapping    = false;
    bool has_nonidentity_vertex_mapping = false;
    for (uint32_t t = 0; t < compact_tets.size(); ++t) {
        const rxmesh::TetHandle th = mesh.map_to_local_tet(t);
        ASSERT_TRUE(th.is_valid());
        EXPECT_EQ(mesh.linear_id(th), t);

        const uint32_t global_t = mesh.map_to_global(th);
        ASSERT_LT(global_t, tets.size());
        has_nonidentity_tet_mapping |= global_t != t;

        for (uint32_t v = 0; v < 4; ++v) {
            EXPECT_EQ(compact_tets_raw[4 * t + v], compact_tets[t][v]);

            const rxmesh::VertexHandle vh =
                mesh.map_to_local_vertex(compact_tets[t][v]);
            ASSERT_TRUE(vh.is_valid());
            EXPECT_EQ(mesh.map_to_global(vh), tets[global_t][v]);
            has_nonidentity_vertex_mapping |=
                compact_tets[t][v] != tets[global_t][v];
        }
    }
    EXPECT_TRUE(has_nonidentity_tet_mapping);
    EXPECT_TRUE(has_nonidentity_vertex_mapping);
}
