#include <array>
#include <vector>

#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

TEST(RXMeshStatic, TriangleBoundaryVertex)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bunnyhead.obj");

    auto v_bd = *rx.add_vertex_attribute<bool>("vBoundary", 1);

    rx.get_boundary_vertices(v_bd);

    uint32_t num_bd_vertices = 0;

    rx.for_each_vertex(
        HOST,
        [&](const VertexHandle& vh) {
            if (v_bd(vh)) {
                num_bd_vertices++;
            }
        },
        NULL,
        false);

    EXPECT_EQ(num_bd_vertices, 98);

    // auto ps = rx.get_polyscope_mesh();
    // ps->addVertexScalarQuantity("vBoundary", *v_bd);
    // polyscope::show();

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(RXMeshStatic, TetBoundaryVertex)
{
    using namespace rxmesh;

    auto check_boundary = [](std::vector<std::vector<uint32_t>>& tets,
                             const std::array<bool, 5>&          expected,
                             const bool                          is_closed) {
        RXMeshStatic rx(tets, "", 1);

        EXPECT_EQ(rx.is_closed(), is_closed);
        EXPECT_GT(rx.get_num_patches(), 1);

        auto v_bd = *rx.add_vertex_attribute<bool>("vBoundary", 1);
        rx.get_boundary_vertices(v_bd);

        std::array<bool, 5> visited = {};

        rx.for_each_vertex(
            HOST,
            [&](const VertexHandle& vh) {
                EXPECT_TRUE(vh.is_valid());
                EXPECT_EQ(rx.get_owner_handle(vh), vh);

                const uint32_t v = rx.map_to_global(vh);
                ASSERT_LT(v, expected.size());
                EXPECT_FALSE(visited[v]);
                EXPECT_EQ(v_bd(vh), expected[v]);
                visited[v] = true;
            },
            NULL,
            false);

        for (const bool v : visited) {
            EXPECT_TRUE(v);
        }

        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    };

    std::vector<std::vector<uint32_t>> open_tets = {
        {4, 1, 2, 3}, {0, 4, 2, 3}, {0, 1, 4, 3}, {0, 1, 2, 4}};
    check_boundary(open_tets, {true, true, true, true, false}, false);

    std::vector<std::vector<uint32_t>> closed_tets = {
        {1, 2, 3, 4}, {2, 0, 3, 4}, {0, 1, 3, 4}, {1, 0, 2, 4}, {0, 1, 2, 3}};
    check_boundary(closed_tets, {false, false, false, false, false}, true);
}
