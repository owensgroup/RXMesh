#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

TEST(RXMeshStatic, BoundaryVertex)
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