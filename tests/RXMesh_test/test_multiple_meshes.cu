#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

TEST(RXMeshStatic, MultipleMeshes)
{
    using namespace rxmesh;

    std::vector<std::string> inputs = {STRINGIFY(INPUT_DIR) "dragon.obj",
                                       STRINGIFY(INPUT_DIR) "giraffe.obj",
                                       STRINGIFY(INPUT_DIR) "bunnyhead.obj"};

    RXMeshStatic rx(inputs);

    auto      x       = *rx.get_input_vertex_coordinates();
    auto      v_label = *rx.get_vertex_region_label();
    glm::vec3 lower, upper;
    rx.bounding_box(lower, upper);

    glm::vec3 bb = upper - lower;

    for (int i = 0; i < rx.get_num_regions(); ++i) {
        rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
            int j = i % 3;
            if (v_label(vh) == i) {
                x(vh, j) += 0.5 * (i + 1) * bb[j];
            }
        });
    }

    rx.get_polyscope_mesh()->updateVertexPositions(x);


    polyscope::show();
}