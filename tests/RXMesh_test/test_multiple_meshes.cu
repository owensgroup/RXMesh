#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

TEST(RXMeshStatic, MultipleMeshes)
{
    using namespace rxmesh;

    std::vector<std::string> inputs = {STRINGIFY(INPUT_DIR) "dragon.obj",
                                       STRINGIFY(INPUT_DIR) "giraffe.obj",
                                       STRINGIFY(INPUT_DIR) "bunnyhead.obj"};

    RXMeshStatic rx(inputs);

    polyscope::show();
}