#include "gtest/gtest.h"

#include "rxmesh/rxmesh_dynamic.h"

TEST(RXMeshDynamic, Validate)
{
    using namespace rxmesh;

    RXMeshDynamic rxmesh(STRINGIFY(INPUT_DIR) "dragon.obj");

    EXPECT_TRUE(rxmesh.validate());
}