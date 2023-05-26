#pragma once
#include "rxmesh/rxmesh_dynamic.h"

TEST(RXMeshDynamic, Validate)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id);

    RXMeshDynamic rxmesh(STRINGIFY(INPUT_DIR) "dragon.obj");

    EXPECT_TRUE(rxmesh.validate());
}