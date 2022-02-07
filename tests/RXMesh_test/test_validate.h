#pragma once
#include "rxmesh/rxmesh_dynamic.h"

TEST(RXMeshDynamic, Validate)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshDynamic rxmesh(STRINGIFY(INPUT_DIR) "dragon.obj", rxmesh_args.quite);

    EXPECT_TRUE(rxmesh.validate());
}