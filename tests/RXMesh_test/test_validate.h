#pragma once
#include "rxmesh/rxmesh_dynamic.h"

TEST(RXMeshDynamic, Validate)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;
    ASSERT_TRUE(import_obj(
        STRINGIFY(INPUT_DIR) "dragon.obj", Verts, Faces, rxmesh_args.quite));

    RXMeshDynamic rxmesh(Faces, rxmesh_args.quite);

    EXPECT_TRUE(rxmesh.validate());
}