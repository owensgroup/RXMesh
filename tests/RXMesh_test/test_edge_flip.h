#include "gtest/gtest.h"

#include "edge_flip.cuh"
#include "rxmesh/rxmesh_dynamic.h"

TEST(RXMeshDynamic, EdgeFlip)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;
    ASSERT_TRUE(import_obj(
        STRINGIFY(INPUT_DIR) "sphere3.obj", Verts, Faces, rxmesh_args.quite));

    RXMeshDynamic rxmesh(Faces, rxmesh_args.quite);


}