#include "gtest/gtest.h"

#include "edge_flip.cuh"
#include "rxmesh/rxmesh_dynamic.h"

TEST(RXMeshDynamic, EdgeFlip)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshDynamic rxmesh(STRINGIFY(INPUT_DIR) "diamond.obj", rxmesh_args.quite);

    ASSERT_TRUE(rxmesh.is_edge_manifold());

    EXPECT_TRUE(rxmesh.validate());

    constexpr uint32_t      blockThreads = 256;
    LaunchBox<blockThreads> launch_box;
    rxmesh.prepare_launch_box(
        {}, {DynOp::EdgeFlip}, launch_box, (void*)edge_flip<blockThreads>);

    edge_flip<blockThreads>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rxmesh.get_context());

    CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_TRUE(rxmesh.validate());
}