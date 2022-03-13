#include "gtest/gtest.h"

#include "rxmesh/context.h"
#include "rxmesh/kernels/update_dispatcher.cuh"
#include "rxmesh/rxmesh_dynamic.h"

template <uint32_t blockThreads>
__global__ static void edge_collpase(rxmesh::Context context)
{
    using namespace rxmesh;

    auto should_collpase = [&](const EdgeHandle& edge) -> bool {
        if (edge.unpack().second == 1) {
            return true;
        } else {
            return false;
        }
    };

    update_block_dispatcher<DynOp::EdgeCollapse, blockThreads>(context,
                                                               should_collpase);
}

TEST(RXMeshDynamic, EdgeCollpase)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshDynamic rxmesh(rxmesh_args.obj_file_name, rxmesh_args.quite);

    ASSERT_TRUE(rxmesh.is_edge_manifold());

    EXPECT_TRUE(rxmesh.validate());

    constexpr uint32_t      blockThreads = 256;
    LaunchBox<blockThreads> launch_box;
    rxmesh.prepare_launch_box(
        {}, {DynOp::EdgeFlip}, launch_box, (void*)edge_collpase<blockThreads>);

    edge_collpase<blockThreads>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rxmesh.get_context());

    CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_TRUE(rxmesh.validate());
}