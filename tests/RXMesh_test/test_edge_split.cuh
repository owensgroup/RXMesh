#include "gtest/gtest.h"

#include "rxmesh/context.h"
#include "rxmesh/kernels/update_dispatcher.cuh"
#include "rxmesh/rxmesh_dynamic.h"

template <uint32_t blockThreads>
__global__ static void edge_split(rxmesh::Context context)
{
    using namespace rxmesh;

    auto should_split = [&](const EdgeHandle& edge) -> bool {
        if (edge.unpack().second == 1) {
            return true;
        } else {
            return false;
        }
    };

    update_block_dispatcher<DynOp::EdgeSplit, blockThreads>(context,
                                                            should_split);
}

TEST(RXMeshDynamic, EdgeSplit)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshDynamic rxmesh(rxmesh_args.obj_file_name, rxmesh_args.quite);

    ASSERT_TRUE(rxmesh.is_edge_manifold());

    EXPECT_TRUE(rxmesh.validate());

    constexpr uint32_t      blockThreads = 256;
    LaunchBox<blockThreads> launch_box;
    rxmesh.prepare_launch_box(
        {}, {DynOp::EdgeSplit}, launch_box, (void*)edge_split<blockThreads>);

    edge_split<blockThreads>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rxmesh.get_context());

    CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_TRUE(rxmesh.validate());
}