#include "gtest/gtest.h"

#include "rxmesh/context.h"
#include "rxmesh/kernels/update_dispatcher.cuh"
#include "rxmesh/rxmesh_dynamic.h"

template <uint32_t blockThreads>
__global__ static void delete_edge(rxmesh::Context context)
{
    using namespace rxmesh;

    auto should_delete = [&](const EdgeHandle& edge) -> bool {
        if (threadIdx.x == 1) {
            return true;
        } else {
            return false;
        }
    };

    update_block_dispatcher<DynOp::DeleteEdge, blockThreads>(context,
                                                             should_delete);
}

TEST(RXMeshDynamic, DeleteEdge)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshDynamic rxmesh(STRINGIFY(INPUT_DIR) "diamond.obj", rxmesh_args.quite);

    ASSERT_TRUE(rxmesh.is_edge_manifold());

    EXPECT_TRUE(rxmesh.validate());

    constexpr uint32_t      blockThreads = 256;
    LaunchBox<blockThreads> launch_box;
    rxmesh.prepare_launch_box(
        {}, {DynOp::DeleteEdge}, launch_box, (void*)delete_edge<blockThreads>);

    delete_edge<blockThreads>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rxmesh.get_context());

    CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_TRUE(rxmesh.validate());
}