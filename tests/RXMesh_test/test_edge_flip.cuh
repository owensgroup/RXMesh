#include "gtest/gtest.h"

#include "rxmesh/context.h"
#include "rxmesh/kernels/update_dispatcher.cuh"
#include "rxmesh/rxmesh_dynamic.h"

template <uint32_t blockThreads>
__global__ static void edge_flip(rxmesh::Context context)
{
    using namespace rxmesh;

    // flip one edge (the edge assigned to thread 0) in each patch
    auto should_flip = [&](const EdgeHandle& edge) -> bool {
        if (threadIdx.x == 1) {
            return true;
        } else {
            return false;
        }
    };

    update_block_dispatcher<DynOp::EdgeFlip, blockThreads>(context,
                                                           should_flip);
}

TEST(RXMeshDynamic, EdgeFlip)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshDynamic rx(STRINGIFY(INPUT_DIR) "diamond.obj", rxmesh_args.quite);

    ASSERT_TRUE(rx.is_edge_manifold());

    EXPECT_TRUE(rx.validate());

    constexpr uint32_t      blockThreads = 256;
    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box(
        {}, {DynOp::EdgeFlip}, launch_box, (void*)edge_flip<blockThreads>);

    edge_flip<blockThreads>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context());

    CUDA_ERROR(cudaDeviceSynchronize());

    rx.update_host();

    //rx.export_obj("flipped.obj", *(rx.get_input_vertex_coordinates()));

    EXPECT_TRUE(rx.validate());
}