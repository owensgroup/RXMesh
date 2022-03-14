#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/import_obj.h"

#include "rxmesh/kernels/for_each_dispatcher.cuh"

TEST(RXMeshStatic, ForEach)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshStatic rxmesh_static(STRINGIFY(INPUT_DIR) "cube.obj",
                               rxmesh_args.quite);

    std::atomic_uint32_t num_v = 0;
    std::atomic_uint32_t num_e = 0;
    std::atomic_uint32_t num_f = 0;

    rxmesh_static.for_each_vertex(HOST,
                                  [&](const VertexHandle vh) { num_v++; });

    rxmesh_static.for_each_edge(HOST, [&](const EdgeHandle eh) { num_e++; });

    rxmesh_static.for_each_face(HOST, [&](const FaceHandle fh) { num_f++; });

    EXPECT_EQ(num_v, rxmesh_static.get_num_vertices());

    EXPECT_EQ(num_e, rxmesh_static.get_num_edges());

    EXPECT_EQ(num_f, rxmesh_static.get_num_faces());
}


template <uint32_t blockThreads, rxmesh::Op op, typename HandleT>
__global__ static void for_each_kernel(const rxmesh::Context context)
{
    using namespace rxmesh;

    auto for_each_lambda = [&](HandleT& id) {

    };

    for_each_dispatcher<op, blockThreads>(context, for_each_lambda);
}

TEST(RXMeshStatic, ForEachOnDevice)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    constexpr uint32_t      blockThreads = 256;
    LaunchBox<blockThreads> launch_box;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj", rxmesh_args.quite);

    rx.prepare_launch_box(
        {Op::V},
        launch_box,
        (void*)for_each_kernel<blockThreads, Op::V, VertexHandle>);

    for_each_kernel<blockThreads, Op::V, VertexHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context());
    EXPECT_EQ(cudaDeviceSynchronize() == cudaSuccess);

    rx.prepare_launch_box(
        {Op::E},
        launch_box,
        (void*)for_each_kernel<blockThreads, Op::E, EdgeHandle>);
    for_each_kernel<blockThreads, Op::E, EdgeHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context());
    EXPECT_EQ(cudaDeviceSynchronize() == cudaSuccess);


    rx.prepare_launch_box(
        {Op::F},
        launch_box,
        (void*)for_each_kernel<blockThreads, Op::F, FaceHandle>);
    for_each_kernel<blockThreads, Op::F, FaceHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context());        
    EXPECT_EQ(cudaDeviceSynchronize() == cudaSuccess);
}