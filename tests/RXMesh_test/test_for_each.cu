#include "gtest/gtest.h"

#include <atomic>

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/import_obj.h"

#include "rxmesh/kernels/for_each.cuh"

TEST(RXMeshStatic, ForEachTriangle)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "cube.obj");

    std::atomic_uint32_t num_v = 0;
    std::atomic_uint32_t num_e = 0;
    std::atomic_uint32_t num_f = 0;

    rx.for_each_vertex(HOST, [&](const VertexHandle) { num_v++; });

    rx.for_each_edge(HOST, [&](const EdgeHandle) { num_e++; });

    rx.for_each_face(HOST, [&](const FaceHandle) { num_f++; });

    EXPECT_EQ(num_v, rx.get_num_vertices());

    EXPECT_EQ(num_e, rx.get_num_edges());

    EXPECT_EQ(num_f, rx.get_num_faces());
}


TEST(RXMeshStatic, ForEachTet)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "car.msh");

    std::atomic_uint32_t num_t = 0;
    std::atomic_uint32_t num_v = 0;
    std::atomic_uint32_t num_e = 0;
    std::atomic_uint32_t num_f = 0;

    rx.for_each_vertex(HOST, [&](const VertexHandle) { num_v++; });
    rx.for_each_edge(HOST, [&](const EdgeHandle) { num_e++; });
    rx.for_each_face(HOST, [&](const FaceHandle) { num_f++; });
    rx.for_each_tet(HOST, [&](const TetHandle) { num_t++; });

    EXPECT_EQ(num_v, rx.get_num_vertices());
    EXPECT_EQ(num_e, rx.get_num_edges());
    EXPECT_EQ(num_f, rx.get_num_faces());
    EXPECT_EQ(num_t, rx.get_num_tets());
}


template <uint32_t blockThreads, rxmesh::Op op, typename HandleT>
__global__ static void for_each_kernel(
    const __grid_constant__ rxmesh::Context context,
    uint32_t*                               count)
{
    using namespace rxmesh;

    auto for_each_lambda = [&](HandleT&) { atomicAdd(count, 1); };

    for_each<op, blockThreads>(context, for_each_lambda);
}

TEST(RXMeshStatic, ForEachOnDeviceTriangle)
{
    using namespace rxmesh;

    constexpr uint32_t      blockThreads = 256;
    LaunchBox<blockThreads> launch_box;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    uint32_t* d_count = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_count, sizeof(uint32_t)));

    const auto reset_count = [&]() {
        CUDA_ERROR(cudaMemset(d_count, 0, sizeof(uint32_t)));
    };

    const auto get_count = [&]() {
        uint32_t count = 0;
        CUDA_ERROR(cudaMemcpy(
            &count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return count;
    };

    reset_count();
    rx.prepare_launch_box(
        {Op::V},
        launch_box,
        (void*)for_each_kernel<blockThreads, Op::V, VertexHandle>);
    for_each_kernel<blockThreads, Op::V, VertexHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context(), d_count);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(get_count(), rx.get_num_vertices());

    reset_count();
    rx.prepare_launch_box(
        {Op::E},
        launch_box,
        (void*)for_each_kernel<blockThreads, Op::E, EdgeHandle>);
    for_each_kernel<blockThreads, Op::E, EdgeHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context(), d_count);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(get_count(), rx.get_num_edges());

    reset_count();
    rx.prepare_launch_box(
        {Op::F},
        launch_box,
        (void*)for_each_kernel<blockThreads, Op::F, FaceHandle>);
    for_each_kernel<blockThreads, Op::F, FaceHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context(), d_count);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(get_count(), rx.get_num_faces());

    reset_count();
    rx.prepare_launch_box(
        {Op::T},
        launch_box,
        (void*)for_each_kernel<blockThreads, Op::T, TetHandle>);
    for_each_kernel<blockThreads, Op::T, TetHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context(), d_count);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(get_count(), 0);

    CUDA_ERROR(cudaFree(d_count));
}

TEST(RXMeshStatic, ForEachOnDeviceTet)
{
    using namespace rxmesh;

    constexpr uint32_t      blockThreads = 256;
    LaunchBox<blockThreads> launch_box;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "car.msh");

    uint32_t* d_count = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_count, sizeof(uint32_t)));

    const auto reset_count = [&]() {
        CUDA_ERROR(cudaMemset(d_count, 0, sizeof(uint32_t)));
    };

    const auto get_count = [&]() {
        uint32_t count = 0;
        CUDA_ERROR(cudaMemcpy(
            &count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return count;
    };

    reset_count();
    rx.prepare_launch_box(
        {Op::V},
        launch_box,
        (void*)for_each_kernel<blockThreads, Op::V, VertexHandle>);
    for_each_kernel<blockThreads, Op::V, VertexHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context(), d_count);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(get_count(), rx.get_num_vertices());

    reset_count();
    rx.prepare_launch_box(
        {Op::E},
        launch_box,
        (void*)for_each_kernel<blockThreads, Op::E, EdgeHandle>);
    for_each_kernel<blockThreads, Op::E, EdgeHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context(), d_count);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(get_count(), rx.get_num_edges());

    reset_count();
    rx.prepare_launch_box(
        {Op::F},
        launch_box,
        (void*)for_each_kernel<blockThreads, Op::F, FaceHandle>);
    for_each_kernel<blockThreads, Op::F, FaceHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context(), d_count);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(get_count(), rx.get_num_faces());

    reset_count();
    rx.prepare_launch_box(
        {Op::T},
        launch_box,
        (void*)for_each_kernel<blockThreads, Op::T, TetHandle>);
    for_each_kernel<blockThreads, Op::T, TetHandle>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context(), d_count);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(get_count(), rx.get_num_tets());

    CUDA_ERROR(cudaFree(d_count));
}
