#pragma once

#include "gtest/gtest.h"

#include "rxmesh/diff/scalar.h"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"

template <typename T, int blockThreads>
__global__ static void smoothing(const rxmesh::Context      context,
                                 rxmesh::VertexAttribute<T> v_opt_pos,
                                 rxmesh::VertexAttribute<T> v_pos_grad,
                                 rxmesh::EdgeAttribute<T>   e_obj_func)
{
    using namespace rxmesh;

    constexpr int VariableDim    = 3;
    constexpr int ElementValence = 2;
    constexpr int NElements      = VariableDim * ElementValence;
    using ScalarT                = Scalar<T, NElements, false>;

    auto func = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        const VertexHandle v0 = iter[0];
        const VertexHandle v1 = iter[1];

        Eigen::Vector3<ScalarT> d0;
        d0[0].val     = v_opt_pos(v0, 0);
        d0[0].grad[0] = 1;

        d0[1].val     = v_opt_pos(v0, 1);
        d0[1].grad[1] = 1;

        d0[2].val     = v_opt_pos(v0, 2);
        d0[2].grad[2] = 1;

        Eigen::Vector3<ScalarT> d1;
        d1[0].val     = v_opt_pos(v1, 0);
        d1[0].grad[3] = 1;

        d1[1].val     = v_opt_pos(v1, 1);
        d1[1].grad[4] = 1;

        d1[2].val     = v_opt_pos(v1, 2);
        d1[2].grad[5] = 1;

        Eigen::Vector3<ScalarT> dist = (d0 - d1);

        ScalarT dist_sq = dist.squaredNorm();

        // add the edge contribution to the objective function
        //::atomicAdd(d_obj_func, dist_sq.val);
        e_obj_func(eh) = dist_sq.val;

        // add the edge contribution to its two end vertices' grad
        for (uint16_t vertex = 0; vertex < iter.size(); ++vertex) {
            for (int local = 0; local < VariableDim; ++local) {
                ::atomicAdd(
                    &v_pos_grad(iter[vertex], local),
                    dist_sq.grad[index_mapping<VariableDim>(vertex, local)]);
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);

    ShmemAllocator shrd_alloc;

    query.dispatch<Op::EV>(block, shrd_alloc, func);
}

template <typename T>
void take_step(const rxmesh::RXMeshStatic&       rx,
               rxmesh::VertexAttribute<T>&       v_opt_pos,
               const rxmesh::VertexAttribute<T>& v_pos_grad,
               const float                       learning_rate)
{
    using namespace rxmesh;
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
        for (int i = 0; i < 3; ++i) {
            v_opt_pos(vh, i) -= learning_rate * v_pos_grad(vh, i);
        }
    });
}

TEST(DiffAttribute, SmoothingGradDescent)
{
    using namespace rxmesh;

    // RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bunnyhead.obj");
    RXMeshStatic rx(rxmesh_args.obj_file_name);

    auto v_opt_pos = *rx.add_vertex_attribute<float>("vPos", 3);

    auto v_pos_grad = *rx.add_vertex_attribute<float>("vGradPos", 3);

    auto v_input_pos = *rx.get_input_vertex_coordinates();

    auto e_obj_func = *rx.add_edge_attribute<float>("eObjFunc", 1);

    v_opt_pos.copy_from(v_input_pos, DEVICE, DEVICE);

    constexpr uint32_t blockThreads = 256;

    float learning_rate = 0.01;

    int num_iterations = 100;

    LaunchBox<blockThreads> lb;
    rx.prepare_launch_box({Op::EV}, lb, (void*)smoothing<float, blockThreads>);

    EdgeReduceHandle<float> obj_func(e_obj_func);

    GPUTimer timer;
    timer.start();

    for (int iter = 0; iter < num_iterations; ++iter) {

        smoothing<float, blockThreads>
            <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                rx.get_context(), v_opt_pos, v_pos_grad, e_obj_func);

        float energy = obj_func.reduce(e_obj_func, cub::Sum(), 0);

        if (iter % 10 == 0) {
            RXMESH_INFO("Iteration = {}: Energy = {}", iter, energy);
        }


        take_step(rx, v_opt_pos, v_pos_grad, learning_rate);

        v_pos_grad.reset(0, DEVICE);
    }
    timer.stop();

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::cout << "\nSmoothing RXMesh: " << timer.elapsed_millis() << " (ms),"
              << timer.elapsed_millis() / float(num_iterations)
              << " ms per iteration\n";

#if USE_POLYSCOPE
    v_opt_pos.move(DEVICE, HOST);
    rx.get_polyscope_mesh()->updateVertexPositions(v_opt_pos);
    polyscope::show();
#endif
}
