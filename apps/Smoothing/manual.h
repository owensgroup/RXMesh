
#pragma once

#include "rxmesh/rxmesh_static.h"

template <typename T>
void manual(RXMeshStatic& rx)
{
    constexpr int blockThreads = 256;

    auto input_pos = *rx.get_input_vertex_coordinates();

    auto pos = *rx.add_vertex_attribute_like("pos", input_pos);

    int cols = pos.get_num_attributes();

    DenseMatrix<T> grad(rx, rx.get_num_vertices(), cols);

    pos.copy_from(input_pos, DEVICE, DEVICE);

    double lr = Arg.learning_rate;

    GPUTimer timer;
    timer.start();

    for (int iter = 0; iter < Arg.num_iter; ++iter) {

        // calc gradients
        grad.reset(0, DEVICE);

        rx.run_query_kernel<Op::VV, blockThreads>(
            [=] __device__(const VertexHandle&   vh,
                           const VertexIterator& iter) mutable {
                for (int v = 0; v < iter.size(); ++v) {
                    const VertexHandle uh = iter[v];

                    for (int i = 0; i < cols; ++i) {
                        grad(vh, i) += 2 * (pos(vh, i) - pos(uh, i));
                    }
                }
            });

        // take step
        rx.for_each_vertex(
            DEVICE, [grad, pos, lr, cols] __device__(const VertexHandle& vh) {
                for (int i = 0; i < cols; ++i) {
                    pos(vh, i) -= lr * grad(vh, i);
                }
            });
    }
    timer.stop();

    CUDA_ERROR(cudaDeviceSynchronize());

    RXMESH_INFO("Manual Smoothing GD took {} (ms), ms/iter = {} ",
                timer.elapsed_millis(),
                timer.elapsed_millis() / float(Arg.num_iter));

#if USE_POLYSCOPE
    pos.move(DEVICE, HOST);
    polyscope::registerSurfaceMesh(
        "Manual", pos, rx.get_polyscope_mesh()->faces);
#endif
}