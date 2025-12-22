
#pragma once

#include "rxmesh/rxmesh_static.h"

template <typename T>
void manual(RXMeshStatic& rx)
{
    Report report("SmoothingManual");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rx, "Input");
    report.add_member("num_faces", rx.get_num_faces());


    constexpr int blockThreads = 256;

    auto input_pos = *rx.get_input_vertex_coordinates();

    auto pos = *rx.add_vertex_attribute_like("pos", input_pos);

    int cols = pos.get_num_attributes();

    DenseMatrix<T> grad(rx, rx.get_num_vertices(), cols, LOCATION_ALL);

    pos.copy_from(input_pos, DEVICE, DEVICE);

    double lr = Arg.learning_rate;

    GPUTimer timer;
    timer.start();

    for (int iter = 0; iter < Arg.num_iter; ++iter) {

        // calc gradients
        grad.reset(0, DEVICE);

        if (Arg.area) {
            rx.run_query_kernel<Op::VV, blockThreads>(
                [=] __device__(const VertexHandle&   vh,
                               const VertexIterator& iter) mutable {
                    const int k = iter.size();
                    if (k < 2) {
                        return;
                    }

                    Eigen::Vector3<T> xv(pos(vh, 0), pos(vh, 1), pos(vh, 2));

                    Eigen::Vector3<T> gv(T(0), T(0), T(0));

                    const T eps = T(1e-20);

                    for (int t = 0; t < k; ++t) {
                        const VertexHandle uh = iter[t];
                        const VertexHandle wh = iter[(t + 1) % k];

                        Eigen::Vector3<T> xu(
                            pos(uh, 0), pos(uh, 1), pos(uh, 2));
                        Eigen::Vector3<T> xw(
                            pos(wh, 0), pos(wh, 1), pos(wh, 2));
                        
                        const Eigen::Vector3<T> e1 = xu - xv;
                        const Eigen::Vector3<T> e2 = xw - xv;

                        const Eigen::Vector3<T> n  = e1.cross(e2);

                        const T                 nn = n.norm();

                        if (nn <= eps) {
                            continue;
                        }

                        const Eigen::Vector3<T> nhat = n / nn;

                        
                        gv += T(0.5) * (xu - xw).cross(nhat);
                    }

                    grad(vh, 0) += gv[0];
                    grad(vh, 1) += gv[1];
                    grad(vh, 2) += gv[2];
                },
                true);
        } else {
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
        }
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

    report.add_member("method", std::string("manual"));
    report.add_member("total_time_ms", timer.elapsed_millis());
    report.add_member("num_iter", Arg.num_iter);

    report.write(Arg.output_folder + "/manual_smoothing",
                 "Manual_RXMesh_" + extract_file_name(Arg.obj_file_name));

#if USE_POLYSCOPE
    pos.move(DEVICE, HOST);
    polyscope::registerSurfaceMesh(
        "Manual", pos, rx.get_polyscope_mesh()->faces);
#endif
}