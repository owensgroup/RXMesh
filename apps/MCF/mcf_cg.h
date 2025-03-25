#pragma once

#include <cuda_profiler_api.h>
#include "rxmesh/attribute.h"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"

#include "mcf_kernels.cuh"

template <typename T>
void axpy(rxmesh::RXMeshStatic&             rx,
          rxmesh::VertexAttribute<T>&       y,
          const rxmesh::VertexAttribute<T>& x,
          const T                           alpha,
          const T                           beta,
          cudaStream_t                      stream = NULL)
{
    // Y = alpha*X + beta*Y
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [y, x, alpha, beta] __device__(const rxmesh::VertexHandle vh) {
            for (uint32_t i = 0; i < 3; ++i) {
                y(vh, i) = alpha * x(vh, i) + beta * y(vh, i);
            }
        });
}

template <typename T>
void init_PR(rxmesh::RXMeshStatic&             rx,
             const rxmesh::VertexAttribute<T>& B,
             const rxmesh::VertexAttribute<T>& S,
             rxmesh::VertexAttribute<T>&       R,
             rxmesh::VertexAttribute<T>&       P)
{
    rx.for_each_vertex(rxmesh::DEVICE,
                       [B, S, R, P] __device__(const rxmesh::VertexHandle vh) {
                           R(vh, 0) = B(vh, 0) - S(vh, 0);
                           R(vh, 1) = B(vh, 1) - S(vh, 1);
                           R(vh, 2) = B(vh, 2) - S(vh, 2);

                           P(vh, 0) = R(vh, 0);
                           P(vh, 1) = R(vh, 1);
                           P(vh, 2) = R(vh, 2);
                       });
}

template <typename T>
void mcf_cg(rxmesh::RXMeshStatic& rx)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    // Report
    Report report("MCF_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rx);
    report.add_member("method", std::string("RXMesh"));
    report.add_member("time_step", Arg.time_step);
    report.add_member("cg_tolerance", Arg.cg_tolerance);
    report.add_member("use_uniform_laplace", Arg.use_uniform_laplace);
    report.add_member("max_num_cg_iter", Arg.max_num_cg_iter);
    report.add_member("blockThreads", blockThreads);

    ASSERT_TRUE(rx.is_closed())
        << "mcf_rxmesh only takes watertight/closed mesh without boundaries";

    // Different attributes used throughout the application
    auto input_coord = rx.get_input_vertex_coordinates();

    // S in CG
    auto S = rx.add_vertex_attribute<T>("S", 3, rxmesh::DEVICE, rxmesh::SoA);
    S->reset(0.0, rxmesh::DEVICE);

    // P in CG
    auto P = rx.add_vertex_attribute<T>("P", 3, rxmesh::DEVICE, rxmesh::SoA);
    P->reset(0.0, rxmesh::DEVICE);

    // R in CG
    auto R = rx.add_vertex_attribute<T>("R", 3, rxmesh::DEVICE, rxmesh::SoA);
    R->reset(0.0, rxmesh::DEVICE);

    // B in CG
    auto B = rx.add_vertex_attribute<T>("B", 3, rxmesh::DEVICE, rxmesh::SoA);
    B->reset(0.0, rxmesh::DEVICE);

    // X in CG (the output)
    auto X = rx.add_vertex_attribute<T>("X", 3, rxmesh::LOCATION_ALL);
    X->copy_from(*input_coord, rxmesh::DEVICE, rxmesh::DEVICE);

    VertexReduceHandle<T> reduce_handle(*X);

    // RXMesh launch box
    LaunchBox<blockThreads> launch_box_init_B;
    LaunchBox<blockThreads> launch_box_matvec;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_init_B,
                          (void*)init_B<T, blockThreads>,
                          !Arg.use_uniform_laplace);
    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_matvec,
                          (void*)rxmesh_matvec<T, blockThreads>,
                          !Arg.use_uniform_laplace);


    // init kernel to initialize RHS (B)
    init_B<T, blockThreads><<<launch_box_init_B.blocks,
                              launch_box_init_B.num_threads,
                              launch_box_init_B.smem_bytes_dyn>>>(
        rx.get_context(), *X, *B, Arg.use_uniform_laplace);

    // CG scalars
    T alpha(0), beta(0), delta_new(0), delta_old(0);

    GPUTimer timer;
    timer.start();

    // s = Ax
    rxmesh_matvec<T, blockThreads>
        <<<launch_box_matvec.blocks,
           launch_box_matvec.num_threads,
           launch_box_matvec.smem_bytes_dyn>>>(rx.get_context(),
                                               *input_coord,
                                               *X,
                                               *S,
                                               Arg.use_uniform_laplace,
                                               Arg.time_step);

    // r = b - s = b - Ax
    // p=rk
    init_PR(rx, *B, *S, *R, *P);


    // delta_new = <r,r>
    delta_new = reduce_handle.norm2(*R);
    delta_new *= delta_new;

    const T delta_0(delta_new);

    uint32_t num_cg_iter_taken = 0;

    GPUTimer matvec_timer;
    float    matvec_time = 0;


    while (num_cg_iter_taken < Arg.max_num_cg_iter) {
        // s = Ap
        matvec_timer.start();
        rxmesh_matvec<T, blockThreads>
            <<<launch_box_matvec.blocks,
               launch_box_matvec.num_threads,
               launch_box_matvec.smem_bytes_dyn>>>(rx.get_context(),
                                                   *input_coord,
                                                   *P,
                                                   *S,
                                                   Arg.use_uniform_laplace,
                                                   Arg.time_step);
        matvec_timer.stop();
        matvec_time += matvec_timer.elapsed_millis();

        // alpha = delta_new / <s,p>
        alpha = reduce_handle.dot(*S, *P);
        alpha = delta_new / alpha;

        // x =  alpha*p + x
        axpy(rx, *X, *P, alpha, 1.f);

        // r = - alpha*s + r
        axpy(rx, *R, *S, -alpha, 1.f);


        // delta_old = delta_new
        CUDA_ERROR(cudaStreamSynchronize(0));
        delta_old = delta_new;


        // delta_new = <r,r>
        delta_new = reduce_handle.norm2(*R);
        delta_new *= delta_new;

        CUDA_ERROR(cudaStreamSynchronize(0));


        // exit if error is getting too low across three coordinates
        if (delta_new < Arg.cg_tolerance * Arg.cg_tolerance * delta_0) {
            break;
        }

        // beta = delta_new/delta_old
        beta = delta_new / delta_old;

        // p = beta*p + r
        axpy(rx, *P, *R, 1.f, beta);

        ++num_cg_iter_taken;

        CUDA_ERROR(cudaStreamSynchronize(0));
    }

    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaProfilerStop());


    RXMESH_TRACE(
        "mcf_rxmesh() took {} (ms) and {} iterations (i.e., {} ms/iter), "
        "mat_vec time {} (ms) (i.e., {} ms/iter)",
        timer.elapsed_millis(),
        num_cg_iter_taken,
        timer.elapsed_millis() / float(num_cg_iter_taken),
        matvec_time,
        matvec_time / float(num_cg_iter_taken));

    // move output to host
    X->move(rxmesh::DEVICE, rxmesh::HOST);

    // output to obj
    // rxmesh.export_obj("mcf_rxmesh.obj", *X);
#if USE_POLYSCOPE
    polyscope::registerSurfaceMesh("old mesh",
                                   rx.get_polyscope_mesh()->vertices,
                                   rx.get_polyscope_mesh()->faces);
    rx.get_polyscope_mesh()->updateVertexPositions(*X);
    
    polyscope::show();
#endif
    
    // Finalize report
    report.add_member("start_residual", delta_0);
    report.add_member("end_residual", delta_new);
    report.add_member("num_cg_iter_taken", num_cg_iter_taken);
    report.add_member("total_time (ms)", timer.elapsed_millis());
    report.add_member("matvec_time (ms)", matvec_time);
    TestData td;
    td.test_name   = "MCF";
    td.num_threads = launch_box_matvec.num_threads;
    td.num_blocks  = launch_box_matvec.blocks;
    td.dyn_smem    = launch_box_matvec.smem_bytes_dyn;
    td.static_smem = launch_box_matvec.smem_bytes_static;
    td.num_reg     = launch_box_matvec.num_registers_per_thread;
    td.passed.push_back(true);
    td.time_ms.push_back(timer.elapsed_millis() / float(num_cg_iter_taken));
    report.add_test(td);
    report.write(Arg.output_folder + "/rxmesh",
                 "MCF_RXMesh_" + extract_file_name(Arg.obj_file_name));
}