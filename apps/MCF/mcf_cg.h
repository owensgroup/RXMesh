#pragma once

#include <cuda_profiler_api.h>
#include "rxmesh/attribute.h"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"

#include "mcf_kernels.cuh"

#include "rxmesh/matrix/cg_solver.h"

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

    // RHS
    auto B = rx.add_vertex_attribute<T>("B", 3, rxmesh::DEVICE, rxmesh::SoA);
    B->reset(0.0, rxmesh::DEVICE);

    // the unknowns
    auto X = rx.add_vertex_attribute<T>("X", 3, rxmesh::LOCATION_ALL);
    X->copy_from(*input_coord, rxmesh::DEVICE, rxmesh::DEVICE);

    // init kernel to initialize RHS (B)
    LaunchBox<blockThreads> lb;
    rx.prepare_launch_box(
        {Op::VV}, lb, (void*)init_B<T, blockThreads>, !Arg.use_uniform_laplace);
    init_B<T, blockThreads><<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
        rx.get_context(), *X, *B, Arg.use_uniform_laplace);

    // matvec launch box
    rx.prepare_launch_box({rxmesh::Op::VV},
                          lb,
                          (void*)rxmesh_matvec<T, blockThreads>,
                          !Arg.use_uniform_laplace);


    CGSolver<T, VertexHandle> solver(
        rx,
        [&](const VertexAttribute<T>& in,
            VertexAttribute<T>&       out,
            cudaStream_t              stream) {
            rx.run_kernel(lb,
                          rxmesh_matvec<T, blockThreads>,
                          stream,
                          *input_coord,
                          in,
                          out,
                          Arg.use_uniform_laplace,
                          Arg.time_step);
        },
        X->get_num_attributes(),
        Arg.max_num_cg_iter,
        0.0,
        Arg.cg_tolerance * Arg.cg_tolerance);

    solver.pre_solve(X.get(), B.get());

    GPUTimer timer;
    timer.start();

    solver.solve(X.get(), B.get());

    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaProfilerStop());


    RXMESH_TRACE(
        "mcf_rxmesh() took {} (ms) and {} iterations (i.e., {} ms/iter)",
        timer.elapsed_millis(),
        solver.iter_taken(),
        timer.elapsed_millis() / float(solver.iter_taken()));

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
    report.add_member("start_residual", solver.start_residual());
    report.add_member("end_residual", solver.final_residual());
    report.add_member("num_cg_iter_taken", solver.iter_taken());
    report.add_member("total_time (ms)", timer.elapsed_millis());
    TestData td;
    td.test_name   = "MCF";
    td.num_threads = lb.num_threads;
    td.num_blocks  = lb.blocks;
    td.dyn_smem    = lb.smem_bytes_dyn;
    td.static_smem = lb.smem_bytes_static;
    td.num_reg     = lb.num_registers_per_thread;
    td.passed.push_back(true);
    td.time_ms.push_back(timer.elapsed_millis() / float(solver.iter_taken()));
    report.add_test(td);
    report.write(Arg.output_folder + "/rxmesh",
                 "MCF_RXMesh_" + extract_file_name(Arg.obj_file_name));
}