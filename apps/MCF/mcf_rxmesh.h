#pragma once

#include <cuda_profiler_api.h>
#include "mcf_rxmesh_kernel.cuh"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"
#include "rxmesh/util/vector.h"

template <typename T>
void axpy(rxmesh::RXMeshStatic&                   rxmesh,
          rxmesh::RXMeshVertexAttribute<T>&       y,
          const rxmesh::RXMeshVertexAttribute<T>& x,
          const rxmesh::Vector<3, T>              alpha,
          const rxmesh::Vector<3, T>              beta,
          cudaStream_t                            stream = NULL)
{
    // Y = alpha*X + beta*Y
    rxmesh.for_each_vertex(
        rxmesh::DEVICE,
        [y, x, alpha, beta] __device__(const rxmesh::VertexHandle vh) {
            for (uint32_t i = 0; i < 3; ++i) {
                y(vh, i) = alpha[i] * x(vh, i) + beta[i] * y(vh, i);
            }
        });
}

template <typename T>
void init_PR(rxmesh::RXMeshStatic&                   rxmesh,
             const rxmesh::RXMeshVertexAttribute<T>& B,
             const rxmesh::RXMeshVertexAttribute<T>& S,
             rxmesh::RXMeshVertexAttribute<T>&       R,
             rxmesh::RXMeshVertexAttribute<T>&       P)
{
    rxmesh.for_each_vertex(
        rxmesh::DEVICE, [B, S, R, P] __device__(const rxmesh::VertexHandle vh) {
            R(vh, 0) = B(vh, 0) - S(vh, 0);
            R(vh, 1) = B(vh, 1) - S(vh, 1);
            R(vh, 2) = B(vh, 2) - S(vh, 2);

            P(vh, 0) = R(vh, 0);
            P(vh, 1) = R(vh, 1);
            P(vh, 2) = R(vh, 2);
        });
}

template <typename T>
void mcf_rxmesh(rxmesh::RXMeshStatic&              rxmesh,
                const std::vector<std::vector<T>>& Verts,
                const std::vector<std::vector<T>>& ground_truth)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    // Report
    Report report("MCF_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rxmesh);
    report.add_member("method", std::string("RXMesh"));
    report.add_member("time_step", Arg.time_step);
    report.add_member("cg_tolerance", Arg.cg_tolerance);
    report.add_member("use_uniform_laplace", Arg.use_uniform_laplace);
    report.add_member("max_num_cg_iter", Arg.max_num_cg_iter);
    report.add_member("blockThreads", blockThreads);

    ASSERT_TRUE(rxmesh.is_closed())
        << "mcf_rxmesh only takes watertight/closed mesh without boundaries";

    // Different attributes used throughout the application
    auto input_coord =
        rxmesh.add_vertex_attribute<T>(Verts, "coord", rxmesh::LOCATION_ALL);

    // S in CG
    auto S =
        rxmesh.add_vertex_attribute<T>("S", 3, rxmesh::DEVICE, rxmesh::SoA);
    S->reset_v1(0.0, rxmesh::DEVICE);

    // P in CG
    auto P =
        rxmesh.add_vertex_attribute<T>("P", 3, rxmesh::DEVICE, rxmesh::SoA);
    P->reset_v1(0.0, rxmesh::DEVICE);

    // R in CG
    auto R =
        rxmesh.add_vertex_attribute<T>("R", 3, rxmesh::DEVICE, rxmesh::SoA);
    R->reset_v1(0.0, rxmesh::DEVICE);

    // B in CG
    auto B =
        rxmesh.add_vertex_attribute<T>("B", 3, rxmesh::DEVICE, rxmesh::SoA);
    B->reset_v1(0.0, rxmesh::DEVICE);

    // X in CG (the output)
    auto X = rxmesh.add_vertex_attribute<T>(
        "X", 3, rxmesh::LOCATION_ALL, rxmesh::SoA);
    // TODO use copy_v1
    X->copy(*input_coord, rxmesh::HOST, rxmesh::DEVICE);


    // RXMesh launch box
    LaunchBox<blockThreads> launch_box;
    rxmesh.prepare_launch_box(rxmesh::Op::VV, launch_box, false, true);


    // init kernel to initialize RHS (B)
    init_B<T, blockThreads>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rxmesh.get_context(), *X, *B, Arg.use_uniform_laplace);

    // CG scalars
    Vector<3, T> alpha(T(0)), beta(T(0)), delta_new(T(0)), delta_old(T(0)),
        ones(T(1));

    GPUTimer timer;
    timer.start();

    // s = Ax
    mcf_matvec<T, blockThreads>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rxmesh.get_context(),
            *input_coord,
            *X,
            *S,
            Arg.use_uniform_laplace,
            Arg.time_step);

    // r = b - s = b - Ax
    // p=rk
    init_PR(rxmesh, *B, *S, *R, *P);


    // delta_new = <r,r>
    //R->reduce(delta_new, rxmesh::NORM2);

    const Vector<3, T> delta_0(delta_new);

    uint32_t num_cg_iter_taken = 0;

    while (num_cg_iter_taken < Arg.max_num_cg_iter) {
        // s = Ap

        mcf_matvec<T, blockThreads>
            <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
                rxmesh.get_context(),
                *input_coord,
                *P,
                *S,
                Arg.use_uniform_laplace,
                Arg.time_step);

        // alpha = delta_new / <s,p>
        //S->reduce(alpha, rxmesh::DOT, P.get());

        alpha = delta_new / alpha;

        // x =  alpha*p + x
        axpy(rxmesh, *X, *P, alpha, ones);

        // r = - alpha*s + r
        axpy(rxmesh, *R, *S, -alpha, ones);


        // delta_old = delta_new
        CUDA_ERROR(cudaStreamSynchronize(0));
        delta_old = delta_new;


        // delta_new = <r,r>
        //R->reduce(delta_new, rxmesh::NORM2);

        CUDA_ERROR(cudaStreamSynchronize(0));


        // exit if error is getting too low across three coordinates
        if (delta_new[0] < Arg.cg_tolerance * Arg.cg_tolerance * delta_0[0] &&
            delta_new[1] < Arg.cg_tolerance * Arg.cg_tolerance * delta_0[1] &&
            delta_new[2] < Arg.cg_tolerance * Arg.cg_tolerance * delta_0[2]) {
            break;
        }

        // beta = delta_new/delta_old
        beta = delta_new / delta_old;

        // p = beta*p + r
        axpy(rxmesh, *P, *R, ones, beta);

        ++num_cg_iter_taken;

        CUDA_ERROR(cudaStreamSynchronize(0));
    }

    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaProfilerStop());


    RXMESH_TRACE(
        "mcf_rxmesh() took {} (ms) and {} iterations (i.e., {} ms/iter) ",
        timer.elapsed_millis(),
        num_cg_iter_taken,
        timer.elapsed_millis() / float(num_cg_iter_taken));

    // move output to host
    X->move(rxmesh::DEVICE, rxmesh::HOST);

    // output to obj
    //rxmesh.export_obj("mcf_rxmesh.obj", *X);

    // Verify
    const T tol = 0.001;
    rxmesh.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = rxmesh.map_to_global(vh);

        for (uint32_t i = 0; i < 3; ++i) {
            EXPECT_LT(std::abs(((*X)(vh, i) - ground_truth[v_id][i]) /
                               ground_truth[v_id][i]),
                      tol);
        }
    });


    // Finalize report
    report.add_member("start_residual", to_string(delta_0));
    report.add_member("end_residual", to_string(delta_new));
    report.add_member("num_cg_iter_taken", num_cg_iter_taken);
    report.add_member("total_time (ms)", timer.elapsed_millis());
    TestData td;
    td.test_name = "MCF";
    td.time_ms.push_back(timer.elapsed_millis() / float(num_cg_iter_taken));
    report.add_test(td);
    report.write(Arg.output_folder + "/rxmesh",
                 "MCF_RXMesh_" + extract_file_name(Arg.obj_file_name));
}