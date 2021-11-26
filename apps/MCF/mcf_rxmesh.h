#pragma once

#include <cuda_profiler_api.h>
#include "mcf_rxmesh_kernel.cuh"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"
#include "rxmesh/util/vector.h"


template <typename T>
void mcf_rxmesh(rxmesh::RXMeshStatic&              rxmesh_static,
                const std::vector<std::vector<T>>& Verts,
                const rxmesh::RXMeshAttribute<T>&  ground_truth)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    // Report
    Report report("MCF_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rxmesh_static);
    report.add_member("method", std::string("RXMesh"));
    report.add_member("time_step", Arg.time_step);
    report.add_member("cg_tolerance", Arg.cg_tolerance);
    report.add_member("use_uniform_laplace", Arg.use_uniform_laplace);
    report.add_member("max_num_cg_iter", Arg.max_num_cg_iter);
    report.add_member("blockThreads", blockThreads);

    ASSERT_TRUE(rxmesh_static.is_closed())
        << "mcf_rxmesh only takes watertight/closed mesh without boundaries";

    // Different attributes used throughout the application
    auto input_coord =
        rxmesh_static.add_vertex_attribute<T>("coord", 3, rxmesh::LOCATION_ALL);
    for (uint32_t i = 0; i < Verts.size(); ++i) {
        for (uint32_t j = 0; j < Verts[i].size(); ++j) {
            (*input_coord)(i, j) = Verts[i][j];
        }
    }
    input_coord->change_layout(rxmesh::HOST);
    input_coord->move(rxmesh::HOST, rxmesh::DEVICE);

    // S in CG
    auto S = rxmesh_static.add_vertex_attribute<T>(
        "S", 3, rxmesh::DEVICE, rxmesh::SoA);
    S->reset(0.0, rxmesh::DEVICE);

    // P in CG
    auto P = rxmesh_static.add_vertex_attribute<T>(
        "P", 3, rxmesh::DEVICE, rxmesh::SoA);
    P->reset(0.0, rxmesh::DEVICE);

    // R in CG
    auto R = rxmesh_static.add_vertex_attribute<T>(
        "R", 3, rxmesh::DEVICE, rxmesh::SoA);
    R->reset(0.0, rxmesh::DEVICE);

    // B in CG
    auto B = rxmesh_static.add_vertex_attribute<T>(
        "B", 3, rxmesh::DEVICE, rxmesh::SoA);
    B->reset(0.0, rxmesh::DEVICE);

    // X in CG
    auto X = rxmesh_static.add_vertex_attribute<T>(
        "X", 3, rxmesh::LOCATION_ALL, rxmesh::SoA);
    X->copy(*input_coord, rxmesh::HOST, rxmesh::DEVICE);


    // RXMesh launch box
    LaunchBox<blockThreads> launch_box;
    rxmesh_static.prepare_launch_box(rxmesh::Op::VV, launch_box, false, true);


    // init kernel to initialize RHS (B)
    init_B<T, blockThreads>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rxmesh_static.get_context(), *X, *B, Arg.use_uniform_laplace);

    // CG scalars
    Vector<3, T> alpha(T(0)), beta(T(0)), delta_new(T(0)), delta_old(T(0)),
        ones(T(1));

    GPUTimer timer;
    timer.start();

    // s = Ax
    mcf_matvec<T, blockThreads>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rxmesh_static.get_context(),
            *input_coord,
            *X,
            *S,
            Arg.use_uniform_laplace,
            Arg.time_step);

    // r = b - s = b - Ax
    // p=r
    const uint32_t num_blocks =
        DIVIDE_UP(rxmesh_static.get_num_vertices(), blockThreads);
    init_PR<T><<<num_blocks, blockThreads>>>(
        rxmesh_static.get_num_vertices(), *B, *S, *R, *P);

    // delta_new = <r,r>
    R->reduce(delta_new, rxmesh::NORM2);

    const Vector<3, T> delta_0(delta_new);

    uint32_t num_cg_iter_taken = 0;

    while (num_cg_iter_taken < Arg.max_num_cg_iter) {
        // s = Ap

        mcf_matvec<T, blockThreads>
            <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
                rxmesh_static.get_context(),
                *input_coord,
                *P,
                *S,
                Arg.use_uniform_laplace,
                Arg.time_step);

        // alpha = delta_new / <s,p>
        S->reduce(alpha, rxmesh::DOT, P.get());

        alpha = delta_new / alpha;

        // x =  x + alpha*p
        X->axpy(*P, alpha, ones);

        // r = r - alpha*s
        R->axpy(*S, -alpha, ones);


        // delta_old = delta_new
        CUDA_ERROR(cudaStreamSynchronize(0));
        delta_old = delta_new;


        // delta_new = <r,r>
        R->reduce(delta_new, rxmesh::NORM2);

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
        P->axpy(*R, ones, beta);

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
    //rxmesh_static.export_obj(
    //    "mcf_rxmesh.obj", [&X](uint32_t i, uint32_t j) { return (*X)(i, j); });

    // Verify
    bool    passed = true;
    const T tol    = 0.001;
    for (uint32_t v = 0; v < rxmesh_static.get_num_vertices(); ++v) {
        if (std::fabs((*X)(v, 0) - ground_truth(v, 0)) >
                tol * std::fabs(ground_truth(v, 0)) ||
            std::fabs((*X)(v, 1) - ground_truth(v, 1)) >
                tol * std::fabs(ground_truth(v, 1)) ||
            std::fabs((*X)(v, 2) - ground_truth(v, 2)) >
                tol * std::fabs(ground_truth(v, 2))) {
            passed = false;
            break;
        }
    }

    EXPECT_TRUE(passed);


    // Finalize report
    report.add_member("start_residual", to_string(delta_0));
    report.add_member("end_residual", to_string(delta_new));
    report.add_member("num_cg_iter_taken", num_cg_iter_taken);
    report.add_member("total_time (ms)", timer.elapsed_millis());
    TestData td;
    td.test_name = "MCF";
    td.time_ms.push_back(timer.elapsed_millis() / float(num_cg_iter_taken));
    td.passed.push_back(passed);
    report.add_test(td);
    report.write(Arg.output_folder + "/rxmesh",
                 "MCF_RXMesh_" + extract_file_name(Arg.obj_file_name));
}