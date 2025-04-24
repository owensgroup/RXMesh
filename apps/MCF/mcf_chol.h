#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/cholesky_solver.h"
#include "rxmesh/matrix/lu_solver.h"

#include "mcf_kernels.cuh"


template <typename T>
void mcf_cusolver_chol(rxmesh::RXMeshStatic& rx,
                       rxmesh::PermuteMethod permute_method)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    uint32_t num_vertices = rx.get_num_vertices();

    auto coords = rx.get_input_vertex_coordinates();

    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  B_mat(rx, num_vertices, 3);

    DenseMatrix<float> X_mat = *coords->to_matrix();

    RXMESH_INFO("use_uniform_laplace: {}, time_step: {}",
                Arg.use_uniform_laplace,
                Arg.time_step);

    // B set up
    rx.run_kernel<blockThreads>({Op::VV},
                                mcf_B_setup<float, blockThreads>,
                                *coords,
                                B_mat,
                                Arg.use_uniform_laplace);

    // A set up
    rx.run_kernel<blockThreads>({Op::VV},
                                mcf_A_setup<float, blockThreads>,
                                *coords,
                                A_mat,
                                Arg.use_uniform_laplace,
                                Arg.time_step);

    Report report("MCF_Chol");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rx);
    report.add_member("method", std::string("RXMesh"));
    report.add_member("blockThreads", blockThreads);
    report.add_member("PermuteMethod",
                      permute_method_to_string(permute_method));

    RXMESH_INFO("permute_method took {}",
                permute_method_to_string(permute_method));

    float total_time = 0;

    CholeskySolver solver(&A_mat, permute_method);

    // To Use LU, we have to move the data to the host
    // LUSolver solver(&A_mat, permute_method);
    // A_mat.move(DEVICE, HOST);
    // B_mat.move(DEVICE, HOST);
    // X_mat.move(DEVICE, HOST);

    // pre-solve
    // solver.pre_solve(rx);
    // Solve
    // solver.solve(B_mat, X_mat);

    CPUTimer timer;
    GPUTimer gtimer;

    timer.start();
    gtimer.start();
    solver.permute_alloc();
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("permute_alloc took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member(
        "permute_alloc",
        std::max(timer.elapsed_millis(), gtimer.elapsed_millis()));
    total_time += std::max(timer.elapsed_millis(), gtimer.elapsed_millis());

    timer.start();
    gtimer.start();
    solver.permute(rx);
    solver.premute_value_ptr();
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("permute took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member(
        "permute", std::max(timer.elapsed_millis(), gtimer.elapsed_millis()));
    total_time += std::max(timer.elapsed_millis(), gtimer.elapsed_millis());


    timer.start();
    gtimer.start();
    solver.analyze_pattern();
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("analyze_pattern took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member(
        "analyze_pattern",
        std::max(timer.elapsed_millis(), gtimer.elapsed_millis()));
    total_time += std::max(timer.elapsed_millis(), gtimer.elapsed_millis());


    timer.start();
    gtimer.start();
    solver.post_analyze_alloc();
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("post_analyze_alloc took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member(
        "post_analyze_alloc",
        std::max(timer.elapsed_millis(), gtimer.elapsed_millis()));
    total_time += std::max(timer.elapsed_millis(), gtimer.elapsed_millis());

    timer.start();
    gtimer.start();
    solver.factorize();
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("factorize took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member(
        "factorize", std::max(timer.elapsed_millis(), gtimer.elapsed_millis()));
    total_time += std::max(timer.elapsed_millis(), gtimer.elapsed_millis());


    timer.start();
    gtimer.start();
    solver.solve(B_mat, X_mat);
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("solve took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member(
        "solve", std::max(timer.elapsed_millis(), gtimer.elapsed_millis()));
    total_time += std::max(timer.elapsed_millis(), gtimer.elapsed_millis());

    report.add_member("total_time", total_time);

    RXMESH_INFO("total_time {} (ms)", total_time);


    // move the results to the host
    // if we use LU, the data will be on the host and we should not move the
    // device to the host
    X_mat.move(rxmesh::DEVICE, rxmesh::HOST);

    // copy the results to attributes
    coords->from_matrix(&X_mat);

#if USE_POLYSCOPE
    polyscope::registerSurfaceMesh("old mesh",
                                   rx.get_polyscope_mesh()->vertices,
                                   rx.get_polyscope_mesh()->faces);
    rx.get_polyscope_mesh()->updateVertexPositions(*coords);
    polyscope::show();
#endif

    B_mat.release();
    X_mat.release();
    A_mat.release();

    report.write(Arg.output_folder + "/rxmesh",
                 "MCF_" + solver.name() + extract_file_name(Arg.obj_file_name));
}