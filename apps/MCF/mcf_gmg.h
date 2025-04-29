#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/gmg_solver.h"

#include "mcf_kernels.cuh"

template <typename T>
void mcf_gmg(rxmesh::RXMeshStatic& rx)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    uint32_t num_vertices = rx.get_num_vertices();

    auto coords = rx.get_input_vertex_coordinates();

    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  B_mat(rx, num_vertices, 3);

    DenseMatrix<float> X_mat = *(coords->to_matrix());

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

    Report report("MCF_GMG");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rx);
    report.add_member("method", std::string("RXMesh"));
    report.add_member("blockThreads", blockThreads);

    GMGSolver solver(
        rx, A_mat, Arg.max_num_iter, Arg.levels, 2, 2, CoarseSolver::Jacobi, Arg.gmg_tolerance_abs, Arg.gmg_tolerance_rel);



    float    total_time = 0;
    CPUTimer timer;
    GPUTimer gtimer;


    timer.start();
    gtimer.start();
    solver.pre_solve(B_mat, X_mat);
    timer.stop();
    gtimer.stop();
    RXMESH_INFO("GMG pre-solve took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member(
        "pre-solve", std::max(timer.elapsed_millis(), gtimer.elapsed_millis()));
    total_time += std::max(timer.elapsed_millis(), gtimer.elapsed_millis());


    timer.start();
    gtimer.start();
    solver.solve(B_mat, X_mat);
    timer.stop();
    gtimer.stop();

    RXMESH_INFO("start_residual {}", solver.start_residual());

    RXMESH_INFO("solver {} took {} (ms), {} (ms)",
                solver.name(),
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
    report.add_member(
        "solve", std::max(timer.elapsed_millis(), gtimer.elapsed_millis()));
    total_time += std::max(timer.elapsed_millis(), gtimer.elapsed_millis());

    report.add_member("total_time", total_time);

    RXMESH_INFO("total_time {} (ms)", total_time);


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
