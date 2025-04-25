#pragma once
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"

#include "rxmesh/matrix/cg_solver.h"
#include "rxmesh/matrix/pcg_solver.h"

#include "mcf_kernels.cuh"


template <typename T, typename SolverT>
void run_cg(rxmesh::RXMeshStatic& rx)
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

    Report report("MCF_CG");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rx);
    report.add_member("method", std::string("RXMesh"));
    report.add_member("blockThreads", blockThreads);


    float total_time = 0;

    SolverT solver(A_mat,
                   X_mat.cols(),
                   Arg.max_num_iter,
                   0.0,
                   Arg.cg_tolerance * Arg.cg_tolerance);


    solver.pre_solve(B_mat, X_mat);

    GPUTimer timer;
    timer.start();


    solver.solve(B_mat, X_mat);

    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO("start_residual {}", solver.start_residual());

    RXMESH_INFO("solver {} took {} (ms) and {} iterations (i.e., {} ms/iter)",
                solver.name(),
                timer.elapsed_millis(),
                solver.iter_taken(),
                timer.elapsed_millis() / float(solver.iter_taken()));


    X_mat.move(rxmesh::DEVICE, rxmesh::HOST);


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

template <typename T>
void mcf_cg(rxmesh::RXMeshStatic& rx)
{
    using namespace rxmesh;

    run_cg<T, CGSolver<T>>(rx);
}

template <typename T>
void mcf_pcg(rxmesh::RXMeshStatic& rx)
{
    using namespace rxmesh;

    run_cg<T, PCGSolver<T>>(rx);
}