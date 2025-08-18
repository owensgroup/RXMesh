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
    rx.export_obj("mcf_initial.obj", *rx.get_input_vertex_coordinates());

    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  B_mat(rx, num_vertices, 3);

    DenseMatrix<float> X_mat = *(coords->to_matrix());

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
    report.add_member("method", std::string("GPUGMG"));
    report.add_member("application", std::string("MCF"));
    report.add_member("blockThreads", blockThreads);

    GMGSolver solver(rx,
                     A_mat,
                     Arg.max_num_iter,
                     Arg.levels,
                     2,
                     2,
                     CoarseSolver::Cholesky,
                     Arg.tol_abs,
                     Arg.tol_rel,
                     Arg.threshold,
                     Arg.use_new_ptap,
                     Arg.ptap_verify);

    float    solve_time = 0;
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
    RXMESH_INFO(
        "GMG memory allocation took {} (ms)",
        solver.gmg_memory_alloc_time + solver.v_cycle_memory_alloc_time);


    report.add_member(
        "pre-solve",
        std::max(timer.elapsed_millis() - solver.gmg_memory_alloc_time -
                     solver.v_cycle_memory_alloc_time,
                 gtimer.elapsed_millis() - solver.gmg_memory_alloc_time -
                     solver.v_cycle_memory_alloc_time));
    total_time +=
        std::max(timer.elapsed_millis() - solver.gmg_memory_alloc_time -
                     solver.v_cycle_memory_alloc_time,
                 gtimer.elapsed_millis() - solver.gmg_memory_alloc_time -
                     solver.v_cycle_memory_alloc_time);

    timer.start();
    gtimer.start();
    solver.solve(B_mat, X_mat);
    timer.stop();
    gtimer.stop();
    total_time += std::max(timer.elapsed_millis(), gtimer.elapsed_millis());
    solve_time += std::max(timer.elapsed_millis(), gtimer.elapsed_millis());

    RXMESH_INFO("start_residual {}", solver.start_residual());
    RXMESH_INFO("final_residual {}", solver.final_residual());
    RXMESH_INFO("solver {} took {} (ms), {} (ms)",
                solver.name(),
                timer.elapsed_millis(),
                gtimer.elapsed_millis());

    Arg.levels = solver.get_num_levels();
    report.add_member("levels", Arg.levels);
    report.add_member("iterations", solver.iter_taken());
    report.add_member(
        "solve", std::max(timer.elapsed_millis(), gtimer.elapsed_millis()));
    total_time += std::max(timer.elapsed_millis(), gtimer.elapsed_millis());
    report.add_member("solve", solve_time);
    report.add_member("avg iteration time", solve_time / solver.iter_taken());

    report.add_member("total_time", total_time);

    report.add_member("max_iterations", Arg.max_num_iter);
    report.add_member("threshold", Arg.threshold);
    report.add_member("final_residual", solver.get_final_residual());
    report.add_member("using_new_ptap", Arg.use_new_ptap);
    report.add_member("verification_on", Arg.ptap_verify);
    RXMESH_INFO("total_time {} (ms)", total_time);


    X_mat.move(rxmesh::DEVICE, rxmesh::HOST);

    // copy the results to attributes
    coords->from_matrix(&X_mat);

#if USE_POLYSCOPE
    if (Arg.render_hierarchy) {
        solver.render_hierarchy();
        solver.render_laplacian();
    }
    polyscope::registerSurfaceMesh("old mesh",
                                   rx.get_polyscope_mesh()->vertices,
                                   rx.get_polyscope_mesh()->faces);
    rx.get_polyscope_mesh()->updateVertexPositions(*coords);

    polyscope::show();


#endif

    /*rx.export_obj("gmg_mcf_result.obj",
     *coords);*/

    B_mat.release();
    X_mat.release();
    A_mat.release();

    report.write(Arg.output_folder + "/rxmesh",
                 "MCF_" + solver.name() + extract_file_name(Arg.obj_file_name));
}
