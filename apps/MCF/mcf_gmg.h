#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/gmg_solver.h"

#include "mcf_kernels.cuh"

template <typename T>
void mcf_gmg(rxmesh::RXMeshStatic& rx,
             rxmesh::CoarseSolver  csolver,
             rxmesh::Sampling      sampling)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    uint32_t num_vertices = rx.get_num_vertices();

    auto coords = rx.get_input_vertex_coordinates();
    //rx.export_obj("mcf_initial.obj", *rx.get_input_vertex_coordinates());

    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  B_mat(rx, num_vertices, 3, LOCATION_ALL);

    DenseMatrix<float> X_mat = *(coords->to_matrix());

    rx.run_kernel<blockThreads>({Op::VV},
                                mcf_B_setup<float, blockThreads>,
                                *coords,
                                B_mat,
                                Arg.use_uniform_laplace);

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


    GMGSolver solver(rx,
                     A_mat,
                     Arg.max_num_iter,
                     Arg.gmg_levels,
                     2,
                     2,
                     csolver,
                     sampling,
                     Arg.tol_abs,
                     Arg.tol_rel,
                     Arg.gmg_threshold,
                     Arg.gmg_pruned_ptap,
                     Arg.gmg_verify_ptap);


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
        solver.gmg_memory_alloc_time() + solver.v_cycle_memory_alloc_time());

    float pre_solve_time =
        std::max(timer.elapsed_millis(), gtimer.elapsed_millis());


    timer.start();
    gtimer.start();
    solver.solve(B_mat, X_mat);
    timer.stop();
    gtimer.stop();

    float solve_time =
        std::max(timer.elapsed_millis(), gtimer.elapsed_millis());

    float total_time = pre_solve_time + solve_time;

    RXMESH_INFO("start_residual {}", solver.start_residual());
    RXMESH_INFO("final_residual {}", solver.final_residual());
    RXMESH_INFO("GMG solve took {} (ms), {} (ms). #iter= {}",
                timer.elapsed_millis(),
                gtimer.elapsed_millis(),
                solver.iter_taken());
    RXMESH_INFO("total_time {} (ms)", total_time);


    report.add_member("solver", solver.name());
    report.add_member("application", std::string("MCF"));
    report.add_member("blockThreads", blockThreads);
    report.add_member("levels", solver.get_num_levels());
    report.add_member("iterations", solver.iter_taken());
    report.add_member("pre_solve_time(ms)", pre_solve_time);
    report.add_member("gmg_memory_alloc_time(ms)",
                      solver.gmg_memory_alloc_time());
    report.add_member("v_cycle_memory_alloc_time(ms)",
                      solver.v_cycle_memory_alloc_time());
    report.add_member("solve_time(ms)", solve_time);
    report.add_member("total_time(ms)", total_time);
    report.add_member("avg_iteration_time(ms)",
                      solve_time / solver.iter_taken());
    report.add_member("max_iterations", Arg.max_num_iter);
    report.add_member("threshold", Arg.gmg_threshold);
    report.add_member("final_residual", solver.final_residual());
    report.add_member("gmg_pruned_ptap", Arg.gmg_pruned_ptap);
    report.add_member("gmg_verify_ptap", Arg.gmg_verify_ptap);
    report.add_member("gmg_coarse_solver", Arg.gmg_csolver);
    report.add_member("gmg_sampling", Arg.gmg_sampling);


    X_mat.move(rxmesh::DEVICE, rxmesh::HOST);

    // copy the results to attributes
    coords->from_matrix(&X_mat);

#if USE_POLYSCOPE
    if (Arg.gmg_render_hierarchy) {
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
