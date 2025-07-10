#pragma once

#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix.h"
#include "rxmesh/rxmesh_static.h"

#include "mcf_kernels.cuh"


#include <Eigen/Sparse>

#include <Eigen/src/SparseCholesky/SimplicialCholesky.h>

/**
 * @brief using Eigen to solve the same system
 */
template <typename SpMat, typename DMat>
void simplicial_llt(const rxmesh::RXMeshStatic& rx,
                    const SpMat&                A,
                    const DMat&                 B)
{
    using namespace rxmesh;

    Report report("MCF_Eigen");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rx);
    report.add_member("method", std::string("SimplicialLLT"));
    report.add_member("application", std::string("MCF"));

    float total_time = 0;

    Eigen::SimplicialLLT<SpMat,
                         Eigen::UpLoType::Lower,
                         Eigen::AMDOrdering<typename SpMat::StorageIndex>>
        solver;

    CPUTimer timer;
    timer.start();
    solver.analyzePattern(A);
    timer.stop();

    if (solver.info() != Eigen::Success) {
        std::cout << solver.info() << "\n";
        RXMESH_ERROR("Eigen::SimplicialLLT analyzePattern failed!");
    }

    total_time += timer.elapsed_millis();
    RXMESH_INFO("SimplicialLLT analyze_pattern took {} (ms)",
                timer.elapsed_millis());
    report.add_member("analyze_pattern", timer.elapsed_millis());

    timer.start();
    solver.factorize(A);
    timer.stop();

    total_time += timer.elapsed_millis();
    RXMESH_INFO("SimplicialLLT factorize took {} (ms)", timer.elapsed_millis());
    report.add_member("factorize", timer.elapsed_millis());

    if (solver.info() != Eigen::Success) {
        std::cout << solver.info() << "\n";
        RXMESH_ERROR("Eigen::SimplicialLLT factorization failed!");
    }

    timer.start();
    DMat X = solver.solve(B);
    timer.stop();

    total_time += timer.elapsed_millis();
    RXMESH_INFO("SimplicialLLT solve took {} (ms)", timer.elapsed_millis());
    report.add_member("solve", timer.elapsed_millis());

    if (solver.info() != Eigen::Success) {
        std::cout << solver.info() << "\n";
        RXMESH_ERROR("Eigen::SimplicialLLT solve failed!");
    }

    RXMESH_INFO("SimplicialLLT total_time {} (ms)", total_time);
    report.add_member("total_time", total_time);


    report.write(Arg.output_folder + "/rxmesh",
                 "MCF_EIGEN_LDLT_" + extract_file_name(Arg.obj_file_name));
}
