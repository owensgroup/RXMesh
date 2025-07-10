#pragma once

#include "rxmesh/attribute.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix.h"
#include "rxmesh/rxmesh_static.h"

#include "mcf_kernels.cuh"


#include <Eigen/Sparse>

#include <Eigen/src/SparseCholesky/SimplicialCholesky.h>


template <typename SolverT, typename SpMat, typename DMat>
void run_eigen(const rxmesh::RXMeshStatic& rx,
               const std::string           name,
               const SpMat&                A,
               const DMat&                 B)
{
    using namespace rxmesh;

    Report report("MCF_Eigen_" + name);
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rx);
    report.add_member("method", name);
    report.add_member("application", std::string("MCF"));

    float total_time = 0;

    SolverT solver;

    CPUTimer timer;
    timer.start();
    solver.analyzePattern(A);
    timer.stop();

    if (solver.info() != Eigen::Success) {
        RXMESH_ERROR("{} analyzePattern failed!", name);
    }

    total_time += timer.elapsed_millis();
    RXMESH_INFO(
        "{} analyze_pattern took {} (ms)", name, timer.elapsed_millis());
    report.add_member("analyze_pattern", timer.elapsed_millis());

    timer.start();
    solver.factorize(A);
    timer.stop();

    total_time += timer.elapsed_millis();
    RXMESH_INFO("{} factorize took {} (ms)", name, timer.elapsed_millis());
    report.add_member("factorize", timer.elapsed_millis());

    if (solver.info() != Eigen::Success) {
        RXMESH_ERROR("{} factorization failed!", name);
    }

    timer.start();
    DMat X = solver.solve(B);
    timer.stop();

    total_time += timer.elapsed_millis();
    RXMESH_INFO("{} solve took {} (ms)", name, timer.elapsed_millis());
    report.add_member("solve", timer.elapsed_millis());

    if (solver.info() != Eigen::Success) {
        RXMESH_ERROR("{} solve failed!", name);
    }

    RXMESH_INFO("{} total_time {} (ms)", name, total_time);
    report.add_member("total_time", total_time);


    report.write(Arg.output_folder + "/rxmesh",
                 "MCF_" + name + "_" + extract_file_name(Arg.obj_file_name));
}


template <typename SpMat, typename DMat>
void run_eigen_all(const rxmesh::RXMeshStatic& rx,
                   const SpMat&                A,
                   const DMat&                 B)
{
    using LLT =
        Eigen::SimplicialLLT<SpMat,
                             Eigen::UpLoType::Lower,
                             Eigen::AMDOrdering<typename SpMat::StorageIndex>>;

    run_eigen<LLT>(rx, "SimplicialLLT", A, B);

    using LDLT =
        Eigen::SimplicialLDLT<SpMat,
                              Eigen::UpLoType::Lower,
                              Eigen::AMDOrdering<typename SpMat::StorageIndex>>;

    run_eigen<LDLT>(rx, "SimplicialLDLT", A, B);
}
