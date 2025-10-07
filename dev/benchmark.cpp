//
// Created by behrooz on 2025-09-28.
//


#include <igl/cotmatrix.h>
#include <igl/read_triangle_mesh.h>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <chrono>
#include <iostream>

#include "LinSysSolver.hpp"
#include "get_factor_nnz.h"
#include "check_valid_permutation.h"
#include "ordering.h"
#include "remove_diagonal.h"
#include "parth/parth.h"

struct CLIArgs
{
    std::string input_mesh;
    std::string output_address;
    std::string solver_type   = "CHOLMOD";
    std::string ordering_type = "DEFAULT";
    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"Separator analysis"};
        app.add_option("-a,--ordering", ordering_type, "ordering type");
        app.add_option("-s,--solver", solver_type, "solver type");
        app.add_option("-o,--output", output_address, "output folder name");
        app.add_option("-i,--input", input_mesh, "input mesh name");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            exit(app.exit(e));
        }
    }
};


Eigen::SparseMatrix<double> computeSmootherMatrix(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(V, F, L);
    // Make sure the matrix is semi-positive definite by adding values to the diagonal
    L.diagonal().array() += 100;
    return L;

}


int main(int argc, char* argv[])
{
    // Load the mesh
    CLIArgs args(argc, argv);

    if (args.input_mesh.empty()) {
        std::cerr << "Error: Input mesh file not specified. Use -i or --input "
                     "to specify the mesh file."
                  << std::endl;
        return 1;
    }

    std::cout << "Loading mesh from: " << args.input_mesh << std::endl;
    std::cout << "Output folder: " << args.output_address << std::endl;

    Eigen::MatrixXd OV;
    Eigen::MatrixXi OF;
    if (!igl::read_triangle_mesh(args.input_mesh, OV, OF)) {
        std::cerr << "Failed to read the mesh: " << args.input_mesh
                  << std::endl;
        return 1;
    }

    // Create laplacian matrix
    Eigen::SparseMatrix<double> OL;
    igl::cotmatrix(OV, OF, OL);

    // Print laplacian size and sparsity
    spdlog::info("Number of rows: {}", OL.rows());
    spdlog::info("Number of non-zeros: {}", OL.nonZeros());
    spdlog::info(
        "Sparsity: {:.2f}%",
        (1 - (OL.nonZeros() / static_cast<double>(OL.rows() * OL.rows()))) *
            100);

    // Make sure the matrix is semi-positive definit by add values to diagonal
    OL.diagonal().array() += 100;
    Eigen::VectorXd rhs = Eigen::VectorXd::Random(OL.rows());
    Eigen::VectorXd result;

    // Init permuter
    std::vector<int>         perm;
    RXMESH_SOLVER::Ordering* ordering = nullptr;
    if (args.ordering_type == "DEFAULT") {
        spdlog::info("Using default ordering (default for each solver).");
        ordering = nullptr;
    } else if (args.ordering_type == "METIS") {
        spdlog::info("Using METIS ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::RXMESH_Ordering_Type::METIS);
    } else if (args.ordering_type == "RXMESH_ND") {
        spdlog::info("Using RXMESH ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::RXMESH_Ordering_Type::RXMESH_ND);
    } else if (args.ordering_type == "NEUTRAL"){
        spdlog::info("Using NEUTRAL ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::RXMESH_Ordering_Type::NEUTRAL);
    } else {
        spdlog::error("Unknown Ordering type.");
    }

    //Init solver
    RXMESH_SOLVER::LinSysSolver* solver = nullptr;
    if (args.solver_type == "CHOLMOD") {
        solver = RXMESH_SOLVER::LinSysSolver::create(
            RXMESH_SOLVER::LinSysSolverType::CPU_CHOLMOD);
        spdlog::info("Using CHOLMOD direct solver.");
    } else if (args.solver_type == "CUDSS") {
        solver = RXMESH_SOLVER::LinSysSolver::create(
            RXMESH_SOLVER::LinSysSolverType::GPU_CUDSS);
        spdlog::info("Using CUDSS direct solver.");
    } else if (args.solver_type == "PARTH_SOLVER") {
        solver = RXMESH_SOLVER::LinSysSolver::create(
            RXMESH_SOLVER::LinSysSolverType::PARTH_SOLVER);
        spdlog::info("Using PARTH direct solver.");
    } else {
        spdlog::error("Unknown solver type.");
    }
    assert(solver != nullptr);

    // Create the graph
    std::vector<int> Gp;
    std::vector<int> Gi;
    RXMESH_SOLVER::remove_diagonal(
        OL.rows(), OL.outerIndexPtr(), OL.innerIndexPtr(), Gp, Gi);

    // Init the permuter
    if (ordering != nullptr) {
        // Provide mesh data if the ordering needs it (e.g., RXMesh ND)
        if (ordering->needsMesh()) {
            ordering->setMesh(OV, OF);
        }
        ordering->setGraph(Gp.data(), Gi.data(), OL.rows(), Gi.size());
        auto ordering_start = std::chrono::high_resolution_clock::now();
        ordering->compute_permutation(perm);
        auto ordering_end = std::chrono::high_resolution_clock::now();
        //Check for correct perm
        if (!RXMESH_SOLVER::check_valid_permutation(perm.data(), perm.size())) {
            spdlog::error("Permutation is not valid!");
        }
        spdlog::info("Ordering time: {} ms",
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         ordering_end - ordering_start)
                         .count());
        assert(perm.size() == OL.rows());

        int factor_nnz = RXMESH_SOLVER::get_factor_nnz(OL.outerIndexPtr(),
                                                       OL.innerIndexPtr(),
                                                       OL.valuePtr(),
                                                       OL.rows(),
                                                       OL.nonZeros(),
                                                       perm);
        spdlog::info(
            "The ratio of factor non-zeros to matrix non-zeros given custom reordering: {}",
            (factor_nnz * 1.0 /OL.nonZeros()));
        solver->ordering_name = ordering->typeStr();
    }

    solver->setMatrix(OL.outerIndexPtr(),
                      OL.innerIndexPtr(),
                      OL.valuePtr(),
                      OL.rows(),
                      OL.nonZeros());

    // Symbolic analysis time
    auto start = std::chrono::high_resolution_clock::now();
    solver->analyze_pattern(perm);
    auto end = std::chrono::high_resolution_clock::now();
    spdlog::info(
        "Analysis time: {} ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count());

    // Factorization time
    start = std::chrono::high_resolution_clock::now();
    solver->factorize();
    end = std::chrono::high_resolution_clock::now();
    spdlog::info(
        "Factorization time: {} ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count());

    // Solve time
    start = std::chrono::high_resolution_clock::now();
    solver->solve(rhs, result);
    end = std::chrono::high_resolution_clock::now();
    spdlog::info(
        "Solve time: {} ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count());

    // Compute residual
    assert(OL.rows() == OL.cols());
    double residual = (rhs - OL * result).norm();
    spdlog::info("Residual: {}", residual);
    spdlog::info("Final factor/matrix NNZ ratio: {}",
                 solver->getFactorNNZ() * 1.0 / OL.nonZeros());
    delete solver;
    delete ordering;
    return 0;
}