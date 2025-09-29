//
// Created by behrooz on 2025-09-28.
//


#include <iostream>
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include <igl/read_triangle_mesh.h>
#include <igl/cotmatrix.h>
#include <LinSysSolver.hpp>
#include <spdlog/spdlog.h>
#include <chrono>


struct CLIArgs {
    std::string input_mesh;
    std::string output_address;
    std::string solver_type = "CHOLMOD";
    CLIArgs(int argc, char *argv[]) {
        CLI::App app{"Separator analysis"};
        app.add_option("-s,--solver", solver_type, "solver type");
        app.add_option("-o,--output", output_address, "output folder name");
        app.add_option("-i,--input", input_mesh, "input mesh name");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError &e) {
            exit(app.exit(e));
        }
    }
};


void createSmoother(){

}



int main(int argc, char *argv[]) {
    // Load the mesh
    CLIArgs args(argc, argv);

    if (args.input_mesh.empty()) {
        std::cerr << "Error: Input mesh file not specified. Use -i or --input to specify the mesh file." << std::endl;
        return 1;
    }

    std::cout << "Loading mesh from: " << args.input_mesh << std::endl;
    std::cout << "Output folder: " << args.output_address << std::endl;

    Eigen::MatrixXd OV;
    Eigen::MatrixXi OF;
    if (!igl::read_triangle_mesh(args.input_mesh, OV, OF)) {
        std::cerr << "Failed to read the mesh: " << args.input_mesh << std::endl;
        return 1;
    }

    //Create laplacian matrix
    Eigen::SparseMatrix<double> OL;
    igl::cotmatrix(OV, OF, OL);

    // Print laplacian size and sparsity
    spdlog::info("Number of rows: {}", OL.rows());
    spdlog::info("Number of non-zeros: {}", OL.nonZeros());
    spdlog::info("Sparsity: {:.2f}%", (1 - (OL.nonZeros() / static_cast<double>(OL.rows() * OL.rows()))) * 100);

    //Make sure the matrix is semi-positive definit by add values to diagonal
    OL.diagonal().array() += 100;
    Eigen::VectorXd rhs = Eigen::VectorXd::Random(OL.rows());
    Eigen::VectorXd result;

    //init Parth
    std::vector<int> perm;
    RXMESH_SOLVER::LinSysSolver* solver = nullptr;
    if (args.solver_type == "CHOLMOD") {
        solver = RXMESH_SOLVER::LinSysSolver::create(
            RXMESH_SOLVER::LinSysSolverType::CPU_CHOLMOD);
    } else if (args.solver_type == "CUDSS") {
        solver = RXMESH_SOLVER::LinSysSolver::create(
            RXMESH_SOLVER::LinSysSolverType::CPU_CHOLMOD);
    } else {
        spdlog::error("Unknown solver type.");
    }
    assert(solver != nullptr);

    solver->setMatrix(OL.outerIndexPtr(), OL.innerIndexPtr(),
        OL.valuePtr(), OL.rows(), OL.nonZeros());

    //Symbolic analysis time
    auto start = std::chrono::high_resolution_clock::now();
    solver->analyze_pattern(perm);
    auto end = std::chrono::high_resolution_clock::now();
    spdlog::info("Analysis time: {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    //Factorization time
    start = std::chrono::high_resolution_clock::now();
    solver->factorize();
    end = std::chrono::high_resolution_clock::now();
    spdlog::info("Factorization time: {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    //Solve time
    start = std::chrono::high_resolution_clock::now();
    solver->solve(rhs, result);
    end = std::chrono::high_resolution_clock::now();
    spdlog::info("Solve time: {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    //Compute residual
    solver->computeResidual(OL, result, rhs);
    spdlog::info("Residual: {}", solver->residual);
    return 0;
}