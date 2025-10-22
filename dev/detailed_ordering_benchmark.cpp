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
#include "check_valid_permutation.h"
#include "get_factor_nnz.h"
#include "ordering.h"
#include "parth/parth.h"
#include "remove_diagonal.h"

struct CLIArgs
{
    std::string input_mesh;
    std::string output_address;
    std::string solver_type   = "CHOLMOD";
    std::string ordering_type = "DEFAULT";
    std::string local_permute_method = "metis";
    std::string separator_finding_method = "max_degree";
    std::string separator_refinement_method = "nothing";
    CLIArgs(int argc, char* argv[])
    {
        CLI::App app{"Separator analysis"};
        app.add_option("-a,--ordering", ordering_type, "ordering type");
        app.add_option("-s,--solver", solver_type, "solver type");
        app.add_option("-o,--output", output_address, "output folder name");
        app.add_option("-i,--input", input_mesh, "input mesh name");
        app.add_option("-l,--local_permute_method", local_permute_method, "local permute method");
        app.add_option("-t,--separator_finding_method", separator_finding_method, "separator finding method");
        app.add_option("-r,--separator_refinement_method", separator_refinement_method, "separator refinement method");
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

    // Make sure the matrix is symmetric positive definite by adding to diagonal
    // The cotangent Laplacian is negative semi-definite, so we add a constant
    // to shift all eigenvalues to be positive
    for (int i = 0; i < OL.rows(); ++i) {
        OL.coeffRef(i, i) += 100.0;
    }
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
    } else if (args.ordering_type == "POC_ND") {
        spdlog::info("Using POC_ND ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::RXMESH_Ordering_Type::POC_ND);
        ordering->setOptions({{"local_permute_method", args.local_permute_method}, {"separator_finding_method", args.separator_finding_method}, {"separator_refinement_method", args.separator_refinement_method}});
    } else if (args.ordering_type == "PARTH") {
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::RXMESH_Ordering_Type::PARTH);
    } else if (args.ordering_type == "NEUTRAL"){
        spdlog::info("Using NEUTRAL ordering.");
        ordering = RXMESH_SOLVER::Ordering::create(
            RXMESH_SOLVER::RXMESH_Ordering_Type::NEUTRAL);
    } else {
        spdlog::error("Unknown Ordering type.");
    }

    // Create the graph
    std::vector<int> Gp;
    std::vector<int> Gi;
    RXMESH_SOLVER::remove_diagonal(
        OL.rows(), OL.outerIndexPtr(), OL.innerIndexPtr(), Gp, Gi);

    // Init the permuter
    double factor_ratio = 0;
    if (ordering != nullptr) {
        // Provide mesh data if the ordering needs it (e.g., RXMesh ND)
        if (ordering->needsMesh()) {
            // Pass raw pointers to avoid ABI issues between C++ and CUDA compilation
            ordering->setMesh(OV.data(), OV.rows(), OV.cols(),
                            OF.data(), OF.rows(), OF.cols());
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
        factor_ratio = factor_nnz * 1.0 /OL.nonZeros();
    }
    std::string mesh_name = args.input_mesh.substr(args.input_mesh.find_last_of("/") + 1);
    mesh_name = mesh_name.substr(0, mesh_name.find_last_of("."));
    std::map<std::string, double> extra_info;
    extra_info["fill-ratio"] = factor_ratio;

    ordering->add_record("/home/behrooz/Desktop/Last_Project/RXMesh-dev/output/ordering_benchmark", extra_info, mesh_name);
    delete ordering;
    return 0;
}