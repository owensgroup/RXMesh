//
//  CHOLMODSolver.cpp
//  IPC
//
//  Created by Minchen Li on 6/22/18.
//

#ifdef USE_CUDSS

#include <spdlog/spdlog.h>
#include <cassert>
#include <iostream>
#include "CUDSSSolver.hpp"

#include <spdlog/spdlog.h>
#include "omp.h"

#ifndef CUDA_ERROR
inline void HandleError(cudaError_t err, const char* file, int line)
{
    // Error handling micro, wrap it around function whenever possible
    if (err != cudaSuccess) {
        spdlog::error("Line {} File {}", line, file);
        spdlog::error("CUDA ERROR: {}", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CUDA_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#endif


namespace RXMESH_SOLVER {
CUDSSSolver::~CUDSSSolver()
{
    cudssConfigDestroy(config);
    cudssDataDestroy(handle, data);
    clean_sparse_matrix_mem();
    clean_rhs_sol_mem();
    cudssDestroy(handle);
}

CUDSSSolver::CUDSSSolver()
{
    rowOffsets_dev = nullptr;
    colIndices_dev = nullptr;
    values_dev     = nullptr;
    bvalues_dev    = nullptr;
    xvalues_dev    = nullptr;
    user_perm_dev  = nullptr;
    x_mat          = nullptr;
    b_mat          = nullptr;
    data           = nullptr;
    handle         = nullptr;
    config         = nullptr;
    A              = nullptr;
    N              = 0;
    NNZ            = 0;
    auto status = cudssCreate(&handle);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::constructor - cudssCreate failed with status: {}", status);
        exit(EXIT_FAILURE);
    }
    
    status = cudssConfigCreate(&config);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::constructor - cudssConfigCreate failed with status: {}", status);
        exit(EXIT_FAILURE);
    }
    
    status = cudssDataCreate(handle, &data);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::constructor - cudssDataCreate failed with status: {}", status);
        exit(EXIT_FAILURE);
    }
    
    is_allocated = false;
}

void CUDSSSolver::clean_sparse_matrix_mem()
{
    if (rowOffsets_dev != nullptr)
        CUDA_ERROR(cudaFree(rowOffsets_dev));
    if (colIndices_dev != nullptr)
        CUDA_ERROR(cudaFree(colIndices_dev));
    if (values_dev != nullptr)
        CUDA_ERROR(cudaFree(values_dev));
    if (A != nullptr)
        cudssMatrixDestroy(A);
    rowOffsets_dev = nullptr;
    colIndices_dev = nullptr;
    values_dev     = nullptr;
    A              = nullptr;
    is_allocated   = false;
}

void CUDSSSolver::clean_rhs_sol_mem()
{
    if (x_mat != nullptr)
        cudssMatrixDestroy(x_mat);
    if (b_mat != nullptr)
        cudssMatrixDestroy(b_mat);
    if (xvalues_dev != nullptr)
        CUDA_ERROR(cudaFree(xvalues_dev));
    if (bvalues_dev != nullptr)
        CUDA_ERROR(cudaFree(bvalues_dev));
    xvalues_dev = nullptr;
    bvalues_dev = nullptr;
    x_mat       = nullptr;
    b_mat       = nullptr;
}


void CUDSSSolver::setMatrix(int* p, int* i, double* x, int A_N, int NNZ)
{
    // Validate input parameters
    if (A_N <= 0 || NNZ <= 0) {
        spdlog::error("CUDSSSolver::setMatrix - Invalid dimensions: N={}, NNZ={}", A_N, NNZ);
        exit(EXIT_FAILURE);
    }
    if (p == nullptr || i == nullptr || x == nullptr) {
        spdlog::error("CUDSSSolver::setMatrix - Null pointer passed");
        exit(EXIT_FAILURE);
    }
    
    // Log matrix properties for debugging
    spdlog::info("CUDSSSolver::setMatrix - N={}, NNZ={}", A_N, NNZ);
    
    this->N   = A_N;
    this->NNZ = NNZ;

    // Allocating memory
    if (!is_allocated || this->NNZ != NNZ || this->N != A_N) {
        clean_sparse_matrix_mem();
        CUDA_ERROR(
            cudaMalloc((void**)&rowOffsets_dev, (A_N + 1) * sizeof(int)));
        CUDA_ERROR(cudaMalloc((void**)&colIndices_dev, NNZ * sizeof(int)));
        CUDA_ERROR(cudaMalloc((void**)&values_dev, NNZ * sizeof(double)));
        is_allocated = true;
    }


    // Copying data to device
    CUDA_ERROR(cudaMemcpy(
        rowOffsets_dev, p, (A_N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(
        colIndices_dev, i, NNZ * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(
        values_dev, x, NNZ * sizeof(double), cudaMemcpyHostToDevice));

    // Creating matrix
    auto status = cudssMatrixCreateCsr(&A,
                         N,                    // nrows
                         N,                    // ncols  
                         NNZ,                  // nnz
                         rowOffsets_dev,       // csrRowOffsets
                         nullptr,              // csrRowOffsetsEnd (optional)
                         colIndices_dev,       // csrColInd
                         values_dev,           // csrValues
                         CUDA_R_32I,           // csrRowOffsetsType
                         CUDA_R_64F,           // csrValuesType (double precision)
                         CUDSS_MTYPE_SYMMETRIC,// matrixType
                         CUDSS_MVIEW_FULL,     // viewType
                         CUDSS_BASE_ZERO);     // indexBase
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::matrix creation failed with status: {}", status);
        exit(EXIT_FAILURE);
    }
}

void CUDSSSolver::innerAnalyze_pattern(std::vector<int>& user_defined_perm)
{
    assert(is_allocated);
    cudssStatus_t status;
    
    if (user_defined_perm.size() == N) {
        // When using user-defined permutation, provide it to CUDSS
        spdlog::info("CUDSS: Using user-defined permutation (size={})", N);
        
        // Allocate device memory for permutation if not already allocated
        if (user_perm_dev != nullptr) {
            CUDA_ERROR(cudaFree(user_perm_dev));
            user_perm_dev = nullptr;
        }
        CUDA_ERROR(cudaMalloc((void**)&user_perm_dev, N * sizeof(int)));
        
        // Copy host permutation to device
        CUDA_ERROR(cudaMemcpy(user_perm_dev, 
                              user_defined_perm.data(), 
                              N * sizeof(int), 
                              cudaMemcpyHostToDevice));

        // Set CUDSS to use user-defined permutation
        cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
        status = cudssConfigSet(config,
                                CUDSS_CONFIG_REORDERING_ALG,
                                &reorder_alg,
                                sizeof(cudssAlgType_t));
        if (status != CUDSS_STATUS_SUCCESS) {
            spdlog::error("CUDSSSolver::cudssConfigSet for reordering algorithm failed with status: {}", status);
            exit(EXIT_FAILURE);
        }
        
        // Provide device pointer to CUDSS
        status = cudssDataSet(handle,
                              data,
                              CUDSS_DATA_USER_PERM,
                              user_perm_dev,  // Device pointer!
                              N * sizeof(int));
        if (status != CUDSS_STATUS_SUCCESS) {
            spdlog::error("CUDSSSolver::cudssDataSet for user permutation failed with status: {}", status);
            exit(EXIT_FAILURE);
        }
    } else {
        // Use default CUDSS reordering
        spdlog::info("CUDSS: Using default reordering and analysis");
        cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
        status = cudssConfigSet(config,
                                CUDSS_CONFIG_REORDERING_ALG,
                                &reorder_alg,
                                sizeof(cudssAlgType_t));
        if (status != CUDSS_STATUS_SUCCESS) {
            spdlog::error("CUDSSSolver::cudssConfigSet for reordering algorithm failed with status: {}", status);
            exit(EXIT_FAILURE);
        }
    }
    // Run both reordering and symbolic factorization
    status = cudssExecute(
        handle,
        CUDSS_PHASE_REORDERING | CUDSS_PHASE_SYMBOLIC_FACTORIZATION,
        config,
        data,
        A,
        nullptr,
        nullptr);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::symbolic analysis failed with status: {}", status);
        exit(EXIT_FAILURE);
    }
}

void CUDSSSolver::innerFactorize(void)
{
    assert(is_allocated);
    auto status = cudssExecute(
        handle, CUDSS_PHASE_FACTORIZATION, config, data, A, nullptr, nullptr);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::factorization failed!");
        exit(EXIT_FAILURE);
    }
}

void CUDSSSolver::innerSolve(Eigen::VectorXd& rhs, Eigen::VectorXd& result)
{
    // Validate input dimensions
    if (rhs.size() != N) {
        spdlog::error("CUDSSSolver::solve - RHS size {} doesn't match matrix size {}", rhs.size(), N);
        exit(EXIT_FAILURE);
    }
    
    spdlog::info("CUDSSSolver::solve - Matrix size: {}, RHS size: {}", N, rhs.size());
    
    // Clean up any existing memory first
    clean_rhs_sol_mem();
    CUDA_ERROR(cudaMalloc((void**)&bvalues_dev, N * sizeof(double)));
    CUDA_ERROR(cudaMalloc((void**)&xvalues_dev, N * sizeof(double)));
    // Copying the memory
    CUDA_ERROR(cudaMemcpy(
        bvalues_dev, rhs.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    // Creating the rhs matrix (N×1 column vector)
    spdlog::info("Creating RHS dense matrix: N={}, ncols=1, ld={}", N, N);
    auto status = cudssMatrixCreateDn(
        &b_mat, N, 1, N, bvalues_dev, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::RHS matrix creation failed with status: {}", status);
        exit(EXIT_FAILURE);
    }

    // Creating solution matrix (N×1 column vector)
    spdlog::info("Creating solution dense matrix: N={}, ncols=1, ld={}", N, N);
    status = cudssMatrixCreateDn(
        &x_mat, N, 1, N, xvalues_dev, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::solution matrix creation failed with status: {}", status);
        exit(EXIT_FAILURE);
    }

    // Execute solve phase
    spdlog::info("Executing CUDSS solve phase...");
    status = cudssExecute(
        handle, CUDSS_PHASE_SOLVE, config, data, A, x_mat, b_mat);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::solve failed with status: {} ", status);
        exit(EXIT_FAILURE);
    }
    spdlog::info("CUDSS solve completed successfully");
    // Copy the result back
    result.conservativeResize(rhs.size());
    CUDA_ERROR(cudaMemcpy(result.data(),
                          xvalues_dev,
                          result.rows() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    clean_rhs_sol_mem();
}


void CUDSSSolver::resetSolver()
{
}

LinSysSolverType CUDSSSolver::type() const
{
    return LinSysSolverType::GPU_CUDSS;
};
}  // namespace RXMESH_SOLVER

#endif
