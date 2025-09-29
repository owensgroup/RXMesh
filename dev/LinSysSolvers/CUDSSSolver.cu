//
//  CHOLMODSolver.cpp
//  IPC
//
//  Created by Minchen Li on 6/22/18.
//

#ifdef RXMESH_WITH_SUITESPARSE

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
    A              = nullptr;
    N              = 0;
    NNZ            = 0;
    cudssCreate(&handle);
    cudssConfigCreate(&config);
    cudssDataCreate(handle, &data);
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
    cudssMatrixDestroy(A);
    rowOffsets_dev = nullptr;
    colIndices_dev = nullptr;
    values_dev     = nullptr;
    A              = nullptr;
    is_allocated   = false;
}

void CUDSSSolver::clean_rhs_sol_mem()
{
    cudssMatrixDestroy(x_mat);
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
    this->N   = A_N;
    this->NNZ = NNZ;

    // Allocating memory
    if (!is_allocated || this->NNZ != NNZ || this->N != A_N) {
        clean_sparse_matrix_mem();
        CUDA_ERROR(
            cudaMalloc((void**)&rowOffsets_dev, (A_N + 1) * sizeof(int)));
        CUDA_ERROR(cudaMalloc((void**)&colIndices_dev, NNZ * sizeof(int)));
        CUDA_ERROR(cudaMalloc((void**)&values_dev, NNZ * sizeof(double)));
        CUDA_ERROR(cudaMalloc((void**)&bvalues_dev, A_N * sizeof(double)));
        CUDA_ERROR(cudaMalloc((void**)&xvalues_dev, A_N * sizeof(double)));
        is_allocated = true;
    }


    // Copying data to device
    CUDA_ERROR(cudaMemcpy(
        rowOffsets_dev, p, (A_N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(
        colIndices_dev, i, NNZ * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(
        values_dev, x, NNZ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemset(bvalues_dev, 0, A_N * sizeof(double)));
    CUDA_ERROR(cudaMemset(xvalues_dev, 0, A_N * sizeof(double)));

    // Creating matrix
    cudssMatrixCreateCsr(&A,
                         N,
                         N,
                         NNZ,
                         rowOffsets_dev,
                         nullptr,
                         colIndices_dev,
                         values_dev,
                         CUDA_R_32I,
                         CUDA_R_32F,
                         CUDSS_MTYPE_SYMMETRIC,
                         CUDSS_MVIEW_FULL,
                         CUDSS_BASE_ZERO);
}

void CUDSSSolver::innerAnalyze_pattern(std::vector<int>& user_defined_perm)
{
    assert(is_allocated);
    if (user_defined_perm.size() == N) {
        cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
        cudssConfigSet(config,
                       CUDSS_CONFIG_REORDERING_ALG,
                       &reorder_alg,
                       sizeof(cudssAlgType_t));
        cudssDataSet(handle,
                     data,
                     CUDSS_DATA_USER_PERM,
                     user_defined_perm.data(),
                     sizeof(int));
    } else {
        spdlog::info("Using default reordering and analysis");
        cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
        cudssConfigSet(config,
                       CUDSS_CONFIG_REORDERING_ALG,
                       &reorder_alg,
                       sizeof(cudssAlgType_t));
    }
    auto status = cudssExecute(
        handle,
        CUDSS_PHASE_REORDERING | CUDSS_PHASE_SYMBOLIC_FACTORIZATION,
        config,
        data,
        A,
        nullptr,
        nullptr);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::symbolic analysis failed!");
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
    // allocating the memory
    if (bvalues_dev != nullptr || xvalues_dev != nullptr) {
        clean_rhs_sol_mem();
    }
    CUDA_ERROR(cudaMalloc((void**)&bvalues_dev, N * sizeof(double)));
    CUDA_ERROR(cudaMalloc((void**)&xvalues_dev, N * sizeof(double)));
    // Copying the memory
    CUDA_ERROR(cudaMemcpy(
        bvalues_dev, rhs.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    // Creating the rhs matrix
    auto status = cudssMatrixCreateDn(
        &b_mat, N, 1, 1, bvalues_dev, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::b allocation failed failed!");
        exit(EXIT_FAILURE);
    }

    status = cudssMatrixCreateDn(
        &x_mat, N, 1, 1, xvalues_dev, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::x allocation failed failed!");
        exit(EXIT_FAILURE);
    }

    status = cudssExecute(
        handle, CUDSS_PHASE_SOLVE, config, data, A, x_mat, nullptr);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::solve failed!");
        exit(EXIT_FAILURE);
    }
    // Copy the result back
    result.conservativeResize(rhs.size());
    cudaMemcpy(result.data(),
               xvalues_dev,
               result.rows() * sizeof(double),
               cudaMemcpyDeviceToHost);
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
