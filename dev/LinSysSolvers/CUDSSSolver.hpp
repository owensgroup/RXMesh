
#pragma once

#ifdef USE_CUDSS

#include "LinSysSolver.hpp"
#include <cuda.h>
#include <cudss.h>
#include <Eigen/Eigen>
#include <vector>

namespace RXMESH_SOLVER {

class CUDSSSolver : public LinSysSolver {
    typedef LinSysSolver Base; // The class
public:                // Access specifier

    cudssHandle_t             handle;
    cudssConfig_t             config;
    cudssData_t               data;
    cudssMatrix_t             A;
    cudssMatrix_t             b_mat;
    cudssMatrix_t             x_mat;

    //Device pointers
    int*    rowOffsets_dev;
    int*    colIndices_dev;
    double* values_dev;
    double* bvalues_dev;
    double* xvalues_dev;
    int*    user_perm_dev;  // Device pointer for user-defined permutation

    bool is_allocated;



    ~CUDSSSolver();
    CUDSSSolver();

    void setMatrix(int *p, int *i, double *x, int A_N, int NNZ) override;
    void innerAnalyze_pattern(std::vector<int>& user_defined_perm) override;
    void innerFactorize(void) override;
    void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) override;
    void resetSolver() override;
    void clean_sparse_matrix_mem();
    void clean_rhs_sol_mem();
    virtual LinSysSolverType type() const override;

};

}

#endif
