
#pragma once

#ifdef RXMESH_WITH_SUITESPARSE

#include "LinSysSolver.hpp"

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
    cudssMatrix_t             b;
    cudssMatrix_t             x;

    //Device pointers
    int*    rowOffsets_dev;
    int*    colIndices_dev;
    double* values_dev;
    double* bvalues_dev;
    double* xvalues_dev;

    ~CUDSSSolver();
    CUDSSSolver();

    void setMatrix(int *p, int *i, double *x, int A_N, int NNZ) override;
    void innerAnalyze_pattern(std::vector<int>& user_defined_perm) override;
    void innerFactorize(void) override;
    void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) override;
    void resetSolver() override;
    virtual LinSysSolverType type() const override;

};

}

#endif
