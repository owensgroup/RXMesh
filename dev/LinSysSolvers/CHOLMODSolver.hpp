
#pragma once

#ifdef USE_SUITESPARSE

#include "LinSysSolver.hpp"

#include <cholmod.h>
#include <Eigen/Eigen>
#include <vector>

namespace RXMESH_SOLVER {

class CHOLMODSolver : public LinSysSolver {
    typedef LinSysSolver Base; // The class
public:                // Access specifier
    cholmod_common cm;
    cholmod_sparse *A;
    cholmod_factor *L;
    cholmod_dense *b;

    cholmod_dense *x_solve;

    void *Ai, *Ap, *Ax, *bx;


    ~CHOLMODSolver();
    CHOLMODSolver();

    void setMatrix(int *p, int *i, double *x, int A_N, int NNZ) override;
    void innerAnalyze_pattern(std::vector<int>& user_defined_perm) override;
    void innerFactorize(void) override;
    void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) override;
    void resetSolver() override;
    void save_factor(const std::string &file_name);
    virtual LinSysSolverType type() const override;

    void cholmod_clean_memory();

};

}

#endif
