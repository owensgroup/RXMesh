
#pragma once

#ifdef USE_PARTH

#include <cholmod.h>
#include <Eigen/Eigen>
#include <vector>
#include "LinSysSolver.hpp"
#include "parth_solver.h"

namespace RXMESH_SOLVER {

class ParthSolver : public LinSysSolver {
    typedef LinSysSolver Base; // The class
public:                // Access specifier
    ~ParthSolver();
    ParthSolver();

    PARTH::ParthSolverAPI solver;
    void setMatrix(int *p, int *i, double *x, int A_N, int NNZ) override;
    void innerAnalyze_pattern(std::vector<int>& user_defined_perm) override;
    void innerFactorize(void) override;
    void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) override;
    void resetSolver() override;
    void save_factor(const std::string &filePath);
    virtual LinSysSolverType type() const override;

    void cholmod_clean_memory();

};

}

#endif
