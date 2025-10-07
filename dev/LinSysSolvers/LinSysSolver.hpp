//
//  LinSysSolver.hpp
//  IPC
//
//  Created by Minchen Li on 6/30/18.
//
#pragma once


#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cassert>

namespace RXMESH_SOLVER {

enum class LinSysSolverType
{
    PARTH_SOLVER,
    CPU_CHOLMOD,
    GPU_CUDSS,

};

class LinSysSolver
{
   public:

    int    L_NNZ    = 0;
    int    NNZ      = 0;
    int    N        = 0;
    double residual = 0;
    std::string ordering_name = "DEFAULT";

   public:
    virtual ~LinSysSolver(void) {};

    static LinSysSolver* create(const LinSysSolverType type);

    virtual LinSysSolverType type() const = 0;

   public:
    virtual void setMatrix(int*              p,
                           int*              i,
                           double*           x,
                           int               A_N,
                           int               NNZ) = 0;

    virtual void analyze_pattern(std::vector<int>& user_defined_perm)
    {
        innerAnalyze_pattern(user_defined_perm);
    }

    virtual void innerAnalyze_pattern(std::vector<int>& user_defined_perm) = 0;

    virtual void factorize(void)
    {
        innerFactorize();
    }

    virtual void innerFactorize(void) = 0;
    virtual int getFactorNNZ(void)
    {
        return L_NNZ;
    }

    virtual void solve(Eigen::VectorXd& rhs, Eigen::VectorXd& result)
    {
        innerSolve(rhs, result);
    }

    virtual void computeResidual(Eigen::SparseMatrix<double>& mtr, Eigen::VectorXd& sol, Eigen::VectorXd& rhs)
    {
        assert(mtr.rows() == mtr.cols());
        assert(rhs.rows() == mtr.rows());
        assert(sol.rows() == mtr.rows());
        residual = (rhs - mtr * sol).norm();
    }

    virtual void innerSolve(Eigen::VectorXd& rhs, Eigen::VectorXd& result) = 0;

    virtual void resetSolver() = 0;

   public:
    double getResidual(void)
    {
        return residual;
    }

    virtual void initVariables()
    {
        L_NNZ    = 0;
        NNZ      = 0;
        N        = 0;
        residual = 0;
    }
};

}  // namespace PARTH_SOLVER
