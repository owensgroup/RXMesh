#pragma once

#include "rxmesh/diff/diff_scalar_problem.h"

namespace rxmesh {

template <typename T, int VariableDim, typename OptVarHandleT>
struct GradientDescent
{

    using DiffProblemT =
        DiffScalarProblem<T, VariableDim, OptVarHandleT, false>;
    using HessMatT  = typename DiffProblemT::HessMatT;
    using DenseMatT = typename DiffProblemT::DenseMatT;

    DiffProblemT& problem;
    double        m_learning_rate;

    GradientDescent(DiffProblemT& p, double lr)
        : problem(p), m_learning_rate(lr)
    {
    }


    inline void take_step(cudaStream_t stream = NULL)
    {
        auto&  grad    = problem.grad;
        auto&  opt_var = *(problem.opt_var);
        double lr      = m_learning_rate;

        problem.rx.template for_each<OptVarHandleT>(
            DEVICE, [grad, opt_var, lr] __device__(const OptVarHandleT& h) {
                for (int i = 0; i < VariableDim; ++i) {
                    opt_var(h, i) -= lr * grad(h, i);
                }
            });
    }
};

}  // namespace rxmesh