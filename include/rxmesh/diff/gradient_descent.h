#pragma once

#include "rxmesh/diff/diff_scalar_problem.h"

namespace rxmesh {

template <typename T, int VariableDim, typename ObjHandleT>
struct GradientDescent
{

    using DiffProblemT = DiffScalarProblem<T, VariableDim, ObjHandleT, false>;
    using HessMatT     = typename DiffProblemT::HessMatT;
    using DenseMatT    = typename DiffProblemT::DenseMatT;

    DiffProblemT& problem;
    double        m_learning_rate;

    GradientDescent(DiffProblemT& p, double lr)
        : problem(p), m_learning_rate(lr)
    {
    }


    inline void take_step(cudaStream_t stream = NULL)
    {
        auto&  grad = problem.grad;
        auto&  obj  = *(problem.objective);
        double lr   = m_learning_rate;

        problem.rx.for_each<ObjHandleT>(
            DEVICE, [grad, obj, lr] __device__(const ObjHandleT& h) {
                for (int i = 0; i < VariableDim; ++i) {
                    obj(h, i) -= lr * grad(h, i);
                }
            });
    }
};

}  // namespace rxmesh