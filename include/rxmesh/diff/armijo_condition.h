#pragma once

#include "rxmesh/matrix/dense_matrix.h"

namespace rxmesh {

/**
 * @brief armijo/wolfe condition used line search 
 */
template <typename T>
inline bool armijo_condition(const T                                f_curr,
                             const T                                f_new,
                             const T                                s,
                             const DenseMatrix<T, Eigen::RowMajor>& dir,
                             const DenseMatrix<T, Eigen::RowMajor>& grad,
                             const T armijo_const)
{
    //TODO we don't need to compute the dir.dot(grad) every time 
    return f_new <= f_curr + armijo_const * s * dir.dot(grad);
}

}  // namespace rxmesh