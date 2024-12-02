#pragma once

/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */

#include <Eigen/Dense>
#include <iostream>
#include <sstream>

namespace rxmesh {

template <typename PassiveT, int k, bool WithHessian>
struct Scalar;


/**
 * @brief  Mapping local query index to global index in the gradient/hessian.
 * The local query index (index_in_iter) is the index used for indexing the
 * iterator returned from query operations (e.g., query.dispatch). The local_id
 * is the index of the local variable (e.g., o for u and 1 for v in a
 * parameterization problem where we optimize the UV coordinates). The returned
 * index is an index that can be used to index the gradient/hessian, i.e., a 1D
 * index for an array (e.g., gradient) or matrix (e.g., hessian) that holds all
 * the optimization variables as 1D indexed variables.
 */
template <int VariableDim>
__device__ __inline__ int index_mapping(uint16_t index_in_iter,
                                        int      variable_local_id)
{
    assert(variable_local_id < VariableDim);
    return index_in_iter * VariableDim + variable_local_id;
}


/**
 * @brief Assemble matrix from column vectors.
 */
template <typename Derived>
__device__ __host__ __inline__ auto col_mat(
    const Eigen::MatrixBase<Derived>& _v0,
    const Eigen::MatrixBase<Derived>& _v1)
{
    using T = typename Derived::Scalar;
    Eigen::Matrix<T, Derived::RowsAtCompileTime, 2 * Derived::ColsAtCompileTime>
        M;

    M << _v0, _v1;

    return M;
}

/**
 * @brief Assemble matrix from column vectors.
 */
template <typename Derived>
__device__ __host__ __inline__ auto col_mat(
    const Eigen::MatrixBase<Derived>& _v0,
    const Eigen::MatrixBase<Derived>& _v1,
    const Eigen::MatrixBase<Derived>& _v2)
{
    using T = typename Derived::Scalar;
    Eigen::Matrix<T, Derived::RowsAtCompileTime, 3 * Derived::ColsAtCompileTime>
        M;

    M << _v0, _v1, _v2;

    return M;
}

/**
 * @brief return true if the element wise difference between A and B is less
 * than certain threshold
 */
template <typename DerivedA, typename DerivedB, typename T>
__device__ __host__ __inline__ bool is_same_matrix(
    const Eigen::MatrixBase<DerivedA>& A,
    const Eigen::MatrixBase<DerivedB>& B,
    const T                            eps)
{

    const auto& A_ref = A;
    const auto& B_ref = B;
    if (A_ref.rows() != B_ref.rows()) {
        return false;
    }
    if (A_ref.cols() != B_ref.cols()) {
        return false;
    }
    for (Eigen::Index i = 0; i < A_ref.rows(); ++i) {
        for (Eigen::Index j = 0; j < A_ref.cols(); ++j)
            if (std::abs(A_ref(i, j) - B_ref(i, j)) > eps) {
                return false;
            }
    }

    return true;
}

/**
 * @brief check if a variable is +/- infinity. Works both on host and device
 */
template <typename T>
__device__ __host__ __inline__ bool is_inf(const T& A)
{

#ifdef __CUDA_ARCH__
    return ::isinf(A);
#else
    return std::isinf(A);
#endif
}

/**
 * @brief check if a variable is nan. Works both on host and device
 */
template <typename T>
__device__ __host__ __inline__ bool is_nan(const T& A)
{

#ifdef __CUDA_ARCH__
    return ::isnan(A);
#else
    return std::isnan(A);
#endif
}


/**
 * @brief check if a variable is finite. Works both on host and device
 */
template <typename T>
__device__ __host__ __inline__ bool is_finite(const T& A)
{

#ifdef __CUDA_ARCH__
    return ::isfinite(A);
#else
    return std::isfinite(A);
#endif
}

/**
 * @brief return true if all elements in the matrix are finite. The type should
 * be float or double
 */
template <typename Derived>
__device__ __host__ __inline__ bool is_finite_mat(
    const Eigen::MatrixBase<Derived>& A)
{
    const auto& A_ref = A;
    for (Eigen::Index i = 0; i < A_ref.rows(); ++i) {
        for (Eigen::Index j = 0; j < A_ref.cols(); ++j) {
            if (!is_finite(A_ref(i, j))) {
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief check if input Eigen matrix is symmetric
 */
template <typename T, typename S>
__host__ __device__ __inline__ bool is_sym(const T& A, S eps = double(1e-6))
{
    const auto& A_ref = A;
    if (((A_ref) - (A_ref).transpose()).array().abs().maxCoeff() > eps) {
        return false;
    }
    return true;
}


/**
 * @brief  NAN-check for Scalar type
 */
template <int k, typename PassiveT, bool WithHessian>
__device__ __host__ __inline__ bool is_finite_scalar(
    const Scalar<PassiveT, k, WithHessian>& s)
{
    if (!is_finite(s.val) || !is_finite_mat(s.grad) || !is_finite_mat(s.Hess)) {
        return false;
    }
    return true;
}


/**
 * @brief  Include this file for a fallback no-op version of to_passive(...)
 * without needing to include scalar.h
 */
template <typename PassiveT>
__host__ __device__ const PassiveT& to_passive(const PassiveT& a)
{
    return a;
}


/**
 * @brief Additional passive-type functions for which scalar.h offers active
 * overloads
 */
template <typename PassiveT>
__host__ __device__ const PassiveT sqr(const PassiveT& a)
{
    return a * a;
}

}  // namespace rxmesh
