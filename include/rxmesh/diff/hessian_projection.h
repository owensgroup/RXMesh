/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 *
 * Update: adding support for the Scalar type to run on both host and device
 * Author: Ahmed Mahmoud
 */
#pragma once
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace rxmesh {

constexpr double default_hessian_projection_eps = 1e-9;

/**
 * @brief Check if matrix is diagonally dominant and has positive diagonal
 * entries. This is a sufficient condition for positive-definiteness and can be
 * used as an early out to avoid eigen decomposition.
 */
template <int k, typename T>
__inline__ __host__ __device__ bool positive_diagonally_dominant(
    const Eigen::Matrix<T, k, k>& H,
    const T&                      eps)
{
    for (Eigen::Index i = 0; i < H.rows(); ++i) {
        T off_diag_abs_sum = 0.0;
        for (Eigen::Index j = 0; j < H.cols(); ++j) {
            if (i != j) {
                off_diag_abs_sum += std::abs(H(i, j));
            }
        }

        if (H(i, i) < off_diag_abs_sum + eps) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Project symmetric matrix to positive-definite matrix
 * via eigen decomposition.
 */
template <int k, typename T>
__inline__ __host__ __device__ void project_positive_definite(
    Eigen::Matrix<T, k, k>& H,
    const T eigenvalue_eps = (std::is_same_v<T, double> ? 1e-9 : 1e-6))
{

    if constexpr (k == 0) {
        return;
    } else {
        using MatT = Eigen::Matrix<T, k, k>;

        // Early out if sufficient condition is fulfilled
        if (positive_diagonally_dominant<k, T>(H, eigenvalue_eps)) {
            return;
        }

        // Compute eigen-decomposition (of symmetric matrix)
        Eigen::SelfAdjointEigenSolver<MatT> eig(H);

        //This method is buggy on the GPU. It results into all zero matrix!!
        //MatT D = eig.eigenvalues().asDiagonal();

        MatT D;
        D.setZero();

        for (Eigen::Index i = 0; i < H.rows(); ++i) {
            D(i, i) = eig.eigenvalues()[i];
        }
       
        // Clamp all eigenvalues to eps
        bool all_positive = true;
        for (Eigen::Index i = 0; i < H.rows(); ++i) {
            if (D(i, i) < eigenvalue_eps) {
                D(i, i)      = eigenvalue_eps;
                all_positive = false;
            }
        }

        // Do nothing if all eigenvalues were already at least eps
        if (all_positive) {
            return;
        }

        // Re-assemble matrix using clamped eigenvalues
        MatT eigvect = eig.eigenvectors();
        H = eigvect * D * eigvect.transpose();

        assert(is_finite_mat(H));
    }
}

}  // namespace rxmesh
