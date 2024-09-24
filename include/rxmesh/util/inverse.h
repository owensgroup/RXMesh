#pragma once

#include <cuda.h>
#include "rxmesh/types.h"

#include <Eigen/Dense>

namespace rxmesh {
/**
 * @brief since 3x3 matrix inverse in eigen is buggy on device (it results into
 * "unspecified launch failure"), we convert eigen matrix into glm, inverse it,
 * then convert it back to eigen matrix
 * @tparam T the floating point type of the matrix
 * @tparam n the size of the matrix, expected/tested sizes are 2,3, and 4.
 */

template <typename T, int n>
__device__ __host__ __inline__ Eigen::Matrix<T, n, n> inverse(
    const Eigen::Matrix<T, n, n>& in)
{
    glm::mat<n, n, T, glm::defaultp> glm_mat;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            glm_mat[i][j] = in(i, j);
        }
    }

    auto glm_inv = glm::inverse(glm_mat);

    Eigen::Matrix<T, n, n> eig_inv;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            eig_inv(i, j) = glm_inv[i][j];
        }
    }
    return eig_inv;
}


}  // namespace rxmesh