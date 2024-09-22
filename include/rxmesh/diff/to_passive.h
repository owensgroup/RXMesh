/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

namespace rxmesh {

/**
 * @brief check if input Eigen matrix is symmetric
 */
template <typename T, typename S>
__host__ __device__ bool is_sym(const T& A, S eps = double(1e-9))
{
    const auto& A_ref = A;
    if (((A_ref) - (A_ref).transpose()).array().abs().maxCoeff() > eps) {
        return false;
    }
    return true;
}

// Include this file for a fallback no-op version of to_passive(...)
// without needing to include scalar.h

template <typename PassiveT>
__host__ __device__ const PassiveT& to_passive(const PassiveT& a)
{
    return a;
}

}  // namespace rxmesh

// Additional passive-type functions for which scalar.h
// offers active overloads:

template <typename PassiveT>
__host__ __device__ const PassiveT sqr(const PassiveT& a)
{
    return a * a;
}
