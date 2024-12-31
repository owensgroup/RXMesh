#pragma once

#include <limits>

namespace rxmesh {

template <typename HandleT, typename T>
using KeyValuePair = cub::KeyValuePair<HandleT, T>;

namespace detail {

template <typename HandleT, typename T>
struct ArgMaxOp
{
    constexpr T default_val() const
    {
        return std::numeric_limits<T>::lowest();
    }

    __device__ __forceinline__ KeyValuePair<HandleT, T> operator()(
        const KeyValuePair<HandleT, T>& a,
        const KeyValuePair<HandleT, T>& b) const
    {
        return (b.value > a.value) ? b : a;
    }
};


template <typename HandleT, typename T>
struct ArgMinOp
{
    constexpr T default_val() const
    {
        return std::numeric_limits<T>::max();
    }

    __device__ __forceinline__ KeyValuePair<HandleT, T> operator()(
        const KeyValuePair<HandleT, T>& a,
        const KeyValuePair<HandleT, T>& b) const
    {
        return (b.value < a.value) ? b : a;
    }
};


}  // namespace detail
}  // namespace rxmesh