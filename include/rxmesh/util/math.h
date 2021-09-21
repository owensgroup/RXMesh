#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <numeric>
#include <vector>

namespace RXMESH {
// 180.0/PI (multiply this by the radian angle to convert to degree)
constexpr float RadToDeg = 57.295779513078550;

constexpr float PIf = 3.1415927f;


/**
 * l2_norm()
 * TODO remove
 */
template <typename T>
__host__ __device__ __forceinline__ T l2_norm(const T ax0,
                                              const T ax1,
                                              const T ax2,
                                              const T bx0,
                                              const T bx1,
                                              const T bx2)
{
    // compute sqrt((xa0-xb0)*(xa0-xb0) + (xa1-xb1)*(xa1-xb1) +
    //(xa2-xb2)*(xa2-xb2))
    return sqrt(l2_norm_sq(ax0, ax1, ax2, bx0, bx1, bx2));
}


/**
 * l2_norm_sq()
 * TODO remove
 */
template <typename T>
__host__ __device__ __forceinline__ T l2_norm_sq(const T ax0,
                                                 const T ax1,
                                                 const T ax2,
                                                 const T bx0,
                                                 const T bx1,
                                                 const T bx2)
{
    // compute (xa0-xb0)*(xa0-xb0) + (xa1-xb1)*(xa1-xb1) + (xa2-xb2)*(xa2-xb2)
    T x0 = ax0 - bx0;
    T x1 = ax1 - bx1;
    T x2 = ax2 - bx2;
    return x0 * x0 + x1 * x1 + x2 * x2;
}

/**
 * vector_length()
 * TODO remove
 */
__device__ __host__ __forceinline__ float vector_length(const float x,
                                                        const float y,
                                                        const float z)
{
    return sqrtf(x * x + y * y + z * z);
}


/**
 * vector_length()
 * TODO remove
 */
__device__ __host__ __forceinline__ double vector_length(const double x,
                                                         const double y,
                                                         const double z)
{
    return sqrt(x * x + y * y + z * z);
}

/**
 * cross_product()
 * TODO remove
 */
template <typename T>
__host__ __device__ __forceinline__ void
cross_product(T xv1, T yv1, T zv1, T xv2, T yv2, T zv2, T& xx, T& yy, T& zz)
{
    xx = yv1 * zv2 - zv1 * yv2;
    yy = zv1 * xv2 - xv1 * zv2;
    zz = xv1 * yv2 - yv1 * xv2;
}

/**
 * vector_normal()
 * TODO remove
 */
template <typename T>
__device__ __host__ __forceinline__ T vector_normal(const T& vector_x,
                                                    const T& vector_y,
                                                    const T& vector_z)
{
    return vector_length(vector_x, vector_y, vector_z);
}

/**
 * normalize_vector()
 * TODO remove
 */
template <typename T>
__device__ __host__ __forceinline__ void normalize_vector(T& vector_x,
                                                          T& vector_y,
                                                          T& vector_z)
{
    T nn = vector_normal(vector_x, vector_y, vector_z);
    if (nn == 0) {
        vector_x = vector_y = vector_z = 0;
    } else {
        nn = 1 / nn;
        vector_x *= nn;
        vector_y *= nn;
        vector_z *= nn;
    }
}

/**
 * round_up_multiple()
 */
template <typename T>
__host__ __device__ __forceinline__ T round_up_multiple(const T numToRound,
                                                        const T multiple)
{

    // https://stackoverflow.com/a/3407254/1608232
    // rounding numToRound to the closest number multiple of multiple
    // this code meant only for +ve int. for -ve, check the reference above
    if (multiple == 0) {
        return numToRound;
    }

    const T remainder = numToRound % multiple;
    if (remainder == 0) {
        return numToRound;
    }
    return numToRound + multiple - remainder;
}

/**
 * round_to_next_power_two()
 */
__host__ __device__ __forceinline__ uint32_t
round_to_next_power_two(const uint32_t numToRound)
{

    // https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    uint32_t res = numToRound;
    if (res == 0) {
        return 1;
    }
    res--;
    res |= res >> 1;
    res |= res >> 2;
    res |= res >> 4;
    res |= res >> 8;
    res |= res >> 16;
    res++;
    return res;
}

/**
 * dot()
 * TODO remove
 */
template <typename T>
T dot(const std::vector<T>& u, const std::vector<T>& v)
{
    return std::inner_product(std::begin(u), std::end(u), std::begin(v), 0.0);
}

/**
 * scale()
 * TODO remove
 */
template <typename T>
void scale(std::vector<T>& v, const T factor)
{
    std::transform(
        v.begin(),
        v.end(),
        v.begin(),
        std::bind(std::multiplies<T>(), std::placeholders::_1, factor));
}

/**
 * axpy()
 */
template <typename T>
void axpy(const std::vector<T>& x,
          const T               alpha,
          const T               beta,
          std::vector<T>&       y)
{
    // y = alpha*x + beta*y
    for (uint32_t i = 0; i < x.size(); ++i) {
        y[i] *= beta;
        y[i] += alpha * x[i];
    }
}

}  // namespace RXMESH