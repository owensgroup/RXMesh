#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace rxmesh {
// 180.0/PI (multiply this by the radian angle to convert to degree)
constexpr float RadToDeg = 57.295779513078550;

constexpr float PIf = 3.1415927f;

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

}  // namespace rxmesh