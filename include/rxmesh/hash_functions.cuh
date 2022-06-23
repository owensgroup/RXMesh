// This file is taken from https://github.com/owensgroup/BGHT
/*
 *   Copyright 2021 The Regents of the University of California, Davis
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace rxmesh {
struct mars_rng_32
{
    uint32_t y;
    __host__ __device__ constexpr mars_rng_32() : y(2463534242)
    {
    }
    constexpr uint32_t __host__ __device__ operator()()
    {
        y ^= (y << 13);
        y = (y >> 17);
        return (y ^= (y << 5));
    }
};

struct universal_hash
{
    __host__ __device__ constexpr universal_hash(uint32_t hash_x,
                                                 uint32_t hash_y)
        : m_hash_x(hash_x), m_hash_y(hash_y)
    {
    }

    constexpr uint32_t __host__ __device__ __inline__ operator()(
        const uint16_t k) const
    {        
        return (((m_hash_x ^ k) + m_hash_y) % prime_divisor);
    }

    universal_hash(const universal_hash&) = default;
    universal_hash() : m_hash_x(0u), m_hash_y(0u){};
    universal_hash(universal_hash&&) = default;
    universal_hash& operator=(universal_hash const&) = default;
    universal_hash& operator=(universal_hash&&) = default;
    ~universal_hash()                           = default;

    static constexpr uint32_t prime_divisor = 2063u;

   private:
    uint32_t m_hash_x;
    uint32_t m_hash_y;
};

template <typename Hash, typename RNG>
Hash initialize_hf(RNG& rng)
{
    uint32_t x = rng() % Hash::prime_divisor;
    if (x < 1u) {
        x = 1;
    }
    uint32_t y = rng() % Hash::prime_divisor;
    return Hash(x, y);
}
}  // namespace rxmesh
