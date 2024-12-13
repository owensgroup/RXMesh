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
struct MarsRng32
{
    static constexpr uint32_t init_val = 2463534242;

    uint32_t y;

    __host__ __device__ constexpr MarsRng32() : y(init_val)
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
    __host__ __device__ universal_hash() : m_hash_x(0u), m_hash_y(0u) {};
    universal_hash(universal_hash&&)                 = default;
    universal_hash& operator=(universal_hash const&) = default;
    universal_hash& operator=(universal_hash&&)      = default;
    ~universal_hash()                                = default;

    static constexpr uint32_t prime_divisor = 4294967291u;

   private:
    uint32_t m_hash_x;
    uint32_t m_hash_y;
};


// Taken from https://github.com/NVIDIA/cuCollections
/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


// MurmurHash3_32 implementation from
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
// Note - The x86 and x64 versions do _not_ produce the same results, as the
// algorithms are optimized for their respective platforms. You can still
// compile and run any of them on any platform, but your performance with the
// non-native version will be less than optimal.
struct MurmurHash3_32
{
    using Key = uint16_t;

    __host__ __device__ constexpr MurmurHash3_32() : m_seed(0)
    {
    }

    __host__ __device__ constexpr MurmurHash3_32(uint32_t seed) : m_seed(seed)
    {
    }

    MurmurHash3_32(const MurmurHash3_32&)            = default;
    MurmurHash3_32(MurmurHash3_32&&)                 = default;
    MurmurHash3_32& operator=(MurmurHash3_32 const&) = default;
    MurmurHash3_32& operator=(MurmurHash3_32&&)      = default;
    ~MurmurHash3_32()                                = default;

    uint32_t __host__ __device__ operator()(Key const& key) const noexcept
    {
        constexpr int        len     = sizeof(Key);
        const uint8_t* const data    = (const uint8_t*)&key;
        constexpr int        nblocks = len / 4;

        uint32_t           h1 = m_seed;
        constexpr uint32_t c1 = 0xcc9e2d51;
        constexpr uint32_t c2 = 0x1b873593;
        //----------
        // body
        const uint32_t* const blocks = (const uint32_t*)(data + nblocks * 4);
        for (int i = -nblocks; i; i++) {
            uint32_t k1 = blocks[i];  // getblock32(blocks,i);
            k1 *= c1;
            k1 = rotl32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
            h1 = rotl32(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;
        }
        //----------
        // tail
        const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
        uint32_t       k1   = 0;
        switch (len & 3) {
            case 3:
                k1 ^= tail[2] << 16;
            case 2:
                k1 ^= tail[1] << 8;
            case 1:
                k1 ^= tail[0];
                k1 *= c1;
                k1 = rotl32(k1, 15);
                k1 *= c2;
                h1 ^= k1;
        };
        //----------
        // finalization
        h1 ^= len;
        h1 = fmix32(h1);
        return h1;
    }

   private:
    constexpr __host__ __device__ uint32_t rotl32(uint32_t x,
                                                  int8_t   r) const noexcept
    {
        return (x << r) | (x >> (32 - r));
    }

    constexpr __host__ __device__ uint32_t fmix32(uint32_t h) const noexcept
    {
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }
    uint32_t m_seed;
};


// Taken from https://github.com/skeeto/hash-prospector
struct hash16_xm2
{
    hash16_xm2()                             = default;
    hash16_xm2(const hash16_xm2&)            = default;
    hash16_xm2(hash16_xm2&&)                 = default;
    hash16_xm2& operator=(hash16_xm2 const&) = default;
    hash16_xm2& operator=(hash16_xm2&&)      = default;
    ~hash16_xm2()                            = default;

    constexpr uint16_t __host__ __device__
    operator()(uint16_t key) const noexcept
    {
        key ^= key >> 8;
        key *= 0x88b5U;
        key ^= key >> 7;
        key *= 0xdb2dU;
        key ^= key >> 9;
        return key;
    }
};

template <typename HashT, typename RNG>
HashT initialize_hf(RNG& rng)
{
    if constexpr (std::is_same_v<HashT, universal_hash>) {
        uint32_t x = rng() % universal_hash::prime_divisor;
        if (x < 1u) {
            x = 1;
        }
        uint32_t y = rng() % universal_hash::prime_divisor;
        return universal_hash(x, y);
    }

    if constexpr (std::is_same_v<HashT, MurmurHash3_32>) {
        uint32_t x = rng();
        if (x < 1u) {
            x = 1;
        }
        return MurmurHash3_32(x);
    }

    if constexpr (std::is_same_v<HashT, hash16_xm2>) {
        return hash16_xm2();
    }
}
}  // namespace rxmesh