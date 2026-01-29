#include "rxmesh/hash_functions.h"

#include <type_traits>

namespace rxmesh {
uint32_t __host__ __device__
MurmurHash3_32::operator()(Key const& key) const noexcept
{
    constexpr int        len     = sizeof(Key);
    const uint8_t* const data    = (const uint8_t*)&key;
    constexpr int        nblocks = len / 4;

    uint32_t              h1     = m_seed;
    constexpr uint32_t    c1     = 0xcc9e2d51;
    constexpr uint32_t    c2     = 0x1b873593;
    const uint32_t* const blocks = (const uint32_t*)(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i];
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }
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
    h1 ^= len;
    h1 = fmix32(h1);
    return h1;
}

constexpr __host__ __device__ uint32_t
MurmurHash3_32::rotl32(uint32_t x, int8_t r) const noexcept
{
    return (x << r) | (x >> (32 - r));
}

constexpr __host__ __device__ uint32_t
MurmurHash3_32::fmix32(uint32_t h) const noexcept
{
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

constexpr uint16_t __host__ __device__
hash16_xm2::operator()(uint16_t key) const noexcept
{
    key ^= key >> 8;
    key *= 0x88b5U;
    key ^= key >> 7;
    key *= 0xdb2dU;
    key ^= key >> 9;
    return key;
}

constexpr uint32_t __host__ __device__
Hash64To32XOR::operator()(const uint64_t k) const
{
    return (uint32_t)(k ^ (k >> 32) ^ seed);
}

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

    if constexpr (std::is_same_v<HashT, Hash64To32XOR>) {
        uint32_t x = rng();
        return Hash64To32XOR(x);
    }
}

// Explicit instantiations 
template universal_hash initialize_hf<universal_hash, MarsRng32>(MarsRng32&);
template MurmurHash3_32 initialize_hf<MurmurHash3_32, MarsRng32>(MarsRng32&);
template hash16_xm2     initialize_hf<hash16_xm2, MarsRng32>(MarsRng32&);
template Hash64To32XOR  initialize_hf<Hash64To32XOR, MarsRng32>(MarsRng32&);

}  // namespace rxmesh
