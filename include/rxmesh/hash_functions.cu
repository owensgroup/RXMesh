#include "rxmesh/hash_functions.h"

#include <type_traits>

namespace rxmesh {
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
