#include "rxmesh/lp_hashtable.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "rxmesh/hash_functions.h"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/prime_numbers.h"

namespace rxmesh {

LPHashTable::LPHashTable(const uint16_t capacity, bool is_on_device)
    : m_capacity(std::max(capacity, uint16_t(2))), m_is_on_device(is_on_device)
{
    m_capacity = find_next_prime_number(m_capacity);
    if (m_is_on_device) {
        CUDA_ERROR(cudaMalloc((void**)&m_table, num_bytes()));
        CUDA_ERROR(cudaMalloc((void**)&m_stash, stash_size * sizeof(LPPair)));

    } else {
        m_table = (LPPair*)malloc(num_bytes());
        m_stash = (LPPair*)malloc(stash_size * sizeof(LPPair));
        for (uint8_t i = 0; i < stash_size; ++i) {
            m_stash[i] = LPPair::sentinel_pair();
        }
    }

    clear();
    double         lg_input_size  = (float)(log((double)m_capacity) / log(2.0));
    const unsigned max_iter_const = 7;
    m_max_cuckoo_chains = static_cast<uint16_t>(max_iter_const * lg_input_size);

    MarsRng32 rng;
    randomize_hash_functions(rng);
}

__host__ void LPHashTable::clear()
{
    if (m_is_on_device) {
        CUDA_ERROR(cudaMemset(m_table, INVALID8, num_bytes()));
        CUDA_ERROR(cudaMemset(m_stash, INVALID8, stash_size * sizeof(LPPair)));
    } else {
        std::fill_n(m_table, m_capacity, LPPair());
        std::fill_n(m_stash, stash_size, LPPair());
    }
}

__host__ void LPHashTable::free()
{
    if (m_is_on_device) {
        GPU_FREE(m_table);
        GPU_FREE(m_stash);
    } else {
        ::free(m_table);
        ::free(m_stash);
    }
}


template <typename RNG>
void LPHashTable::randomize_hash_functions(RNG& rng)
{
    m_hasher0 = initialize_hf<HashT>(rng);
    m_hasher1 = initialize_hf<HashT>(rng);
    m_hasher2 = initialize_hf<HashT>(rng);
    m_hasher3 = initialize_hf<HashT>(rng);
}


__host__ void LPHashTable::move(const LPHashTable src)
{
    const size_t stash_num_bytes = LPHashTable::stash_size * sizeof(LPPair);
    if (src.m_is_on_device && m_is_on_device) {
        CUDA_ERROR(cudaMemcpy(
            m_table, src.m_table, num_bytes(), cudaMemcpyDeviceToDevice));
        CUDA_ERROR(cudaMemcpy(
            m_stash, src.m_stash, stash_num_bytes, cudaMemcpyDeviceToDevice));
    }

    if (!src.m_is_on_device && !m_is_on_device) {
        std::memcpy(m_table, src.m_table, num_bytes());
        std::memcpy(m_stash, src.m_stash, stash_num_bytes);
    }

    if (src.m_is_on_device && !m_is_on_device) {
        CUDA_ERROR(cudaMemcpy(
            m_table, src.m_table, num_bytes(), cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(
            m_stash, src.m_stash, stash_num_bytes, cudaMemcpyDeviceToHost));
    }

    if (!src.m_is_on_device && m_is_on_device) {
        CUDA_ERROR(cudaMemcpy(
            m_table, src.m_table, num_bytes(), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(
            m_stash, src.m_stash, stash_num_bytes, cudaMemcpyHostToDevice));
    }
}


// Explicit instantiations: blockSize/blockThreads = 128, 256, 320, 384, 512,
// 768, 1024
#define LPHASHTABLE_CLEAR_INSTANTIATE(blockThreads) \
    template __device__ void LPHashTable::clear<blockThreads>();

#define LPHASHTABLE_WRITE_TO_GLOBAL_INSTANTIATE(blockSize)                   \
    template __device__ void LPHashTable::write_to_global_memory<blockSize>( \
        const LPPair*, const LPPair*);

LPHASHTABLE_CLEAR_INSTANTIATE(128)
LPHASHTABLE_CLEAR_INSTANTIATE(256)
LPHASHTABLE_CLEAR_INSTANTIATE(320)
LPHASHTABLE_CLEAR_INSTANTIATE(384)
LPHASHTABLE_CLEAR_INSTANTIATE(512)
LPHASHTABLE_CLEAR_INSTANTIATE(768)
LPHASHTABLE_CLEAR_INSTANTIATE(1024)

LPHASHTABLE_WRITE_TO_GLOBAL_INSTANTIATE(128)
LPHASHTABLE_WRITE_TO_GLOBAL_INSTANTIATE(256)
LPHASHTABLE_WRITE_TO_GLOBAL_INSTANTIATE(320)
LPHASHTABLE_WRITE_TO_GLOBAL_INSTANTIATE(384)
LPHASHTABLE_WRITE_TO_GLOBAL_INSTANTIATE(512)
LPHASHTABLE_WRITE_TO_GLOBAL_INSTANTIATE(768)
LPHASHTABLE_WRITE_TO_GLOBAL_INSTANTIATE(1024)

#undef LPHASHTABLE_CLEAR_INSTANTIATE
#undef LPHASHTABLE_WRITE_TO_GLOBAL_INSTANTIATE

// randomize_hash_functions instantiated with MarsRng32
template void LPHashTable::randomize_hash_functions<MarsRng32>(MarsRng32&);

}  // namespace rxmesh
