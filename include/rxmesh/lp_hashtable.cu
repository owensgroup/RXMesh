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

__device__ __host__ LPHashTable::LPHashTable()
    : m_table(nullptr),
      m_stash(nullptr),
      m_capacity(0),
      m_max_cuckoo_chains(0),
      m_is_on_device(false)
{
}

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

__host__ __device__ uint16_t LPHashTable::get_capacity() const
{
    return m_capacity;
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

template <uint32_t blockThreads>
__device__ void LPHashTable::clear()
{
#ifdef __CUDA_ARCH__
    assert(m_is_on_device);
    for (uint32_t i = threadIdx.x; i < m_capacity; i += blockThreads) {
        m_table[i].m_pair = INVALID32;
    }
    for (uint32_t i = threadIdx.x; i < stash_size; i += blockThreads) {
        m_stash[i].m_pair = INVALID32;
    }
#endif
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

__host__ __device__ uint32_t LPHashTable::num_bytes() const
{
    return m_capacity * sizeof(LPPair);
}

template <typename RNG>
void LPHashTable::randomize_hash_functions(RNG& rng)
{
    m_hasher0 = initialize_hf<HashT>(rng);
    m_hasher1 = initialize_hf<HashT>(rng);
    m_hasher2 = initialize_hf<HashT>(rng);
    m_hasher3 = initialize_hf<HashT>(rng);
}

__host__ __device__ float LPHashTable::compute_load_factor(
    LPPair* s_table) const
{
    auto lf = [&]() {
        uint32_t m = 0;
        for (uint32_t i = 0; i < m_capacity; ++i) {
            if (s_table != nullptr) {
                if (!s_table[i].is_sentinel()) {
                    ++m;
                }
            } else {
                if (!m_table[i].is_sentinel()) {
                    ++m;
                }
            }
        }
        return m;
    };

    return static_cast<float>(lf()) / static_cast<float>(m_capacity);
}

__host__ __device__ float LPHashTable::compute_stash_load_factor(
    LPPair* s_stash) const
{
    auto lf = [&]() {
        uint32_t m = 0;
        for (uint32_t i = 0; i < stash_size; ++i) {
            if (s_stash != nullptr) {
                if (!s_stash[i].is_sentinel()) {
                    ++m;
                }
            } else {
                if (!m_stash[i].is_sentinel()) {
                    ++m;
                }
            }
        }
        return m;
    };

    return static_cast<float>(lf()) / static_cast<float>(stash_size);
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

__device__ void LPHashTable::load_in_shared_memory(LPPair* s_table,
                                                   bool    with_wait,
                                                   LPPair* s_stash) const
{
#ifdef __CUDA_ARCH__
    if (s_stash != nullptr) {
        detail::load_async(m_stash, stash_size, s_stash, false);
    }
    detail::load_async(m_table, m_capacity, s_table, with_wait);
#endif
}

template <uint32_t blockSize>
__device__ void LPHashTable::write_to_global_memory(const LPPair* s_table,
                                                    const LPPair* s_stash)
{
#ifdef __CUDA_ARCH__
    if (s_stash != nullptr) {
        detail::store<blockSize>(s_stash, stash_size, m_stash);
    }
    detail::store<blockSize>(s_table, m_capacity, m_table);
#endif
}

__host__ __device__ bool LPHashTable::insert(LPPair           pair,
                                             volatile LPPair* table,
                                             volatile LPPair* stash)
{
#ifndef __CUDA_ARCH__
    assert(stash == nullptr);
    assert(table == nullptr);
#endif

    auto     bucket_id      = m_hasher0(pair.key()) % m_capacity;
    uint16_t cuckoo_counter = 0;

    do {
        const auto input_key = pair.key();
#ifdef __CUDA_ARCH__
        if (table != nullptr) {
            pair.m_pair =
                ::atomicExch((uint32_t*)table + bucket_id, pair.m_pair);
        } else {
            pair.m_pair =
                ::atomicExch((uint32_t*)m_table + bucket_id, pair.m_pair);
        }
#else
        uint32_t temp = pair.m_pair;
        if (table != nullptr) {
            pair.m_pair             = table[bucket_id].m_pair;
            table[bucket_id].m_pair = temp;
        } else {
            pair.m_pair               = m_table[bucket_id].m_pair;
            m_table[bucket_id].m_pair = temp;
        }
#endif

        if (pair.is_sentinel() || pair.key() == input_key) {
            return true;
        } else {
            auto bucket0 = m_hasher0(pair.key()) % m_capacity;
            auto bucket1 = m_hasher1(pair.key()) % m_capacity;
            auto bucket2 = m_hasher2(pair.key()) % m_capacity;
            auto bucket3 = m_hasher3(pair.key()) % m_capacity;

            auto new_bucket_id = bucket0;
            if (bucket_id == bucket2) {
                new_bucket_id = bucket3;
            } else if (bucket_id == bucket1) {
                new_bucket_id = bucket2;
            } else if (bucket_id == bucket0) {
                new_bucket_id = bucket1;
            }

            bucket_id = new_bucket_id;
        }
        cuckoo_counter++;
    } while (cuckoo_counter < m_max_cuckoo_chains);

    const auto input_key = pair.key();

#ifdef __CUDA_ARCH__
    for (uint8_t i = 0; i < stash_size; ++i) {
        LPPair prv;
        if (stash != nullptr) {
            prv.m_pair =
                ::atomicCAS((uint32_t*)(stash + i), INVALID32, pair.m_pair);
        } else {
            prv.m_pair = ::atomicCAS(reinterpret_cast<uint32_t*>(m_stash + i),
                                     INVALID32,
                                     pair.m_pair);
        }
        if (prv.is_sentinel() || prv.key() == input_key) {
            return true;
        }
    }
#else
    assert(stash == nullptr);
    for (uint8_t i = 0; i < stash_size; ++i) {
        if (m_stash[i].is_sentinel() || m_stash[i].key() == input_key) {
            m_stash[i] = pair;
            return true;
        }
    }
#endif
    return false;
}

__host__ __device__ LPPair LPHashTable::find(const typename LPPair::KeyT key,
                                             const LPPair*               table,
                                             const LPPair* stash) const
{
    uint32_t bucket_id(0);
    bool     in_stash(false);
    return find(key, bucket_id, in_stash, table, stash);
}

__host__ __device__ void LPHashTable::replace(const LPPair new_pair)
{
    uint32_t bucket_id(0);
    bool     in_stash(false);
    LPPair   old_pair =
        find(new_pair.key(), bucket_id, in_stash, nullptr, nullptr);

    assert(!old_pair.is_sentinel());
    assert(new_pair.key() == old_pair.key());

    if (!in_stash) {
        m_table[bucket_id].m_pair = new_pair.m_pair;
    } else {
        m_stash[bucket_id].m_pair = new_pair.m_pair;
    }
}

__host__ __device__ void LPHashTable::remove(const typename LPPair::KeyT key,
                                             LPPair*                     table,
                                             LPPair*                     stash)
{
    uint32_t bucket_id(0);
    bool     in_stash(false);
    find(key, bucket_id, in_stash, table, stash);

    if (!in_stash) {
        if (table != nullptr) {
            table[bucket_id].m_pair = INVALID32;
        } else {
            m_table[bucket_id].m_pair = INVALID32;
        }
    } else {
        if (stash != nullptr) {
            stash[bucket_id].m_pair = INVALID32;
        } else {
            m_stash[bucket_id].m_pair = INVALID32;
        }
    }
}

__host__ __device__ LPPair LPHashTable::find(const typename LPPair::KeyT key,
                                             uint32_t&     bucket_id,
                                             bool&         in_stash,
                                             const LPPair* table,
                                             const LPPair* stash) const
{
#ifndef __CUDA_ARCH__
    assert(stash == nullptr);
    assert(table == nullptr);
#endif

    constexpr int num_hfs = 4;
    in_stash              = false;
    bucket_id             = m_hasher0(key) % m_capacity;
    for (int hf = 0; hf < num_hfs; ++hf) {

        LPPair found;
        if (table != nullptr) {
            found = table[bucket_id];
        } else {
#ifdef __CUDA_ARCH__
            uint32_t* ptr = reinterpret_cast<uint32_t*>(m_table + bucket_id);
            found         = LPPair(atomic_read(ptr));
#else
            found = m_table[bucket_id];
#endif
        }

        if (found.key() == key) {
            return found;
        } else {
            if (hf == 0) {
                bucket_id = m_hasher1(key) % m_capacity;
            } else if (hf == 1) {
                bucket_id = m_hasher2(key) % m_capacity;
            } else if (hf == 2) {
                bucket_id = m_hasher3(key) % m_capacity;
            }
        }
    }

    for (bucket_id = 0; bucket_id < stash_size; ++bucket_id) {
        LPPair st_pair;
        if (stash != nullptr) {
            st_pair = stash[bucket_id];
        } else {
            st_pair = m_stash[bucket_id];
        }

        if (st_pair.key() == key) {
            in_stash = true;
            return st_pair;
        }
    }
    return LPPair::sentinel_pair();
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
