// This file has been heavily modified from https://github.com/owensgroup/BGHT
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
#include <cooperative_groups.h>

#include <algorithm>
#include <random>

#include "rxmesh/hash_functions.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/prime_numbers.h"

#include <thrust/sort.h>
#include <thrust/unique.h>


namespace rxmesh {

struct Edge
{
    uint64_t m_key;

    __host__ __device__ Edge() : m_key(INVALID64) {};
    __host__ __device__ Edge(uint64_t k) : m_key(k) {};
    Edge(const Edge& other)      = default;
    Edge(Edge&&)                 = default;
    Edge& operator=(const Edge&) = default;
    Edge& operator=(Edge&&)      = default;
    ~Edge()                      = default;

    __device__ __host__ Edge(uint32_t l, uint32_t r)
        : m_key(static_cast<uint64_t>(std::min(l, r)) << 32 | std::max(l, r))
    {
    }

    constexpr __device__ __host__ uint64_t key() const
    {
        return m_key;
    }

    constexpr __device__ __host__ std::pair<int, int> unpack() const
    {
        int a = static_cast<int>(m_key >> 32);
        int b = static_cast<int>(m_key & 0xFFFFFFFF);
        return std::make_pair(a, b);
    }

    constexpr __device__ __host__ bool is_sentinel() const
    {
        return m_key == INVALID64;
    }

    __host__ __device__ friend bool operator==(const Edge& a, const Edge& b)
    {
        return a.m_key == b.m_key;
    }

    __host__ __device__ friend bool operator<(const Edge& a, const Edge& b)
    {
        return a.m_key < b.m_key;
    }

    __host__ __device__ friend bool operator!=(const Edge& a, const Edge& b)
    {

        return a.m_key != b.m_key;
    }
};


template <typename T = Edge>
struct GPUStorage
{
    __device__ __host__ GPUStorage() : m_storage(nullptr), m_capacity(0)
    {
    }
    GPUStorage(const GPUStorage& other)      = default;
    GPUStorage(GPUStorage&&)                 = default;
    GPUStorage& operator=(const GPUStorage&) = default;
    GPUStorage& operator=(GPUStorage&&)      = default;
    ~GPUStorage()                            = default;


    explicit GPUStorage(const uint32_t capacity) : m_capacity(capacity)
    {
        CUDA_ERROR(cudaMalloc((void**)&m_storage, num_bytes()));
        CUDA_ERROR(cudaMalloc((void**)&m_count, sizeof(int)));
        clear();
    }

    __host__ __device__ __inline__ uint32_t get_capacity() const
    {
        return m_capacity;
    }

    __host__ void clear()
    {
        CUDA_ERROR(cudaMemset(m_count, 0, sizeof(int)));
        CUDA_ERROR(cudaMemset(m_storage, INVALID64, num_bytes()));
    }
    __host__ void free()
    {
        GPU_FREE(m_storage);
        GPU_FREE(m_count);
    }

    __host__ __device__ __inline__ uint32_t num_bytes() const
    {
        return m_capacity * sizeof(T);
    }


    template <typename FunT>
    __host__ void for_each(FunT func)
    {
        constexpr uint32_t blockThreads = 256;

        T*   storage = m_storage;
        int* count   = m_count;

        uint32_t blocks = DIVIDE_UP(m_capacity, blockThreads);


        for_each_item<<<blocks, blockThreads>>>(
            m_capacity, [storage, count, func] __device__(int i) mutable {
                if (i < count[0]) {
                    func(storage[i]);
                }
            });
    }

    __device__ __inline__ bool insert(T item)
    {
        int prv = ::atomicAdd(m_count, 1);
        if (prv < m_capacity) {
            m_storage[prv] = item;
            return true;
        }
        ::atomicAdd(m_count, -1);
        return false;
    }

    __host__ void uniquify()
    {
        // https://nvidia.github.io/cccl/thrust/api/function_group__stream__compaction_1ga0981eb0b6034017ef622075d6612f68a.html

        int h_count = 0;

        CUDA_ERROR(
            cudaMemcpy(&h_count, m_count, sizeof(int), cudaMemcpyDeviceToHost));

        thrust::sort(thrust::device, m_storage, m_storage + h_count);

        T* new_end =
            thrust::unique(thrust::device, m_storage, m_storage + h_count);

        h_count = new_end - m_storage;

        CUDA_ERROR(
            cudaMemcpy(m_count, &h_count, sizeof(int), cudaMemcpyHostToDevice));
    }

    __host__ int count()
    {
        int h_count = 0;

        CUDA_ERROR(
            cudaMemcpy(&h_count, m_count, sizeof(int), cudaMemcpyDeviceToHost));

        return h_count;
    }

    T*       m_storage;
    int*     m_count;
    uint32_t m_capacity;
};

#if 0
template <typename T = Edge>
struct GPUHashTable
{
    using HashT = Hash64To32XOR;

    static constexpr uint32_t stash_size = 128;

    __device__ __host__ GPUHashTable()
        : m_table(nullptr),
          m_stash(nullptr),
          m_capacity(0),
          m_max_cuckoo_chains(0)
    {
    }
    GPUHashTable(const GPUHashTable& other)      = default;
    GPUHashTable(GPUHashTable&&)                 = default;
    GPUHashTable& operator=(const GPUHashTable&) = default;
    GPUHashTable& operator=(GPUHashTable&&)      = default;
    ~GPUHashTable()                              = default;

    /**
     * @brief Constructor using the hash table capacity.This is used as
     * allocation size
     */
    explicit GPUHashTable(const uint32_t capacity)
        : m_capacity(std::max(capacity, uint32_t(2)))
    {

        m_capacity = find_next_prime_number(m_capacity);

        CUDA_ERROR(cudaMalloc((void**)&m_table, num_bytes()));
        CUDA_ERROR(cudaMalloc((void**)&m_stash, stash_size * sizeof(T)));


        clear();

        // maximum number of cuckoo chains
        double lg_input_size = (float)(log((double)m_capacity) / log(2.0));
        const unsigned max_iter_const = 7;
        m_max_cuckoo_chains =
            static_cast<uint32_t>(max_iter_const * lg_input_size);


        // std::mt19937 rng(2);
        MarsRng32 rng;
        randomize_hash_functions(rng);
    }


    /**
     * @brief Get the hash table capacity
     */
    __host__ __device__ __inline__ uint32_t get_capacity() const
    {
        return m_capacity;
    }

    /**
     * @brief Reset the hash table to sentinel key-value (tombstone). This API
     * is for the host only
     */
    __host__ void clear()
    {
        uint32_t len     = std::max(m_capacity, stash_size);
        uint32_t threads = 256;
        uint32_t blocks  = DIVIDE_UP(len, threads);

        // it is weird that cuda extended lambda function can not capture member
        // variables and end up with a slight error
        uint32_t capacity = m_capacity;
        uint32_t st       = stash_size;

        T* table = m_table;
        T* stash = m_stash;

        for_each_item<<<blocks, threads>>>(
            len, [capacity, st, table, stash] __device__(uint32_t i) mutable {
                if (i < capacity) {
                    table[i].m_key = INVALID64;
                }
                if (i < st) {
                    stash[i].m_key = INVALID64;
                }
            });
    }


    /**
     * @brief Free the GPU allocation
     */
    __host__ void free()
    {
        GPU_FREE(m_table);
        GPU_FREE(m_stash);
    }

    /**
     * @brief size of the hash table in bytes which may be needed for allocation
     */
    __host__ __device__ __inline__ uint32_t num_bytes() const
    {
        return m_capacity * sizeof(T);
    }

    /**
     * @brief (Re)Generate new hashers
     */
    template <typename RNG>
    void randomize_hash_functions(RNG& rng)
    {
        m_hasher0 = initialize_hf<HashT>(rng);
        m_hasher1 = initialize_hf<HashT>(rng);
        m_hasher2 = initialize_hf<HashT>(rng);
        m_hasher3 = initialize_hf<HashT>(rng);
    }

    template <typename FunT>
    __host__ void for_each(FunT func)
    {
        constexpr uint32_t blockThreads = 256;

        uint32_t capacity = m_capacity;
        uint32_t st       = stash_size;

        T* table = m_table;
        T* stash = m_stash;

        uint32_t len    = capacity;
        uint32_t blocks = DIVIDE_UP(len, blockThreads);


        for_each_item<<<blocks, blockThreads>>>(
            len, [table, func] __device__(uint32_t i) mutable {
                if (!table[i].is_sentinel()) {
                    func(table[i]);
                }
            });

        len    = st;
        blocks = DIVIDE_UP(len, blockThreads);

        for_each_item<<<blocks, blockThreads>>>(
            len, [stash, func] __device__(uint32_t i) mutable {
                if (!stash[i].is_sentinel()) {
                    func(stash[i]);
                }
            });
    }

    /**
     * @brief Insert new pair in the table. This function can be called from
     * host (not thread safe) or from the device (by a single thread). The table
     * itself is part of the input in case it was loaded in shared memory.
     * @param pair to be inserted in the hash table
     * @param table pointer to the hash table (could shared memory on the
     * device)
     * @return true if the insertion succeeded and false otherwise
     */
    __device__ __inline__ bool insert(T pair)
    {
        auto bucket_id = m_hasher0(pair.key()) % m_capacity;

        uint32_t cuckoo_counter = 0;

        const auto input_key = pair.key();

        do {

#ifdef __CUDA_ARCH__

            __threadfence();
            pair.m_key =
                ::atomicExch((uint64_t*)m_table + bucket_id, pair.m_key);
            __threadfence();
#endif

            // compare against initial key to avoid duplicated
            // i.e., if we are inserting a pair such that its key already
            // exists, this comparison would allow updating the pair
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

        for (uint8_t i = 0; i < stash_size; ++i) {
            T prv;

#ifdef __CUDA_ARCH__
            __threadfence();
            prv.m_key = ::atomicCAS(reinterpret_cast<uint64_t*>(m_stash + i),
                                    INVALID64,
                                    pair.m_key);
            __threadfence();

#endif

            if (prv.is_sentinel() || prv.key() == input_key) {
                return true;
            }
        }

        return false;
    }


    T*       m_table;
    T*       m_stash;
    HashT    m_hasher0;
    HashT    m_hasher1;
    HashT    m_hasher2;
    HashT    m_hasher3;
    uint32_t m_capacity;
    uint32_t m_max_cuckoo_chains;
};
#endif
}  // namespace rxmesh
