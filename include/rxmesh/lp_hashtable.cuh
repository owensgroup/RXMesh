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
#include <random>

#include "rxmesh/hash_functions.cuh"
#include "rxmesh/lp_pair.cuh"
#include "rxmesh/util/macros.h"

namespace rxmesh {

/**
 * @brief Hash table storing a patch's ribbon (not-owned) mesh elements where
 * the key is the local index within the patch and the value is the local index
 * within the owner patch along with an index in the PatchStash. The key and
 * value are combined in a single 32-bit using LBPair.
 */
struct LPHashTable
{
    /**
     * @brief Constructor using the hash table capacity.This is used as
     * allocation size
     */
    explicit LPHashTable(const uint16_t capacity, bool is_on_device)
        : m_capacity(std::max(capacity, uint16_t(1))),
          m_is_on_device(is_on_device)
    {
        if (m_is_on_device) {
            CUDA_ERROR(cudaMalloc((void**)&m_table, num_bytes()));
        } else {
            m_table = (LPPair*)malloc(num_bytes());
        }

        clear();

        // maximum number of cuckoo chains
        double lg_input_size = (float)(log((double)m_capacity) / log(2.0));
        const unsigned max_iter_const = 7;
        m_max_cuckoo_chains =
            static_cast<uint16_t>(max_iter_const * lg_input_size);

        std::mt19937 rng(2);
        randomize_hash_functions(rng);
    }

    /**
     * @brief return a pointer to the hash table
     */
    __host__ __device__ __inline__ LPPair* get_table()
    {
        return m_table;
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
        if (m_is_on_device) {
            CUDA_ERROR(cudaMemset(m_table, INVALID8, num_bytes()));
        } else {
            std::memset(m_table, INVALID8, num_bytes());
        }
    }

    /**
     * @brief Reset the hash table to sentinel key-value (tombstone). This API
     * is for the device only and must be called by a single block
     */
    template <uint32_t blockThreads>
    __device__ __inline__ void clear()
    {
        assert(m_is_on_device);
        for (uint32_t i = threadIdx.x; i < m_capacity; i += blockThreads) {
            m_table[i].m_pair = INVALID32;
        }
    }

    /**
     * @brief Free the GPU allocation
     */
    __host__ void free()
    {
        if (m_is_on_device) {
            GPU_FREE(m_table);
        } else {
            ::free(m_table);
        }
    }

    /**
     * @brief size of the hash table in bytes which may be needed for allocation
     */
    __host__ __device__ __inline__ uint32_t num_bytes() const
    {
        return m_capacity * sizeof(LPPair);
    }

    /**
     * @brief (Re)Generate new hashers
     */
    template <typename RNG>
    void randomize_hash_functions(RNG& rng)
    {
        m_hasher0 = initialize_hf<universal_hash<LPPair>>(rng);
        m_hasher1 = initialize_hf<universal_hash<LPPair>>(rng);
        m_hasher2 = initialize_hf<universal_hash<LPPair>>(rng);
        m_hasher3 = initialize_hf<universal_hash<LPPair>>(rng);
    }

   private:
    LPPair*                m_table;
    universal_hash<LPPair> m_hasher0;
    universal_hash<LPPair> m_hasher1;
    universal_hash<LPPair> m_hasher2;
    universal_hash<LPPair> m_hasher3;
    uint16_t               m_capacity;
    uint16_t               m_max_cuckoo_chains;
    bool                   m_is_on_device;
};
}  // namespace rxmesh