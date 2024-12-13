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
#include "rxmesh/lp_pair.cuh"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/prime_numbers.h"

#ifdef __CUDA_ARCH__
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/util.cuh"
#endif

namespace rxmesh {
/**
 * @brief Hash table storing a patch's ribbon (not-owned) mesh elements where
 * the key is the local index within the patch and the value is the local index
 * within the owner patch along with an index in the PatchStash. The key and
 * value are combined in a single 32-bit using LBPair.
 * TODO in order to support deletion, we must distinguish between empty keys and
 * tombstone. Empty key is a slot that has not been touched before while
 * tombstone is a slot that has been inserted into before but it is now deleted.
 * We should then initialize the table with empty keys, then build. During
 * deletion, we first try to find the key and then replace it with tombstone. If
 * another key is trying to find its slot in order to delete itself, it may
 * encounter a tombstone during the cuckoo chain which only means that this
 * place used to have a pair that has been deleted. In which case, this key
 * should continue its chain until it find its pair.
 */
struct LPHashTable
{
    using HashT = universal_hash;
    // using HashT = MurmurHash3_32;

    static constexpr uint8_t stash_size = 128;

    __device__ __host__ LPHashTable()
        : m_table(nullptr),
          m_stash(nullptr),
          m_capacity(0),
          m_max_cuckoo_chains(0),
          m_is_on_device(false)
    {
    }
    LPHashTable(const LPHashTable& other) = default;
    LPHashTable(LPHashTable&&)            = default;
    LPHashTable& operator=(const LPHashTable&) = default;
    LPHashTable& operator=(LPHashTable&&) = default;
    ~LPHashTable()                        = default;

    /**
     * @brief Constructor using the hash table capacity.This is used as
     * allocation size
     */
    explicit LPHashTable(const uint16_t capacity, bool is_on_device)
        : m_capacity(std::max(capacity, uint16_t(2))),
          m_is_on_device(is_on_device)
    {
        m_capacity = find_next_prime_number(m_capacity);
        if (m_is_on_device) {
            CUDA_ERROR(cudaMalloc((void**)&m_table, num_bytes()));
            CUDA_ERROR(
                cudaMalloc((void**)&m_stash, stash_size * sizeof(LPPair)));

            std::vector<LPPair> temp(stash_size, LPPair::sentinel_pair());
            CUDA_ERROR(cudaMemcpy(m_stash,
                                  temp.data(),
                                  stash_size * sizeof(LPPair),
                                  cudaMemcpyHostToDevice));
        } else {
            m_table = (LPPair*)malloc(num_bytes());
            m_stash = (LPPair*)malloc(stash_size * sizeof(LPPair));
            for (uint8_t i = 0; i < stash_size; ++i) {
                m_stash[i] = LPPair::sentinel_pair();
            }
        }

        clear();
        // maximum number of cuckoo chains
        double lg_input_size = (float)(log((double)m_capacity) / log(2.0));
        const unsigned max_iter_const = 7;
        m_max_cuckoo_chains =
            static_cast<uint16_t>(max_iter_const * lg_input_size);


        // std::mt19937 rng(2);
        MarsRng32 rng;
        randomize_hash_functions(rng);
    }


    /**
     * @brief Get the hash table capacity
     */
    __host__ __device__ __inline__ uint16_t get_capacity() const
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
            std::fill_n(m_table, m_capacity, LPPair());            
        }
    }

    /**
     * @brief Reset the hash table to sentinel key-value (tombstone). This API
     * is for the device only and must be called by a single block
     */
    template <uint32_t blockThreads>
    __device__ __inline__ void clear()
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

    /**
     * @brief Free the GPU allocation
     */
    __host__ void free()
    {
        if (m_is_on_device) {
            GPU_FREE(m_table);
            GPU_FREE(m_stash);
        } else {
            ::free(m_table);
            ::free(m_stash);
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
        m_hasher0 = initialize_hf<HashT>(rng);
        m_hasher1 = initialize_hf<HashT>(rng);
        m_hasher2 = initialize_hf<HashT>(rng);
        m_hasher3 = initialize_hf<HashT>(rng);
    }


    /**
     * @brief compute current load factor
     */
    __host__ __device__ __inline__ float compute_load_factor(
        LPPair* s_table = nullptr)const 
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
            };
            return m;
        };

        return static_cast<float>(lf()) / static_cast<float>(m_capacity);
    }

    /**
     * @brief compute current load factor for the stash
     */
    __host__ __device__ __inline__ float compute_stash_load_factor(
        LPPair* s_stash = nullptr)const 
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
            };
            return m;
        };

        return static_cast<float>(lf()) / static_cast<float>(stash_size);
    }


    /**
     * @brief memcpy the hashtable from host to device, host to host, device to
     * host, device to device depending on where the src and this is allocated
     */
    __host__ void move(const LPHashTable src)
    {
        const size_t stash_num_bytes = LPHashTable::stash_size * sizeof(LPPair);
        if (src.m_is_on_device && m_is_on_device) {
            CUDA_ERROR(cudaMemcpy(
                m_table, src.m_table, num_bytes(), cudaMemcpyDeviceToDevice));
            CUDA_ERROR(cudaMemcpy(m_stash,
                                  src.m_stash,
                                  stash_num_bytes,
                                  cudaMemcpyDeviceToDevice));
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


    /**
     * @brief Load the memory used for the hash table into a shared memory
     * buffer
     */
    template <typename DummyT = void>
    __device__ __inline__ void load_in_shared_memory(
        LPPair* s_table,
        bool    with_wait,
        LPPair* s_stash = nullptr) const
    {
#ifdef __CUDA_ARCH__
        if (s_stash != nullptr) {
            detail::load_async(m_stash, stash_size, s_stash, false);
        }
        detail::load_async(m_table, m_capacity, s_table, with_wait);
#endif
    }


    /**
     * @brief write the content of the hash table from (likely shared memory)
     * buffer
     */
    template <uint32_t blockSize>
    __device__ __inline__ void write_to_global_memory(const LPPair* s_table,
                                                      const LPPair* s_stash)
    {
#ifdef __CUDA_ARCH__
        if (s_stash != nullptr) {
            detail::store<blockSize>(s_stash, stash_size, m_stash);
        }
        detail::store<blockSize>(s_table, m_capacity, m_table);
#endif
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
    __host__ __device__ __inline__ bool insert(LPPair           pair,
                                               volatile LPPair* table,
                                               volatile LPPair* stash)
    {
#ifndef __CUDA_ARCH__
        // on the host, these two should be nullprt since there is no shared
        // memory
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

        const auto input_key = pair.key();

#ifdef __CUDA_ARCH__
        for (uint8_t i = 0; i < stash_size; ++i) {
            LPPair prv;
            if (stash != nullptr) {
                prv.m_pair =
                    ::atomicCAS((uint32_t*)(stash + i), INVALID32, pair.m_pair);
            } else {
                prv.m_pair =
                    ::atomicCAS(reinterpret_cast<uint32_t*>(m_stash + i),
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

    /**
     * @brief Find a pair in the hash table given its key.
     * @param key input key
     * @param table pointer to the hash table (could shared memory on the
     * device)
     * @return a LPPair pair that contains the key and its associated value
     */
    __host__ __device__ __inline__ LPPair find(const typename LPPair::KeyT key,
                                               const LPPair* table,
                                               const LPPair* stash) const
    {
        uint32_t bucket_id(0);
        bool     in_stash(false);
        return find(key, bucket_id, in_stash, table, stash);
    }


    /**
     * @brief Replace an existing pair with another. We use the new_pair to
     * find the old pair. Then, we replace it with new_pair
     * @param new_pair the new pair that will be inserted
     */
    __host__ __device__ __inline__ void replace(const LPPair new_pair)
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


    /**
     * @brief remove item from the hash table i.e., replace it with
     * sentinel_pair
     * @param key key to the item to remove
     * @param table pointer to the hash table (could shared memory on the
     * device). Otherwise, it should be null
     * @param stash pointer to the hash table stash (could shared memory on the
     * device). Otherwise, it should be null
     */
    __host__ __device__ __inline__ void remove(const typename LPPair::KeyT key,
                                               LPPair* table,
                                               LPPair* stash)
    {
        uint32_t bucket_id(0);
        bool     in_stash(false);
        find(key, bucket_id, in_stash, table, stash);

        // TODO not sure if we need to do this atomically on the GPU. But if we
        // do, here is the implementation
        // #ifdef __CUDA_ARCH__
        //        if (table != nullptr) {
        //            ::atomicExch((uint32_t*)table + bucket_id, INVALID32);
        //        } else {
        //            ::atomicExch((uint32_t*)m_table + bucket_id, INVALID32);
        //        }
        // #else
        //        if (table != nullptr) {
        //            table[bucket_id].m_pair = INVALID32;
        //        } else {
        //            m_table[bucket_id].m_pair = INVALID32;
        //        }
        // #endif


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


    // private:
    /**
     * @brief Find a pair in the hash table given its key.
     * @param key input key
     * @param bucket_id returned bucket ID of the found LPPair
     * @param table pointer to the hash table (could shared memory on the
     * device) Otherwise, it should be null
     * @param stash pointer to the hash table stash (could shared memory on the
     * device) Otherwise, it should be null
     * @return a LPPair pair that contains the key and its associated value
     */
    __host__ __device__ __inline__ LPPair find(const typename LPPair::KeyT key,
                                               uint32_t&     bucket_id,
                                               bool&         in_stash,
                                               const LPPair* table,
                                               const LPPair* stash) const
    {

#ifndef __CUDA_ARCH__
        // on the host, these two should be nullprt since there is no shared
        // memory
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
                uint32_t* ptr =
                    reinterpret_cast<uint32_t*>(m_table + bucket_id);
                found = LPPair(atomic_read(ptr));
#else
                found = m_table[bucket_id];
#endif
            }

            if (found.key() == key) {
                return found;
                // since we only look for pairs that we know that they exist in
                // the table, we skip this since the pair could be in the stash
                //  } else if (found.m_pair == INVALID32) {
                //    return found;
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


    LPPair*  m_table;
    LPPair*  m_stash;
    HashT    m_hasher0;
    HashT    m_hasher1;
    HashT    m_hasher2;
    HashT    m_hasher3;
    uint16_t m_capacity;
    uint16_t m_max_cuckoo_chains;
    bool     m_is_on_device;
};
}  // namespace rxmesh
