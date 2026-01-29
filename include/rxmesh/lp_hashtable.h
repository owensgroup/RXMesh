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

#include "rxmesh/hash_functions.cuh"
#include "rxmesh/lp_pair.cuh"
#include "rxmesh/util/macros.h"

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

    static constexpr uint8_t stash_size = 128;

    __device__ __host__ LPHashTable();
    LPHashTable(const LPHashTable& other)      = default;
    LPHashTable(LPHashTable&&)                 = default;
    LPHashTable& operator=(const LPHashTable&) = default;
    LPHashTable& operator=(LPHashTable&&)      = default;
    ~LPHashTable()                             = default;

    /**
     * @brief Constructor using the hash table capacity.This is used as
     * allocation size
     */
    explicit LPHashTable(uint16_t capacity, bool is_on_device);

    /**
     * @brief Get the hash table capacity
     */
    __host__ __device__ uint16_t get_capacity() const;

    /**
     * @brief Reset the hash table to sentinel key-value (tombstone). This API
     * is for the host only
     */
    __host__ void clear();

    /**
     * @brief Reset the hash table to sentinel key-value (tombstone). This API
     * is for the device only and must be called by a single block
     */
    template <uint32_t blockThreads>
    __device__ void clear();

    /**
     * @brief Free the GPU allocation
     */
    __host__ void free();

    /**
     * @brief size of the hash table in bytes which may be needed for allocation
     */
    __host__ __device__ uint32_t num_bytes() const;

    /**
     * @brief (Re)Generate new hashers
     */
    template <typename RNG>
    void randomize_hash_functions(RNG& rng);

    /**
     * @brief compute current load factor
     */
    __host__ __device__ float compute_load_factor(
        LPPair* s_table = nullptr) const;

    /**
     * @brief compute current load factor for the stash
     */
    __host__ __device__ float compute_stash_load_factor(
        LPPair* s_stash = nullptr) const;

    /**
     * @brief memcpy the hashtable from host to device, host to host, device to
     * host, device to device depending on where the src and this is allocated
     */
    __host__ void move(const LPHashTable src);

    /**
     * @brief Load the memory used for the hash table into a shared memory
     * buffer
     */
    __device__ void load_in_shared_memory(LPPair* s_table,
                                          bool    with_wait,
                                          LPPair* s_stash = nullptr) const;

    /**
     * @brief write the content of the hash table from (likely shared memory)
     * buffer
     */
    template <uint32_t blockSize>
    __device__ void write_to_global_memory(const LPPair* s_table,
                                           const LPPair* s_stash);

    /**
     * @brief Insert new pair in the table. This function can be called from
     * host (not thread safe) or from the device (by a single thread). The table
     * itself is part of the input in case it was loaded in shared memory.
     * @param pair to be inserted in the hash table
     * @param table pointer to the hash table (could shared memory on the
     * device)
     * @return true if the insertion succeeded and false otherwise
     */
    __host__ __device__ bool insert(LPPair           pair,
                                    volatile LPPair* table = nullptr,
                                    volatile LPPair* stash = nullptr);

    /**
     * @brief Find a pair in the hash table given its key.
     * @param key input key
     * @param table pointer to the hash table (could shared memory on the
     * device)
     * @return a LPPair pair that contains the key and its associated value
     */
    __host__ __device__ LPPair find(const typename LPPair::KeyT key,
                                    const LPPair*               table = nullptr,
                                    const LPPair* stash = nullptr) const;

    /**
     * @brief Replace an existing pair with another. We use the new_pair to
     * find the old pair. Then, we replace it with new_pair
     * @param new_pair the new pair that will be inserted
     */
    __host__ __device__ void replace(const LPPair new_pair);

    /**
     * @brief remove item from the hash table i.e., replace it with
     * sentinel_pair
     * @param key key to the item to remove
     * @param table pointer to the hash table (could shared memory on the
     * device). Otherwise, it should be null
     * @param stash pointer to the hash table stash (could shared memory on the
     * device). Otherwise, it should be null
     */
    __host__ __device__ void remove(const typename LPPair::KeyT key,
                                    LPPair*                     table,
                                    LPPair*                     stash);

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
    __host__ __device__ LPPair find(const typename LPPair::KeyT key,
                                    uint32_t&                   bucket_id,
                                    bool&                       in_stash,
                                    const LPPair*               table = nullptr,
                                    const LPPair* stash = nullptr) const;

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
