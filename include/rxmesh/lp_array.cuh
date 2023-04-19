#pragma once
#include <algorithm>
#include <random>

#include "rxmesh/lp_pair.cuh"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/prime_numbers.h"

namespace rxmesh {

struct LPArray
{
    __device__ __host__ LPArray()
        : m_table(nullptr), m_capacity(0), m_is_on_device(false)
    {
    }
    LPArray(const LPArray& other) = default;
    LPArray(LPArray&&)            = default;
    LPArray& operator=(const LPArray&) = default;
    LPArray& operator=(LPArray&&) = default;
    ~LPArray()                    = default;

    /**
     * @brief Constructor using the hash table capacity.This is used as
     * allocation size
     */
    explicit LPArray(const uint16_t capacity, bool is_on_device)
        : m_capacity(capacity), m_is_on_device(is_on_device)
    {

        if (m_is_on_device) {
            CUDA_ERROR(cudaMalloc((void**)&m_table, num_bytes()));
        } else {
            m_table = (LPPair::ValueT*)malloc(num_bytes());
        }

        clear();
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
#ifdef __CUDA_ARCH__
        assert(m_is_on_device);
        for (uint32_t i = threadIdx.x; i < m_capacity; i += blockThreads) {
            m_table[i] = INVALID16;
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
        } else {
            ::free(m_table);
        }
    }

    /**
     * @brief size of the hash table in bytes which may be needed for allocation
     */
    __host__ __device__ __inline__ uint32_t num_bytes() const
    {
        return m_capacity * sizeof(LPPair::ValueT);
    }


    /**
     * @brief write the content of the hash table from (likely shared memory)
     * buffer
     */
    template <uint32_t blockSize>
    __device__ __inline__ void write_to_global_memory(
        const LPPair::ValueT* s_table)
    {
#ifdef __CUDA_ARCH__
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
    __host__ __device__ __inline__ bool insert(
        LPPair                   pair,
        volatile LPPair::ValueT* table = nullptr)
    {
        if (table != nullptr) {
            return table[pair.key()] = pair.value();
        } else {
            return m_table[pair.key()] = pair.value();
        }
    }

    /**
     * @brief Find a pair in the hash table given its key.
     * @param key input key
     * @param table pointer to the hash table (could shared memory on the
     * device)
     * @return a LPPair pair that contains the key and its associated value
     */
    __host__ __device__ __inline__ LPPair find(
        const typename LPPair::KeyT key,
        const LPPair::ValueT*       table = nullptr) const
    {
        return find(key, table);
    }


    /**
     * @brief Replace an existing pair with another. We use the new_pair to
     * find the old pair. Then, we replace it with new_pair
     * @param new_pair the new pair that will be inserted
     */
    __host__ __device__ __inline__ void replace(const LPPair new_pair)
    {
        m_table[new_pair.key()] = new_pair.value();
    }


    /**
     * @brief remove item from the hash table i.e., replace it with
     * sentinel_pair
     * @param key key to the item to remove
     * @param table pointer to the hash table (could shared memory on the
     * device)
     */
    __host__ __device__ __inline__ void remove(const typename LPPair::KeyT key,
                                               LPPair::ValueT* table = nullptr)
    {

        if (table != nullptr) {
            table[key] = INVALID16;
        } else {
            m_table[key] = INVALID16;
        }
    }


   private:
    /**
     * @brief Find a pair in the hash table given its key.
     * @param key input key
     * @param bucket_id returned bucket ID of the found LPPair
     * @param table pointer to the hash table (could shared memory on the
     * device)
     * @return a LPPair pair that contains the key and its associated value
     */
    __host__ __device__ __inline__ LPPair find(
        const typename LPPair::KeyT key,
        const LPPair::ValueT*       table = nullptr) const
    {
        if (table != nullptr) {
            return LPPair(key, table[key]);
        } else {
            return LPPair(key, m_table[key]);
        }
    }


    LPPair::ValueT* m_table;
    uint16_t        m_capacity;
    bool            m_is_on_device;
};
}  // namespace rxmesh
