#pragma once
#include <algorithm>
#include <random>

#include "rxmesh/lp_pair.cuh"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/prime_numbers.h"

namespace rxmesh {

struct LPHashTable
{
    __device__ __host__ LPHashTable()
        : m_table(nullptr), m_capacity(0), m_is_on_device(false)
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
        : m_capacity(capacity), m_is_on_device(is_on_device)
    {

        if (m_is_on_device) {
            CUDA_ERROR(cudaMalloc((void**)&m_table, num_bytes()));
        } else {
            m_table = (LPPair*)malloc(num_bytes());
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
        return m_capacity * sizeof(LPPair);
    }


    /**
     * @brief memcpy the hashtable from host to device, host to host, device to
     * host, device to device depending on where the src and this is allocated
     */
    void move(const LPHashTable src)
    {
        if (src.m_is_on_device && m_is_on_device) {
            CUDA_ERROR(cudaMemcpy(
                m_table, src.m_table, num_bytes(), cudaMemcpyDeviceToDevice));
        }

        if (!src.m_is_on_device && !m_is_on_device) {
            std::memcpy(m_table, src.m_table, num_bytes());
        }

        if (src.m_is_on_device && !m_is_on_device) {
            CUDA_ERROR(cudaMemcpy(
                m_table, src.m_table, num_bytes(), cudaMemcpyDeviceToHost));
        }

        if (!src.m_is_on_device && m_is_on_device) {
            CUDA_ERROR(cudaMemcpy(
                m_table, src.m_table, num_bytes(), cudaMemcpyHostToDevice));
        }
    }


    /**
     * @brief Load the memory used for the hash table into a shared memory
     * buffer
     */
    template <typename DummyT = void>
    __device__ __inline__ void load_in_shared_memory(LPPair* s_table,
                                                     bool    with_wait) const
    {
#ifdef __CUDA_ARCH__
        detail::load_async(m_table, m_capacity, s_table, with_wait);
#endif
    }


    /**
     * @brief write the content of the hash table from (likely shared memory)
     * buffer
     */
    template <uint32_t blockSize>
    __device__ __inline__ void write_to_global_memory(const LPPair* s_table)
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
    __host__ __device__ __inline__ bool insert(LPPair  pair,
                                               LPPair* table = nullptr)
    {
        if (table != nullptr) {
            table[pair.key()] = pair;
        } else {
            m_table[pair.key()] = pair;
        }
        return true;
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
        const LPPair*               table = nullptr) const
    {
        if (table != nullptr) {
            return table[key];
        } else {
            return m_table[key];
        }
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
                                               LPPair* table = nullptr)
    {

        if (table != nullptr) {
            table[key] = INVALID32;
        } else {
            m_table[key] = INVALID32;
        }
    }


   private:
    LPPair*  m_table;
    uint16_t m_capacity;
    bool     m_is_on_device;
};
}  // namespace rxmesh