#pragma once

#include <cooperative_groups.h>
#include <stdint.h>

#include "rxmesh/kernels/shmem_allocator.cuh"
#include "rxmesh/util/bitmask_util.h"

namespace rxmesh {
struct Bitmask
{

    __device__ __host__ __inline__ Bitmask() : m_size(0), m_bitmask(nullptr)
    {
    }

    __device__ __host__ Bitmask(const Bitmask& other) = default;
    __device__ __host__ Bitmask(Bitmask&&)            = default;
    __device__ __host__ Bitmask& operator=(const Bitmask&) = default;
    __device__ __host__ Bitmask& operator=(Bitmask&&) = default;
    __device__                   __host__ ~Bitmask()  = default;

    /**
     * @brief constructor for a user-defined/allocated mask and size
     * @param size is the number of bits represented by this mask
     * @param mask a pointer to a mask allocated by the caller
     * @return
     */
    __device__ __host__ __inline__ Bitmask(uint32_t size, uint32_t* mask)
        : m_size(size), m_bitmask(mask)
    {
    }

    /**
     * @brief constructor that works for bitmask stored in shared memory on the
     * device. It takes a shared memory allocator and perform the allocation
     * internally
     * @param size is the number of bits represented by this mask
     * @param shrd_alloc shared memory allocator used to allocate the bitmask
     * bytes
     * @return
     */
    __device__ __inline__ Bitmask(uint32_t size, ShmemAllocator& shrd_alloc)
        : m_size(size)
    {
        m_bitmask =
            reinterpret_cast<uint32_t*>(shrd_alloc.alloc(num_bytes(size)));
    }

    /**
     * @brief the number of bits represented by this bitmask
     */
    __device__ __host__ __inline__ uint32_t size()
    {
        return m_size;
    }


    /**
     * @brief return the number of bytes needed to represent a mask of a given
     * size/number of bits
     * @param size as the number of bits
     */
    constexpr __device__ __host__ __inline__ uint32_t num_bytes(uint32_t size)
    {
        return detail::mask_num_bytes(size);
    }


    /**
     * @brief reset all the bits in the mask to zero. This should only be
     * invoked from the host side
     * @return
     */
    __host__ inline void reset()
    {
        assert(m_bitmask != nullptr);
        uint32_t mask_num_elements = DIVIDE_UP(m_size, 32);
        for (uint32_t i = 0; i < mask_num_elements; ++i) {
            m_bitmask[i] = 0;
        }
    }

    /**
     * @brief reset all the bits in the mask to zero. This can be only invoked
     * from the device using cooperative groups
     * @return
     */
    __device__ __inline__ void reset(cooperative_groups::thread_group& g)
    {
        assert(m_bitmask != nullptr);
        uint32_t mask_num_elements = DIVIDE_UP(m_size, 32);
        for (uint32_t i = g.thread_rank(); i < mask_num_elements;
             i += g.size()) {
            m_bitmask[i] = 0;
        }
    }


    /**
     * @brief set all the bits in the mask one. This should only be invoked from
     * the host side
     * @return
     */
    __host__ inline void set()
    {
        assert(m_bitmask != nullptr);
        uint32_t mask_num_elements = DIVIDE_UP(m_size, 32);
        for (uint32_t i = 0; i < mask_num_elements; ++i) {
            m_bitmask[i] = INVALID32;
        }
    }

    /**
     * @brief set all the bits in the mask one. This can be only invoked from
     * the device using cooperative groups
     * @return
     */
    __device__ __inline__ void set(cooperative_groups::thread_group& g)
    {
        assert(m_bitmask != nullptr);
        const uint32_t mask_num_elements = DIVIDE_UP(m_size, 32);
        for (uint32_t i = g.thread_rank(); i < mask_num_elements;
             i += g.size()) {
            m_bitmask[i] = INVALID32;
        }
    }


    /**
     * @brief set a bit in the bitmask
     * @param bit the bit position
     * @param is_atomic perform the operation atomically in case it is done on
     * the device to avoid race condition. On the host, this option is ignored
     */
    __device__ __host__ __inline__ void set(const uint32_t bit,
                                            bool           is_atomic = false)
    {
        assert(bit < m_size);
        detail::bitmask_set_bit(bit, m_bitmask, is_atomic);
    }


    /**
     * @brief clear a bit in the bitmask
     * @param bit the bit position
     * @param is_atomic perform the operation atomically in case it is done on
     * the device to avoid race condition. On the host, this option is ignored
     */
    __device__ __host__ __inline__ void reset(const uint32_t bit,
                                              bool           is_atomic = false)
    {
        assert(bit < m_size);
        detail::bitmask_clear_bit(bit, m_bitmask, is_atomic);
    }


    /**
     * @brief flip a bit in the bitmask
     * @param bit the bit position
     * @param is_atomic perform the operation atomically in case it is done on
     * the device to avoid race condition. On the host, this option is ignored
     */
    __device__ __host__ __inline__ void flip(const uint32_t bit,
                                             bool           is_atomic = false)
    {
        assert(bit < m_size);
        detail::bitmask_flip_bit(bit, m_bitmask, is_atomic);
    }


    /**
     * @brief check if bit is set
     * @param bit the bit position
     * @return true is the bit is set, false otherwise
     */
    __device__ __host__ __inline__ bool operator()(const uint32_t bit) const
    {
        return detail::is_set_bit(bit, m_bitmask);
    }

   private:
    // number of elements this Bitmask represents
    uint32_t  m_size;
    uint32_t* m_bitmask;
};
}  // namespace rxmesh