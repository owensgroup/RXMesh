#pragma once

#include <stdint.h>

#include "rxmesh/util/macros.h"

namespace rxmesh {

/**
 * @brief PatchLock implements a locking mechanism for the patch. This is meant
 * to be used only on the device
 */
struct PatchLock
{
    __device__ __host__ PatchLock() : lock(nullptr), spin(nullptr) {};
    __device__ __host__ PatchLock(const PatchLock& other)      = default;
    __device__ __host__ PatchLock(PatchLock&&)                 = default;
    __device__ __host__ PatchLock& operator=(const PatchLock&) = default;
    __device__ __host__ PatchLock& operator=(PatchLock&&)      = default;
    __device__                     __host__ ~PatchLock()       = default;

    /**
     * @brief acquire the lock give an id that represent the block index
     */
    __device__ bool acquire_lock(uint32_t id);

    /**
     * @brief release the lock. Should only be called by the block/thread that
     * has successfully acquired the lock
     */
    __device__ void release_lock();

    /**
     * @brief check if the patch is locked
     */
    __device__ bool is_locked() const;

    /**
     * @brief initialize the lock by allocating memory and initialized the
     * values. Should only be called from the host
     */
    __host__ void init();

    /**
     * @brief free the memory allocated for the lock. Should be only called from
     * the host
     */
    __host__ void free();

   private:
    static constexpr uint32_t FREE        = 0;
    static constexpr uint32_t LOCKED      = INVALID32;
    static constexpr int      MAX_ATTEMPT = 10;

    uint32_t *lock, *spin;
};

}  // namespace rxmesh
