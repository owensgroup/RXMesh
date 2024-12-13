#pragma once
#ifdef __CUDA_ARCH__
#include "rxmesh/kernels/util.cuh"
#endif

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
    __device__ bool acquire_lock(uint32_t id)
    {
#ifdef __CUDA_ARCH__
        int attempt = 0;
        while (::atomicCAS(lock, FREE, LOCKED) == LOCKED) {
            __threadfence();
            if (attempt == MAX_ATTEMPT) {
                int other = ::atomicMin(spin, id);
                __threadfence();
                if (other < id) {
                    return false;
                }
                attempt = 0;
            }
            attempt++;
        }
        atomicExch(spin, id);
        return true;
#else
        return true;
#endif
    
    }

    /**
     * @brief release the lock. Should only be called by the block/thread that
     * has successfully acquired the lock
     */
    __device__ void release_lock()
    {
#ifdef __CUDA_ARCH__
        atomicExch(spin, INVALID32);
        atomicExch(lock, FREE);
        __threadfence();
#endif
    }

    /**
     * @brief check if the patch is locked
     */
    __device__ bool is_locked() const
    {
#ifdef __CUDA_ARCH__
        return atomic_read(lock) == LOCKED;
#else
        return false;
#endif
    }

    /**
     * @brief initialize the lock by allocating memory and initialized the
     * values. Should only be called from the host
     */
    __host__ void init()
    {
        CUDA_ERROR(cudaMalloc((void**)&lock, sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc((void**)&spin, sizeof(uint32_t)));
        uint32_t h_lock = FREE, h_spin = INVALID32;
        CUDA_ERROR(cudaMemcpy(
            lock, &h_lock, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(
            spin, &h_spin, sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    /**
     * @brief free the memory allocated for the lock. Should be only called from
     * the host
     */
    __host__ void free()
    {
        GPU_FREE(lock);
        GPU_FREE(spin);
    }


   private:
    static constexpr uint32_t FREE        = 0;
    static constexpr uint32_t LOCKED      = INVALID32;
    static constexpr int      MAX_ATTEMPT = 10;

    uint32_t *lock, *spin;
};

}  // namespace rxmesh