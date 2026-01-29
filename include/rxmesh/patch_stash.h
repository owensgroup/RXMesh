#pragma once
#include <stdint.h>

#include "rxmesh/kernels/shmem_mutex.cuh"
#include "rxmesh/lp_pair.cuh"

namespace rxmesh {

/**
 * @brief Store the neighbor patches neighbor to a patch in a fixed size array.
 */
struct PatchStash
{
    static constexpr uint8_t stash_size = (1 << LPPair::PatchStashNumBits);

    explicit __host__ PatchStash(bool on_device);

    __device__ __host__ PatchStash()                             = default;
    __device__ __host__ PatchStash(const PatchStash& other)      = default;
    __device__ __host__ PatchStash(PatchStash&&)                 = default;
    __device__ __host__ PatchStash& operator=(const PatchStash&) = default;
    __device__ __host__ PatchStash& operator=(PatchStash&&)      = default;
    __device__                      __host__ ~PatchStash()       = default;

    __host__ __device__ __inline__ uint32_t get_patch(uint8_t id) const
    {
        assert(id >= 0 && id < stash_size);
        return m_stash[id];
    }

    __host__ __device__ __inline__ uint32_t& get_patch(uint8_t id)
    {
        assert(id >= 0 && id < stash_size);
        return m_stash[id];
    }

    __host__ __device__ __inline__ uint32_t get_patch(const LPPair p) const
    {
        return get_patch(p.patch_stash_id());
    }

    __host__ __device__ __inline__ uint32_t& get_patch(const LPPair p)
    {
        return get_patch(p.patch_stash_id());
    }


    __device__ uint8_t insert_patch(uint32_t patch, ShmemMutex& mutex);

    /**
     * @brief insert a new patch in the stash and return the stash id. This
     * function makes sure that there is no redundant patches in the stash. If
     * the stash is full, it returns INVALID8
     * @param patch The id of the patch to be added
     * @return
     */
    __host__ __device__ uint8_t insert_patch(uint32_t patch);

    __host__ __device__ uint8_t find_patch_index(uint32_t patch) const;

    __host__ void free();

    // store the stash level of the initial level of coarsening
    uint32_t* m_stash;

    bool m_is_on_device;
};

}  // namespace rxmesh
