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

    __host__ PatchStash(bool on_device) : m_is_on_device(on_device)
    {
        if (m_is_on_device) {
            CUDA_ERROR(
                cudaMalloc((void**)&m_stash, stash_size * sizeof(uint32_t)));
            CUDA_ERROR(
                cudaMemset(m_stash, INVALID8, stash_size * sizeof(uint32_t)));

            CUDA_ERROR(cudaMalloc((void**)&m_edge_weight,
                                  stash_size * sizeof(uint32_t)));
            CUDA_ERROR(cudaMemset(
                m_edge_weight, INVALID32, stash_size * sizeof(uint32_t)));

            CUDA_ERROR(cudaMalloc((void**)&m_coarse_level_id_list,
                                  stash_size * sizeof(uint32_t)));
            CUDA_ERROR(cudaMemset(m_coarse_level_id_list,
                                  INVALID32,
                                  stash_size * sizeof(uint32_t)));

            CUDA_ERROR(cudaMalloc((void**)&m_coarse_level_pair_list,
                                  stash_size * sizeof(uint32_t)));
            CUDA_ERROR(cudaMemset(m_coarse_level_pair_list,
                                  INVALID32,
                                  stash_size * sizeof(uint32_t)));

            CUDA_ERROR(cudaMalloc((void**)&m_coarse_level_num_v,
                                  stash_size * sizeof(uint32_t)));
            CUDA_ERROR(cudaMemset(m_coarse_level_num_v,
                                  INVALID32,
                                  stash_size * sizeof(uint32_t)));

            CUDA_ERROR(cudaMalloc((void**)&m_tmp_level_stash,
                                  stash_size * sizeof(uint32_t)));
            CUDA_ERROR(cudaMemset(m_tmp_level_stash, 
                                  INVALID32, 
                                  stash_size * sizeof(uint32_t)));

            CUDA_ERROR(cudaMalloc((void**)&m_tmp_level_edge_weight,
                                  stash_size * sizeof(uint32_t)));
            CUDA_ERROR(cudaMemset(m_tmp_level_edge_weight,
                                  INVALID32,
                                  stash_size * sizeof(uint32_t)));

            CUDA_ERROR(cudaMalloc((void**)&m_v_ordering_bound,
                                  2 * sizeof(uint32_t)));
            CUDA_ERROR(cudaMemset(m_v_ordering_bound,
                                    INVALID32,
                                    2 * sizeof(uint32_t)));

            m_node_num_separator_v = 0;
            m_is_active = true;
        } else {
            m_stash = (uint32_t*)malloc(stash_size * sizeof(uint32_t));
            for (uint8_t i = 0; i < stash_size; ++i) {
                m_stash[i] = INVALID32;
            }

            m_edge_weight = (uint32_t*)malloc(stash_size * sizeof(uint32_t));
            for (uint8_t i = 0; i < stash_size; ++i) {
                m_edge_weight[i] = INVALID32;
            }
        }
    }

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

    __host__ __device__ __inline__ uint32_t get_edge_weight(uint8_t id) const
    {
        assert(id >= 0 && id < stash_size);

        if (m_edge_weight[id] == INVALID32) {
            m_edge_weight[id] = 0;
        }

        return m_edge_weight[id];
    }

    __host__ __device__ __inline__ uint32_t& get_edge_weight(uint8_t id)
    {
        assert(id >= 0 && id < stash_size);

        if (m_edge_weight[id] == INVALID32) {
            m_edge_weight[id] = 0;
        }

        return m_edge_weight[id];
    }

    __host__ __device__ __inline__ uint32_t get_coarse_level_id(
        uint16_t level) const
    {
        assert(m_coarse_level_id_list[level] != INVALID32);
        return m_coarse_level_id_list[level];
    }

    __host__ __device__ __inline__ uint32_t& get_level_patch_id(uint16_t level)
    {
        assert(m_coarse_level_id_list[level] != INVALID32);
        return m_coarse_level_id_list[level];
    }

    __host__ __device__ __inline__ uint32_t get_tmp_level_patch(
        uint8_t id) const
    {
        assert(id >= 0 && id < stash_size);
        return m_tmp_level_stash[id];
    }

    __host__ __device__ __inline__ uint32_t& get_tmp_level_patch(uint8_t id)
    {
        assert(id >= 0 && id < stash_size);
        return m_tmp_level_stash[id];
    }

    __host__ __device__ __inline__ uint32_t get_tmp_level_edge_weight(
        uint8_t id) const
    {
        assert(id >= 0 && id < stash_size);
        assert(m_tmp_level_stash[id] != INVALID32);

        if (m_tmp_level_edge_weight[id] == INVALID32) {
            m_tmp_level_edge_weight[id] = 0;
        }

        return m_tmp_level_edge_weight[id];
    }

    __host__ __device__ __inline__ uint32_t& get_tmp_level_edge_weight(
        uint8_t id)
    {
        assert(id >= 0 && id < stash_size);
        assert(m_tmp_level_stash[id] != INVALID32);

        if (m_tmp_level_edge_weight[id] == INVALID32) {
            m_tmp_level_edge_weight[id] = 0;
        }

        return m_tmp_level_edge_weight[id];
    }

    /*__host__ __device__ __inline__ uint8_t insert_patch(uint32_t patch)
    {
        assert(patch != INVALID32);

        for (uint8_t i = 0; i < stash_size; ++i) {
#ifdef __CUDA_ARCH__
            uint32_t old = ::atomicCAS(m_stash + i, INVALID32, patch);
            if (old == INVALID32 || old == patch) {
                return i;
            }
#else
            if (m_stash[i] == patch) {
                // prevent redundancy
                return i;
            }

            if (m_stash[i] == INVALID32) {
                m_stash[i] = patch;
                return i;
            }
#endif
        }
        return INVALID8;
    }*/

    __device__ __inline__ uint8_t insert_patch(uint32_t    patch,
                                               ShmemMutex& mutex)
    {
        mutex.lock();
        const uint8_t ret = insert_patch(patch);
        mutex.unlock();
        return ret;
    }

    /**
     * @brief insert a new patch in the stash and return the stash id. This
     * function makes sure that there is no redundant patches in the stash. If
     * the stash is full, it returns INVALID8
     * @param patch The id of the patch to be added
     * @return
     */
    __host__ __device__ __inline__ uint8_t insert_patch(uint32_t patch)
    {
        assert(patch != INVALID32);

        uint8_t empty_slot = INVALID8;
        for (uint8_t i = 0; i < stash_size; ++i) {
            if (m_stash[i] == patch) {
                // prevent redundancy
                return i;
            }

            // update the empty_slot if there is a an empty slot in the stash
            // and update the empty_slot only once
            if (m_stash[i] == INVALID32 && empty_slot == INVALID8) {
                empty_slot = i;
            }
        }

        // if we have found an empty_slot and also this patch has not be
        // encountered
        if (empty_slot != INVALID8) {
            m_stash[empty_slot] = patch;
        }

        // return the empty_slot even if it is not updated. If it is not
        // updated, then return INVALID8 would indicate that the patch has not
        // been added
        return empty_slot;
    }

    __host__ __device__ __inline__ uint8_t find_patch_index(
        uint32_t patch) const
    {
        assert(patch != INVALID32);
        for (uint8_t i = 0; i < stash_size; ++i) {
            if (m_stash[i] == patch) {
                return i;
            }
        }
        return INVALID8;
    }

    __host__ __device__ __inline__ uint8_t find_tmp_level_patch_index(
        uint32_t patch) const
    {
        assert(patch != INVALID32);
        for (uint8_t i = 0; i < stash_size; ++i) {
            if (m_tmp_level_stash[i] == patch) {
                return i;
            }
        }
        return INVALID8;
    }


    __host__ void free()
    {
        if (m_is_on_device) {
            GPU_FREE(m_stash);
            GPU_FREE(m_edge_weight);
        } else {
            ::free(m_stash);
            ::free(m_edge_weight);
        }
    }

    // store the stash level of the initial level of coarsening
    uint32_t* m_stash;
    uint32_t* m_edge_weight;

    // store the patch id of each level of coarsening
    uint32_t* m_coarse_level_pair_list;
    uint32_t* m_coarse_level_id_list;
    uint32_t* m_coarse_level_num_v;
    uint32_t m_node_num_separator_v;
    uint32_t* m_v_ordering_bound;
    
    bool m_is_node;  // determine whether the patch is a node or not(remained or
                     // merged)

    // store the stash info of the current level during coarsening
    bool      m_is_active;
    uint32_t  m_tmp_paired_patch;
    uint32_t* m_tmp_level_stash;
    uint32_t* m_tmp_level_edge_weight;

    bool m_is_on_device;
};
}  // namespace rxmesh