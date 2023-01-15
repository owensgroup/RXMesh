#pragma once

#include "rxmesh/util/util.h"

namespace rxmesh {
struct PatchScheduler
{
    __device__ __host__ PatchScheduler()
        : list(nullptr), size(nullptr), capacity(0){};
    __device__ __host__ PatchScheduler(const PatchScheduler& other) = default;
    __device__ __host__ PatchScheduler(PatchScheduler&&)            = default;
    __device__ __host__ PatchScheduler& operator=(const PatchScheduler&) =
        default;
    __device__ __host__ PatchScheduler& operator=(PatchScheduler&&) = default;
    __device__                          __host__ ~PatchScheduler()  = default;


    /**
     * @brief add/push new patch of the list
     */
    __device__ __inline__ void push(const uint32_t pid)
    {
#ifdef __CUDA_ARCH__
        int id = ::atomicAdd(size, 1);
        __threadfence();
        assert(id < capacity);
        list[id] = pid;
#endif
    }

    /**
     * @brief get a patch from the list. If the list is empty, return INVALID32
     */
    __device__ __inline__ uint32_t pop()
    {
#ifdef __CUDA_ARCH__
        const int id = ::atomicAdd(size, -1);
        __threadfence();
        if (id >= 0) {
            return list[id];
        } else {
            return INVALID32;
        }
#endif
    }

    /**
     * @brief reset the list by sequential numbers
     */
    __host__ void reset()
    {
        std::vector<uint32_t> h_list(capacity);
        fill_with_sequential_numbers(h_list.data(), capacity);
        CUDA_ERROR(cudaMemcpy(list,
                              h_list.data(),
                              h_list.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_ERROR(
            cudaMemcpy(size, &capacity, sizeof(int), cudaMemcpyHostToDevice));
    }

    /**
     * @brief initialize all the memories
     */
    __host__ void init(uint32_t num_patches)
    {
        capacity = num_patches;
        CUDA_ERROR(cudaMalloc((void**)&size, sizeof(int)));
        CUDA_ERROR(cudaMalloc((void**)&list, sizeof(uint32_t) * capacity));
    }

    /**
     * @brief free all the memories
     */
    __host__ void free()
    {
        GPU_FREE(size);
        GPU_FREE(list);
    }

    int*      size;
    uint32_t  capacity;
    uint32_t* list;
};

}  // namespace rxmesh