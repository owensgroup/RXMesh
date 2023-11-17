#pragma once

#include <cooperative_groups.h>

#include <stdint.h>
#include <stdio.h>
#include "cuda_runtime.h"

namespace rxmesh {

__device__ __inline__ void block_print(cooperative_groups::thread_block& block,
                                       char                              msg[])
{
    block.sync();
    if (threadIdx.x == 0) {
        printf("\n Block= %d: %s \n", blockIdx.x, msg);
    }
    block.sync();
}


/**
 * print_arr_uint()
 */
template <typename T>
__device__ __inline__ void print_arr_uint(char     msg[],
                                          uint32_t size,
                                          T*       arr,
                                          uint32_t block_id  = 0,
                                          uint32_t thread_id = 0)
{
    if (blockIdx.x == block_id && threadIdx.x == thread_id) {
        printf("\n %s \n", msg);
        for (uint32_t i = 0; i < size; i++) {
            printf("\n arr[%u]=%u", i, arr[i]);
        }
        printf("\n");
    }
}

/**
 * print_arr_float()
 */
template <typename T>
__device__ __inline__ void print_arr_float(T size, float* arr)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint32_t i = 0; i < size; i++) {
            printf("\n arr[%u]=%f", i, arr[i]);
        }
    }
}

/**
 * print_arr_host()
 */
template <typename dataT>
__global__ void print_arr_host(uint32_t size, dataT* arr)
{
    // only one thread print everything
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        print_arr_uint("from host", size, arr);
    }
}

/**
 * total_smem_size()
 */
__device__ __forceinline__ unsigned total_smem_size()
{
    unsigned ret;
    asm volatile("mov.u32 %0, %total_smem_size;" : "=r"(ret));
    return ret;
}
}  // namespace rxmesh