#pragma once
#include <assert.h>
#define CUB_STDERR
#include <cub/cub.cuh>
#include "rxmesh/util/macros.h"

namespace rxmesh {
/**
 * cub_block_exclusive_sum()
 */
template <typename T, uint32_t blockThreads>
__device__ __forceinline__ void cub_block_exclusive_sum(T*             data,
                                                        const uint32_t size)
{
    __shared__ T s_prv_run_aggregate;
    if (threadIdx.x == 0) {
        s_prv_run_aggregate = 0;
    }

    const uint32_t num_run = DIVIDE_UP(size, blockThreads);
    typedef cub::BlockScan<T, blockThreads>    BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;


    for (uint32_t r = 0; r < num_run; ++r) {
        T        value = 0;
        T        run_aggregate;
        uint32_t index = r * blockThreads + threadIdx.x;

        if (index < size) {
            value = data[index];
        }


        BlockScan(temp_storage).ExclusiveSum(value, value, run_aggregate);


        if (index < size) {
            if (r > 0) {
                value += s_prv_run_aggregate;
            }
            data[index] = value;
        }

        __syncthreads();
        if (threadIdx.x == blockThreads - 1) {
            if (r == num_run - 1) {
                data[size] = s_prv_run_aggregate + run_aggregate;
            } else {
                s_prv_run_aggregate += run_aggregate;
            }
        }
    }


    //__syncthreads();
    // if (threadIdx.x == 0) {
    //    data[size] = s_prv_run_aggregate;
    //}*/

    /*__shared__ T s_prv_run_aggregate;
    if (threadIdx.x == 0) {
        s_prv_run_aggregate = 0;
    }
    __syncthreads();
    const uint32_t num_run = DIVIDE_UP(size, blockThreads);

    typedef cub::BlockScan<T, blockThreads>    BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    for (uint32_t r = 0; r < num_run; ++r) {
        T        value = 0;
        T        run_aggregate;
        uint32_t index = r * blockThreads + threadIdx.x;
        int pred = int(index < size);
        value = pred*data[index];
        //if (index < size) {
        //    value = data[index];
        //}

        BlockScan(temp_storage).ExclusiveSum(value, value, run_aggregate);
        value += s_prv_run_aggregate;
        if (index < size) {
            data[index] = value;
        }
        __syncthreads();
        if (threadIdx.x == blockThreads - 1) {
            s_prv_run_aggregate += run_aggregate;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        data[size] = s_prv_run_aggregate;
    }*/
}

}  // namespace rxmesh