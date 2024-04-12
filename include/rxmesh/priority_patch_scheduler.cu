#include "rxmesh/util/util.h"

#include "cub/device/device_merge_sort.cuh"

#include "rxmesh/priority_patch_scheduler.cuh"

namespace rxmesh {


__host__ void PriorityPatchScheduler::sort(cudaStream_t stream)
{
    LessThan less_than;

    int h_front(0), h_back(0);
    CUDA_ERROR(cudaMemcpyAsync(
        &h_front, front, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_ERROR(cudaMemcpyAsync(
        &h_back, back, sizeof(int), cudaMemcpyDeviceToHost, stream));

    if (h_back > h_front) {
        size_t current_size = h_back - h_front;

        cub::DeviceMergeSort::SortPairs(sort_temp_storage,
                                        sort_temp_storage_bytes,
                                        priority + h_front,
                                        list + h_front,
                                        current_size,
                                        less_than,
                                        stream);
    } else {
        assert(h_front < capacity);
        size_t front_size = capacity - h_front;

        cub::DeviceMergeSort::SortPairs(sort_temp_storage,
                                        sort_temp_storage_bytes,
                                        priority + h_front,
                                        list + h_front,
                                        front_size,
                                        less_than,
                                        stream);

        cub::DeviceMergeSort::SortPairs(sort_temp_storage,
                                        sort_temp_storage_bytes,
                                        priority,
                                        list,
                                        h_back,
                                        less_than,
                                        stream);
    }
}


__host__ void PriorityPatchScheduler::init(uint32_t cap)
{
    capacity = cap;
    CUDA_ERROR(cudaMalloc((void**)&count, sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&front, sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&back, sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&list, sizeof(uint32_t) * capacity));
    CUDA_ERROR(cudaMalloc((void**)&priority, sizeof(PriorityT) * capacity));

    LessThan less_than;

    sort_temp_storage_bytes = 0;
    cub::DeviceMergeSort::SortPairs(sort_temp_storage,
                                    sort_temp_storage_bytes,
                                    priority,
                                    list,
                                    capacity,
                                    less_than);
}
}  // namespace rxmesh