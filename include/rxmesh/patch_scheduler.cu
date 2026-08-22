#include "rxmesh/patch_scheduler.h"

#include <cassert>
#include <cstdio>
#include <vector>

#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

namespace rxmesh {
__host__ void PatchScheduler::refill(const uint32_t size)
{
    static std::vector<uint32_t> h_list(capacity);
    if (h_list.size() < capacity) {
        h_list.resize(capacity);
    }
    fill_with_sequential_numbers(h_list.data(), size);
    random_shuffle(h_list.data(), size);
    std::fill(h_list.begin() + size, h_list.end(), INVALID32);
    CUDA_ERROR(cudaMemcpy(list,
                          h_list.data(),
                          capacity * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(
        cudaMemcpy(count, &size, sizeof(int), cudaMemcpyHostToDevice));

    CUDA_ERROR(
        cudaMemcpy(back, &size, sizeof(int), cudaMemcpyHostToDevice));

    CUDA_ERROR(cudaMemset(front, 0, sizeof(int)));
}

__host__ void PatchScheduler::init(uint32_t cap)
{
    capacity = cap;
    CUDA_ERROR(cudaMalloc((void**)&count, sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&front, sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&back, sizeof(int)));
    CUDA_ERROR(cudaMalloc((void**)&list, sizeof(uint32_t) * capacity));
}

__host__ void PatchScheduler::print_list() const
{
    std::vector<uint32_t> h_list(capacity);
    CUDA_ERROR(cudaMemcpy(h_list.data(),
                          list,
                          h_list.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < h_list.size(); ++i) {
        printf("\n list[%u]= %u", i, h_list[i]);
    }
}

__host__ void PatchScheduler::free()
{
    GPU_FREE(count);
    GPU_FREE(front);
    GPU_FREE(back);
    GPU_FREE(list);
}

}  // namespace rxmesh
