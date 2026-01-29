#include "rxmesh/patch_scheduler.h"

#include <cassert>
#include <cstdio>
#include <vector>

#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

namespace rxmesh {

__device__ bool PatchScheduler::push(const uint32_t pid)
{
#ifdef __CUDA_ARCH__
#ifdef PROCESS_SINGLE_PATCH
    return true;
#else
    assert(pid != INVALID32);
    if (::atomicAdd(count, 1) < static_cast<int>(capacity)) {
        int pos = ::atomicAdd(back, 1) % capacity;

        // the loop because another thread/block may have just decremented
        // the count but has not yet finish reading from the list
        while (::atomicCAS(list + pos, INVALID32, pid) != INVALID32) {
            // TODO do we really need to sleep if it is only one thread
            // in the block doing the job??
            //__nanosleep(10);
        }
        return true;
    } else {
        // for our configuration, this should not happen since a block only
        // pop a patch and, if not able to process it due to dependency
        // conflict, the block push the same patch again. So at all times
        // count is less than capacity if capacity is total number of
        // patches
        assert(0);
        return false;
    }


#endif
#else
    // to silence the compiler warning
    return true;
#endif
}

__device__ uint32_t PatchScheduler::pop()
{
#ifdef __CUDA_ARCH__
#ifdef PROCESS_SINGLE_PATCH
    return blockIdx.x;
#else
    int readable = ::atomicSub(count, 1);

    uint32_t pid = INVALID32;

    if (readable <= 0) {
        ::atomicAdd(count, 1);
    } else {

        int pos = ::atomicAdd(front, 1) % capacity;

        // the loop because another thread/block may have just incremented
        // the count but has not yet wrote to the list
        while (pid == INVALID32) {
            pid = atomicExch(list + pos, INVALID32);
            // TODO do we really need to sleep if it is only one thread
            // in the block doing the job??
            //__nanosleep(10);
        }
    }
    return pid;
#endif
#else
    // to silence the compiler warning
    return INVALID32;
#endif
}

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

__host__ __device__ int PatchScheduler::size(cudaStream_t stream) const
{
#ifdef __CUDA_ARCH__
    return count[0];
#else
    int h_count = 0;
    CUDA_ERROR(cudaMemcpyAsync(
        &h_count, count, sizeof(int), cudaMemcpyDeviceToHost, stream));
    return h_count;
#endif
}

__host__ __device__ bool PatchScheduler::is_empty(cudaStream_t stream)
{
    return size(stream) == 0;
}

}  // namespace rxmesh
