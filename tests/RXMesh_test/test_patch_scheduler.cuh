#include "gtest/gtest.h"

#include <numeric>

#include "rxmesh/patch_scheduler.h"

__global__ void schedule_kernel(uint32_t* d_status, rxmesh::PatchScheduler sch)
{
    // the first thread in the block first pop a patch id, sync with the whole
    // block, the push the same patch again
    using namespace rxmesh;
    uint32_t pid;
    if (blockIdx.x < sch.capacity) {

        if (threadIdx.x == 0) {
            pid = sch.pop();
            if (pid == INVALID32) {
                ::atomicAdd(d_status, 1);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            bool pushed = sch.push(pid);
            if (!pushed) {
                ::atomicAdd(d_status, 1);
            }
        }
    }
}

TEST(RXMeshDynamic, PatchScheduler)
{
    using namespace rxmesh;

    auto prop = cuda_query(rxmesh_args.device_id);

    uint32_t* d_status;
    CUDA_ERROR(cudaMalloc((void**)&d_status, sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_status, 0, sizeof(uint32_t)));

    // make sure to have more blocks launched than number of SM we have
    // on the GPU just as a stress test 
    const uint32_t num_patches = prop.multiProcessorCount * 5;

    PatchScheduler sch;
    sch.init(num_patches);
    sch.refill(num_patches);

    for (int i = 0; i < 1000; ++i) {

        schedule_kernel<<<num_patches, 256>>>(d_status, sch);

        uint32_t h_status = 0;
        CUDA_ERROR(cudaMemcpy(
            &h_status, d_status, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        uint32_t h_count = 0;
        CUDA_ERROR(cudaMemcpy(
            &h_count, sch.count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        EXPECT_EQ(h_status, 0);
        EXPECT_EQ(h_count, num_patches);
    }

    GPU_FREE(d_status);
    sch.free();
}
