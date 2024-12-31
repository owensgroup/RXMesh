#include "gtest/gtest.h"
#include "rxmesh/iterator.cuh"
#include "rxmesh/util/util.h"

template <typename HandleT>
__global__ static void test_iterator(uint32_t*       suceess,
                                     const uint16_t* patch_output,
                                     const uint32_t  num_elements,
                                     const uint32_t  offset_size,
                                     const uint32_t  patch_id)
{
    using namespace rxmesh;
    uint16_t      local_id = threadIdx.x;
    const HandleT truth(patch_id, {local_id});

    // empty context to just get things working
    Context context;

    if (local_id >= num_elements) {
        LPHashTable       ht;
        PatchStash        stash = PatchStash();
        Iterator<HandleT> iter(
            context,
            local_id,
            reinterpret_cast<const typename HandleT::LocalT*>(patch_output),
            nullptr,
            offset_size,
            patch_id,
            nullptr,
            ht,
            nullptr,
            stash);

        if (iter.size() != offset_size) {
            atomicAdd(suceess, 1u);
            return;
        }
        if (iter.front() != truth) {
            atomicAdd(suceess, 1u);
            return;
        }

        if (iter.back() != truth) {
            atomicAdd(suceess, 1u);
            return;
        }

        for (uint32_t i = 0; i < iter.size(); ++i) {
            if (iter[i] != truth) {
                atomicAdd(suceess, 1u);
                return;
            }
        }        
    }
}

TEST(RXMesh, Iterator)
{
    // The patch contains 32 elements and the patch_id is 1
    // and patch_output:
    // 0 0 0 | 1 1 1 | 2 2 2 | ......
    // i.e., fixed_offset = 3

    using namespace rxmesh;
    constexpr uint32_t offset_size  = 3;
    const uint32_t     num_elements = 32;
    const uint32_t     patch_id     = 1;

    std::vector<uint16_t> h_patch_output(offset_size * num_elements);
    for (uint32_t i = 0; i < h_patch_output.size(); ++i) {
        h_patch_output[i] = i / offset_size;
    }


    uint32_t* d_suceess(nullptr);
    uint16_t* d_patch_output(nullptr);

    CUDA_ERROR(cudaMalloc((void**)&d_patch_output,
                          h_patch_output.size() * sizeof(uint32_t)));

    CUDA_ERROR(cudaMemcpy(d_patch_output,
                          h_patch_output.data(),
                          h_patch_output.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMalloc((void**)&d_suceess, sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_suceess, 0, sizeof(uint32_t)));


    test_iterator<VertexHandle><<<1, num_elements>>>(
        d_suceess, d_patch_output, num_elements, offset_size, patch_id);
    CUDA_ERROR(cudaDeviceSynchronize());

    uint32_t h_success = 0;
    CUDA_ERROR(cudaMemcpy(
        &h_success, d_suceess, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_success, 0);

    CUDA_ERROR(cudaFree(d_patch_output));
    CUDA_ERROR(cudaFree(d_suceess));
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaDeviceReset());
}