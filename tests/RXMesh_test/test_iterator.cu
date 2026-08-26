#include <array>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "rxmesh/iterator.cuh"
#include "rxmesh/util/util.h"

template <typename HandleT>
__global__ static void test_iterator(uint32_t*       num_failures,
                                     const uint16_t* patch_output,
                                     const uint16_t* patch_offset,
                                     const uint32_t* output_owned_bitmask,
                                     const uint32_t  num_elements,
                                     const uint32_t  offset_size,
                                     const uint32_t  patch_id)
{
    using namespace rxmesh;

    const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_elements) {
        return;
    }
    const uint16_t local_id = static_cast<uint16_t>(thread_id);

    LPHashTable       ht;
    const PatchStash  stash;
    Iterator<HandleT> iter(
        local_id,
        reinterpret_cast<const typename HandleT::LocalT*>(patch_output),
        patch_offset,
        offset_size,
        patch_id,
        output_owned_bitmask,
        ht,
        nullptr,
        stash);

    const uint16_t expected_size =
        offset_size == 0 ? static_cast<uint16_t>(patch_offset[local_id + 1] -
                                                 patch_offset[local_id]) :
                           static_cast<uint16_t>(offset_size);
    if (iter.size() != expected_size) {
        atomicAdd(num_failures, 1u);
        return;
    }

    if (expected_size == 0) {
        if (iter.front().is_valid() || iter.back().is_valid() ||
            iter[0].is_valid() || iter.local(0) != INVALID16) {
            atomicAdd(num_failures, 1u);
        }
        return;
    }

    const HandleT truth(patch_id, {local_id});
    if (iter.front() != truth || iter.back() != truth) {
        atomicAdd(num_failures, 1u);
        return;
    }

    for (uint16_t i = 0; i < iter.size(); ++i) {
        if (iter[i] != truth || iter.local(i) != local_id) {
            atomicAdd(num_failures, 1u);
            return;
        }
    }

    if (iter[iter.size()].is_valid() || iter.local(iter.size()) != INVALID16) {
        atomicAdd(num_failures, 1u);
    }
}

TEST(RXMesh, Iterator)
{
    // The patch contains 32 elements and the patch_id is 1
    // and patch_output:
    // 0 0 0 | 1 1 1 | 2 2 2 | ......
    // i.e., fixed_offset = 3

    using namespace rxmesh;
    constexpr uint32_t offset_size      = 3;
    constexpr uint32_t num_elements     = 32;
    constexpr uint32_t num_csr_elements = 3;
    constexpr uint32_t patch_id         = 1;

    std::vector<uint16_t> h_patch_output(offset_size * num_elements);
    for (uint32_t i = 0; i < h_patch_output.size(); ++i) {
        h_patch_output[i] = i / offset_size;
    }

    const std::vector<uint16_t> h_csr_output = {0, 0, 2, 2, 2};
    const std::vector<uint16_t> h_csr_offset = {0, 2, 2, 5};

    uint32_t* d_num_failures(nullptr);
    uint32_t* d_owned_mask(nullptr);
    uint16_t* d_patch_output(nullptr);
    uint16_t* d_patch_offset(nullptr);

    CUDA_ERROR(cudaMalloc((void**)&d_patch_output,
                          h_patch_output.size() * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_patch_offset,
                          h_csr_offset.size() * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_owned_mask, sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_owned_mask, 0xFF, sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_num_failures, sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_num_failures, 0, sizeof(uint32_t)));

    CUDA_ERROR(cudaMemcpy(d_patch_output,
                          h_patch_output.data(),
                          h_patch_output.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));

    test_iterator<VertexHandle><<<1, num_elements>>>(d_num_failures,
                                                     d_patch_output,
                                                     nullptr,
                                                     d_owned_mask,
                                                     num_elements,
                                                     offset_size,
                                                     patch_id);
    test_iterator<TetHandle><<<1, num_elements>>>(d_num_failures,
                                                  d_patch_output,
                                                  nullptr,
                                                  d_owned_mask,
                                                  num_elements,
                                                  offset_size,
                                                  patch_id);

    CUDA_ERROR(cudaMemcpy(d_patch_output,
                          h_csr_output.data(),
                          h_csr_output.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_patch_offset,
                          h_csr_offset.data(),
                          h_csr_offset.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));

    test_iterator<VertexHandle><<<1, num_csr_elements>>>(d_num_failures,
                                                         d_patch_output,
                                                         d_patch_offset,
                                                         d_owned_mask,
                                                         num_csr_elements,
                                                         0,
                                                         patch_id);
    test_iterator<TetHandle><<<1, num_csr_elements>>>(d_num_failures,
                                                      d_patch_output,
                                                      d_patch_offset,
                                                      d_owned_mask,
                                                      num_csr_elements,
                                                      0,
                                                      patch_id);
    CUDA_ERROR(cudaDeviceSynchronize());

    uint32_t h_num_failures = 0;
    CUDA_ERROR(cudaMemcpy(&h_num_failures,
                          d_num_failures,
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_num_failures, 0);

    CUDA_ERROR(cudaFree(d_patch_output));
    CUDA_ERROR(cudaFree(d_patch_offset));
    CUDA_ERROR(cudaFree(d_owned_mask));
    CUDA_ERROR(cudaFree(d_num_failures));
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaDeviceReset());
}
