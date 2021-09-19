#include "gtest/gtest.h"
#include "rxmesh/kernels/rxmesh_iterator.cuh"
#include "rxmesh/util/util.h"
template <uint32_t fixedOffset>
__global__ static void test_iterator(uint32_t* suceess,
                                     uint32_t* ltog_map,
                                     uint16_t* patch_output,
                                     uint32_t  num_elements)
{
    using namespace RXMESH;
    uint32_t       local_id = threadIdx.x;
    RXMeshIterator iter(local_id, patch_output, patch_output, ltog_map,
                        fixedOffset, 0);

    if (iter.local_id() != local_id) {
        atomicAdd(suceess, 1u);
        return;
    }

    if (iter.size() != fixedOffset) {
        atomicAdd(suceess, 1u);
        return;
    }

    uint32_t truth = num_elements - threadIdx.x - 1;
    if (iter[0] != truth || iter[1] != truth || iter[2] != truth ||
        iter.back() != truth || iter.front() != truth) {
        atomicAdd(suceess, 1u);
        return;
    }

    for (uint32_t i = 0; i < iter.size(); ++i) {
        if (*iter != truth) {
            atomicAdd(suceess, 1u);
            return;
        }
        ++iter;
    }
}

TEST(RXMesh, Iterator)
{
    // patch_output:
    // 0 0 0 | 1 1 1 |  2  2 2 | ......

    // ltog_map:
    // n-1 n-2 n-3 ..... 3 2 1 0

    // and so the patch_output in global index space should be
    // n-1 n-1 n-1 | n-2 n-2 n-2 | ...... | 1 1 1 | 0 0 0


    using namespace RXMESH;
    constexpr uint32_t fixedOffset = 3;
    const uint32_t     N = 32;

    std::vector<uint16_t> h_patch_output(fixedOffset * N);
    for (uint32_t i = 0; i < h_patch_output.size(); ++i) {
        h_patch_output[i] = i / fixedOffset;
    }

    std::vector<uint32_t> h_ltog_map(N);
    for (uint32_t i = 0; i < h_ltog_map.size(); ++i) {
        h_ltog_map[i] = N - i - 1;
    }


    uint32_t *d_ltog_map(nullptr), *d_suceess(nullptr);
    uint16_t* d_patch_output(nullptr);

    CUDA_ERROR(
        cudaMalloc((void**)&d_ltog_map, h_ltog_map.size() * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_patch_output,
                          h_patch_output.size() * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(d_ltog_map, h_ltog_map.data(),
                          h_ltog_map.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_patch_output, h_patch_output.data(),
                          h_patch_output.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMalloc((void**)&d_suceess, sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_suceess, 0, sizeof(uint32_t)));


    test_iterator<3u><<<1, N>>>(d_suceess, d_ltog_map, d_patch_output, N);
    CUDA_ERROR(cudaDeviceSynchronize());

    uint32_t h_success = 0;
    CUDA_ERROR(cudaMemcpy(&h_success, d_suceess, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_success, 0);

    CUDA_ERROR(cudaFree(d_patch_output));
    CUDA_ERROR(cudaFree(d_suceess));
    CUDA_ERROR(cudaFree(d_ltog_map));
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaDeviceReset());
}