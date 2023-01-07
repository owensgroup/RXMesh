#include "gtest/gtest.h"

#include <numeric>

#include "rxmesh/patch_lock.h"
#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/util/util.h"

__global__ static void lock_kernel(rxmesh::Context context,
                                   uint32_t*       d_status,
                                   uint32_t*       d_block_patch)
{
    using namespace rxmesh;

    const uint32_t block_id       = blockIdx.x;
    const uint32_t num_patches    = context.m_num_patches[0];
    const uint32_t block_patch_id = d_block_patch[block_id];
    PatchInfo      patch          = context.m_patches_info[block_patch_id];

    bool st = true;

    if (threadIdx.x == 0) {
        // lock my patch before anything else
        st = patch.lock.acquire_lock(block_id);
        if (st) {
            // loop over all other patches this block want to lock
            for (uint32_t p = 0; p < num_patches; ++p) {
                // make sure it is not this block's patch
                if (p != block_patch_id) {
                    // try to acquire the lock of another patch
                    st = context.m_patches_info[p].lock.acquire_lock(block_id);

                    if (!st) {
                        // if failed, we need to release the lock all the
                        // patches that this block has locked
                        patch.lock.release_lock();
                        for (uint32_t q = 0; q < p; ++q) {
                            if (q != block_patch_id) {
                                context.m_patches_info[q].lock.release_lock();
                            }
                        }
                        break;
                    }
                }
            }
        }

        if (st) {
            // make sure to release all the locks if all are successful
            patch.lock.release_lock();
            for (uint32_t p = 0; p < num_patches; ++p) {
                if (p != block_patch_id) {
                    context.m_patches_info[p].lock.release_lock();
                }
            }
        }
        d_status[block_id] = st;
    }
}

TEST(RXMeshDynamic, PatchLock)
{
    using namespace rxmesh;

    auto prop = cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshDynamic rx(STRINGIFY(INPUT_DIR) "cloth.obj", rxmesh_args.quite);

    ASSERT_GE(prop.multiProcessorCount, rx.get_num_patches());

    uint32_t* d_status;
    CUDA_ERROR(
        cudaMalloc((void**)&d_status, rx.get_num_patches() * sizeof(uint32_t)));
    CUDA_ERROR(
        cudaMemset(d_status, 0, rx.get_num_patches() * sizeof(uint32_t)));

    uint32_t* d_block_patch;
    CUDA_ERROR(cudaMalloc((void**)&d_block_patch,
                          rx.get_num_patches() * sizeof(uint32_t)));

    std::vector<uint32_t> h_status(rx.get_num_patches());
    std::vector<uint32_t> h_block_patch(rx.get_num_patches());
    fill_with_sequential_numbers(h_block_patch.data(), h_block_patch.size());

    for (int i = 0; i < 1000; ++i) {
        random_shuffle(h_block_patch.data(), h_block_patch.size());

        CUDA_ERROR(cudaMemcpy(d_block_patch,
                              h_block_patch.data(),
                              h_status.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        lock_kernel<<<rx.get_num_patches(), 256>>>(
            rx.get_context(), d_status, d_block_patch);

        CUDA_ERROR(cudaMemcpy(h_status.data(),
                              d_status,
                              h_status.size() * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        uint32_t sum = std::accumulate(h_status.begin(), h_status.end(), 0);

        EXPECT_EQ(sum, 1);
    }

    GPU_FREE(d_block_patch);
    GPU_FREE(d_status);
}
