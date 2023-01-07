#include "gtest/gtest.h"

#include "rxmesh/patch_lock.h"
#include "rxmesh/rxmesh_dynamic.h"

__global__ static void lock_kernel(rxmesh::Context context, uint32_t* d_status)
{
    using namespace rxmesh;

    const uint32_t id          = blockIdx.x;
    const uint32_t num_patches = context.m_num_patches[0];
    PatchInfo      patch       = context.m_patches_info[id];

    bool st = true;

    if (threadIdx.x == 0) {
        // lock my patch before anything else
        st = patch.lock.acquire_lock(id);
        if (st) {
            // loop over all other patches this block want to lock
            for (uint32_t p = 0; p < num_patches; ++p) {
                // make sure it is not this block's patch
                if (p != id) {
                    // try to acquire the lock of another patch
                    st = context.m_patches_info[p].lock.acquire_lock(id);
                    if (!st) {
                        // if failed, we need to release the lock all the
                        // patches that this block has locked
                        patch.lock.release_lock();
                        for (uint32_t q = 0; q < p; ++q) {
                            if (p != id) {
                                context.m_patches_info[q].lock.release_lock();
                            }
                        }
                        break;
                    }
                }
            }
        }

        d_status[id] = st;
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

    lock_kernel<<<rx.get_num_patches(), 256>>>(rx.get_context(), d_status);

    CUDA_ERROR(cudaDeviceSynchronize());


    std::vector<uint32_t> h_status(rx.get_num_patches());
    CUDA_ERROR(cudaMemcpy(h_status.data(),
                          d_status,
                          h_status.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    GPU_FREE(d_status);
}
