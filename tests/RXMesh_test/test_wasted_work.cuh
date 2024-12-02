#include "gtest/gtest.h"


#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/util/report.h"

template <uint32_t blockThreads>
__global__ static void measure_wasted_work_kernel(rxmesh::Context context,
                                                  uint32_t*       d_scheduled,
                                                  uint32_t*       d_wasted,
                                                  bool            flag)
{
    using namespace rxmesh;

    if (threadIdx.x == 0) {
        const uint32_t block_id = blockIdx.x;
        uint32_t       patch_id = context.m_patch_scheduler.pop();
        uint32_t       neighbour_locked[PatchStash::stash_size];

        if (patch_id != INVALID32) {
            PatchInfo patch_info = context.m_patches_info[patch_id];

            for (uint8_t n = 0; n < PatchStash::stash_size; ++n) {
                neighbour_locked[n] = INVALID32;
            }

            // release this patch (patch_id) lock and any neighbor patch we
            // locked for this patch (patch_id)
            auto release = [&]() {
                for (uint8_t n = 0; n < PatchStash::stash_size; ++n) {
                    if (neighbour_locked[n] != INVALID32) {
                        context.m_patches_info[neighbour_locked[n]]
                            .lock.release_lock();
                    }
                }
                // and finally release the lock of this patch
                patch_info.lock.release_lock();
            };


            if (patch_info.patch_id != INVALID32) {
                // try to lock this patch

                if (!patch_info.lock.acquire_lock(block_id)) {
                    // if we can not, we add it again to the queue
                    context.m_patch_scheduler.push(patch_id);
                } else {

                    ::atomicAdd(d_scheduled, 1u);
                    // loop over all neighbor patches to this patch (patch_id)
                    for (uint8_t n = 0; n < PatchStash::stash_size; ++n) {
                        uint32_t q = patch_info.patch_stash.get_patch(n);

                        if (q != INVALID32) {
                            // try to acquire the lock of another patch
                            if (!context.m_patches_info[q].lock.acquire_lock(
                                    block_id)) {
                                ::atomicAdd(d_wasted, 1u);
                                // readding the patch to the queue
                                context.m_patch_scheduler.push(patch_id);
                                break;
                            } else {
                                neighbour_locked[n] = q;
                            }
                        }
                    }

                    // make sure to release the lock before finishing
                    release();
                }
            }
        }
    }

    if (flag) {
        // this is just here so cuda allocate registers and shared memory
        auto block = cooperative_groups::this_thread_block();

        ShmemAllocator shrd_alloc;

        CavityManager<blockThreads, CavityOp::E> cavity(
            block, context, shrd_alloc, false);

        for_each_edge(cavity.patch_info(),
                      [&](const EdgeHandle eh) { cavity.create(eh); });

        block.sync();

        if (cavity.prologue(block, shrd_alloc)) {


            cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
                assert(size == 4);

                DEdgeHandle new_edge =
                    cavity.add_edge(cavity.get_cavity_vertex(c, 1),
                                    cavity.get_cavity_vertex(c, 3));


                cavity.add_face(cavity.get_cavity_edge(c, 0),
                                new_edge,
                                cavity.get_cavity_edge(c, 3));


                cavity.add_face(cavity.get_cavity_edge(c, 1),
                                cavity.get_cavity_edge(c, 2),
                                new_edge.get_flip_dedge());
            });
        }

        cavity.epilogue(block);
    }
}

TEST(RXMeshDynamic, MeasureWastedWork)
{
    using namespace rxmesh;

    auto prop = cuda_query(rxmesh_args.device_id);

    RXMeshDynamic rx(rxmesh_args.obj_file_name);


    Report report("MeasureWastedWork");
    report.command_line(rxmesh_args.argc, rxmesh_args.argv);
    report.device();
    report.system();
    report.model_data(rxmesh_args.obj_file_name, rx);


    uint32_t* d_scheduled;
    CUDA_ERROR(cudaMalloc((void**)&d_scheduled, sizeof(uint32_t)));

    uint32_t* d_wasted;
    CUDA_ERROR(cudaMalloc((void**)&d_wasted, sizeof(uint32_t)));

    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box(
        {}, launch_box, (void*)measure_wasted_work_kernel<blockThreads>);

    int num_blocks;
    CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks,
        (void*)measure_wasted_work_kernel<blockThreads>,
        blockThreads,
        launch_box.smem_bytes_dyn));
    num_blocks *= prop.multiProcessorCount;

    uint32_t            done_patches = 0;
    std::vector<double> ratio;

    while (!rx.is_queue_empty()) {

        CUDA_ERROR(cudaMemset(d_scheduled, 0, sizeof(uint32_t)));
        CUDA_ERROR(cudaMemset(d_wasted, 0, sizeof(uint32_t)));

        measure_wasted_work_kernel<blockThreads>
            <<<num_blocks, launch_box.num_threads, launch_box.smem_bytes_dyn>>>(
                rx.get_context(),
                d_scheduled,
                d_wasted,
                rx.get_num_patches() > rx.get_num_faces());

        uint32_t h_scheduled, h_wasted;
        CUDA_ERROR(cudaMemcpy(&h_scheduled,
                              d_scheduled,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        CUDA_ERROR(cudaMemcpy(
            &h_wasted, d_wasted, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        done_patches += h_scheduled - h_wasted;
        RXMESH_INFO("Scheduled {}, Wasted= {}", h_scheduled, h_wasted);
        ratio.push_back(double(h_wasted) / double(h_scheduled));
    }

    ASSERT_EQ(done_patches, rx.get_num_patches());

    GPU_FREE(d_scheduled);
    GPU_FREE(d_wasted);

    report.add_member("WastedWorkRatio", ratio);

    report.write(rxmesh_args.output_folder + "/WastedWork",
                 extract_file_name(rxmesh_args.obj_file_name));
}
