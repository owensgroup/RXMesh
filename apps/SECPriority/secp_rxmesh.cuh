#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>


#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_dynamic.h"

// Priority Queue related includes
#include <cuco/priority_queue.cuh>
#include <cuco/detail/pair.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/**
 * @brief Return unique index of the local mesh element composed by the
 * patch id and the local index
 *
 * @param local_id the local within-patch mesh element id
 * @param patch_id the patch owning the mesh element
 * @return
 */
constexpr __device__ __host__ __forceinline__ uint32_t
unique_id32(const uint16_t local_id, const uint16_t patch_id)
{
    uint32_t ret = patch_id;
    ret          = (ret << 16);
    ret |= local_id;
    return ret;
}

/**
 * @brief unpack a 32 uint to its high and low 16 bits. 
 * This is used to convert the unique id to its local id (16
 * low bit) and patch id (high 16 bit)
 * @param uid unique id
 * @return a std::pair storing the patch id and local id
 */
constexpr __device__ __host__ __forceinline__ std::pair<uint16_t, uint16_t>
                                              unpack32(uint32_t uid)
{
    uint16_t local_id = uid & ((1 << 16) - 1);
    uint16_t patch_id = uid >> 16;
    return std::make_pair(patch_id, local_id);
}

// Priority queue setup. Use 'pair_less' to prioritize smaller values.
template <typename T>
struct pair_less 
{
    __host__ __device__ bool operator()(const T& a, const T& b) const
    {
        return a.first < b.first;
    }
};

using PriorityPair_t        = cuco::pair<float, uint32_t>;
using PriorityCompare       = pair_less<PriorityPair_t>;
using PriorityQueue_t       = cuco::priority_queue<PriorityPair_t, PriorityCompare>;
using PQView_t              = PriorityQueue_t::device_mutable_view;


template <typename T>
using Vec3 = glm::vec<3, T, glm::defaultp>;

using EdgeStatus = int8_t;
enum : EdgeStatus
{
    UNSEEN = 0,  // means we have not tested it before for e.g., split/flip/col
    OKAY   = 1,  // means we have tested it and it is okay to skip
    UPDATE = 2,  // means we should update it i.e., we have tested it before
    ADDED  = 3,  // means it has been added to during the split/flip/collapse
};

#include "secp_kernels.cuh"


inline void secp_rxmesh(rxmesh::RXMeshDynamic& rx,
                       const uint32_t         final_num_faces)
{
    EXPECT_TRUE(rx.validate());

    using namespace rxmesh;
    constexpr uint32_t blockThreads = 32;


    auto coords = rx.get_input_vertex_coordinates();

    auto edge_status = rx.add_edge_attribute<EdgeStatus>("EdgeStatus", 1);

    LaunchBox<blockThreads> launch_box;

    float total_time   = 0;
    float app_time     = 0;
    float slice_time   = 0;
    float cleanup_time = 0;

    const int num_bins = 256;

    PriorityQueue_t pq(rx.get_num_edges());

    auto e_attr = rx.add_edge_attribute<float>("eMark", 1);


#if USE_POLYSCOPE
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    // polyscope::show();
#endif

    bool validate = false;

    int* d_num_cavities = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_num_cavities, 2 * sizeof(int)));
    CUDA_ERROR(cudaMemset(d_num_cavities, 0, 2 * sizeof(int)));

    float reduce_ratio = 0.05;

    CUDA_ERROR(cudaProfilerStart());
    GPUTimer timer;
    timer.start();
    while (rx.get_num_faces() > final_num_faces) {

    rx.prepare_launch_box({Op::EV},
                          launch_box,
                          (void*)compute_edge_priorities<float, blockThreads>,
                          false,
                          false,
                          false,
                          false,
                          [&](uint32_t v, uint32_t e, uint32_t f){
                            // Allocate enough additional memory
                            // for the priority queue and the intermediate
                            // array of PriorityPait_t.
                            return pq.get_shmem_size(blockThreads) + (e*sizeof(PriorityPair_t));
                          });

    RXMESH_TRACE("pair_alignment<float, rxmesh::EdgeHandle>(){}", cuco::detail::pair_alignment<float, rxmesh::EdgeHandle>());
    RXMESH_TRACE("pair_alignment<float, uint32_t>(){}", cuco::detail::pair_alignment<float, uint32_t>());
    RXMESH_TRACE("pair_alignment<float, uint64_t>(){}", cuco::detail::pair_alignment<float, uint64_t>());
    RXMESH_TRACE("sizeof(PriorityPair_t){}", sizeof(PriorityPair_t));
    compute_edge_priorities<float, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords, pq.get_mutable_device_view(), pq.get_shmem_size(blockThreads));

    cudaDeviceSynchronize();
    RXMESH_TRACE("launch_box.smem_bytes_dyn = {}", launch_box.smem_bytes_dyn);
    RXMESH_TRACE("pq.get_shmem_size = {}", pq.get_shmem_size(blockThreads));

    // next kernel needs to pop some percentage of the top
    // elements in the priority queue and store popped elements
    // to be used by the next kernel that actually does the collapses
    //
    // mark some sort of
    // associated edge attribute

    // now pop all the elements to ouput on host
    //
    thrust::device_vector<PriorityPair_t> d_popped(rx.get_num_edges());
    pq.pop(d_popped.begin(), d_popped.end());
    cudaDeviceSynchronize();
    const thrust::host_vector<PriorityPair_t> h_popped(d_popped);
   // for(size_t i = 0; i < h_popped.size(); i++)
   // {
   //     std::cout << i << "\t" << h_popped[i].first
   //         << "\t" << h_popped[i].second << "\n";
   // }
    return;
        // compute max-min histogram
        //histo.init();

        //rx.prepare_launch_box({Op::EV},
        //                      launch_box,
        //                      (void*)compute_min_max_cost<float, blockThreads>,
        //                      false);
        //compute_min_max_cost<float, blockThreads>
        //    <<<launch_box.blocks,
        //       launch_box.num_threads,
        //       launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords, histo);

        //// compute histogram bins
        //rx.prepare_launch_box({Op::EV},
        //                      launch_box,
        //                      (void*)populate_histogram<float, blockThreads>,
        //                      false);
        //populate_histogram<float, blockThreads>
        //    <<<launch_box.blocks,
        //       launch_box.num_threads,
        //       launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords, histo);

        //histo.scan();


        // how much we can reduce the number of edge at each iterations
        reduce_ratio = reduce_ratio + 0.05;

        // loop over the mesh, and try to collapse
        const int reduce_threshold =
            std::max(1, int(reduce_ratio * float(rx.get_num_edges())));

        // reset edge status
        edge_status->reset(UNSEEN, DEVICE);

        rx.reset_scheduler();
        while (!rx.is_queue_empty() && rx.get_num_faces() > final_num_faces) {
            RXMESH_INFO(" Queue size = {}",
                        rx.get_context().m_patch_scheduler.size());

            rx.prepare_launch_box(
                {Op::EV},
                launch_box,
                (void*)secp<float, blockThreads>,
                true,
                false,
                false,
                false,
                [&](uint32_t v, uint32_t e, uint32_t f) {
                    return detail::mask_num_bytes(e) +
                           2 * detail::mask_num_bytes(v) +
                           3 * ShmemAllocator::default_alignment;
                });

            e_attr->reset(0, DEVICE);

            GPUTimer app_timer;
            app_timer.start();
            secp<float, blockThreads>
                 <<<launch_box.blocks,
                    launch_box.num_threads,
                    launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                 *coords,
                                                 //histo,
                                                 reduce_threshold,
                                                 *edge_status,
                                                 *e_attr,
                                                 d_num_cavities);

            app_timer.stop();

            GPUTimer cleanup_timer;
            cleanup_timer.start();
            rx.cleanup();
            cleanup_timer.stop();

            GPUTimer slice_timer;
            slice_timer.start();
            rx.slice_patches(*coords, *edge_status, *e_attr);
            slice_timer.stop();

            GPUTimer cleanup_timer2;
            cleanup_timer2.start();
            rx.cleanup();
            cleanup_timer2.stop();


            CUDA_ERROR(cudaDeviceSynchronize());
            CUDA_ERROR(cudaGetLastError());

            app_time += app_timer.elapsed_millis();
            slice_time += slice_timer.elapsed_millis();
            cleanup_time += cleanup_timer.elapsed_millis();
            cleanup_time += cleanup_timer2.elapsed_millis();


            if (validate) {
                rx.update_host();
                EXPECT_TRUE(rx.validate());
                RXMESH_INFO(" num_vertices = {}, num_edges= {}, num_faces= {}",
                            rx.get_num_vertices(),
                            rx.get_num_edges(),
                            rx.get_num_faces());
            }
        }

        {
            int h_num_cavities[2];
            CUDA_ERROR(cudaMemcpy(&h_num_cavities,
                                  d_num_cavities,
                                  2 * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            RXMESH_INFO(" Requested cavities = {}, executed cavities  = {}",
                        h_num_cavities[1],
                        h_num_cavities[0]);

            coords->move(DEVICE, HOST);
            e_attr->move(DEVICE, HOST);
            rx.update_host();
            rx.update_polyscope();
            auto ps_mesh = rx.get_polyscope_mesh();
            ps_mesh->updateVertexPositions(*coords);
            ps_mesh->setEnabled(false);
            ps_mesh->addEdgeScalarQuantity("eMark", *e_attr);
            rx.render_vertex_patch();
            rx.render_edge_patch();
            rx.render_face_patch();
            polyscope::show();
        }
    }
    timer.stop();
    total_time += timer.elapsed_millis();
    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO("secp_rxmesh() RXMesh simplification took {} (ms)", total_time);
    RXMESH_INFO("secp_rxmesh() App time {} (ms)", app_time);
    RXMESH_INFO("secp_rxmesh() Slice timer {} (ms)", slice_time);
    RXMESH_INFO("secp_rxmesh() Cleanup timer {} (ms)", cleanup_time);

    if (!validate) {
        rx.update_host();
    }
    coords->move(DEVICE, HOST);

#if USE_POLYSCOPE
    rx.update_polyscope();

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(*coords);
    ps_mesh->setEnabled(false);

    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    polyscope::show();
#endif

//    histo.free();
}