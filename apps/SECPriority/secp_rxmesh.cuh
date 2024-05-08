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
    constexpr uint32_t blockThreads = 256;
    auto coords = rx.get_input_vertex_coordinates();
    LaunchBox<blockThreads> launch_box;

    PriorityQueue_t pq(rx.get_num_edges());

    auto e_pop_attr = rx.add_edge_attribute<bool>("ePop", false);

#if USE_POLYSCOPE
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    // polyscope::show();
#endif

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

    float reduce_ratio = 0.1f;

    // Mark the edge attributes to be collapsed
    uint32_t pop_num_edges = reduce_ratio * rx.get_num_edges();
    RXMESH_TRACE("pop_num_edges: {}", pop_num_edges);

    constexpr uint32_t threads_per_block = 1024;
    uint32_t number_of_blocks = (pop_num_edges + threads_per_block - 1) / threads_per_block;
    int shared_mem_bytes = pq.get_shmem_size(threads_per_block) +
                           (threads_per_block * sizeof(PriorityPair_t));
    RXMESH_TRACE("threads_per_block: {}", threads_per_block);
    RXMESH_TRACE("number_of_blocks: {}", number_of_blocks);
    RXMESH_TRACE("shared_mem_bytes: {}", shared_mem_bytes);

    pop_and_mark_edges_to_collapse<threads_per_block>
        <<<number_of_blocks, threads_per_block, shared_mem_bytes>>>
            (pq.get_mutable_device_view(),
             *e_pop_attr,
             pop_num_edges);

    cudaDeviceSynchronize();
    RXMESH_TRACE("Made it past cudaDeviceSynchronize()");
    
}