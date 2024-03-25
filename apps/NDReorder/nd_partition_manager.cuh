#pragma once
#include <stdint.h>

#include <cooperative_groups.h>

#include "rxmesh/bitmask.cuh"
#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/util.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/rxmesh_dynamic.h"

#include "rxmesh/attribute.h"

#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

namespace rxmesh {

// TODO: change the uniform shared memory allocation to per level allocation for
// less shared memory use
// TODO: update the init list
template <uint32_t blockThreads>
struct ALIGN(16) PartitionManager
{
    /**
     * @brief Default constructor for PartitionManager.
     *
     * This constructor initializes all member pointers to nullptr.
     */
    __device__ __inline__ PartitionManager()
        : m_s_ev(nullptr),
          m_s_vwgt(nullptr),
          m_s_vdegree(nullptr),
          m_s_vadjewgt(nullptr),
          m_s_mapping(nullptr),
          m_s_ewgt(nullptr),
          m_s_num_vertices(nullptr),
          m_s_num_edges(nullptr)
    {
    }

    __device__ __inline__ PartitionManager(
        cooperative_groups::thread_block& block,
        Context&                          context,
        ShmemAllocator&                   shrd_alloc,
        uint16_t                          req_level);

    __device__ __inline__ void matching(
        cooperative_groups::thread_block& block,
        rxmesh::VertexAttribute<uint16_t> attr_matched_v,
        rxmesh::EdgeAttribute<uint16_t>   attr_active_e,
        uint16_t                          curr_level);

    __device__ __inline__ void coarsening(
        cooperative_groups::thread_block& block,
        uint16_t                          curr_level);

    __device__ __inline__ void uncoarsening(
        cooperative_groups::thread_block& block,
        uint16_t                          curr_level);

    __device__ __inline__ void partition(
        cooperative_groups::thread_block& block,
        uint16_t                          curr_level);

    __device__ __inline__ void genrate_reordering(
        cooperative_groups::thread_block& block,
        rxmesh::VertexAttribute<uint16_t> v_ordering);

    __device__ __inline__ uint16_t* num_vertices_at(uint16_t curr_level)
    {
        return m_s_num_vertices[curr_level];
    }

    __device__ __inline__ uint16_t* num_edges_at(uint16_t curr_level)
    {
        return m_s_num_edges[curr_level];
    }

    __device__ __inline__ uint16_t* get_ev(uint16_t curr_level)
    {
        uint16_t offset = 0;
        for (int i = 0; i < curr_level; ++i) {
            offset += m_s_num_edges[i];
        }
        return m_s_ev + (2 * offset);
    }

    __device__ __inline__ uint16_t* get_mapping(uint16_t curr_level)
    {
        uint16_t offset = 0;
        for (int i = 0; i < curr_level; ++i) {
            offset += m_s_num_vertices[i];
        }
        return m_s_mapping + offset;
    }

    __device__ __inline__ Bitmask& get_matched_edges_bitmask(
        uint16_t curr_level)
    {
        return m_s_matched_edges[curr_level];
    }

    __device__ __inline__ Bitmask& get_matched_vertices_bitmask(
        uint16_t curr_level)
    {
        return m_s_matched_vertices[curr_level];
    }

    __device__ __inline__ Bitmask& get_p0_vertices_bitmask(uint16_t curr_level)
    {
        return m_s_p0_vertices[curr_level];
    }

    __device__ __inline__ Bitmask& get_p1_vertices_bitmask(uint16_t curr_level)
    {
        return m_s_p1_vertices[curr_level];
    }

    __device__ __inline__ Bitmask& get_separator_vertices_bitmask(
        uint16_t curr_level)
    {
        return m_s_separator_vertices[curr_level];
    }

    __device__ __inline__ bool coarsen_owned(rxmesh::LocalEdgeT v_handle,
                                             uint16_t           level)
    {
        if (level == 0) {
            return m_patch_info.is_owned(v_handle);
        }

        return true;
    }

    __device__ __inline__ bool coarsen_owned(rxmesh::LocalVertexT e_handle,
                                             uint16_t             level)
    {
        if (level == 0) {
            return m_patch_info.is_owned(e_handle);
        }

        return true;
    }


   private:
    static constexpr int max_level_size = 10;

    __device__ __inline__ std::array<Bitmask, max_level_size> alloc_bitmask_arr(
        cooperative_groups::thread_block& block,
        ShmemAllocator&                   shrd_alloc,
        uint16_t                          req_arr_size,
        uint16_t                          bm_size)
    {
        assert(max_level_size >= req_arr_size);
        std::array<Bitmask, max_level_size> bm_arr;

        for (int i = 0; i < req_arr_size; ++i) {
            bm_arr[i] = Bitmask(bm_size, shrd_alloc);
            bm_arr[i].reset(block);
        }

        return bm_arr;
    }

    __device__ __inline__ void calc_new_temp_ve(
        cooperative_groups::thread_block& block,
        uint16_t*                         s_ev,
        uint16_t                          num_vertices,
        uint16_t                          num_edges)
    {
        // Copy EV to offset array
        for (uint16_t i = threadIdx.x; i < num_edges * 2; i += blockThreads) {
            m_s_tmp_offset[i] = s_ev[i];
        }
        block.sync();

        // TODO: may use new API later
        const uint32_t* active_mask_e =
            m_patch_info.get_owned_mask<EdgeHandle>();
        detail::v_e<blockThreads>(num_vertices,
                                  num_edges,
                                  m_s_tmp_offset,
                                  m_s_tmp_value,
                                  active_mask_e);
        block.sync();
    }

    __device__ __inline__ void calc_new_temp_vv(
        cooperative_groups::thread_block& block,
        uint16_t*                         s_ev,
        uint16_t                          num_vertices,
        uint16_t                          num_edges)
    {
        calc_new_temp_ve(block, s_ev, num_vertices, num_edges);

        for (uint32_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
            uint32_t start = m_s_tmp_offset[v];
            uint32_t end   = m_s_tmp_offset[v + 1];

            for (uint32_t e = start; e < end; ++e) {
                uint16_t edge = m_s_tmp_value[e];
                uint16_t v0   = s_ev[2 * edge];
                uint16_t v1   = s_ev[2 * edge + 1];

                assert(v0 != INVALID16 && v1 != INVALID16);
                assert(v0 == v || v1 == v);
                // s_output_value[e] = (v0 == v) ? v1 : v0;
                m_s_tmp_value[e] = (v0 == v) * v1 + (v1 == v) * v0;
            }
        }
    }

    __device__ __inline__ uint16_t* get_new_tmp_attribute_v_arr(
        cooperative_groups::thread_block& block,
        uint16_t                          num_vertices,
        uint16_t                          init_val = 0)
    {
        fill_n<blockThreads>(m_s_tmp_attribute_v, num_vertices, init_val);
        block.sync();
        return m_s_tmp_attribute_v;
    }

    __device__ __inline__ uint16_t* get_new_tmp_attribute_e_arr(
        cooperative_groups::thread_block& block,
        uint16_t                          num_edges,
        uint16_t                          init_val = 0)
    {
        fill_n<blockThreads>(m_s_tmp_attribute_e, num_edges, init_val);
        block.sync();
        return m_s_tmp_attribute_e;
    }

    // set as max v size
    __device__ __inline__ Bitmask& get_new_tmp_bitmask_active_v(
        cooperative_groups::thread_block& block)
    {
        m_s_tmp_active_vertices.reset(block);
        block.sync();
        return m_s_tmp_active_vertices;
    }

    // set as max e size
    __device__ __inline__ Bitmask& get_new_tmp_bitmask_active_e(
        cooperative_groups::thread_block& block)
    {
        m_s_tmp_active_edges.reset(block);
        block.sync();
        return m_s_tmp_active_edges;
    }

    __device__ __inline__ void reset_temp_partition_bitmask(
        cooperative_groups::thread_block& block)
    {
        m_s_tmp_assigned_v.reset(block);
        m_s_tmp_current_frontier_v.reset(block);
        m_s_tmp_next_frontier_v.reset(block);
        m_s_tmp_coarse_p0_v.reset(block);
        m_s_tmp_coarse_p1_v.reset(block);
        block.sync();
    }

    // TODO: use public for variable temporary
   public:
    // reference of the basic rxmesh info
    PatchInfo m_patch_info;
    Context   m_context;
    uint16_t  m_patch_id;

    // The topology information: edge incident vertices and face incident
    // edges from the index in the original patchinfo ev
    uint16_t* m_s_ev;

    // Number of mesh elements in the patch in a array format
    // use 32 if the CUDA function requires
    uint16_t* m_s_num_vertices;
    uint16_t* m_s_num_edges;

    // mask indicate the ownership of the v & e for the base layer, may not need
    // this uint32_t *owned_mask_v, *owned_mask_e;

    // vertex related attribute
    uint16_t* m_s_vwgt;
    uint16_t* m_s_vdegree;
    uint16_t* m_s_vadjewgt;

    // edge related attribute
    uint16_t* m_s_ewgt;

    // array stores the mapping relationship
    uint16_t* m_s_mapping;

    // indicates the number of levels we have, notice right now the level size
    // is the max count
    uint16_t m_s_req_level;

    std::array<Bitmask, max_level_size> m_s_matched_edges;
    std::array<Bitmask, max_level_size> m_s_matched_vertices;

    // for partition
    std::array<Bitmask, max_level_size> m_s_p0_vertices;
    std::array<Bitmask, max_level_size> m_s_p1_vertices;
    std::array<Bitmask, max_level_size> m_s_separator_vertices;

    // tmp variable for VE query
    uint16_t* m_s_tmp_offset;
    uint16_t* m_s_tmp_value;

    // tmp variable for vertex attribute
    uint16_t* m_s_tmp_attribute_v;

    // tmp variable for edge attribute
    uint16_t* m_s_tmp_attribute_e;

    // tmp variable for active edges
    Bitmask m_s_tmp_active_edges;

    // tmp variable for active vertices
    Bitmask m_s_tmp_active_vertices;

    // tmp bitmask used by partition
    Bitmask m_s_tmp_assigned_v;
    Bitmask m_s_tmp_current_frontier_v;
    Bitmask m_s_tmp_next_frontier_v;
    Bitmask m_s_tmp_coarse_p0_v;
    Bitmask m_s_tmp_coarse_p1_v;
};

// TODO: destroyer

// VE query example computed from EV (s_ev)
// 1. Copy s_ev into s_ve_offset
// 2. call v_e(num_vertices, num_edges, s_ve_offset, s_ve_output, nullptr);
// for(uint16_t v=threadIdx.x; v<num_vertices; v+=blockthreads){
//     uint16_t start = s_ve_offset[v];
//     uint16_t end = s_ve_offset[v+1];
//     for(uint16_t e=start; e<end;e++){
//         uint16_t edge = s_ve_output[e];
//
//     }
// }

// EV query example
// for(uint16_t e=threadIdx.x; e< num_edges; e+= blockThreads){
//     uint16_t v0_local_id = s_ev[2*e];
//     uint16_t v1_local_id = s_ev[2*e+1];
// }


/**
 * @brief Constructor for the PartitionManager struct.
 *
 * This constructor initializes the PartitionManager with the given context,
 * shared memory allocator, and requested level. It allocates shared memory for
 * various attributes related to vertices and edges, such as weights, degrees,
 * and adjacency weights.
 *
 * @param block A reference to the cooperative_groups::thread_block object
 * representing the CUDA thread block.
 * @param context A reference to the Context object containing information about
 * the graph or mesh to be partitioned.
 * @param shrd_alloc A reference to the ShmemAllocator object used to allocate
 * shared memory.
 * @param req_level The requested level of the partitioning process.
 */

template <uint32_t blockThreads>
__device__ __inline__ PartitionManager<blockThreads>::PartitionManager(
    cooperative_groups::thread_block& block,
    Context&                          context,
    ShmemAllocator&                   shrd_alloc,
    uint16_t                          req_level)
{
    m_context = context;

    __shared__ uint16_t level_count[1];
    m_s_req_level = level_count[0];

    m_patch_id = blockIdx.x;

    m_patch_info = m_context.m_patches_info[m_patch_id];

    m_s_num_vertices = shrd_alloc.alloc<uint16_t>(req_level);
    m_s_num_edges    = shrd_alloc.alloc<uint16_t>(req_level);

    // load num_v/e from global memory
    m_s_num_vertices[0] = *m_patch_info.num_vertices;
    m_s_num_edges[0]    = *m_patch_info.num_edges;

    const uint16_t req_vertex_cap = m_s_num_vertices[0] * req_level;
    const uint16_t req_edge_cap   = m_s_num_edges[0] * req_level;

    // copy ev to shared memory - 10*4*max_e
    m_s_ev = shrd_alloc.alloc<uint16_t>(2 * req_edge_cap);
    detail::load_async(block,
                       reinterpret_cast<uint16_t*>(m_patch_info.ev),
                       2 * m_s_num_edges[0],
                       m_s_ev,
                       false);

    // alloc shared memory for all vertex attributes - 10*3*max_v
    m_s_vwgt     = shrd_alloc.alloc<uint16_t>(req_vertex_cap);
    m_s_vdegree  = shrd_alloc.alloc<uint16_t>(req_vertex_cap);
    m_s_vadjewgt = shrd_alloc.alloc<uint16_t>(req_vertex_cap);

    // alloc shared memory for edge attributes - 10*1*max_e
    m_s_ewgt = shrd_alloc.alloc<uint16_t>(req_edge_cap);

    // edges chosen or vertex chosen 10*2*max_bitmask
    m_s_matched_vertices =
        alloc_bitmask_arr(block, shrd_alloc, req_level, m_s_num_vertices[0]);
    m_s_matched_edges =
        alloc_bitmask_arr(block, shrd_alloc, req_level, m_s_num_edges[0]);

    // partition bitmask 10*3*max_bitmask
    m_s_p0_vertices =
        alloc_bitmask_arr(block, shrd_alloc, req_level, m_s_num_vertices[0]);
    m_s_p1_vertices =
        alloc_bitmask_arr(block, shrd_alloc, req_level, m_s_num_vertices[0]);
    m_s_separator_vertices =
        alloc_bitmask_arr(block, shrd_alloc, req_level, m_s_num_vertices[0]);

    // alloc shared memory for mapping array 10*1*max_v and set to invalid
    m_s_mapping = shrd_alloc.alloc<uint16_t>(req_vertex_cap);
    fill_n<blockThreads>(m_s_mapping, req_vertex_cap, (uint16_t)INVALID16);

    // tmp VE operation array which will be reused for multiple times
    // 4*max_e
    m_s_tmp_offset = shrd_alloc.alloc<uint16_t>(m_s_num_edges[0] * 2);
    m_s_tmp_value  = shrd_alloc.alloc<uint16_t>(m_s_num_edges[0] * 2);

    // tmp vertex attribute array 1*max_v
    m_s_tmp_attribute_v = shrd_alloc.alloc<uint16_t>(m_s_num_vertices[0]);

    // tmp active e/v bitmask 2*max_bitmask
    m_s_tmp_active_edges    = Bitmask(m_s_num_edges[0], shrd_alloc);
    m_s_tmp_active_vertices = Bitmask(m_s_num_vertices[0], shrd_alloc);

    // partition used tmp bitmasks 5*max_bitmask
    m_s_tmp_assigned_v         = Bitmask(m_s_num_vertices[0], shrd_alloc);
    m_s_tmp_current_frontier_v = Bitmask(m_s_num_vertices[0], shrd_alloc);
    m_s_tmp_next_frontier_v    = Bitmask(m_s_num_vertices[0], shrd_alloc);
    m_s_tmp_coarse_p0_v        = Bitmask(m_s_num_vertices[0], shrd_alloc);
    m_s_tmp_coarse_p1_v        = Bitmask(m_s_num_vertices[0], shrd_alloc);

    block.sync();
}

/**
 * @brief Performs the matching operation in the coarsening phase of the
 * multilevel graph partitioning process.
 *
 * The matching process is performed in a loop until 75% of the vertices are
 * matched or hit 10 iterations. First the vertex would pick an active edge with
 * highest id, then the edge would check if the two vertices are matched. If
 * yes, the edge is matched and the two vertices are marked as matched. The one
 * ring edges of the matched vertices are then deactivated. This process is
 * repeated until the termination condition is met.
 *
 * It take s_ev as input and output the matched vertices and edges.
 *
 * @param block A reference to the cooperative_groups::thread_block object
 * representing the CUDA thread block
 * @param attr_matched_v tmp test vertex attribute
 * @param attr_active_e  tmp test edge attribute
 * @param curr_level The current level of the partitioning process
 */
template <uint32_t blockThreads>
__device__ __inline__ void PartitionManager<blockThreads>::matching(
    cooperative_groups::thread_block& block,
    rxmesh::VertexAttribute<uint16_t> attr_matched_v,
    rxmesh::EdgeAttribute<uint16_t>   attr_active_e,
    uint16_t                          curr_level)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Get level by level parameter
    const uint16_t num_vertices     = m_s_num_vertices[curr_level];
    const uint16_t num_edges        = m_s_num_edges[curr_level];
    uint16_t*      s_ev             = get_ev(curr_level);
    Bitmask&       matched_edges    = get_matched_edges_bitmask(curr_level);
    Bitmask&       matched_vertices = get_matched_vertices_bitmask(curr_level);

    __shared__ uint16_t s_num_matched_vertices[1];

    // TODO: use edge priority to replace the id for selecting edges
    // TODO: choose edge priority as the surrounding avtive edges
    rxmesh::Bitmask& active_edges = get_new_tmp_bitmask_active_e(block);
    active_edges.set(block);

    // Get VE data here to avoid redundant computation
    calc_new_temp_ve(block, s_ev, num_vertices, num_edges);
    const uint16_t* s_ve_offset = m_s_tmp_offset;
    const uint16_t* s_ve_value  = m_s_tmp_value;

    s_num_matched_vertices[0] = 0;
    int iter_count            = 0;
    while (float(s_num_matched_vertices[0]) / float(num_vertices) < 0.75 &&
           iter_count < 10) {
        iter_count++;

        if (idx == 0) {
            printf("iter_count: %u, \n num_v: %u, active_v: %u \n",
                   iter_count,
                   num_vertices,
                   s_num_matched_vertices[0]);
        }

        uint16_t* s_e_chosen_by_v =
            get_new_tmp_attribute_v_arr(block, num_vertices);
        block.sync();
        s_num_matched_vertices[0] = 0;

        // VE operation
        for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
            uint16_t start = s_ve_offset[v];
            uint16_t end   = s_ve_offset[v + 1];

            if (!coarsen_owned(LocalVertexT(v), curr_level)) {
                continue;
            }

            uint16_t tgt_e_id = 0;

            // query for one ring edges
            for (uint16_t e = start; e < end; e++) {
                uint16_t e_local_id = s_ve_value[e];

                if (!coarsen_owned(LocalEdgeT(e_local_id), curr_level) ||
                    !active_edges(e_local_id)) {
                    continue;
                }

                if (e_local_id > tgt_e_id) {
                    tgt_e_id = e_local_id;
                }
            }

            // TODO: assert memory access
            assert(v < num_vertices);
            s_e_chosen_by_v[v] = tgt_e_id;
        }

        block.sync();

        // EV operation -  find the matching edges
        for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
            uint16_t v0_local_id = s_ev[2 * e];
            uint16_t v1_local_id = s_ev[2 * e + 1];

            if (!coarsen_owned(LocalEdgeT(e), curr_level) || !active_edges(e) ||
                matched_edges(e) ||
                !coarsen_owned(LocalVertexT(v0_local_id), curr_level) ||
                !coarsen_owned(LocalVertexT(v1_local_id), curr_level)) {
                continue;
            }


            uint16_t v0_chosen_e_id = s_e_chosen_by_v[v0_local_id];
            uint16_t v1_chosen_e_id = s_e_chosen_by_v[v1_local_id];

            if (e == v1_chosen_e_id && e == v0_chosen_e_id) {
                matched_vertices.set(v0_local_id, true);
                matched_vertices.set(v1_local_id, true);
                matched_edges.set(e, true);

                VertexHandle v0(m_patch_id, v0_local_id);
                attr_matched_v(v0) = iter_count;


                VertexHandle v1(m_patch_id, v1_local_id);
                attr_matched_v(v1) = iter_count;


                EdgeHandle e0(m_patch_id, e);
                attr_active_e(e0) = 10;
            }
        }

        block.sync();

        // VE operation - deactive the surrounding edges of a matched edge
        for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
            uint16_t start = s_ve_offset[v];
            uint16_t end   = s_ve_offset[v + 1];

            if (!coarsen_owned(LocalVertexT(v), curr_level)) {
                continue;
            }

            if (matched_vertices(v)) {
                for (uint16_t e = start; e < end; e++) {
                    uint16_t e_local_id = s_ve_value[e];

                    if (!coarsen_owned(LocalEdgeT(e_local_id), curr_level) ||
                        !active_edges(e_local_id)) {
                        continue;
                    }

                    active_edges.reset(e_local_id, true);
                    if (!matched_edges(e_local_id)) {
                        EdgeHandle e0(m_patch_id, e_local_id);
                        attr_active_e(e0) = 5;
                    }
                }

                // count active vertices
                atomicAdd(&s_num_matched_vertices[0], 1);
            }
        }

        block.sync();
    }

    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];

        if (!coarsen_owned(LocalEdgeT(e), curr_level) ||
            !coarsen_owned(LocalVertexT(v0_local_id), curr_level) ||
            !coarsen_owned(LocalVertexT(v1_local_id), curr_level)) {
            continue;
        }
        if (blockIdx.x == 0) {
            printf(
                "e: %u, v0: %u, v1: %u matched_e: %d, matched_v0: %d, "
                "matched_v1: %d \n",
                e,
                v0_local_id,
                v1_local_id,
                matched_edges(e),
                matched_vertices(v0_local_id),
                matched_vertices(v1_local_id));
        }
    }
}


/**
 * @brief Performs the coarsening operation in the multilevel graph partitioning
 * process.
 *
 * This function performs coarsen the graph into ev format.
 * For a pair of vertices, if they are matched, they are coarsened into a single
 * vertex choosing the smaller id as the representative. Then the edges are
 * coarsened into a single edge with the representative vertices.
 *
 * The input is s_ev and matched_edges, and the output is the coarsened s_ev for
 * the next level.
 *
 * @param block A reference to the cooperative_groups::thread_block object
 * representing the CUDA thread block.
 * @param curr_level The current level of the partitioning process.
 */
template <uint32_t blockThreads>
__device__ __inline__ void PartitionManager<blockThreads>::coarsening(
    cooperative_groups::thread_block& block,
    uint16_t                          curr_level)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Get level by level param
    uint16_t  num_vertices     = m_s_num_vertices[curr_level];
    uint16_t  num_edges        = m_s_num_edges[curr_level];
    uint16_t* s_ev             = get_ev(curr_level);
    uint16_t* s_ev_coarse      = get_ev(curr_level + 1);
    uint16_t* s_mapping        = get_mapping(curr_level);
    Bitmask&  matched_edges    = get_matched_edges_bitmask(curr_level);
    Bitmask&  matched_vertices = get_matched_vertices_bitmask(curr_level);

    // EV operation: set matched vertices' coarsen id to the lower vertex id and
    // remain the unmatched vertices' coarsen id as unchanged.
    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];

        if (!coarsen_owned(LocalEdgeT(e), curr_level) ||
            !coarsen_owned(LocalVertexT(v0_local_id), curr_level) ||
            !coarsen_owned(LocalVertexT(v1_local_id), curr_level)) {
            continue;
        }

        // if (blockIdx.x == 0) {
        //     printf(
        //         "e: %u, v0: %u, v1: %u matched_e: %d, matched_v0: %d, "
        //         "matched_v1: %d \n",
        //         e,
        //         v0_local_id,
        //         v1_local_id,
        //         matched_edges(e),
        //         matched_vertices(v0_local_id),
        //         matched_vertices(v1_local_id));
        // }

        if (matched_edges(e)) {
            // assert(matched_vertices(v0_local_id));
            // assert(matched_vertices(v1_local_id));

            uint16_t coarse_id =
                v0_local_id < v1_local_id ? v0_local_id : v1_local_id;
            s_mapping[v0_local_id] = coarse_id;
            s_mapping[v1_local_id] = coarse_id;
        } else {
            if (!matched_vertices(v0_local_id)) {
                s_mapping[v0_local_id] = v0_local_id;
            }

            if (!matched_vertices(v1_local_id)) {
                s_mapping[v1_local_id] = v1_local_id;
            }
        }
    }

    block.sync();

    for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        if (matched_vertices(v) && coarsen_owned(LocalVertexT(v), curr_level) &&
            blockIdx.x == 0) {
            printf("v:  %u, s_mapping[v]: %u\n", v, s_mapping[v]);
        }
    }

    // e_id counter on shared memory
    __shared__ uint16_t s_e_id_counter[1];
    __shared__ uint16_t s_coarse_v_counter[1];

    s_coarse_v_counter[0] = num_vertices;
    s_e_id_counter[0]     = 0;

    calc_new_temp_ve(block, s_ev, num_vertices, num_edges);
    const uint16_t* s_ve_offset = m_s_tmp_offset;
    const uint16_t* s_ve_value  = m_s_tmp_value;

    auto unique_edge_id = [](uint16_t v0, uint16_t v1, uint16_t num_vertices) {
        v0 = v0 < v1 ? v0 : v1;
        v1 = v0 < v1 ? v1 : v0;
        return v0 * num_vertices + v1;
    };


    // // EV operation: determine whether choose an edge or not
    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];

        if (!coarsen_owned(LocalEdgeT(e), curr_level) ||
            !coarsen_owned(LocalVertexT(v0_local_id), curr_level) ||
            !coarsen_owned(LocalVertexT(v1_local_id), curr_level)) {
            continue;
        }

        // matched edges are not preserved
        if (matched_edges(e)) {
            s_coarse_v_counter[0]--;
            continue;
        }

        bool edge_chosen = true;

        uint32_t current_unique_eid =
            unique_edge_id(v0_local_id, v1_local_id, num_vertices);
        uint32_t coarse_unique_eid = unique_edge_id(
            s_mapping[v0_local_id], s_mapping[v1_local_id], num_vertices);

        uint16_t priority_v =
            v0_local_id < v1_local_id ? v0_local_id : v1_local_id;

        // request one ring edge of priority_v
        uint16_t start = s_ve_offset[priority_v];
        uint16_t end   = s_ve_offset[priority_v + 1];
        for (uint16_t e = start; e < end; e++) {
            uint16_t priority_local_e = s_ve_value[e];

            // skip current edge
            if (priority_local_e == e) {
                continue;
            }

            uint16_t priority_local_v0 = s_ev[2 * priority_local_e];
            uint16_t priority_local_v1 = s_ev[2 * priority_local_e + 1];

            uint32_t priority_current_unique_eid = unique_edge_id(
                priority_local_v0, priority_local_v1, num_vertices);
            uint32_t priority_coarse_unique_eid =
                unique_edge_id(s_mapping[priority_local_v0],
                               s_mapping[priority_local_v1],
                               num_vertices);

            // edge not chosen due to merged to other edge
            if (priority_coarse_unique_eid == coarse_unique_eid &&
                priority_current_unique_eid < current_unique_eid) {
                edge_chosen = false;
            }
        }

        // checked the paired matched vertex for duplicate edge
        if (edge_chosen && matched_vertices(priority_v) &&
            s_mapping[priority_v] != priority_v) {
            uint16_t matched_priority_v = s_mapping[priority_v];

            if (blockIdx.x == 0) {
                printf("e: %u, priority_v: %u, matched_priority_v: %u \n",
                       e,
                       priority_v,
                       matched_priority_v);
            }


            // // request one ring edge of mached_priority_v
            // uint16_t start = s_ve_offset[matched_priority_v];
            // uint16_t end   = s_ve_offset[matched_priority_v + 1];
            // for (uint16_t e = start; e < end; e++) {
            //     uint16_t matched_priority_local_e = s_ve_value[e];

            //     assert(matched_priority_local_e != e);

            //     uint16_t matched_priority_local_v0 =
            //         s_ev[2 * matched_priority_local_e];
            //     uint16_t matched_priority_local_v1 =
            //         s_ev[2 * matched_priority_local_e + 1];

            //     uint32_t matched_priority_current_unique_eid =
            //         unique_edge_id(matched_priority_local_v0,
            //                        matched_priority_local_v1,
            //                        num_vertices);
            //     uint32_t matched_priority_coarse_unique_eid =
            //         unique_edge_id(s_mapping[matched_priority_local_v0],
            //                        s_mapping[matched_priority_local_v1],
            //                        num_vertices);

            //     // edge not chosen due to merged to other edge
            //     if (matched_priority_coarse_unique_eid == coarse_unique_eid
            //     &&
            //         matched_priority_current_unique_eid < current_unique_eid)
            //         { edge_chosen = false;
            //     }
            // }
        }

        if (edge_chosen) {
            uint16_t curr_idx = atomicAdd(&s_e_id_counter[0], 1);
            // printf("num_edges: %u, curr_idx: %u, e: %u \n", num_edges,
            // curr_idx, e);
            s_ev_coarse[2 * curr_idx]     = s_mapping[v0_local_id];
            s_ev_coarse[2 * curr_idx + 1] = s_mapping[v1_local_id];
        }
    }

    // block.sync();

    // m_s_num_vertices[curr_level + 1] = s_coarse_v_counter[0];
    // m_s_num_edges[curr_level + 1] = s_e_id_counter[0] + 1;
}

/**
 * @brief Performs the partition operation in the multilevel graph partitioning
 * process.
 *
 * This function performs the partitioning process for the coarsest level of the
 * graph.
 *
 * It takes s_ev as input and output the partitioned vertices into p0 and p1,
 * and the separator vertices.
 *
 * @param block A reference to the cooperative_groups::thread_block object
 * representing the CUDA thread block.
 * @param curr_level The current level of the partitioning process.
 */
template <uint32_t blockThreads>
__device__ __inline__ void PartitionManager<blockThreads>::partition(
    cooperative_groups::thread_block& block,
    uint16_t                          curr_level)
{
    const uint16_t num_vertices       = m_s_num_vertices[curr_level];
    const uint16_t num_edges          = m_s_num_edges[curr_level];
    uint16_t*      s_ev               = get_ev(curr_level);
    Bitmask&       coarse_p0_vertices = get_p0_vertices_bitmask(curr_level + 1);
    Bitmask&       coarse_p1_vertices = get_p1_vertices_bitmask(curr_level + 1);
    Bitmask&       active_vertices    = get_new_tmp_bitmask_active_v(block);
    int            num_iter           = 10;

    // EV operation setting active vertices
    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];
        active_vertices.set(v0_local_id, true);
        active_vertices.set(v1_local_id, true);
    }

    // VV operation from VE operation
    calc_new_temp_vv(block, s_ev, num_vertices, num_edges);
    reset_temp_partition_bitmask(block);

    block.sync();

    const uint16_t* s_vv_offset = m_s_tmp_offset;
    const uint16_t* s_vv_value  = m_s_tmp_value;

    detail::bi_assignment_ggp<blockThreads>(
        /*cooperative_groups::thread_block& */ block,
        /* const uint16_t                   */ num_vertices,
        /* const Bitmask& s_owned_v         */ active_vertices,
        /* const Bitmask& s_active_v        */ active_vertices,
        /* const uint16_t*                  */ s_vv_offset,
        /* const uint16_t*                  */ s_vv_value,
        /* Bitmask&                         */ m_s_tmp_assigned_v,
        /* Bitmask&                         */ m_s_tmp_current_frontier_v,
        /* Bitmask&                         */ m_s_tmp_next_frontier_v,
        /* Bitmask&                         */ m_s_tmp_coarse_p0_v,
        /* Bitmask&                         */ m_s_tmp_coarse_p1_v,
        /* int                              */ num_iter);

    // TODO get separator from p1 and p0
}

/**
 * @brief Performs the uncoarsening operation in the multilevel graph
 * partitioning process.
 *
 * The function then sets the bitmasks for the partitions and separator vertices
 * at the current level based on the corresponding bitmasks at the next level.
 *
 * @param block A reference to the cooperative_groups::thread_block object
 * representing the CUDA thread block.
 * @param curr_level The current level of the partitioning process.
 */

template <uint32_t blockThreads>
__device__ __inline__ void PartitionManager<blockThreads>::uncoarsening(
    cooperative_groups::thread_block& block,
    uint16_t                          curr_level)
{
    // Get level by level param
    uint16_t  num_vertices       = m_s_num_vertices[curr_level];
    uint16_t  num_edges          = m_s_num_edges[curr_level];
    uint16_t* s_ev               = get_ev(curr_level);
    uint16_t* s_mapping          = get_mapping(curr_level);
    Bitmask&  matched_vertices   = get_matched_edges_bitmask(curr_level);
    Bitmask&  p0_vertices        = get_p0_vertices_bitmask(curr_level);
    Bitmask&  p1_vertices        = get_p1_vertices_bitmask(curr_level);
    Bitmask&  coarse_p0_vertices = get_p0_vertices_bitmask(curr_level + 1);
    Bitmask&  coarse_p1_vertices = get_p1_vertices_bitmask(curr_level + 1);
    Bitmask&  separator_v        = get_separator_vertices_bitmask(curr_level);
    Bitmask&  coarse_separator_v =
        get_separator_vertices_bitmask(curr_level + 1);

    // Get VE data here to avoid redundant computation
    // calc_new_temp_ve(block, s_ev, num_vertices, num_edges);

    for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        // uint16_t start = m_s_tmp_offset[v];
        // uint16_t end   = m_s_tmp_offset[v + 1];

        p0_vertices.set(v, coarse_p0_vertices(s_mapping[v]));
        p1_vertices.set(v, coarse_p1_vertices(s_mapping[v]));
        separator_v.set(v, coarse_separator_v(s_mapping[v]));
    }

    block.sync();
}

/**
 * @brief Generates a reordering of vertices in the graph.
 *
 * This function generates a reordering of vertices based on the partitioning of
 * the graph. It first retrieves the number of vertices and edges at level 0, as
 * well as the edge vertices and the bitmasks for the two partitions and the
 * separator vertices. It then initializes a shared counter. The function then
 * iterates over the vertices in the graph. For each vertex in the first
 * partition, it increments the counter and assigns the counter value as the new
 * order of the vertex. The function then synchronizes the thread block and
 * repeats the process for the vertices in the second partition and the
 * separator vertices.
 *
 * @param block A reference to the cooperative_groups::thread_block object
 * representing the CUDA thread block.
 * @param v_ordering A VertexAttribute object representing the new ordering of
 * the vertices.
 */
template <uint32_t blockThreads>
__device__ __inline__ void PartitionManager<blockThreads>::genrate_reordering(
    cooperative_groups::thread_block& block,
    VertexAttribute<uint16_t>         v_ordering)
{
    // Get level 0 param
    uint16_t  num_vertices = m_s_num_vertices[0];
    uint16_t  num_edges    = m_s_num_edges[0];
    uint16_t* s_ev         = get_ev(0);
    Bitmask&  p0_vertices  = get_p0_vertices_bitmask(0);
    Bitmask&  p1_vertices  = get_p1_vertices_bitmask(0);
    Bitmask&  separator_v  = get_separator_vertices_bitmask(0);

    __shared__ int counter[1];

    // calc_new_temp_ve(block, s_ev, num_vertices, num_edges);

    for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        // uint16_t start = m_s_tmp_offset[v];
        // uint16_t end   = m_s_tmp_offset[v + 1];

        if (p0_vertices(v)) {
            int          curr_idx = ::atomicAdd(&counter[0], 1);
            VertexHandle curr_v(m_patch_id, v);
            v_ordering(curr_v, 0) = curr_idx;
        }
    }

    block.sync();

    for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        // uint16_t start = m_s_tmp_offset[v];
        // uint16_t end   = m_s_tmp_offset[v + 1];

        if (p1_vertices(v)) {
            int          curr_idx = ::atomicAdd(&counter[0], 1);
            VertexHandle curr_v(m_patch_id, v);
            v_ordering(curr_v, 0) = curr_idx;
        }
    }

    block.sync();

    for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        // uint16_t start = m_s_tmp_offset[v];
        // uint16_t end   = m_s_tmp_offset[v + 1];

        if (separator_v(v)) {
            int          curr_idx = ::atomicAdd(&counter[0], 1);
            VertexHandle curr_v(m_patch_id, v);
            v_ordering(curr_v, 0) = curr_idx;
        }
    }

    block.sync();
}

}  // namespace rxmesh
