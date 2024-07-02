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


namespace rxmesh {

// PartitionManager lives in shared memory and is used to manage the
// partitioning. Every patch owns a PartitionManager like the patch_info

// TODO: change the uniform shared memory allocation to per level allocation for
// less shared memory use
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

    __device__ __inline__ void local_matching(
        cooperative_groups::thread_block& block,
        rxmesh::VertexAttribute<uint16_t> attr_matched_v,
        rxmesh::EdgeAttribute<uint16_t>   attr_active_e,
        uint16_t                          curr_level);

    __device__ __inline__ void local_coarsening(
        cooperative_groups::thread_block& block,
        uint16_t                          curr_level);

    __device__ __inline__ void local_uncoarsening(
        cooperative_groups::thread_block& block,
        uint16_t                          curr_level);

    __device__ __inline__ void local_partition(
        cooperative_groups::thread_block& block,
        uint16_t                          curr_level);

    __device__ __inline__ void local_multi_level_partition(
        cooperative_groups::thread_block& block,
        uint16_t                          curr_level,
        uint16_t                          partition_level);

    __device__ __inline__ void local_genrate_reordering(
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
            offset += i == 0 ? m_num_edges_limit : m_s_num_edges[i];
        }
        return m_s_ev + (2 * offset);
    }

    __device__ __inline__ uint16_t* get_mapping(uint16_t curr_level)
    {
        uint16_t offset = 0;
        for (int i = 0; i < curr_level; ++i) {
            offset += i == 0 ? m_num_vertices_limit : m_s_num_vertices[i];
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

    __device__ __inline__ Bitmask& get_current_vertices_bitmask(
        uint16_t curr_level)
    {
        return m_s_current_vertices[curr_level];
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

    __device__ __inline__ bool coarsen_owned(rxmesh::LocalEdgeT e_handle,
                                             uint16_t           level)
    {
        if (level == 0) {
            return m_patch_info.is_owned(e_handle);
        }

        return true;
    }

    __device__ __inline__ bool coarsen_owned(rxmesh::LocalVertexT v_handle,
                                             uint16_t             level)
    {
        if (level == 0) {
            return m_patch_info.is_owned(v_handle);
        }

        return true;
    }


   private:
    static constexpr int max_level_size = 10;

    __device__ __inline__ void alloc_bitmask_arr(
        cooperative_groups::thread_block&    block,
        ShmemAllocator&                      shrd_alloc,
        std::array<Bitmask, max_level_size>& bm_arr,
        uint16_t                             req_arr_size,
        uint16_t                             bm_size)
    {
        assert(max_level_size >= req_arr_size);

        for (int i = 0; i < req_arr_size; ++i) {
            bm_arr[i] = Bitmask(bm_size, shrd_alloc);
            bm_arr[i].reset(block);
        }
    }

    // TODO: vertex id sanity check
    // v_id < num_vertices
    // edge have two active & unique vertices
    __device__ __inline__ void calc_new_temp_ve(
        cooperative_groups::thread_block& block,
        uint16_t*                         s_ev,
        uint16_t                          num_vertices,
        uint16_t                          num_edges,
        uint16_t                          level)
    {
        // Copy EV to offset array
        for (uint16_t i = threadIdx.x; i < num_edges * 2; i += blockThreads) {
            m_s_tmp_offset[i] = s_ev[i];
        }
        block.sync();

        const uint32_t* active_mask_e =
            m_patch_info.get_owned_mask<EdgeHandle>();
        if (level != 0) {
            m_s_tmp_active_edges.set(block);
            active_mask_e = m_s_tmp_active_edges.m_bitmask;
        }

        block.sync();

        // TODO: may use new API later for v_e
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
        uint16_t                          num_edges,
        uint16_t                          level)
    {
        calc_new_temp_ve(block, s_ev, num_vertices, num_edges, level);

        for (uint32_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
            uint32_t start = m_s_tmp_offset[v];
            uint32_t end   = m_s_tmp_offset[v + 1];

            for (uint32_t e = start; e < end; ++e) {
                uint16_t edge = m_s_tmp_value[e];
                uint16_t v0   = s_ev[2 * edge];
                uint16_t v1   = s_ev[2 * edge + 1];

                assert(v0 != INVALID16 && v1 != INVALID16);
                // assert(v0 == v || v1 == v);
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
    uint16_t  m_num_vertices_limit;
    uint16_t  m_num_edges_limit;

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
    uint16_t m_req_level;

    std::array<Bitmask, max_level_size> m_s_matched_edges;
    std::array<Bitmask, max_level_size> m_s_matched_vertices;

    // since the v_id is stores in sparse we need keeptrack of the current
    // vertex id
    std::array<Bitmask, max_level_size> m_s_current_vertices;

    // for partition
    std::array<Bitmask, max_level_size> m_s_p0_vertices;
    std::array<Bitmask, max_level_size> m_s_p1_vertices;
    std::array<Bitmask, max_level_size> m_s_separator_vertices;

    // for output the vertex ordering
    uint16_t* m_s_v_partition_id;
    uint16_t* m_s_v_ordering;

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

    m_req_level = req_level + 1;

    m_patch_id = blockIdx.x;

    m_patch_info = m_context.m_patches_info[m_patch_id];

    m_s_num_vertices = shrd_alloc.alloc<uint16_t>(m_req_level);
    m_s_num_edges    = shrd_alloc.alloc<uint16_t>(m_req_level);

    // load num_v/e from global memory
    // the number of v/e actually used for coarsening
    m_s_num_vertices[0] = 0;
    m_s_num_edges[0]    = 0;
    // the total number of v/e regardless of the owenership and availability
    m_num_vertices_limit = m_patch_info.num_vertices[0];
    m_num_edges_limit    = m_patch_info.num_edges[0];

    const uint16_t req_vertex_cap = m_num_vertices_limit * m_req_level;
    const uint16_t req_edge_cap   = m_num_edges_limit * m_req_level;

    // copy ev to shared memory - 10*4*max_e
    m_s_ev = shrd_alloc.alloc<uint16_t>(2 * req_edge_cap);
    detail::load_async(block,
                       reinterpret_cast<uint16_t*>(m_patch_info.ev),
                       2 * m_num_edges_limit,
                       m_s_ev,
                       false);

    // alloc shared memory for all vertex attributes - 10*3*max_v
    m_s_vwgt     = shrd_alloc.alloc<uint16_t>(req_vertex_cap);
    m_s_vdegree  = shrd_alloc.alloc<uint16_t>(req_vertex_cap);
    m_s_vadjewgt = shrd_alloc.alloc<uint16_t>(req_vertex_cap);

    // alloc shared memory for edge attributes - 10*1*max_e
    m_s_ewgt = shrd_alloc.alloc<uint16_t>(req_edge_cap);

    // edges chosen or vertex chosen 10*2*max_bitmask
    alloc_bitmask_arr(block,
                      shrd_alloc,
                      m_s_matched_vertices,
                      m_req_level,
                      m_num_vertices_limit);
    alloc_bitmask_arr(
        block, shrd_alloc, m_s_matched_edges, m_req_level, m_num_edges_limit);

    alloc_bitmask_arr(block,
                      shrd_alloc,
                      m_s_current_vertices,
                      m_req_level + 1,
                      m_num_vertices_limit);

    // partition bitmask 10*3*max_bitmask
    alloc_bitmask_arr(
        block, shrd_alloc, m_s_p0_vertices, m_req_level, m_num_vertices_limit);
    alloc_bitmask_arr(
        block, shrd_alloc, m_s_p1_vertices, m_req_level, m_num_vertices_limit);
    alloc_bitmask_arr(block,
                      shrd_alloc,
                      m_s_separator_vertices,
                      m_req_level,
                      m_num_vertices_limit);

    // alloc shared memory for mapping array 10*1*max_v and set to invalid
    m_s_mapping = shrd_alloc.alloc<uint16_t>(req_vertex_cap);
    fill_n<blockThreads>(m_s_mapping, req_vertex_cap, (uint16_t)INVALID16);

    // vertex partition id array 1*max_v
    m_s_v_partition_id = shrd_alloc.alloc<uint16_t>(m_num_vertices_limit);
    fill_n<blockThreads>(
        m_s_v_partition_id, m_num_vertices_limit, (uint16_t)INVALID16);

    // vertex ordering array 1*max_v
    m_s_v_ordering = shrd_alloc.alloc<uint16_t>(m_num_vertices_limit);
    fill_n<blockThreads>(
        m_s_v_ordering, m_num_vertices_limit, (uint16_t)INVALID16);

    // tmp VE operation array which will be reused for multiple times
    // 4*max_e
    m_s_tmp_offset = shrd_alloc.alloc<uint16_t>(m_num_edges_limit * 2);
    m_s_tmp_value  = shrd_alloc.alloc<uint16_t>(m_num_edges_limit * 2);

    // tmp vertex attribute array 1*max_v
    m_s_tmp_attribute_v = shrd_alloc.alloc<uint16_t>(m_num_vertices_limit);

    // tmp vertex attribute array 1*max_e
    m_s_tmp_attribute_e = shrd_alloc.alloc<uint16_t>(m_num_edges_limit);

    // tmp active e/v bitmask 2*max_bitmask
    m_s_tmp_active_edges    = Bitmask(m_num_edges_limit, shrd_alloc);
    m_s_tmp_active_vertices = Bitmask(m_num_vertices_limit, shrd_alloc);

    // partition used tmp bitmasks 5*max_bitmask
    m_s_tmp_assigned_v         = Bitmask(m_num_vertices_limit, shrd_alloc);
    m_s_tmp_current_frontier_v = Bitmask(m_num_vertices_limit, shrd_alloc);
    m_s_tmp_next_frontier_v    = Bitmask(m_num_vertices_limit, shrd_alloc);
    m_s_tmp_coarse_p0_v        = Bitmask(m_num_vertices_limit, shrd_alloc);
    m_s_tmp_coarse_p1_v        = Bitmask(m_num_vertices_limit, shrd_alloc);

    block.sync();

    // set the fully owned edges active
    // the fully owned edges should cover all the owned vertices
    for (uint16_t e = threadIdx.x; e < m_num_edges_limit; e += blockThreads) {
        uint16_t v0_local_id = m_s_ev[2 * e];
        uint16_t v1_local_id = m_s_ev[2 * e + 1];

        if (coarsen_owned(LocalEdgeT(e), 0) &&
            coarsen_owned(LocalVertexT(v0_local_id), 0) &&
            coarsen_owned(LocalVertexT(v1_local_id), 0)) {
            atomicAdd(&m_s_num_edges[0], 1);

            m_s_current_vertices[0].set(v0_local_id, true);
            m_s_current_vertices[0].set(v1_local_id, true);
        }
    }
    block.sync();

    for (uint16_t v = threadIdx.x; v < m_num_vertices_limit;
         v += blockThreads) {
        if (m_s_current_vertices[0](v)) {
            atomicAdd(&m_s_num_vertices[0], 1);
        }
    }
    block.sync();

    uint16_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    assert(m_s_num_vertices[0] == m_patch_info.get_num_owned<VertexHandle>());
    if (idx == 0) {
        printf(
            "num_v: %u, num_e: %u, num_v_limit: %u, num_e_limit: %u, "
            "num_v_owned: %u \n",
            m_s_num_vertices[0],
            m_s_num_edges[0],
            m_num_vertices_limit,
            m_num_edges_limit,
            m_patch_info.get_num_owned<VertexHandle>());
    }
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
__device__ __inline__ void PartitionManager<blockThreads>::local_matching(
    cooperative_groups::thread_block& block,
    rxmesh::VertexAttribute<uint16_t> attr_matched_v,
    rxmesh::EdgeAttribute<uint16_t>   attr_active_e,
    uint16_t                          curr_level)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Get level by level parameter
    const uint16_t num_vertices       = m_s_num_vertices[curr_level];
    const uint16_t num_edges          = m_s_num_edges[curr_level];
    const uint16_t num_vertices_query = m_num_vertices_limit;
    const uint16_t num_edges_query =
        curr_level == 0 ? m_num_edges_limit : num_edges;

    uint16_t* s_ev             = get_ev(curr_level);
    Bitmask&  matched_edges    = get_matched_edges_bitmask(curr_level);
    Bitmask&  matched_vertices = get_matched_vertices_bitmask(curr_level);
    Bitmask&  current_vertices = get_current_vertices_bitmask(curr_level);
    Bitmask&  active_edges     = get_new_tmp_bitmask_active_e(block);


    // DONE: set e inactive if e if not fully owned
    // if a v is owned by the patch, then at least one edge owned by the patch
    // would connect to it extreme cases v0-e0-v1 v1-e1-v2, v1-e2-v3, is it
    // possible that e0 and v0 are in the patch but v1 is not?
    if (curr_level == 0) {
        for (uint16_t e = threadIdx.x; e < num_edges_query; e += blockThreads) {
            uint16_t v0_local_id = s_ev[2 * e];
            uint16_t v1_local_id = s_ev[2 * e + 1];

            if (coarsen_owned(LocalEdgeT(e), curr_level) &&
                coarsen_owned(LocalVertexT(v0_local_id), curr_level) &&
                coarsen_owned(LocalVertexT(v1_local_id), curr_level)) {
                active_edges.set(e, true);
            }
        }
        block.sync();
    } else {
        active_edges.set(block);
    }

    // params for testing
    __shared__ uint16_t s_num_matched_vertices[1];
    s_num_matched_vertices[0] = 0;

    __shared__ uint16_t s_num_matched_edges[1];
    s_num_matched_edges[0] = 0;

    // DONE: use edge priority to replace the id for selecting edges
    // DONE: choose edge priority as the surrounding avtive edges
    uint16_t* s_e_priority =
        get_new_tmp_attribute_e_arr(block, m_num_edges_limit);

    // Get VE data here to avoid redundant computation
    calc_new_temp_ve(
        block, s_ev, m_num_vertices_limit, num_edges_query, curr_level);
    const uint16_t* s_ve_offset = m_s_tmp_offset;
    const uint16_t* s_ve_value  = m_s_tmp_value;

    int iter_count = 0;
    while (float(s_num_matched_vertices[0]) / float(num_vertices) < 0.75 &&
           iter_count < 10) {

        // tmp variable for VE query
        uint16_t* s_e_chosen_by_v =
            get_new_tmp_attribute_v_arr(block, m_num_vertices_limit);
        s_num_matched_vertices[0] = 0;

        // VE operation - let v choose the e with the highest priority
        for (uint16_t v = threadIdx.x; v < num_vertices_query;
             v += blockThreads) {
            uint16_t start = s_ve_offset[v];
            uint16_t end   = s_ve_offset[v + 1];

            if (!coarsen_owned(LocalVertexT(v), curr_level) || start == end ||
                !current_vertices(v)) {
                continue;
            }

            assert(start < end);
            assert(current_vertices(v));

            bool ownership_sanity_check = false;

            // query for one ring edges
            uint16_t tgt_e_id = s_ve_value[start];
            for (uint16_t e_id_idx = start; e_id_idx < end; e_id_idx++) {
                uint16_t e_local_id = s_ve_value[e_id_idx];

                if (coarsen_owned(LocalEdgeT(e_local_id), curr_level)) {
                    uint16_t v0_local_id = s_ev[2 * e_local_id];
                    uint16_t v1_local_id = s_ev[2 * e_local_id + 1];
                    if (coarsen_owned(LocalVertexT(v0_local_id), curr_level) &&
                        coarsen_owned(LocalVertexT(v1_local_id), curr_level)) {
                        ownership_sanity_check = true;
                    }
                }

                if (!active_edges(e_local_id)) {
                    continue;
                }

                uint16_t e_priority =
                    e_local_id + s_e_priority[e_local_id] * m_num_edges_limit;
                uint16_t tgt_priority =
                    tgt_e_id + s_e_priority[tgt_e_id] * m_num_edges_limit;

                if (e_priority > tgt_priority) {
                    tgt_e_id = e_local_id;
                }

                assert(e_priority < INVALID16);
                assert(tgt_priority < INVALID16);
            }

            assert(ownership_sanity_check);

            assert(v < m_num_vertices_limit);
            s_e_chosen_by_v[v] = tgt_e_id;
        }

        block.sync();

        // EV operation -  find the matching edges if the two vertices are
        // agreed on the same edge
        for (uint16_t e = threadIdx.x; e < num_edges_query; e += blockThreads) {
            uint16_t v0_local_id = s_ev[2 * e];
            uint16_t v1_local_id = s_ev[2 * e + 1];

            if (!active_edges(e)) {
                continue;
            }

            uint16_t v0_chosen_e_id = s_e_chosen_by_v[v0_local_id];
            uint16_t v1_chosen_e_id = s_e_chosen_by_v[v1_local_id];

            if (e == v1_chosen_e_id && e == v0_chosen_e_id) {
                matched_vertices.set(v0_local_id, true);
                matched_vertices.set(v1_local_id, true);
                matched_edges.set(e, true);

                // for matching visualization only
                atomicAdd(&s_num_matched_edges[0], 1);
                VertexHandle v0(m_patch_id, v0_local_id);
                attr_matched_v(v0) = iter_count;
                VertexHandle v1(m_patch_id, v1_local_id);
                attr_matched_v(v1) = iter_count;
                EdgeHandle e0(m_patch_id, e);
                attr_active_e(e0) = 10;
            }
        }

        block.sync();

        // TODO: current vertices doesnot work with start == end condition
        // VE operation - deactive the surrounding edges of a matched edge
        for (uint16_t v = threadIdx.x; v < num_vertices_query;
             v += blockThreads) {
            uint16_t start = s_ve_offset[v];
            uint16_t end   = s_ve_offset[v + 1];

            if (!coarsen_owned(LocalVertexT(v), curr_level) ||
                !current_vertices(v) || start == end) {
                continue;
            }

            if (matched_vertices(v)) {
                for (uint16_t e_id_idx = start; e_id_idx < end; e_id_idx++) {
                    uint16_t e_local_id = s_ve_value[e_id_idx];

                    if (!active_edges(e_local_id)) {
                        continue;
                    }

                    active_edges.reset(e_local_id, true);

                    // for matching visualization only
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

        // tmp variable for VE query
        uint16_t* v_active_degree =
            get_new_tmp_attribute_v_arr(block, m_num_vertices_limit);

        // VE_operation - update the priority of the surrounding edges
        for (uint16_t v = threadIdx.x; v < num_vertices_query;
             v += blockThreads) {
            uint16_t start = s_ve_offset[v];
            uint16_t end   = s_ve_offset[v + 1];

            if (!coarsen_owned(LocalVertexT(v), curr_level) ||
                !current_vertices(v) || start == end) {
                continue;
            }

            if (!matched_vertices(v)) {
                uint16_t num_e_active = 0;
                for (uint16_t e_id_idx = start; e_id_idx < end; e_id_idx++) {
                    uint16_t e_local_id = s_ve_value[e_id_idx];

                    if (!active_edges(e_local_id)) {
                        continue;
                    }
                    ++num_e_active;
                }
                v_active_degree[v] = num_e_active;
            }
        }
        block.sync();

        // EV operation - update the priority of the edges
        for (uint16_t e = threadIdx.x; e < num_edges_query; e += blockThreads) {
            uint16_t v0_local_id = s_ev[2 * e];
            uint16_t v1_local_id = s_ev[2 * e + 1];

            if (!active_edges(e)) {
                continue;
            }

            s_e_priority[e] =
                v_active_degree[v0_local_id] + v_active_degree[v1_local_id];
        }
        block.sync();

        // if (idx == 0) {
        //     printf("iter_count: %u, \n num_v: %u, matched_v: %u, num_e: %u
        //     \n",
        //            iter_count,
        //            num_vertices,
        //            s_num_matched_vertices[0],
        //            num_edges);
        // }
        iter_count++;
    }

    if (idx == 0) {
        printf("num_matched_edges: %u, num_matched_vertices: %u \n",
               s_num_matched_edges[0],
               s_num_matched_vertices[0]);
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
__device__ __inline__ void PartitionManager<blockThreads>::local_coarsening(
    cooperative_groups::thread_block& block,
    uint16_t                          curr_level)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Get level by level param
    const uint16_t num_vertices       = m_s_num_vertices[curr_level];
    const uint16_t num_edges          = m_s_num_edges[curr_level];
    const uint16_t num_vertices_query = m_num_vertices_limit;
    const uint16_t num_edges_query =
        curr_level == 0 ? m_num_edges_limit : num_edges;

    uint16_t* s_ev             = get_ev(curr_level);
    uint16_t* s_ev_coarse      = get_ev(curr_level + 1);
    uint16_t* s_mapping        = get_mapping(curr_level);
    Bitmask&  matched_edges    = get_matched_edges_bitmask(curr_level);
    Bitmask&  matched_vertices = get_matched_vertices_bitmask(curr_level);
    Bitmask&  active_edges     = get_new_tmp_bitmask_active_e(block);
    Bitmask&  current_vertices = get_current_vertices_bitmask(curr_level);
    Bitmask&  coarse_vertices  = get_current_vertices_bitmask(curr_level + 1);

    // set the fully owned edges active
    if (curr_level == 0) {
        for (uint16_t e = threadIdx.x; e < num_edges_query; e += blockThreads) {
            uint16_t v0_local_id = s_ev[2 * e];
            uint16_t v1_local_id = s_ev[2 * e + 1];

            if (coarsen_owned(LocalEdgeT(e), curr_level) &&
                coarsen_owned(LocalVertexT(v0_local_id), curr_level) &&
                coarsen_owned(LocalVertexT(v1_local_id), curr_level)) {
                active_edges.set(e, true);
            }
        }
    } else {
        active_edges.set(block);
    }
    block.sync();

    // EV operation: set matched vertices' coarsen id to each other's id for
    // computation
    uint16_t* s_matches_check_v =
        get_new_tmp_attribute_v_arr(block, m_num_vertices_limit, INVALID16);
    for (uint16_t e = threadIdx.x; e < num_edges_query; e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];

        if (!active_edges(e)) {
            continue;
        }

        if (matched_edges(e)) {
            assert(matched_vertices(v0_local_id));
            assert(matched_vertices(v1_local_id));

            uint16_t coarse_id =
                v0_local_id < v1_local_id ? v0_local_id : v1_local_id;
            s_mapping[v0_local_id] = coarse_id;
            s_mapping[v1_local_id] = coarse_id;

            s_matches_check_v[v0_local_id] = v1_local_id;
            s_matches_check_v[v1_local_id] = v0_local_id;
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

    // set init val
    m_s_num_vertices[curr_level + 1] = 0;
    m_s_num_edges[curr_level + 1]    = 0;

    calc_new_temp_ve(
        block, s_ev, m_num_vertices_limit, num_edges_query, curr_level);
    const uint16_t* s_ve_offset = m_s_tmp_offset;
    const uint16_t* s_ve_value  = m_s_tmp_value;

    // check: print out ve result
    // if (idx == 0) {
    //     printf("---------- \n");
    //     for (uint16_t v = 0; v < m_num_vertices_limit;
    //          v += 1) {
    //         uint16_t start = s_ve_offset[v];
    //         uint16_t end   = s_ve_offset[v + 1];

    //         printf("v: %u, start: %u, end: %u \n", v, start, end);
    //         for (uint16_t e = start; e < end; e++) {
    //             uint16_t edge = s_ve_value[e];
    //             uint16_t v0   = s_ev[2 * edge];
    //             uint16_t v1   = s_ev[2 * edge + 1];
    //             printf("      e: %u, v0: %u, v1: %u \n", edge, v0, v1);
    //         }
    //     }
    //     printf("---------- \n");
    // }

    block.sync();

    auto unique_edge_id = [](uint16_t v0, uint16_t v1, uint16_t hash_scale) {
        uint16_t sml_id = v0 < v1 ? v0 : v1;
        uint16_t big_id = v0 < v1 ? v1 : v0;
        return sml_id * hash_scale + big_id;
    };

    // EV operation: determine whether an edge is preserved in the coarsen graph
    for (uint16_t e = threadIdx.x; e < num_edges_query; e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];

        if (!active_edges(e)) {
            continue;
        }

        // matched edges are not preserved
        if (matched_edges(e)) {
            assert(s_mapping[v0_local_id] == s_mapping[v1_local_id]);
            continue;
        }

        bool edge_chosen = true;

        uint32_t current_unique_eid =
            unique_edge_id(v0_local_id, v1_local_id, m_num_vertices_limit);
        uint32_t coarse_unique_eid = unique_edge_id(s_mapping[v0_local_id],
                                                    s_mapping[v1_local_id],
                                                    m_num_vertices_limit);

        uint16_t priority_v =
            v0_local_id < v1_local_id ? v0_local_id : v1_local_id;

        assert(current_vertices(priority_v));

        // request one ring edge of priority_v
        uint16_t start = s_ve_offset[priority_v];
        uint16_t end   = s_ve_offset[priority_v + 1];
        for (uint16_t priority_e_idx = start; priority_e_idx < end;
             priority_e_idx++) {
            uint16_t priority_e = s_ve_value[priority_e_idx];

            // skip current edge
            if (priority_e == e) {
                continue;
            }

            uint16_t priority_local_v0 = s_ev[2 * priority_e];
            uint16_t priority_local_v1 = s_ev[2 * priority_e + 1];

            uint32_t priority_current_unique_eid = unique_edge_id(
                priority_local_v0, priority_local_v1, m_num_vertices_limit);
            uint32_t priority_coarse_unique_eid =
                unique_edge_id(s_mapping[priority_local_v0],
                               s_mapping[priority_local_v1],
                               m_num_vertices_limit);

            // the edge with the same coarse unique id and a lower
            // current_unique_eid has higher priority
            if (priority_coarse_unique_eid == coarse_unique_eid &&
                priority_current_unique_eid < current_unique_eid) {
                edge_chosen = false;
            }
        }

        // checked the paired matched vertex for duplicate edge
        if (edge_chosen && matched_vertices(priority_v)) {
            assert(priority_v ==
                   s_matches_check_v[s_matches_check_v[priority_v]]);

            uint16_t matched_priority_v = s_matches_check_v[priority_v];

            // printf("priority_v: %u, matched_priority_v: %u \n",
            //        priority_v,
            //        matched_priority_v);

            // request one ring edge of mached_priority_v
            uint16_t start = s_ve_offset[matched_priority_v];
            uint16_t end   = s_ve_offset[matched_priority_v + 1];
            for (uint16_t matched_priority_e_idx = start;
                 matched_priority_e_idx < end;
                 matched_priority_e_idx++) {
                uint16_t matched_priority_e =
                    s_ve_value[matched_priority_e_idx];

                if (e == matched_priority_e) {
                    printf(
                        "priority_v: %u, matched_priority_v: %u, matches_v %u, "
                        "matches_pri_v %u \n",
                        priority_v,
                        matched_priority_v,
                        s_matches_check_v[priority_v],
                        s_matches_check_v[matched_priority_v]);
                    printf(
                        "e: %u, matched_priority_e: %u, matched(e): %d, v0: "
                        "%d, v1: %d \n",
                        e,
                        matched_priority_e,
                        matched_edges(e),
                        s_ev[2 * e],
                        s_ev[2 * e + 1]);
                }
                assert(matched_priority_e != e);

                uint16_t matched_priority_local_v0 =
                    s_ev[2 * matched_priority_e];
                uint16_t matched_priority_local_v1 =
                    s_ev[2 * matched_priority_e + 1];

                uint32_t matched_priority_current_unique_eid =
                    unique_edge_id(matched_priority_local_v0,
                                   matched_priority_local_v1,
                                   m_num_vertices_limit);
                uint32_t matched_priority_coarse_unique_eid =
                    unique_edge_id(s_mapping[matched_priority_local_v0],
                                   s_mapping[matched_priority_local_v1],
                                   m_num_vertices_limit);

                // the edge with a lower current_unique_eid is chosen
                if (matched_priority_coarse_unique_eid == coarse_unique_eid &&
                    matched_priority_current_unique_eid < current_unique_eid) {
                    edge_chosen = false;
                }
            }
        }

        if (edge_chosen) {
            uint16_t curr_idx = atomicAdd(&m_s_num_edges[curr_level + 1], 1);
            s_ev_coarse[2 * curr_idx]     = s_mapping[v0_local_id];
            s_ev_coarse[2 * curr_idx + 1] = s_mapping[v1_local_id];
        }
    }
    block.sync();

    // EV operation: mark the vertices available for the next level
    for (uint16_t e = threadIdx.x; e < m_s_num_edges[curr_level + 1];
         e += blockThreads) {
        uint16_t v0_local_id = s_ev_coarse[2 * e];
        uint16_t v1_local_id = s_ev_coarse[2 * e + 1];

        // check: check for duplicate edges
        //  printf("coarse_e: %u, %u, %u \n", e, v0_local_id, v1_local_id);

        coarse_vertices.set(v0_local_id, true);
        coarse_vertices.set(v1_local_id, true);
    }

    // VE operation: count the number of vertices for the next level
    for (uint16_t v = threadIdx.x; v < m_num_vertices_limit;
         v += blockThreads) {
        if (coarse_vertices(v)) {
            atomicAdd(&m_s_num_vertices[curr_level + 1], 1);
        }
    }
    block.sync();

    // assert(m_s_num_vertices[curr_level + 1] <= m_s_num_vertices[curr_level]);
    // assert(m_s_num_edges[curr_level + 1] <= m_s_num_edges[curr_level]);

    // if (idx == 0) {
    //     for (int i = 0; i < num_edges; ++i) {
    //         printf("s_ev[%d]: %d, %d \n", i, s_ev[2 * i], s_ev[2 * i + 1]);
    //     }
    // }

    // check the num vet and num edges diff
    if (idx == 0) {
        printf("num_v: %u, num_e: %u, ",
               m_s_num_vertices[curr_level],
               m_s_num_edges[curr_level]);
        printf("num_v_coarse: %u, num_e_coarse: %u \n",
               m_s_num_vertices[curr_level + 1],
               m_s_num_edges[curr_level + 1]);
    }

    if (idx == 0) {
        for (int i = 0; i < num_edges; ++i) {
            printf("s_ev[%d], %d, %d \n", i, s_ev[2 * i], s_ev[2 * i + 1]);
        }
    }
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
__device__ __inline__ void PartitionManager<blockThreads>::local_partition(
    cooperative_groups::thread_block& block,
    uint16_t                          curr_level)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    const uint16_t num_vertices       = m_s_num_vertices[curr_level];
    const uint16_t num_edges          = m_s_num_edges[curr_level];
    uint16_t*      s_ev               = get_ev(curr_level);
    Bitmask&       coarse_p0_vertices = get_p0_vertices_bitmask(curr_level);
    Bitmask&       coarse_p1_vertices = get_p1_vertices_bitmask(curr_level);
    Bitmask&       separator_v = get_separator_vertices_bitmask(curr_level);
    Bitmask&       current_vertices = get_current_vertices_bitmask(curr_level);
    int            num_iter         = 10;

    if (idx == 0) {
        for (int i = 0; i < num_edges; ++i) {
            printf("s_ev[%d], %d, %d \n", i, s_ev[2 * i], s_ev[2 * i + 1]);
        }
    }

    // EV operation setting active vertices
    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];
        current_vertices.set(v0_local_id, true);
        current_vertices.set(v1_local_id, true);
    }

    // VV operation from VE operation
    calc_new_temp_vv(block, s_ev, m_num_vertices_limit, num_edges, curr_level);
    reset_temp_partition_bitmask(block);

    block.sync();

    const uint16_t* s_vv_offset = m_s_tmp_offset;
    const uint16_t* s_vv_value  = m_s_tmp_value;

    // TODO: double check vv result here

    // if (idx == 0) {
    //     printf("start partitioning \n");
    //     for (int i = 0; i < num_vertices; ++i) {
    //         printf("s_vv_offset[%d]: %d, s_vv_offset[%d]: %d \n",
    //                i,
    //                s_vv_offset[i],
    //                i + 1,
    //                s_vv_offset[i + 1]);
    //         for (int j = s_vv_offset[i]; j < s_vv_offset[i + 1]; ++j) {
    //             printf("    s_vv_value[%d]: %d \n", j, s_vv_value[j]);
    //         }
    //     }
    // }

    if (idx == 0) {
        printf("start partitioning \n");
    }

    detail::bi_assignment_ggp<blockThreads>(
        /* cooperative_groups::thread_block& */ block,
        /* const uint16_t                   */ m_num_vertices_limit,
        /* const Bitmask& s_owned_v         */ current_vertices,
        /* const Bitmask& s_active_v        */ current_vertices,
        /* const uint16_t*                  */ s_vv_offset,
        /* const uint16_t*                  */ s_vv_value,
        /* Bitmask&                         */ m_s_tmp_assigned_v,
        /* Bitmask&                         */ m_s_tmp_current_frontier_v,
        /* Bitmask&                         */ m_s_tmp_next_frontier_v,
        /* Bitmask&                         */ coarse_p0_vertices,
        /* Bitmask&                         */ coarse_p1_vertices,
        /* int                              */ num_iter);

    block.sync();

    if (idx == 0) {
        printf("end partitioning \n");
    }

    // choose the separator vertices from p0 coundary vertices
    // TODO: move the separator to the last step and find out how metis do the
    // limit constrain
    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];

        if (!coarse_p0_vertices(v0_local_id) !=
            !coarse_p1_vertices(v1_local_id)) {
            if (coarse_p0_vertices(v0_local_id)) {
                separator_v.set(v0_local_id, true);
                coarse_p0_vertices.reset(v0_local_id, true);
            }

            if (coarse_p0_vertices(v1_local_id)) {
                separator_v.set(v1_local_id, true);
                coarse_p1_vertices.reset(v1_local_id, true);
            }
        }
    }

    if (idx == 0) {
        for (uint16_t v = 0; v < m_num_vertices_limit; v += 1) {
            if (coarse_p0_vertices(v) && current_vertices(v)) {
                printf("coarse_p0_vertices: %u \n", v);
            }
        }

        for (uint16_t v = 0; v < m_num_vertices_limit; v += 1) {
            if (coarse_p1_vertices(v) && current_vertices(v)) {
                printf("coarse_p1_vertices: %u \n", v);
            }
        }
    }
}


template <uint32_t blockThreads>
__device__ __inline__ void
PartitionManager<blockThreads>::local_multi_level_partition(
    cooperative_groups::thread_block& block,
    uint16_t                          curr_level,
    uint16_t                          partition_level)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    const uint16_t num_vertices       = m_s_num_vertices[curr_level];
    const uint16_t num_edges          = m_s_num_edges[curr_level];
    uint16_t*      s_ev               = get_ev(curr_level);
    Bitmask&       coarse_p0_vertices = get_p0_vertices_bitmask(curr_level);
    Bitmask&       coarse_p1_vertices = get_p1_vertices_bitmask(curr_level);
    Bitmask&       separator_v = get_separator_vertices_bitmask(curr_level);
    Bitmask&       current_vertices = get_current_vertices_bitmask(curr_level);
    Bitmask&       active_vertices = get_new_tmp_bitmask_active_v(block);
    active_vertices.reset(block);
    int            num_iter         = 10;


    // EV operation setting active vertices
    for (uint16_t e = threadIdx.x; e < (num_edges); e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];
        current_vertices.set(v0_local_id, true);
        current_vertices.set(v1_local_id, true);
    }

    // VV operation from VE operation
    calc_new_temp_vv(block, s_ev, m_num_vertices_limit, num_edges, curr_level);
    reset_temp_partition_bitmask(block);

    block.sync();

    const uint16_t* s_vv_offset = m_s_tmp_offset;
    const uint16_t* s_vv_value  = m_s_tmp_value;

    if (idx == 0) {
        printf("start partitioning \n");
    }

    detail::bi_assignment_ggp<blockThreads>(
        /* cooperative_groups::thread_block& */ block,
        /* const uint16_t                   */ m_num_vertices_limit,
        /* const Bitmask& s_owned_v         */ current_vertices,
        /* const Bitmask& s_active_v        */ current_vertices,
        /* const uint16_t*                  */ s_vv_offset,
        /* const uint16_t*                  */ s_vv_value,
        /* Bitmask&                         */ m_s_tmp_assigned_v,
        /* Bitmask&                         */ m_s_tmp_current_frontier_v,
        /* Bitmask&                         */ m_s_tmp_next_frontier_v,
        /* Bitmask&                         */ coarse_p0_vertices,
        /* Bitmask&                         */ coarse_p1_vertices,
        /* int                              */ num_iter);
    block.sync();

    for (uint16_t v = threadIdx.x; v < m_num_vertices_limit; v += blockThreads) {
        assert(coarse_p0_vertices(v) != coarse_p1_vertices(v));
        if (coarse_p0_vertices(v) && current_vertices(v)) {
            m_s_v_partition_id[v] = 0;
        }

        if (coarse_p1_vertices(v) && current_vertices(v)) {
            m_s_v_partition_id[v] = 1;
        }
    }
    block.sync();

    // check result
    if (idx == 0) {
        printf("the first partition\n");
        for (uint16_t v = 0; v < m_num_vertices_limit; v += 1) {
            if (m_s_v_partition_id[v] == 0) {
                printf("coarse_p0_vertices: %u \n", v);
            }
        }

        for (uint16_t v = 0; v < m_num_vertices_limit; v += 1) {
            if (m_s_v_partition_id[v] == 1) {
                printf("coarse_p1_vertices: %u \n", v);
            }
        }

        for (uint16_t v = 0; v < m_num_vertices_limit; v += 1) {
            if (m_s_v_partition_id[v] == 2) {
                printf("coarse_p2_vertices: %u \n", v);
            }
        }

        for (uint16_t v = 0; v < m_num_vertices_limit; v += 1) {
            if (m_s_v_partition_id[v] == 3) {
                printf("coarse_p3_vertices: %u \n", v);
            }
        }
    }

    // test partition for label 1 only

    // EV operation setting active vertices for recursion
    for (uint16_t e = threadIdx.x; e < (num_edges); e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];
        if (m_s_v_partition_id[v0_local_id] == 1) {
            active_vertices.set(v0_local_id, true);
        }
        
        if (m_s_v_partition_id[v1_local_id] == 1) {
            active_vertices.set(v1_local_id, true);
        }
    }
    block.sync();

    if (idx == 0) {
        for (int i = 0; i < m_num_vertices_limit; i++) {
            if (active_vertices(i)) {
                printf("active_vertices: %u \n", i);
            }
        }
    }

    // reuse the VV result
    m_s_tmp_assigned_v.reset(block);
    m_s_tmp_current_frontier_v.reset(block);
    m_s_tmp_next_frontier_v.reset(block);
    coarse_p0_vertices.reset(block);
    coarse_p1_vertices.reset(block);
    block.sync();
    detail::bi_assignment_ggp<blockThreads>(
        /* cooperative_groups::thread_block& */ block,
        /* const uint16_t                   */ m_num_vertices_limit,
        /* const Bitmask& s_owned_v         */ active_vertices,
        /* const Bitmask& s_active_v        */ active_vertices,
        /* const uint16_t*                  */ s_vv_offset,
        /* const uint16_t*                  */ s_vv_value,
        /* Bitmask&                         */ m_s_tmp_assigned_v,
        /* Bitmask&                         */ m_s_tmp_current_frontier_v,
        /* Bitmask&                         */ m_s_tmp_next_frontier_v,
        /* Bitmask&                         */ coarse_p0_vertices,
        /* Bitmask&                         */ coarse_p1_vertices,
        /* int                              */ num_iter);

    block.sync();

    for (uint16_t v = threadIdx.x; v < m_num_vertices_limit; v += blockThreads) {
        assert(coarse_p0_vertices(v) != coarse_p1_vertices(v));
        if (coarse_p0_vertices(v)) {
            m_s_v_partition_id[v] = 2;
            // printf("--- check coarse_p0_vertices: %u \n", v);
        }

        if (coarse_p1_vertices(v)) {
            m_s_v_partition_id[v] = 3;
            // printf("--- check coarse_p1_vertices: %u \n", v);
        }
    }
    block.sync();

    if (idx == 0) {
        printf("the second partition\n");
        for (uint16_t v = 0; v < m_num_vertices_limit; v += 1) {
            if (m_s_v_partition_id[v] == 0) {
                printf("coarse_p0_vertices: %u \n", v);
            }
        }

        for (uint16_t v = 0; v < m_num_vertices_limit; v += 1) {
            if (m_s_v_partition_id[v] == 1) {
                printf("coarse_p1_vertices: %u \n", v);
            }
        }

        for (uint16_t v = 0; v < m_num_vertices_limit; v += 1) {
            if (m_s_v_partition_id[v] == 2) {
                printf("coarse_p2_vertices: %u \n", v);
            }
        }

        for (uint16_t v = 0; v < m_num_vertices_limit; v += 1) {
            if (m_s_v_partition_id[v] == 3) {
                printf("coarse_p3_vertices: %u \n", v);
            }
        }
    }
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
__device__ __inline__ void PartitionManager<blockThreads>::local_uncoarsening(
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

    for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        // uint16_t start = m_s_tmp_offset[v];
        // uint16_t end   = m_s_tmp_offset[v + 1];

        if (coarse_p0_vertices(s_mapping[v])) {
            p0_vertices.set(v, true);
        }

        if (coarse_p1_vertices(s_mapping[v])) {
            p1_vertices.set(v, true);
        }

        if (coarse_separator_v(s_mapping[v])) {
            separator_v.set(v, true);
        }
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
__device__ __inline__ void
PartitionManager<blockThreads>::local_genrate_reordering(
    cooperative_groups::thread_block& block,
    VertexAttribute<uint16_t>         v_ordering)
{
    // Get level 0 param
    uint16_t  num_vertices = m_num_vertices_limit;
    uint16_t  num_edges    = m_num_edges_limit;
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
