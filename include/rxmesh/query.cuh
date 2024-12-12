#pragma once

#include <assert.h>
#include <cooperative_groups.h>

#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/query_dispatcher.cuh"
#include "rxmesh/kernels/shmem_allocator.cuh"
#include "rxmesh/types.h"
#include "rxmesh/util/meta.h"

namespace rxmesh {

template <uint32_t blockThreads>
struct Query
{
    Query(const Query&)            = delete;
    Query& operator=(const Query&) = delete;

    __device__ __inline__ Query(const Context& context,
                                const uint32_t pid = blockIdx.x)
        : m_context(context),
          m_patch_info(context.m_patches_info[pid]),
          m_num_src_in_patch(0),
          m_s_participant_bitmask(nullptr),
          m_s_output_owned_bitmask(nullptr),
          m_s_output_offset(nullptr),
          m_s_output_value(nullptr),
          m_s_valence(nullptr),
          m_s_table(nullptr)
    {
    }

    /**
     * @brief return the patch id associated with this instance
     */
    __device__ __inline__ int get_patch_id() const
    {
        return m_patch_info.patch_id;
    }

    /**
     * @brief return the patch info associated with this instance
     */
    __device__ __inline__ const PatchInfo& get_patch_info() const
    {
        return m_patch_info;
    }

    /**
     * @brief compute the vertex valence
     */
    __device__ __inline__ void compute_vertex_valence(
        cooperative_groups::thread_block& block,
        ShmemAllocator&                   shrd_alloc)
    {
        const uint16_t num_vertices = m_patch_info.num_vertices[0];
        const uint16_t num_edges    = m_patch_info.num_edges[0];

        m_s_valence = shrd_alloc.alloc<uint8_t>(num_vertices);

        fill_n<blockThreads>(m_s_valence, num_vertices, uint8_t(0));
        block.sync();

        for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
            if (!m_patch_info.is_deleted(LocalEdgeT(e))) {
                auto [v0, v1] = m_patch_info.get_edge_vertices(e);
                atomicAdd(m_s_valence + v0, uint8_t(1));
                atomicAdd(m_s_valence + v1, uint8_t(1));
                assert(m_s_valence[v0] < 255);
                assert(m_s_valence[v1] < 255);
            }
        }
        block.sync();
    }


    /**
     * @brief return the vertex valence. compute_vertex_valence has to be called
     * first
     * @param v vertex for which valence will be returned
     */
    __device__ __inline__ uint16_t vertex_valence(uint16_t v) const
    {
        assert(m_s_valence);
        assert(v < m_patch_info.num_vertices[0]);
        return m_s_valence[v];
    }

    /**
     * @brief return the vertex valence. compute_vertex_valence has to be called
     * first
     * @param vh vertex for which valence will be returned
     */
    __device__ __inline__ uint16_t vertex_valence(VertexHandle vh) const
    {
        assert(m_s_valence);
        assert(vh.patch_id() == m_patch_info.patch_id);
        assert(vh.local_id() < m_patch_info.num_vertices[0]);
        return m_s_valence[vh.local_id()];
    }

    /**
     * @brief The query dispatch function to be called by the whole block. In
     * this function, threads will be assigned to mesh elements which will be
     * accessible through the input computation lambda function (compute_op).
     * @tparam Op the type of query operation
     * @tparam computeT the type of compute lambda function (inferred)
     * @param block cooperative group block
     * @param shrd_alloc dynamic shared memory allocator
     * @param context which store various parameters needed for the query
     * operation. The context can be obtained from RXMeshStatic
     * @param compute_op the computation lambda function that will be executed
     * by each thread in the block. This lambda function takes two input
     * parameters:
     * 1. Handle to the mesh element assigned to the thread. The handle type
     * matches the source of the query (e.g., VertexHandle for VE query) 2. an
     * iterator to the query output. The iterator type matches the type of the
     * mesh element "iterated" on (e.g., EdgeIterator for VE query)
     * @param oriented specifies if the query are oriented. Currently only VV
     * and EV query is supported for oriented queries. FV, FE and EV is oriented
     * by default
     */
    template <Op op, typename computeT>
    __device__ __inline__ void dispatch(cooperative_groups::thread_block& block,
                                        ShmemAllocator& shrd_alloc,
                                        computeT        compute_op,
                                        const bool      oriented = false)
    {
        // Extract the type of the first input parameters of the compute lambda
        // function. It should be Vertex/Edge/FaceHandle
        using ComputeTraits  = detail::FunctionTraits<computeT>;
        using ComputeHandleT = typename ComputeTraits::template arg<0>::type;

        dispatch<op>(
            block,
            shrd_alloc,
            compute_op,
            [](ComputeHandleT) { return true; },
            oriented);
    }


    /**
     * @brief The query dispatch function to be called by the whole block. In
     * this function, threads will be assigned to mesh elements which will be
     * accessible through the input computation lambda function (compute_op).
     * This function also provides a predicate to specify the active set i.e.,
     * the set on which the query operations should be done. This is mainly used
     * to skip query on a subset of the input mesh elements which may lead to
     * better performance
     * @tparam Op the type of query operation
     * @tparam computeT the type of compute lambda function (inferred)
     * @tparam activeSetT the type of active set lambda function (inferred)
     * @param block cooperative group block
     * @param shrd_alloc dynamic shared memory allocator
     * @param context which store various parameters needed for the query
     * operation. The context can be obtained from RXMeshStatic
     * @param compute_op the computation lambda function that will be executed
     * by each thread in the block. This lambda function takes two input
     * parameters:
     * 1. Handle to the mesh element assigned to the thread. The handle type
     * matches the source of the query (e.g., VertexHandle for VE query) 2. an
     * iterator to the query output. The iterator type matches the type of the
     * mesh element "iterated" on (e.g., EdgeIterator for VE query)
     * @param compute_active_set a predicate used to specify the active set.
     * This lambda function take a single parameter which is a handle of the
     * type similar to the input of the query operation (e.g., VertexHandle for
     * VE query)
     * @param oriented specifies if the query are oriented. Currently only VV
     * and EV query is supported for oriented queries. FV, FE and EV is oriented
     * by default
     */
    template <Op op, typename computeT, typename activeSetT>
    __device__ __inline__ void dispatch(cooperative_groups::thread_block& block,
                                        ShmemAllocator& shrd_alloc,
                                        computeT        compute_op,
                                        activeSetT      compute_active_set,
                                        const bool      oriented        = false,
                                        const bool      allow_not_owned = false)
    {

        // Extract the type of the input parameters of the compute lambda
        // function.
        // The first parameter should be Vertex/Edge/FaceHandle and second
        // parameter should be RXMeshVertex/Edge/FaceIterator
        using ComputeTraits    = detail::FunctionTraits<computeT>;
        using ComputeHandleT   = typename ComputeTraits::template arg<0>::type;
        using ComputeIteratorT = typename ComputeTraits::template arg<1>::type;
        using LocalT           = typename ComputeIteratorT::LocalT;

        // Extract the type of the single input parameter of the active_set
        // lambda function. It should be Vertex/Edge/FaceHandle and it should
        // match the first parameter of the compute lambda function
        using ActiveSetTraits = detail::FunctionTraits<activeSetT>;
        using ActiveSetHandleT =
            typename ActiveSetTraits::template arg<0>::type;
        static_assert(
            std::is_same_v<ActiveSetHandleT, ComputeHandleT>,
            "First argument of compute_op lambda function should "
            "match the first argument of active_set lambda function ");

        prologue<op>(
            block, shrd_alloc, compute_active_set, oriented, allow_not_owned);

        run_compute(block, compute_op);

        epilogue(block, shrd_alloc);
    }

    /**
     * @brief run the query and prepare internal data structure to run the
     * computation on top of the queries
     */
    template <Op op>
    __device__ __inline__ void prologue(cooperative_groups::thread_block& block,
                                        ShmemAllocator& shrd_alloc,
                                        const bool      oriented        = false,
                                        const bool      allow_not_owned = true)
    {
        prologue<op>(
            block,
            shrd_alloc,
            [](VertexHandle) { return true; },
            oriented,
            allow_not_owned);
    }


    /**
     * @brief run the query and prepare internal data structure to run the
     * computation on top of the queries
     */
    template <Op op, typename activeSetT>
    __device__ __inline__ void prologue(cooperative_groups::thread_block& block,
                                        ShmemAllocator& shrd_alloc,
                                        activeSetT      compute_active_set,
                                        const bool      oriented        = false,
                                        const bool      allow_not_owned = true)
    {
        m_op           = op;
        m_shmem_before = shrd_alloc.get_allocated_size_bytes();

        detail::query_block_dispatcher<op, blockThreads>(
            block,
            shrd_alloc,
            m_patch_info,
            compute_active_set,
            oriented,
            m_num_src_in_patch,
            m_s_output_offset,
            m_s_output_value,
            m_s_participant_bitmask,
            m_s_output_owned_bitmask,
            m_output_lp_hashtable,
            m_s_table,
            allow_not_owned);
    }


    /**
     * @brief run the computation on the query operations. Should be done after
     * calling prologue() and it should be called by the whole block where every
     * thread will be assigned to one element
     */
    template <typename computeT>
    __device__ __inline__ void run_compute(
        cooperative_groups::thread_block& block,
        computeT                          compute_op)
    {
        using ComputeTraits    = detail::FunctionTraits<computeT>;
        using ComputeHandleT   = typename ComputeTraits::template arg<0>::type;
        using ComputeIteratorT = typename ComputeTraits::template arg<1>::type;

        for (uint16_t local_id = threadIdx.x; local_id < m_num_src_in_patch;
             local_id += blockThreads) {

            if (detail::is_set_bit(local_id, m_s_participant_bitmask)) {

                assert(m_s_output_value);

                ComputeHandleT   handle(m_patch_info.patch_id, local_id);
                ComputeIteratorT iter =
                    get_iterator<ComputeIteratorT>(local_id);
                compute_op(handle, iter);
            }
        }
    }


    /**
     * @brief return an iterator over the queries elements give a local index of
     * a source element
     */
    template <typename IteratorT>
    __device__ __inline__ IteratorT get_iterator(uint16_t local_id) const
    {
        const uint32_t fixed_offset =
            ((m_op == Op::EV) ? 2 :
                                ((m_op == Op::FV || m_op == Op::FE) ?
                                     3 :
                                     ((m_op == Op::EVDiamond) ? 4 : 0)));

        using LocalT = typename IteratorT::LocalT;

        if (detail::is_set_bit(local_id, m_s_participant_bitmask)) {
            return IteratorT(m_context,
                             local_id,
                             reinterpret_cast<LocalT*>(m_s_output_value),
                             m_s_output_offset,
                             fixed_offset,
                             m_patch_info.patch_id,
                             m_s_output_owned_bitmask,
                             m_output_lp_hashtable,
                             m_s_table,
                             m_patch_info.patch_stash,
                             int(m_op == Op::FE));
        } else {
            return IteratorT(m_context, local_id, m_patch_info.patch_id);
        }
    }


    /**
     * @brief free up shared memory allocated to store the query operations.
     */
    __device__ __inline__ void epilogue(cooperative_groups::thread_block& block,
                                        ShmemAllocator& shrd_alloc)
    {
        // cleanup shared memory allocation
        shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() -
                           m_shmem_before);
        m_num_src_in_patch       = 0;
        m_s_participant_bitmask  = nullptr;
        m_s_output_owned_bitmask = nullptr;
        m_s_output_offset        = nullptr;
        m_s_output_value         = nullptr;
        m_s_table                = nullptr;
    }

   private:
    const Context&   m_context;
    const PatchInfo& m_patch_info;
    uint32_t         m_shmem_before;
    uint32_t         m_num_src_in_patch;
    uint32_t*        m_s_participant_bitmask;
    uint32_t*        m_s_output_owned_bitmask;
    uint16_t*        m_s_output_offset;
    uint16_t*        m_s_output_value;
    uint8_t*         m_s_valence;
    LPHashTable      m_output_lp_hashtable;
    LPPair*          m_s_table;
    Op               m_op;
};
}  // namespace rxmesh