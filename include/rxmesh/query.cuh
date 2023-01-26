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
    Query(const Query&) = delete;
    Query& operator=(const Query&) = delete;

    __device__ __inline__ Query(const Context& context,
                                const uint32_t pid = blockIdx.x)
        : m_patch_info(context.m_patches_info[pid]),
          m_num_src_in_patch(0),
          m_s_participant_bitmask(nullptr),
          m_s_output_owned_bitmask(nullptr),
          m_s_output_offset(nullptr),
          m_s_output_value(nullptr),
          m_s_table(nullptr)
    {
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
                                        const bool      oriented = false)
    {
        static_assert(op != Op::EE, "Op::EE is not supported!");

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

        const uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

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
            m_s_table);

        constexpr uint32_t fixed_offset =
            ((op == Op::EV) ? 2 :
                              ((op == Op::FV || op == Op::FE) ?
                                   3 :
                                   ((op == Op::EVDiamond) ? 4 : 0)));

        const uint32_t patch_id = m_patch_info.patch_id;

        for (uint16_t local_id = threadIdx.x; local_id < m_num_src_in_patch;
             local_id += blockThreads) {

            assert(m_s_output_value);

            if (detail::is_set_bit(local_id, m_s_participant_bitmask)) {

                ComputeHandleT   handle(patch_id, local_id);
                ComputeIteratorT iter(
                    local_id,
                    reinterpret_cast<LocalT*>(m_s_output_value),
                    m_s_output_offset,
                    fixed_offset,
                    patch_id,
                    m_s_output_owned_bitmask,
                    m_output_lp_hashtable,
                    m_s_table,
                    m_patch_info.patch_stash,
                    int(op == Op::FE));

                compute_op(handle, iter);
            }
        }

        // cleanup shared memory allocation
        shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() -
                           shmem_before);
    }

   private:
    const PatchInfo& m_patch_info;
    uint32_t         m_num_src_in_patch;
    uint32_t*        m_s_participant_bitmask;
    uint32_t*        m_s_output_owned_bitmask;
    uint16_t*        m_s_output_offset;
    uint16_t*        m_s_output_value;
    LPHashTable      m_output_lp_hashtable;
    LPPair*          m_s_table;
};
}  // namespace rxmesh