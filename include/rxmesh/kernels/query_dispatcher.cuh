#pragma once
#include <assert.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <cub/block/block_discontinuity.cuh>

#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/iterator.cuh"
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/debug.cuh"
#include "rxmesh/kernels/dynamic_util.cuh"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/rxmesh_queries.cuh"
#include "rxmesh/kernels/shmem_allocator.cuh"
#include "rxmesh/types.h"
#include "rxmesh/util/bitmask_util.h"
#include "rxmesh/util/meta.h"

namespace rxmesh {

namespace detail {

/**
 * query_block_dispatcher()
 */
template <Op op, uint32_t blockThreads, typename activeSetT>
__device__ __inline__ void query_block_dispatcher(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc,
    const PatchInfo&                  patch_info,
    activeSetT                        compute_active_set,
    const bool                        oriented,
    uint32_t&                         num_src_in_patch,
    uint16_t*&                        s_output_offset,
    uint16_t*&                        s_output_value,
    uint32_t*&                        s_participant_bitmask,
    uint32_t*&                        s_output_owned_bitmask,
    LPHashTable&                      output_lp_hashtable,
    LPPair*&                          s_table)
{
    static_assert(op != Op::EE, "Op::EE is not supported!");

    num_src_in_patch                = 0;
    uint16_t    num_output_in_patch = 0;
    uint32_t *  input_active_mask, *input_owned_mask;
    LPHashTable hashtable;
    if constexpr (op == Op::VV || op == Op::VE || op == Op::VF) {
        num_src_in_patch  = patch_info.num_vertices[0];
        input_active_mask = patch_info.active_mask_v;
        input_owned_mask  = patch_info.owned_mask_v;
    }
    if constexpr (op == Op::EV || op == Op::EF || op == Op::EVDiamond) {
        num_src_in_patch  = patch_info.num_edges[0];
        input_active_mask = patch_info.active_mask_e;
        input_owned_mask  = patch_info.owned_mask_e;
    }
    if constexpr (op == Op::FV || op == Op::FE || op == Op::FF) {
        num_src_in_patch  = patch_info.num_faces[0];
        input_active_mask = patch_info.active_mask_f;
        input_owned_mask  = patch_info.owned_mask_f;
    }

    // alloc participant bitmask
    s_participant_bitmask = reinterpret_cast<uint32_t*>(
        shrd_alloc.alloc(mask_num_bytes(num_src_in_patch)));

    for (uint32_t i = threadIdx.x; i < DIVIDE_UP(num_src_in_patch, 32);
         i += blockThreads) {
        s_participant_bitmask[i] = 0;
    }
    __syncthreads();

    // alloc and load owned mask async
    // select lp hashtable
    if constexpr (op == Op::VV || op == Op::EV || op == Op::FV ||
                  op == Op::EVDiamond) {
        const uint32_t mask_size = mask_num_bytes(patch_info.num_vertices[0]);
        s_output_owned_bitmask =
            reinterpret_cast<uint32_t*>(shrd_alloc.alloc(mask_size));
        load_async(reinterpret_cast<char*>(patch_info.owned_mask_v),
                   mask_size,
                   reinterpret_cast<char*>(s_output_owned_bitmask),
                   false);
        output_lp_hashtable = patch_info.lp_v;
    }
    if constexpr (op == Op::VE || op == Op::EE || op == Op::FE) {
        const uint32_t mask_size = mask_num_bytes(patch_info.num_edges[0]);
        s_output_owned_bitmask =
            reinterpret_cast<uint32_t*>(shrd_alloc.alloc(mask_size));
        load_async(reinterpret_cast<char*>(patch_info.owned_mask_e),
                   mask_size,
                   reinterpret_cast<char*>(s_output_owned_bitmask),
                   false);
        output_lp_hashtable = patch_info.lp_e;
    }
    if constexpr (op == Op::VF || op == Op::EF || op == Op::FF) {
        const uint32_t mask_size = mask_num_bytes(patch_info.num_faces[0]);
        s_output_owned_bitmask =
            reinterpret_cast<uint32_t*>(shrd_alloc.alloc(mask_size));
        load_async(reinterpret_cast<char*>(patch_info.owned_mask_f),
                   mask_size,
                   reinterpret_cast<char*>(s_output_owned_bitmask),
                   false);
        output_lp_hashtable = patch_info.lp_f;
    }


    // load table async
    auto alloc_then_load_table = [&](bool with_wait) {
        s_table = shrd_alloc.template alloc<LPPair>(
            output_lp_hashtable.get_capacity());
        load_async(block,
                   output_lp_hashtable.get_table(),
                   output_lp_hashtable.get_capacity(),
                   s_table,
                   with_wait);
    };

    if constexpr (op != Op::FV && op != Op::VV && op != Op::FF &&
                  op != Op::EVDiamond) {
        if (op != Op::EV && oriented) {
            alloc_then_load_table(false);
        }
    }


    // we  cache the result of (is_active && is_owned && is_compute_set) in
    // shared memory to check on it later
    bool is_participant = false;
    block_loop<uint16_t,
               blockThreads,
               true>(num_src_in_patch, [&](const uint16_t local_id) {
        bool is_par = false;
        if (local_id < num_src_in_patch) {
            bool is_del = is_deleted(local_id, input_active_mask);
            bool is_own = is_owned(local_id, input_owned_mask);
            bool is_act = compute_active_set({patch_info.patch_id, local_id});
            is_par      = !is_del && is_own && is_act;
        }
        is_participant     = is_participant || is_par;
        uint32_t warp_mask = __ballot_sync(0xFFFFFFFF, is_par);
        uint32_t lane_id   = threadIdx.x % 32;
        if (lane_id == 0) {
            uint32_t mask_id               = local_id / 32;
            s_participant_bitmask[mask_id] = warp_mask;
        }
    });


    if (__syncthreads_or(is_participant) == 0) {
        // reset num_src_in_patch to zero to indicate that this block/patch has
        // no work to do
        num_src_in_patch = 0;
        return;
    }


    // Perform the query operation
    query<blockThreads, op>(block,
                            patch_info,
                            shrd_alloc,
                            s_output_offset,
                            s_output_value,
                            oriented);

    if constexpr (op == Op::FV || op == Op::VV || op == Op::FF ||
                  op == Op::EVDiamond) {
        block.sync();
        alloc_then_load_table(true);
    }
    if (op == Op::EV && oriented) {
        block.sync();
        alloc_then_load_table(true);
    }
    block.sync();
}


/**
 * query_block_dispatcher()
 */
template <Op op, uint32_t blockThreads, typename computeT, typename activeSetT>
__device__ __inline__ void query_block_dispatcher(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc,
    const Context&                    context,
    const uint32_t                    patch_id,
    computeT                          compute_op,
    activeSetT                        compute_active_set,
    const bool                        oriented = false)
{
    // Extract the type of the input parameters of the compute lambda function.
    // The first parameter should be Vertex/Edge/FaceHandle and second parameter
    // should be RXMeshVertex/Edge/FaceIterator
    using ComputeTraits    = detail::FunctionTraits<computeT>;
    using ComputeHandleT   = typename ComputeTraits::template arg<0>::type;
    using ComputeIteratorT = typename ComputeTraits::template arg<1>::type;
    using LocalT           = typename ComputeIteratorT::LocalT;

    // Extract the type of the single input parameter of the active_set lambda
    // function. It should be Vertex/Edge/FaceHandle and it should match the
    // first parameter of the compute lambda function
    using ActiveSetTraits  = detail::FunctionTraits<activeSetT>;
    using ActiveSetHandleT = typename ActiveSetTraits::template arg<0>::type;
    static_assert(
        std::is_same_v<ActiveSetHandleT, ComputeHandleT>,
        "First argument of compute_op lambda function should match the first "
        "argument of active_set lambda function ");

    static_assert(op != Op::EE, "Op::EE is not supported!");


    assert(patch_id < context.m_num_patches[0]);


    uint32_t    num_src_in_patch = 0;
    uint16_t*   s_output_offset(nullptr);
    uint16_t*   s_output_value(nullptr);
    uint32_t*   s_participant_bitmask;
    uint32_t*   s_output_owned_bitmask;
    LPHashTable output_lp_hashtable;
    LPPair*     s_table;

    query_block_dispatcher<op, blockThreads>(block,
                                             shrd_alloc,
                                             context.m_patches_info[patch_id],
                                             compute_active_set,
                                             oriented,
                                             num_src_in_patch,
                                             s_output_offset,
                                             s_output_value,
                                             s_participant_bitmask,
                                             s_output_owned_bitmask,
                                             output_lp_hashtable,
                                             s_table);

    // Call compute on the output in shared memory by looping over all
    // source elements in this patch.
    constexpr uint32_t fixed_offset =
        ((op == Op::EV) ? 2 :
                          ((op == Op::FV || op == Op::FE) ?
                               3 :
                               ((op == Op::EVDiamond) ? 4 : 0)));

    for (uint16_t local_id = threadIdx.x; local_id < num_src_in_patch;
         local_id += blockThreads) {

        assert(s_output_value);

        if (is_set_bit(local_id, s_participant_bitmask)) {

            ComputeHandleT   handle(patch_id, local_id);
            ComputeIteratorT iter(context,
                                  local_id,
                                  reinterpret_cast<LocalT*>(s_output_value),
                                  s_output_offset,
                                  fixed_offset,
                                  patch_id,
                                  s_output_owned_bitmask,
                                  output_lp_hashtable,
                                  s_table,
                                  context.m_patches_info[patch_id].patch_stash,
                                  int(op == Op::FE));

            compute_op(handle, iter);
        }
    }
}


/**
 * query_block_dispatcher()
 */
template <Op op, uint32_t blockThreads, typename computeT, typename activeSetT>
__device__ __inline__ void query_block_dispatcher(const Context& context,
                                                  const uint32_t patch_id,
                                                  computeT       compute_op,
                                                  activeSetT compute_active_set,
                                                  const bool oriented = false)
{
    namespace cg           = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    ShmemAllocator   shrd_alloc;

    query_block_dispatcher<op, blockThreads>(block,
                                             shrd_alloc,
                                             context,
                                             patch_id,
                                             compute_op,
                                             compute_active_set,
                                             oriented);
}

}  // namespace detail
/**
 * @brief The main query function to be called by the whole block. In this
 * function, threads will be assigned to mesh elements which will be accessible
 * through the input computation lambda function (compute_op). This function
 * also provides a predicate to specify the active set i.e., the set on which
 * the query operations should be done. This is mainly used to skip query on
 * a subset of the input mesh elements which may lead to better performance
 * @tparam Op the type of query operation
 * @tparam blockThreads the number of CUDA threads in the block
 * @tparam computeT the type of compute lambda function (inferred)
 * @tparam activeSetT the type of active set lambda function (inferred)
 * @param block cooperative group block
 * @param shrd_alloc dynamic shared memory allocator
 * @param context which store various parameters needed for the query
 * operation. The context can be obtained from RXMeshStatic
 * @param compute_op the computation lambda function that will be executed by
 * each thread in the block. This lambda function takes two input parameters:
 * 1. Handle to the mesh element assigned to the thread. The handle type matches
 * the source of the query (e.g., VertexHandle for VE query) 2. an iterator to
 * the query output. The iterator type matches the type of the mesh element
 * "iterated" on (e.g., EdgeIterator for VE query)
 * @param compute_active_set a predicate used to specify the active set. This
 * lambda function take a single parameter which is a handle of the type similar
 * to the input of the query operation (e.g., VertexHandle for VE query)
 * @param oriented specifies if the query are oriented. Currently only VV query
 * is supported for oriented queries. FV, FE and EV is oriented by default
 */
template <Op op, uint32_t blockThreads, typename computeT, typename activeSetT>
__device__ __inline__ void query_block_dispatcher(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc,
    const Context&                    context,
    computeT                          compute_op,
    activeSetT                        compute_active_set,
    const bool                        oriented = false)
{
    if (blockIdx.x >= context.m_num_patches[0]) {
        return;
    }

    detail::query_block_dispatcher<op, blockThreads>(block,
                                                     shrd_alloc,
                                                     context,
                                                     blockIdx.x,
                                                     compute_op,
                                                     compute_active_set,
                                                     oriented);
}

/**
 * @brief same as the above function but no cooperative group or shared memory
 * allocator needed
 */
template <Op op, uint32_t blockThreads, typename computeT, typename activeSetT>
__device__ __inline__ void query_block_dispatcher(const Context& context,
                                                  computeT       compute_op,
                                                  activeSetT compute_active_set,
                                                  const bool oriented = false)
{
    if (blockIdx.x >= context.m_num_patches[0]) {
        return;
    }

    detail::query_block_dispatcher<op, blockThreads>(
        context, blockIdx.x, compute_op, compute_active_set, oriented);
}


/**
 * @brief The main query function to be called by the whole block. In this
 * function, threads will be assigned to mesh elements which will be accessible
 * through the input computation lambda function (compute_op).
 * @tparam Op the type of query operation
 * @tparam blockThreads the number of CUDA threads in the block
 * @tparam computeT the type of compute lambda function (inferred)
 * @param block cooperative group block
 * @param shrd_alloc dynamic shared memory allocator
 * @param context which store various parameters needed for the query
 * operation. The context can be obtained from RXMeshStatic
 * @param compute_op the computation lambda function that will be executed by
 * each thread in the block. This lambda function takes two input parameters:
 * 1. Handle to the mesh element assigned to the thread. The handle type matches
 * the source of the query (e.g., VertexHandle for VE query) 2. an iterator to
 * the query output. The iterator type matches the type of the mesh element
 * "iterated" on (e.g., EdgeIterator for VE query)
 * @param oriented specifies if the query are oriented. Currently only VV query
 * is supported for oriented queries. FV, FE and EV is oriented by default
 */
template <Op op, uint32_t blockThreads, typename computeT>
__device__ __inline__ void query_block_dispatcher(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc,
    const Context&                    context,
    computeT                          compute_op,
    const bool                        oriented = false)
{
    // Extract the type of the first input parameters of the compute lambda
    // function. It should be Vertex/Edge/FaceHandle
    using ComputeTraits  = detail::FunctionTraits<computeT>;
    using ComputeHandleT = typename ComputeTraits::template arg<0>::type;

    query_block_dispatcher<op, blockThreads>(
        block,
        shrd_alloc,
        context,
        compute_op,
        [](ComputeHandleT) { return true; },
        oriented);
}

/**
 * @brief same as the above function but no cooperative group or shared memory
 * allocator needed
 */
template <Op op, uint32_t blockThreads, typename computeT>
__device__ __inline__ void query_block_dispatcher(const Context& context,
                                                  computeT       compute_op,
                                                  const bool oriented = false)
{
    // Extract the type of the first input parameters of the compute lambda
    // function. It should be Vertex/Edge/FaceHandle
    using ComputeTraits  = detail::FunctionTraits<computeT>;
    using ComputeHandleT = typename ComputeTraits::template arg<0>::type;

    query_block_dispatcher<op, blockThreads>(
        context, compute_op, [](ComputeHandleT) { return true; }, oriented);
}


/**
 * @brief This function is used to perform a query operation on a specific mesh
 * element. This is only needed for higher query (e.g., 2-ring query) where the
 * first query is done using query_block_dispatcher in which each thread is
 * assigned to a mesh element. Subsequent queries should be handled by this
 * function. This function should be called by the whole CUDA block.
 * @tparam Op the type of query operation
 * @tparam blockThreads the number of CUDA threads in the block
 * @tparam computeT the type of compute lambda function (inferred)
 * @tparam HandleT the type of input handle (inferred) which should match the
 * input of the query operations (e.g., VertexHandle for VE query)
 * @param context which store various parameters needed for the query
 * operation. The context can be obtained from RXMeshStatic
 * @param src_id the input mesh element to the query. Inactive threads can
 * simply pass HandleT() in which case they are skipped
 * @param compute_op the computation lambda function that will be executed by
 * the thread. This lambda function takes two input parameters:
 * 1. HandleT which is the same as src_id 2. an iterator to the query output.
 * The iterator type matches the type of the mesh element "iterated" on (e.g.,
 * EdgeIterator for VE query)
 * @param oriented specifies if the query are oriented. Currently only VV query
 * is supported for oriented queries. FV, FE and EV is oriented by default
 */
template <Op op, uint32_t blockThreads, typename computeT, typename HandleT>
__device__ __inline__ void higher_query_block_dispatcher(
    const Context& context,
    const HandleT  src_id,
    computeT       compute_op,
    const bool     oriented = false)
{
    using ComputeTraits    = detail::FunctionTraits<computeT>;
    using ComputeIteratorT = typename ComputeTraits::template arg<1>::type;

    // The whole block should be calling this function. If one thread is not
    // participating, its src_id should be INVALID32

    auto compute_active_set = [](HandleT) { return true; };

    // the source and local id of the source mesh element
    std::pair<uint32_t, uint16_t> pl = src_id.unpack();

    // Here, we want to identify the set of unique patches for this thread
    // block. We do this by first sorting the patches, compute discontinuity
    // head flag, then threads with head flag =1 can add their patches to the
    // shared memory buffer that will contain the unique patches

    __shared__ uint32_t s_block_patches[blockThreads];
    __shared__ uint32_t s_num_patches;
    if (threadIdx.x == 0) {
        s_num_patches = 0;
    }
    typedef cub::BlockRadixSort<uint32_t, blockThreads, 1>  BlockRadixSort;
    typedef cub::BlockDiscontinuity<uint32_t, blockThreads> BlockDiscontinuity;
    union TempStorage
    {
        typename BlockRadixSort::TempStorage     sort_storage;
        typename BlockDiscontinuity::TempStorage discont_storage;
    };
    __shared__ TempStorage all_temp_storage;
    uint32_t               thread_data[1], thread_head_flags[1];
    thread_data[0]       = pl.first;
    thread_head_flags[0] = 0;
    BlockRadixSort(all_temp_storage.sort_storage).Sort(thread_data);
    BlockDiscontinuity(all_temp_storage.discont_storage)
        .FlagHeads(thread_head_flags, thread_data, cub::Inequality());

    if (thread_head_flags[0] == 1 && thread_data[0] != INVALID32) {
        uint32_t id         = ::atomicAdd(&s_num_patches, uint32_t(1));
        s_block_patches[id] = thread_data[0];
    }

    // We could eliminate the discontinuity operation and atomicAdd and instead
    // use thrust::unique. However, this method causes illegal memory access
    // and it looks like a bug in thrust
    /*__syncthreads();
    // uniquify
    uint32_t* new_end = thrust::unique(thrust::device, s_block_patches,
                                       s_block_patches + blockThreads);
    __syncthreads();

    if (threadIdx.x == 0) {
        s_num_patches = new_end - s_block_patches - 1;
    }*/
    __syncthreads();

    namespace cg           = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    ShmemAllocator   shrd_alloc;

    for (uint32_t p = 0; p < s_num_patches; ++p) {

        uint32_t patch_id = s_block_patches[p];

        assert(patch_id < context.m_num_patches[0]);

        uint32_t    num_src_in_patch = 0;
        uint16_t*   s_output_offset(nullptr);
        uint16_t*   s_output_value(nullptr);
        uint32_t*   s_participant_bitmask;
        uint32_t*   s_output_owned_bitmask;
        LPHashTable output_lp_hashtable;
        LPPair*     s_table;

        detail::template query_block_dispatcher<op, blockThreads>(
            block,
            shrd_alloc,
            context.m_patches_info[patch_id],
            compute_active_set,
            oriented,
            num_src_in_patch,
            s_output_offset,
            s_output_value,
            s_participant_bitmask,
            s_output_owned_bitmask,
            output_lp_hashtable,
            s_table);


        if (pl.first == patch_id) {

            constexpr uint32_t fixed_offset =
                ((op == Op::EV) ? 2 :
                                  ((op == Op::FV || op == Op::FE) ?
                                       3 :
                                       ((op == Op::EVDiamond) ? 4 : 0)));

            ComputeIteratorT iter(
                context,
                pl.second,
                reinterpret_cast<typename ComputeIteratorT::LocalT*>(
                    s_output_value),
                s_output_offset,
                fixed_offset,
                patch_id,
                s_output_owned_bitmask,
                output_lp_hashtable,
                s_table,
                context.m_patches_info[patch_id].patch_stash,
                int(op == Op::FE));

            compute_op(src_id, iter);
        }
        __syncthreads();
    }
}


}  // namespace rxmesh
