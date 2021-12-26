#pragma once
#include <assert.h>
#include <stdint.h>
#include <cub/block/block_discontinuity.cuh>

#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/iterator.cuh"
#include "rxmesh/kernels/collective.cuh"
#include "rxmesh/kernels/debug.cuh"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/rxmesh_queries.cuh"
#include "rxmesh/types.h"
#include "rxmesh/util/meta.h"

namespace rxmesh {

namespace detail {

/**
 * query_block_dispatcher()
 */
template <Op op, uint32_t blockThreads, typename activeSetT>
__device__ __inline__ void query_block_dispatcher(const PatchInfo& patch_info,
                                                  activeSetT compute_active_set,
                                                  const bool oriented,
                                                  uint32_t&  num_src_in_patch,
                                                  uint16_t*& s_output_offset,
                                                  uint16_t*& s_output_value)
{
    static_assert(op != Op::EE, "Op::EE is not supported!");


    ELEMENT src_element, output_element;
    io_elements(op, src_element, output_element);

    extern __shared__ uint16_t shrd_mem[];

    s_output_offset = shrd_mem;


    LocalVertexT* s_ev = reinterpret_cast<LocalVertexT*>(shrd_mem);
    LocalEdgeT*   s_fe = reinterpret_cast<LocalEdgeT*>(shrd_mem);

    constexpr bool load_fe  = (op == Op::VF || op == Op::EE || op == Op::EF ||
                              op == Op::FV || op == Op::FE || op == Op::FF);
    constexpr bool loead_ev = (op == Op::VV || op == Op::VE || op == Op::VF ||
                               op == Op::EV || op == Op::FV);
    static_assert(loead_ev || load_fe,
                  "At least faces or edges needs to be loaded");

    constexpr bool is_fixed_offset =
        (op == Op::EV || op == Op::FV || op == Op::FE);


    // Check if any of the mesh elements are in the active set
    // input mapping does not need to be stored in shared memory since it will
    // be read coalesced, we can rely on L1 cache here
    num_src_in_patch = 0;
    switch (src_element) {
        case ELEMENT::VERTEX: {
            num_src_in_patch = patch_info.num_owned_vertices;
            break;
        }
        case ELEMENT::EDGE: {
            num_src_in_patch = patch_info.num_owned_edges;
            break;
        }
        case ELEMENT::FACE: {
            num_src_in_patch = patch_info.num_owned_faces;
            break;
        }
    }


    bool     is_active = false;
    uint16_t local_id  = threadIdx.x;
    while (local_id < num_src_in_patch) {
        is_active =
            local_id || compute_active_set({patch_info.patch_id, local_id});
        local_id += blockThreads;
    }

    if (__syncthreads_or(is_active) == 0) {
        return;
    }

    // 2) Load the patch info
    load_mesh<blockThreads>(patch_info, loead_ev, load_fe, s_ev, s_fe);
    __syncthreads();
    // 3)Perform the query operation
    if (oriented) {
        assert(op == Op::VV);
        if constexpr (op == Op::VV) {
            v_v_oreinted<blockThreads>(patch_info,
                                       s_output_offset,
                                       s_output_value,
                                       reinterpret_cast<uint16_t*>(s_ev));
        }
    } else {
        query<blockThreads, op>(s_output_offset,
                                s_output_value,
                                reinterpret_cast<uint16_t*>(s_ev),
                                reinterpret_cast<uint16_t*>(s_fe),
                                patch_info.num_vertices,
                                patch_info.num_edges,
                                patch_info.num_faces);
    }

    __syncthreads();
}

}  // namespace detail

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


    assert(patch_id < context.get_num_patches());

    uint32_t  num_src_in_patch = 0;
    uint16_t* s_output_offset(nullptr);
    uint16_t* s_output_value(nullptr);

    detail::template query_block_dispatcher<op, blockThreads>(
        context.get_patches_info()[patch_id],
        compute_active_set,
        oriented,
        num_src_in_patch,
        s_output_offset,
        s_output_value);

    assert(s_output_offset);
    assert(s_output_value);

    // Call compute on the output in shared memory by looping over all
    // source elements in this patch.

    uint16_t local_id = threadIdx.x;
    while (local_id < num_src_in_patch) {

        if (compute_active_set({patch_id, local_id})) {
            constexpr uint32_t fixed_offset =
                ((op == Op::EV)                 ? 2 :
                 (op == Op::FV || op == Op::FE) ? 3 :
                                                  0);


            ComputeHandleT   handle(patch_id, local_id);
            ComputeIteratorT iter(local_id,
                                  reinterpret_cast<LocalT*>(s_output_value),
                                  s_output_offset,
                                  fixed_offset,
                                  patch_id,
                                  int(op == Op::FE));

            compute_op(handle, iter);
        }

        local_id += blockThreads;
    }
}

/**
 * query_block_dispatcher()
 */
template <Op op, uint32_t blockThreads, typename computeT, typename activeSetT>
__device__ __inline__ void query_block_dispatcher(const Context& context,
                                                  computeT       compute_op,
                                                  activeSetT compute_active_set,
                                                  const bool oriented = false)
{
    if (blockIdx.x >= context.get_num_patches()) {
        return;
    }

    query_block_dispatcher<op, blockThreads>(
        context, blockIdx.x, compute_op, compute_active_set, oriented);
}


/**
 * query_block_dispatcher()
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
 * query_block_dispatcher() for higher queries
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
    std::pair<uint32_t, uint16_t> pl_not_owning =
        detail::unpack(src_id.unique_id());
    // which could be on a ribbon and so performing query on that patch is
    // meaningless so we grab the patch that owns this source mesh elements and
    // find it local id in there as well
    std::pair<uint32_t, uint16_t> pl_owning =
        context.get_patches_info()[pl_not_owning.first].get_patch_and_local_id(
            src_id);

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
    thread_data[0]       = pl_owning.first;
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


    for (uint32_t p = 0; p < s_num_patches; ++p) {

        uint32_t patch_id = s_block_patches[p];

        assert(patch_id < context.get_num_patches());

        uint32_t  num_src_in_patch = 0;
        uint16_t *s_output_offset(nullptr), *s_output_value(nullptr);

        detail::template query_block_dispatcher<op, blockThreads>(
            context.get_patches_info()[patch_id],
            compute_active_set,
            oriented,
            num_src_in_patch,
            s_output_offset,
            s_output_value);


        if (pl_owning.first == patch_id) {

            constexpr uint32_t fixed_offset =
                ((op == Op::EV)                 ? 2 :
                 (op == Op::FV || op == Op::FE) ? 3 :
                                                  0);

            ComputeIteratorT iter(
                pl_owning.second,
                reinterpret_cast<typename ComputeIteratorT::LocalT*>(
                    s_output_value),
                s_output_offset,
                fixed_offset,
                patch_id,
                int(op == Op::FE));

            compute_op(src_id, iter);
        }
        __syncthreads();
    }
}


}  // namespace rxmesh
