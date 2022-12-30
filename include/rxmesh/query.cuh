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
    Query(const ShmemAllocator&) = delete;
    Query& operator=(const Query&) = delete;

    __device__ __inline__ Query(const Context& context)
        : m_patch_info(context.m_patches_info[blockIdx.x]),
          m_status(Empty),
          m_num_src_in_patch(0),
          m_s_participant_bitmask(nullptr),
          m_s_output_owned_bitmask(nullptr),
          m_s_output_offset(nullptr),
          m_s_output_value(nullptr),
          m_s_table(nullptr)
    {

        assert(m_patch_info.patch_id < context.m_num_patches[0]);
    }


    template <Op op, typename computeT, typename activeSetT>
    __device__ __inline__ void dispatch(cooperative_groups::thread_block& block,
                                        ShmemAllocator& shrd_alloc,
                                        computeT        compute_op,
                                        activeSetT      compute_active_set,
                                        const bool      oriented)
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

        m_status |= SharedMemReserved;
        set_status<op>();

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
    }

   private:
    // Indicate which query information is currently stored in shared memory
    using QueryStatus = uint32_t;
    enum : uint32_t
    {
        Empty             = 0,
        SharedMemReserved = 1,
        VV                = 2,
        VE                = 4,
        VF                = 8,
        EV                = 16,
        EF                = 32,
        FV                = 64,
        FE                = 128,
        FF                = 512,
        EVDiamond         = 1024,
    };

    template <Op op>
    __device__ __inline__ void set_status()
    {
        if constexpr (op == Op::VV) {
            m_status = m_status | VV;
        }

        if constexpr (op == Op::VE) {
            m_status = m_status | VE;
        }

        if constexpr (op == Op::VF) {
            m_status = m_status | VF;
        }

        if constexpr (op == Op::EV) {
            m_status = m_status | EV;
        }

        if constexpr (op == Op::EF) {
            m_status = m_status | EF;
        }

        if constexpr (op == Op::FV) {
            m_status = m_status | FV;
        }

        if constexpr (op == Op::FE) {
            m_status = m_status | FE;
        }

        if constexpr (op == Op::FF) {
            m_status = m_status | FF;
        }

        if constexpr (op == Op::EVDiamond) {
            m_status = m_status | EVDiamond;
        }
    }

    const PatchInfo& m_patch_info;
    QueryStatus      m_status;
    uint32_t         m_num_src_in_patch;
    uint32_t*        m_s_participant_bitmask;
    uint32_t*        m_s_output_owned_bitmask;
    uint16_t*        m_s_output_offset;
    uint16_t*        m_s_output_value;
    LPHashTable      m_output_lp_hashtable;
    LPPair*          m_s_table;
};
}  // namespace rxmesh