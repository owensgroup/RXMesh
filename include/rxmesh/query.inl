namespace rxmesh {

template <uint32_t blockThreads>
template <Op op, typename computeT>
__device__ __inline__ void Query<blockThreads>::dispatch(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc,
    computeT                          compute_op,
    const bool                        oriented)
{
    // Extract the type of the first input parameters of the compute lambda
    using ComputeTraits  = detail::FunctionTraits<computeT>;
    using ComputeHandleT = typename ComputeTraits::template arg<0>::type;

    dispatch<op>(
        block,
        shrd_alloc,
        compute_op,
        [](ComputeHandleT) { return true; },
        oriented);
}

template <uint32_t blockThreads>
template <Op op, typename computeT, typename activeSetT>
__device__ __inline__ void Query<blockThreads>::dispatch(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc,
    computeT                          compute_op,
    activeSetT                        compute_active_set,
    const bool                        oriented,
    const bool                        allow_not_owned)
{
    if (get_patch_id() == INVALID32) {
        return;
    }
    using ComputeTraits    = detail::FunctionTraits<computeT>;
    using ComputeHandleT   = typename ComputeTraits::template arg<0>::type;
    using ComputeIteratorT = typename ComputeTraits::template arg<1>::type;
    using LocalT           = typename ComputeIteratorT::LocalT;

    using ActiveSetTraits  = detail::FunctionTraits<activeSetT>;
    using ActiveSetHandleT = typename ActiveSetTraits::template arg<0>::type;
    static_assert(std::is_same_v<ActiveSetHandleT, ComputeHandleT>,
                  "First argument of compute_op lambda function should "
                  "match the first argument of active_set lambda function ");

    prologue<op>(
        block, shrd_alloc, compute_active_set, oriented, allow_not_owned);

    run_compute(block, compute_op);

    epilogue(block, shrd_alloc);
}

template <uint32_t blockThreads>
template <Op op>
__device__ __inline__ void Query<blockThreads>::prologue(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc,
    const bool                        oriented,
    const bool                        allow_not_owned)
{
    prologue<op>(
        block,
        shrd_alloc,
        [](VertexHandle) { return true; },
        oriented,
        allow_not_owned);
}

template <uint32_t blockThreads>
template <Op op, typename activeSetT>
__device__ __inline__ void Query<blockThreads>::prologue(
    cooperative_groups::thread_block& block,
    ShmemAllocator&                   shrd_alloc,
    activeSetT                        compute_active_set,
    const bool                        oriented,
    const bool                        allow_not_owned)
{
    if (get_patch_id() == INVALID32) {
        return;
    }

    m_op           = op;
    m_shmem_before = shrd_alloc.get_allocated_size_bytes();

    detail::query_block_dispatcher<op, blockThreads>(block,
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

template <uint32_t blockThreads>
template <typename computeT>
__device__ __inline__ void Query<blockThreads>::run_compute(
    cooperative_groups::thread_block& block,
    computeT                          compute_op)
{
    if (get_patch_id() == INVALID32) {
        return;
    }

    using ComputeTraits    = detail::FunctionTraits<computeT>;
    using ComputeHandleT   = typename ComputeTraits::template arg<0>::type;
    using ComputeIteratorT = typename ComputeTraits::template arg<1>::type;

    for (uint16_t local_id = threadIdx.x; local_id < m_num_src_in_patch;
         local_id += blockThreads) {

        if (detail::is_set_bit(local_id, m_s_participant_bitmask)) {

            assert(m_s_output_value);

            ComputeHandleT   handle(m_patch_info.patch_id, local_id);
            ComputeIteratorT iter = get_iterator<ComputeIteratorT>(local_id);
            compute_op(handle, iter);
        }
    }
}

template <uint32_t blockThreads>
template <typename IteratorT>
__device__ __inline__ IteratorT Query<blockThreads>::get_iterator(
    uint16_t local_id) const
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

}  // namespace rxmesh
