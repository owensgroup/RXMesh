#pragma once
#include "rxmesh/kernels/rxmesh_query_dispatcher.cuh"
namespace rxmesh {
namespace detail {

/**
 * @brief represents the minimal user function for op query.
 * This function is only used in order to calculate the static shared memory and
 * registers used given a certain query operation and block size
 */
template <Op       op,
          uint32_t blockThreads,
          typename InputHandleT,
          typename OutputHandleT>
__global__ static void query_prototype(const Context context,
                                       const bool    oriented = false)
{
    static_assert(op != Op::EE, "Op::EE is not supported!");

    auto user_lambda = [&](InputHandleT& id, Iterator<OutputHandleT>& iter) {
        printf("\n iter.size() = %u", iter.size());
        for (uint32_t i = 0; i < iter.size(); ++i) {
            printf("\n iter[%u] = %llu", i, iter[i].unique_id());
        }
    };
    query_block_dispatcher<op, blockThreads>(context, user_lambda, oriented);
}

/**
 * @brief  represents the minimal user function for higher
 * queries. Here we assume that all query of similar type. This function is
 * only used in order to calculate the static shared memory and registers used
 */
template <Op       op,
          uint32_t blockThreads,
          typename InputHandleT,
          typename OutputHandleT>
__global__ static void higher_query_prototype(const Context context,
                                              const bool    oriented = false)
{
    static_assert(op != Op::EE, "Op::EE is not supported!");

    InputHandleT thread_element;
    auto first_ring = [&](InputHandleT id, Iterator<OutputHandleT>& iter) {
        thread_element = id;
        printf("\n iter.size() = %u", iter.size());
        for (uint32_t i = 0; i < iter.size(); ++i) {
            printf("\n iter[%u] = %llu", i, iter[i].unique_id());
        }
    };

    query_block_dispatcher<op, blockThreads>(context, first_ring, oriented);

    auto n_ring = [&](InputHandleT id, Iterator<OutputHandleT>& iter) {
        printf("\n iter.size() = %u", iter.size());
        for (uint32_t i = 0; i < iter.size(); ++i) {
            printf("\n iter[%u] = %llu", i, iter[i].unique_id());
        }
    };

    higher_query_block_dispatcher<op, blockThreads>(
        context, thread_element, n_ring, oriented);
}

}  // namespace detail
}  // namespace rxmesh