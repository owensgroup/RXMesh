#pragma once
#include "rxmesh/kernels/rxmesh_query_dispatcher.cuh"
namespace RXMESH {
namespace detail {

/**
 * query_prototype() represents the minimal user function for op query
 * this function is only used in order to calculate the static shared memory and
 * registers used
 */
template <Op op, uint32_t blockThreads>
__launch_bounds__(blockThreads) __global__
    static void query_prototype(const RXMeshContext context,
                                const bool          oriented = false)
{
    static_assert(op != Op::EE, "Op::EE is not supported!");

    auto user_lambda = [&](uint32_t id, RXMeshIterator& iter) {
        printf("\n iter.size() = %u", iter.size());
        for (uint32_t i = 0; i < iter.size(); ++i) {
            printf("\n iter[%u] = %u", i, iter[i]);
        }
    };

    query_block_dispatcher<op, blockThreads>(context, user_lambda, oriented);
}

}  // namespace detail
}  // namespace RXMESH