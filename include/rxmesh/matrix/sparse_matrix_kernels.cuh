#pragma once
#include "cusolverSp.h"
#include "cusparse.h"

#include "rxmesh/context.h"
#include "rxmesh/matrix/block_shape.h"
#include "rxmesh/query.h"

namespace rxmesh {

namespace detail {

// this is the function for the CSR calculation
template <Op op, uint32_t blockThreads, typename IndexT = int>
__global__ static void sparse_mat_prescan(const rxmesh::Context context,
                                          IndexT*               row_ptr,
                                          BlockShape            block_shape,
                                          bool                  add_diagonal)
{
    bool is_aos = true;

    using namespace rxmesh;

    using HandleT = typename InputHandle<op>::type;
    using IterT   = typename IteratorType<op>::type;

    if constexpr (op == Op::V || op == Op::E || op == Op::F) {
        // block-diagonal matrix
        for_each<op, blockThreads>(context, [&](const HandleT& h) {
            IndexT offset =
                context.prefix<HandleT>()[h.patch_id()] + h.local_id();
            offset *= block_shape.x;

            for (IndexT i = 0; i < block_shape.x; ++i) {
                row_ptr[offset + i] = block_shape.y;
            }
        });
    } else {
        auto init_lambda = [&](HandleT& v_id, const IterT& iter) {
            auto     ids      = v_id.unpack();
            uint32_t patch_id = ids.first;
            uint16_t local_id = ids.second;
            IndexT   size     = iter.size();

            if (add_diagonal) {
                size += 1;
            }

            size *= block_shape.y;
            IndexT offset = context.prefix<HandleT>()[patch_id] + local_id;

            if (is_aos) {
                offset *= block_shape.x;
                for (IndexT i = 0; i < block_shape.x; ++i) {
                    row_ptr[offset + i] = size;
                }
            } else {
                const uint32_t num_elements = context.get_num<HandleT>();
                for (IndexT i = 0; i < block_shape.x; ++i) {
                    row_ptr[num_elements * i + offset] = size;
                }
            }
        };

        auto                block = cooperative_groups::this_thread_block();
        Query<blockThreads> query(context);
        ShmemAllocator      shrd_alloc;
        query.dispatch<op>(block, shrd_alloc, init_lambda);
    }
}


template <Op op, uint32_t blockThreads, typename IndexT = int>
__global__ static void sparse_mat_col_fill(const rxmesh::Context context,
                                           IndexT*               row_ptr,
                                           IndexT*               col_idx,
                                           BlockShape            block_shape,
                                           bool                  add_diagonal)
{
    using namespace rxmesh;

    using HandleT = typename InputHandle<op>::type;
    using IterT   = typename IteratorType<op>::type;

    using IterHandle = typename IterT::Handle;

    if constexpr (op == Op::V || op == Op::E || op == Op::F) {
        // block-diagonal matrix
        for_each<op, blockThreads>(context, [&](const HandleT& h) {
            IndexT v_id =
                context.prefix<HandleT>()[h.patch_id()] + h.local_id();

            IndexT v_global = v_id * block_shape.x;

            for (IndexT i = 0; i < block_shape.x; ++i) {
                IndexT v_base_offset = row_ptr[v_global + i];
                for (IndexT j = 0; j < block_shape.y; ++j) {
                    col_idx[v_base_offset + j] = v_id * block_shape.y + j;
                }
            }
        });

    } else {

        auto col_fillin = [&](HandleT& h, const IterT& iter) {
            auto     ids      = h.unpack();
            uint32_t patch_id = ids.first;
            uint16_t local_id = ids.second;

            IndexT v_id     = context.prefix<HandleT>()[patch_id] + local_id;
            IndexT v_global = v_id * block_shape.x;

            // "block" diagonal entries (which is stored as the first entry in
            // the col_idx for each row) with block_shape.x =1,  there is only
            // one entry per diagonal. But with higher block_shape, it becomes
            // (block_shape.x x block_shape.y) block

            int diagonal_offset = 0;

            if (add_diagonal) {
                diagonal_offset += block_shape.x;

                for (IndexT i = 0; i < block_shape.x; ++i) {
                    IndexT v_base_offset = row_ptr[v_global + i];
                    for (IndexT j = 0; j < block_shape.y; ++j) {
                        col_idx[v_base_offset + j] = v_id * block_shape.y + j;
                    }
                }
            }

            for (uint32_t q = 0; q < iter.size(); ++q) {
                auto     q_ids      = iter[q].unpack();
                uint32_t q_patch_id = q_ids.first;
                uint16_t q_local_id = q_ids.second;

                IndexT q_global =
                    context.prefix<IterHandle>()[q_patch_id] + q_local_id;
                q_global *= block_shape.y;

                for (IndexT i = 0; i < block_shape.x; ++i) {
                    IndexT q_base_offset =
                        row_ptr[v_global + i] + q * block_shape.y;
                    for (IndexT j = 0; j < block_shape.y; ++j) {
                        col_idx[q_base_offset + j + diagonal_offset] =
                            q_global + j;
                        //                          ^^ to account for the
                        //                          diagonal entries
                    }
                }
            }
        };

        auto                block = cooperative_groups::this_thread_block();
        Query<blockThreads> query(context);
        ShmemAllocator      shrd_alloc;
        query.dispatch<op>(block, shrd_alloc, col_fillin);
    }
}

}  // namespace detail

}  // namespace rxmesh