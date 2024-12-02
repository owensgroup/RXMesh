#pragma once
#include "cusolverSp.h"
#include "cusparse.h"

#include "rxmesh/context.h"
#include "rxmesh/query.cuh"

namespace rxmesh {

namespace detail {

// this is the function for the CSR calculation
template <uint32_t blockThreads, typename IndexT = int>
__global__ static void sparse_mat_prescan(const rxmesh::Context context,
                                          IndexT*               row_ptr,
                                          IndexT                replicate)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        auto     ids      = v_id.unpack();
        uint32_t patch_id = ids.first;
        uint16_t local_id = ids.second;
        IndexT   size     = iter.size() + 1;
        size *= replicate;
        IndexT offset = context.vertex_prefix()[patch_id] + local_id;
        offset *= replicate;

        for (IndexT i = 0; i < replicate; ++i) {
            row_ptr[offset + i] = size;
        }
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, init_lambda);
}

template <uint32_t blockThreads, typename IndexT = int>
__global__ static void sparse_mat_col_fill(const rxmesh::Context context,
                                           IndexT*               row_ptr,
                                           IndexT*               col_idx,
                                           IndexT                replicate)
{
    using namespace rxmesh;

    auto col_fillin = [&](VertexHandle& v_id, const VertexIterator& iter) {
        auto     ids      = v_id.unpack();
        uint32_t patch_id = ids.first;
        uint16_t local_id = ids.second;

        IndexT v_global = context.vertex_prefix()[patch_id] + local_id;
        v_global *= replicate;

        // "block" diagonal entries (which is stored as the first entry in the
        // col_idx for each row)
        // with replicate =1,  there is only one entry per diagonal. But with
        // higher replicate, it becomes (replicate x replicate) block

        for (IndexT i = 0; i < replicate; ++i) {
            IndexT v_base_offset = row_ptr[v_global + i];
            for (IndexT j = 0; j < replicate; ++j) {
                col_idx[v_base_offset + j] = v_global + j;
            }
        }

        for (uint32_t q = 0; q < iter.size(); ++q) {
            auto     q_ids      = iter[q].unpack();
            uint32_t q_patch_id = q_ids.first;
            uint16_t q_local_id = q_ids.second;

            IndexT q_global = context.vertex_prefix()[q_patch_id] + q_local_id;
            q_global *= replicate;

            for (IndexT i = 0; i < replicate; ++i) {
                IndexT q_base_offset = row_ptr[v_global + i] + q * replicate;
                for (IndexT j = 0; j < replicate; ++j) {
                    col_idx[q_base_offset + j + replicate] = q_global + j;
                    //                          ^^ to account for the diagonal
                    //                          entries
                }
            }
        }
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, col_fillin);
}

}  // namespace detail

}  // namespace rxmesh