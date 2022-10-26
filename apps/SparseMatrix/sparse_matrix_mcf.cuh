#pragma once

#include <cuda_profiler_api.h>
#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"
#include "sparse_matrix.cuh"

#include "cusolverSp.h"
#include "cusparse.h"

template <uint32_t blockThreads>
__global__ static void sparse_mat_weight_cal(
    const rxmesh::Context          context,
    rxmesh::VertexAttribute<float> coords,
    rxmesh::SparseMatInfo<float>     sparse_mat)
{
    using namespace rxmesh;
    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        // reference value calculation
        auto     r_ids      = v_id.unpack();
        uint32_t r_patch_id = r_ids.first;
        uint16_t r_local_id = r_ids.second;

        uint32_t row_index =
            sparse_mat.m_d_patch_ptr_v[r_patch_id] + r_local_id;

        float len_sum = 0;

        float v_weight = iter.size();

        Vector<3, float> v_coord(
            coords(v_id, 0), coords(v_id, 1), coords(v_id, 2));
        for (uint32_t v = 0; v < iter.size(); ++v) {
            Vector<3, float> vi_coord(
                coords(iter[v], 0), coords(iter[v], 1), coords(iter[v], 2));

            len_sum += dist(v_coord, vi_coord);
        }

        sparse_mat(v_id, v_id) = len_sum / v_weight;
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, init_lambda);
}
