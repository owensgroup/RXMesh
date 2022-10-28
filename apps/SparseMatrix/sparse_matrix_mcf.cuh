#pragma once

#include <cuda_profiler_api.h>
#include "dense_matrix.cuh"
#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"
#include "sparse_matrix.cuh"

#include "cusolverSp.h"
#include "cusparse.h"

template <typename T, uint32_t blockThreads>
__global__ static void A_mat_A_B_setup(
    const rxmesh::Context          context,
    rxmesh::VertexAttribute<float> coords,  // for non-uniform
    rxmesh::SparseMatInfo<float>   A_mat,
    rxmesh::DenseMatInfo<float>    B_mat,
    const bool                     use_uniform_laplace,  // for non-uniform
    const T                        time_step)
{
    using namespace rxmesh;
    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        T sum_e_weight(0);

        T v_weight = iter.size();

        // reference value calculation
        auto     r_ids      = v_id.unpack();
        uint32_t r_patch_id = r_ids.first;
        uint16_t r_local_id = r_ids.second;

        uint32_t row_index =
            A_mat.m_d_patch_ptr_v[r_patch_id] + r_local_id;

        B_mat(row_index, 0) = coords(v_id, 0) * v_weight;
        B_mat(row_index, 1) = coords(v_id, 1) * v_weight;
        B_mat(row_index, 2) = coords(v_id, 2) * v_weight;

        Vector<3, float> vi_coord(
            coords(v_id, 0), coords(v_id, 1), coords(v_id, 2));
        for (uint32_t v = 0; v < iter.size(); ++v) {
            T e_weight          = 1;
            A_mat(v_id, v) = -time_step * e_weight;

            sum_e_weight += e_weight;
        }

        A_mat(v_id, v_id) = v_weight + time_step * sum_e_weight;
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, init_lambda);
}