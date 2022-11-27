#pragma once

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

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
__global__ static void mcf_A_B_setup(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<T> coords,  // for non-uniform
    rxmesh::SparseMatInfo<T>   A_mat,
    rxmesh::DenseMatInfo<T>    B_mat,
    const bool                 use_uniform_laplace,  // for non-uniform
    const T                    time_step)
{
    using namespace rxmesh;
    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        T sum_e_weight(0);

        T v_weight = iter.size();

        // reference value calculation
        auto     r_ids      = v_id.unpack();
        uint32_t r_patch_id = r_ids.first;
        uint16_t r_local_id = r_ids.second;

        uint32_t row_index = A_mat.m_d_patch_ptr_v[r_patch_id] + r_local_id;

        B_mat(row_index, 0) = coords(v_id, 0) * v_weight;
        B_mat(row_index, 1) = coords(v_id, 1) * v_weight;
        B_mat(row_index, 2) = coords(v_id, 2) * v_weight;

        Vector<3, float> vi_coord(
            coords(v_id, 0), coords(v_id, 1), coords(v_id, 2));
        for (uint32_t v = 0; v < iter.size(); ++v) {
            T e_weight           = 1;
            A_mat(v_id, iter[v]) = -time_step * e_weight;

            sum_e_weight += e_weight;
        }

        A_mat(v_id, v_id) = v_weight + time_step * sum_e_weight;
    };

    query_block_dispatcher<Op::VV, blockThreads>(context, init_lambda);
}

template <typename T>
void sparse_coord_solve(rxmesh::SparseMatInfo<T> A_mat,
                        rxmesh::DenseMatInfo<T>  B_mat,
                        rxmesh::DenseMatInfo<T>  X_mat)
{   
    if (typeid(T) == typeid(float)) {
        printf("float!!!\n");
    }

    cusolverSpHandle_t handle         = NULL;
    cusparseHandle_t   cusparseHandle = NULL; /* used in residual evaluation */
    cudaStream_t       stream         = NULL;
    cusparseMatDescr_t descrA         = NULL;

    cusolverSpCreate(&handle);
    cusparseCreate(&cusparseHandle);

    cudaStreamCreate(&stream);
    cusolverSpSetStream(handle, stream);
    cusparseSetStream(cusparseHandle, stream);

    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
}

void sparse_linear_solve(const int          solver,
                         const int          reorder,
                         cusolverSpHandle_t handle,
                         cusparseHandle_t   cusparseHandle,
                         cusparseMatDescr_t descrA,
                         int                rowsA,
                         int                colsA,
                         int                nnzA,
                         int*               d_csrRowPtrA,
                         int*               d_csrColIndA,
                         double*            d_csrValA,
                         int*               d_Q,
                         double*            d_Qb,
                         double*            d_x,
                         double*            d_z)
{
    double tol         = 1.e-12;
    int    singularity = 0; /* -1 if A is invertible under tol. */

    printf("step 7: solve A*x = b on GPU\n");

    /* solve B*z = Q*b */
    if (solver == 0) {
        cusolverSpDcsrlsvchol(handle,
                              rowsA,
                              nnzA,
                              descrA,
                              d_csrValA,
                              d_csrRowPtrA,
                              d_csrColIndA,
                              d_Qb,
                              tol,
                              reorder,
                              d_z,
                              &singularity);
    } else if (solver == 1) {
        cusolverSpDcsrlsvqr(handle,
                            rowsA,
                            nnzA,
                            descrA,
                            d_csrValA,
                            d_csrRowPtrA,
                            d_csrColIndA,
                            d_Qb,
                            tol,
                            reorder,
                            d_z,
                            &singularity);
    } else {
        fprintf(stderr, "Error: only chol(0) and qr(1) is supported\n");
        exit(1);
    }
    cudaDeviceSynchronize();

    if (0 <= singularity) {
        printf("WARNING: the matrix is singular at row %d under tol (%E)\n",
               singularity,
               tol);
    }
}
