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

namespace rxmesh {

template <typename T>
void spmat_linear_solve(rxmesh::SparseMatInfo<T> A_mat,
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

}  // namespace rxmesh