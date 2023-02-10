#pragma once
#include "cusolverSp.h"
#include "cusparse.h"
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/types.h"

#include "rxmesh/matrix/dense_matrix.cuh"

namespace rxmesh {

/**
 * @brief The enum class for choosing different solver types
 */
enum class Solver
{
    CHOL = 0,
    LU   = 1,
    QR   = 2
};

/**
 * @brief The enum class for choosing different reorder types
 * NONE for No Reordering Applied, SYMRCM for Symmetric Reverse Cuthill-McKee
 * permutation, SYMAMD for Symmetric Approximate Minimum Degree Algorithm based
 * on Quotient Graph, NSTDIS for Nested Dissection
 */
enum class Reorder
{
    NONE   = 0,
    SYMRCM = 1,
    SYMAMD = 2,
    NSTDIS = 3
};

static int reorder_to_int(const Reorder& reorder)
{
    switch (reorder) {
        case Reorder::NONE:
            return 0;
        case Reorder::SYMRCM:
            return 1;
        case Reorder::SYMAMD:
            return 2;
        case Reorder::NSTDIS:
            return 3;
        default: {
            RXMESH_ERROR("reorder_to_int() unknown input reorder");
            return 0;
        }
    }
}

namespace detail {
// this is the function for the CSR calculation
template <uint32_t blockThreads, typename IndexT = int>
__global__ static void sparse_mat_prescan(const rxmesh::Context context,
                                          IndexT*               row_ptr)
{
    using namespace rxmesh;

    auto init_lambda = [&](VertexHandle& v_id, const VertexIterator& iter) {
        auto     ids                                          = v_id.unpack();
        uint32_t patch_id                                     = ids.first;
        uint16_t local_id                                     = ids.second;
        row_ptr[context.m_vertex_prefix[patch_id] + local_id] = iter.size() + 1;
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, init_lambda);
}

template <uint32_t blockThreads, typename IndexT = int>
__global__ static void sparse_mat_col_fill(const rxmesh::Context context,
                                           IndexT*               row_ptr,
                                           IndexT*               col_idx)
{
    using namespace rxmesh;

    auto col_fillin = [&](VertexHandle& v_id, const VertexIterator& iter) {
        auto     ids      = v_id.unpack();
        uint32_t patch_id = ids.first;
        uint16_t local_id = ids.second;
        col_idx[row_ptr[context.m_vertex_prefix[patch_id] + local_id]] =
            context.m_vertex_prefix[patch_id] + local_id;
        for (uint32_t v = 0; v < iter.size(); ++v) {
            auto     s_ids      = iter[v].unpack();
            uint32_t s_patch_id = s_ids.first;
            uint16_t s_local_id = s_ids.second;
            col_idx[row_ptr[context.m_vertex_prefix[patch_id] + local_id] + v +
                    1] = context.m_vertex_prefix[s_patch_id] + s_local_id;
        }
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, col_fillin);
}

}  // namespace detail


// TODO: add compatibility for EE, FF, VE......
// TODO: purge operation?
template <typename T, typename IndexT = int>
struct SparseMatrix
{
    SparseMatrix(RXMeshStatic& rx)
        : m_d_row_ptr(nullptr),
          m_d_col_idx(nullptr),
          m_d_val(nullptr),
          m_row_size(0),
          m_col_size(0),
          m_nnz(0),
          m_context(rx.get_context()),
          m_cusparse_handle(NULL),
          m_descr(NULL),
          m_spdescr(NULL),
          m_spmm_buffer_size(0),
          m_spmv_buffer_size(0)
    {
        using namespace rxmesh;
        constexpr uint32_t blockThreads = 256;

        IndexT num_patches  = rx.get_num_patches();
        IndexT num_vertices = rx.get_num_vertices();
        IndexT num_edges    = rx.get_num_edges();

        m_row_size = num_vertices;
        m_col_size = num_vertices;

        // row pointer allocation and init with prefix sum for CSR
        CUDA_ERROR(cudaMalloc((void**)&m_d_row_ptr,
                              (num_vertices + 1) * sizeof(IndexT)));

        CUDA_ERROR(
            cudaMemset(m_d_row_ptr, 0, (num_vertices + 1) * sizeof(IndexT)));

        LaunchBox<blockThreads> launch_box;
        rx.prepare_launch_box({Op::VV},
                              launch_box,
                              (void*)detail::sparse_mat_prescan<blockThreads>);

        detail::sparse_mat_prescan<blockThreads>
            <<<launch_box.blocks,
               launch_box.num_threads,
               launch_box.smem_bytes_dyn>>>(m_context, m_d_row_ptr);

        // prefix sum using CUB.
        void*  d_cub_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                      temp_storage_bytes,
                                      m_d_row_ptr,
                                      m_d_row_ptr,
                                      num_vertices + 1);
        CUDA_ERROR(cudaMalloc((void**)&d_cub_temp_storage, temp_storage_bytes));

        cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                      temp_storage_bytes,
                                      m_d_row_ptr,
                                      m_d_row_ptr,
                                      num_vertices + 1);

        CUDA_ERROR(cudaFree(d_cub_temp_storage));

        // get nnz
        CUDA_ERROR(cudaMemcpy(&m_nnz,
                              (m_d_row_ptr + num_vertices),
                              sizeof(IndexT),
                              cudaMemcpyDeviceToHost));

        // column index allocation and init
        CUDA_ERROR(cudaMalloc((void**)&m_d_col_idx, m_nnz * sizeof(IndexT)));
        rx.prepare_launch_box({Op::VV},
                              launch_box,
                              (void*)detail::sparse_mat_col_fill<blockThreads>);

        detail::sparse_mat_col_fill<blockThreads>
            <<<launch_box.blocks,
               launch_box.num_threads,
               launch_box.smem_bytes_dyn>>>(
                m_context, m_d_row_ptr, m_d_col_idx);

        // val pointer allocation, actual value init should be in another
        // function
        CUDA_ERROR(cudaMalloc((void**)&m_d_val, m_nnz * sizeof(IndexT)));

        CUSPARSE_ERROR(cusparseCreateMatDescr(&m_descr));
        CUSPARSE_ERROR(
            cusparseSetMatType(m_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        CUSPARSE_ERROR(
            cusparseSetMatIndexBase(m_descr, CUSPARSE_INDEX_BASE_ZERO));

        CUSPARSE_ERROR(cusparseCreateCsr(&m_spdescr,
                                         m_row_size,
                                         m_col_size,
                                         m_nnz,
                                         m_d_row_ptr,
                                         m_d_col_idx,
                                         m_d_val,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         CUDA_R_32F));

        CUSPARSE_ERROR(cusparseCreate(&m_cusparse_handle));
        CUSOLVER_ERROR(cusolverSpCreate(&m_cusolver_sphandle));
    }

    void set_ones()
    {
        std::vector<T> init_tmp_arr(m_nnz, 1);
        CUDA_ERROR(cudaMemcpy(m_d_val,
                              init_tmp_arr.data(),
                              m_nnz * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    __device__ IndexT get_val_idx(const VertexHandle& row_v,
                                  const VertexHandle& col_v)
    {
        auto     r_ids      = row_v.unpack();
        uint32_t r_patch_id = r_ids.first;
        uint16_t r_local_id = r_ids.second;

        auto     c_ids      = col_v.unpack();
        uint32_t c_patch_id = c_ids.first;
        uint16_t c_local_id = c_ids.second;

        uint32_t col_index = m_context.m_vertex_prefix[c_patch_id] + c_local_id;
        uint32_t row_index = m_context.m_vertex_prefix[r_patch_id] + r_local_id;

        const IndexT start = m_d_row_ptr[row_index];
        const IndexT end   = m_d_row_ptr[row_index + 1];

        for (IndexT i = start; i < end; ++i) {
            if (m_d_col_idx[i] == col_index) {
                return i;
            }
        }
        assert(1 != 1);
    }

    __device__ T& operator()(const VertexHandle& row_v,
                             const VertexHandle& col_v)
    {
        return m_d_val[get_val_idx(row_v, col_v)];
    }

    __device__ T& operator()(const VertexHandle& row_v,
                             const VertexHandle& col_v) const
    {
        return m_d_val[get_val_idx(row_v, col_v)];
    }

    __device__ T& direct_access(IndexT x, IndexT y)
    {
        const IndexT start = m_d_row_ptr[x];
        const IndexT end   = m_d_row_ptr[x + 1];

        for (IndexT i = start; i < end; ++i) {
            if (m_d_col_idx[i] == y) {
                return m_d_val[i];
            }
        }
        assert(1 != 1);
    }

    void free()
    {
        CUDA_ERROR(cudaFree(m_d_row_ptr));
        CUDA_ERROR(cudaFree(m_d_col_idx));
        CUDA_ERROR(cudaFree(m_d_val));
        CUSPARSE_ERROR(cusparseDestroy(m_cusparse_handle));
        CUSPARSE_ERROR(cusparseDestroyMatDescr(m_descr));
        CUSOLVER_ERROR(cusolverSpDestroy(m_cusolver_sphandle));
    }


    /**
     * @brief wrap up the cusparse api for sparse matrix dense matrix
     * multiplication buffer size calculation.
     */
    void denmat_mul_buffer_size(rxmesh::DenseMatrix<T> B_mat,
                                rxmesh::DenseMatrix<T> C_mat,
                                cudaStream_t           stream = 0)
    {
        float alpha = 1.0f;
        float beta  = 0.0f;

        cusparseSpMatDescr_t matA    = m_spdescr;
        cusparseDnMatDescr_t matB    = B_mat.m_dendescr;
        cusparseDnMatDescr_t matC    = C_mat.m_dendescr;
        void*                dBuffer = NULL;

        cusparseSetStream(m_cusparse_handle, stream);

        // allocate an external buffer if needed
        CUSPARSE_ERROR(cusparseSpMM_bufferSize(m_cusparse_handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               matA,
                                               matB,
                                               &beta,
                                               matC,
                                               CUDA_R_32F,
                                               CUSPARSE_SPMM_ALG_DEFAULT,
                                               &m_spmm_buffer_size));
    }

    /**
     * @brief wrap up the cusparse api for sparse matrix dense matrix
     * multiplication.
     */
    void denmat_mul(rxmesh::DenseMatrix<T> B_mat,
                    rxmesh::DenseMatrix<T> C_mat,
                    cudaStream_t           stream = 0)
    {
        float alpha = 1.0f;
        float beta  = 0.0f;

        // A_mat.create_cusparse_handle();
        cusparseSpMatDescr_t matA    = m_spdescr;
        cusparseDnMatDescr_t matB    = B_mat.m_dendescr;
        cusparseDnMatDescr_t matC    = C_mat.m_dendescr;
        void*                dBuffer = NULL;

        cusparseSetStream(m_cusparse_handle, stream);

        // allocate an external buffer if needed
        if (m_spmm_buffer_size == 0) {
            RXMESH_WARN(
                "Sparse matrix - Dense matrix multiplication buffer size not "
                "initialized.",
                "Calculate it now.");
            denmat_mul_buffer_size(B_mat, C_mat, stream);
        }
        CUDA_ERROR(cudaMalloc(&dBuffer, m_spmm_buffer_size));

        // execute SpMM
        CUSPARSE_ERROR(cusparseSpMM(m_cusparse_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    matA,
                                    matB,
                                    &beta,
                                    matC,
                                    CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT,
                                    dBuffer));

        CUDA_ERROR(cudaFree(dBuffer));
    }

    void arr_mul_buffer_size(T* in_arr, T* rt_arr, cudaStream_t stream = 0)
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;

        cusparseDnVecDescr_t vecx = NULL;
        cusparseDnVecDescr_t vecy = NULL;

        CUSPARSE_ERROR(
            cusparseCreateDnVec(&vecx, m_col_size, in_arr, CUDA_R_32F));
        CUSPARSE_ERROR(
            cusparseCreateDnVec(&vecy, m_row_size, rt_arr, CUDA_R_32F));

        CUSPARSE_ERROR(cusparseSpMV_bufferSize(m_cusparse_handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               m_spdescr,
                                               vecx,
                                               &beta,
                                               vecy,
                                               CUDA_R_32F,
                                               CUSPARSE_SPMV_ALG_DEFAULT,
                                               &m_spmv_buffer_size));
    }

    /**
     * @brief wrap up the cusparse api for sparse matrix array
     * multiplication.
     */
    void arr_mul(T* in_arr, T* rt_arr, cudaStream_t stream = 0)
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;

        void*                buffer = NULL;
        cusparseDnVecDescr_t vecx   = NULL;
        cusparseDnVecDescr_t vecy   = NULL;

        CUSPARSE_ERROR(
            cusparseCreateDnVec(&vecx, m_col_size, in_arr, CUDA_R_32F));
        CUSPARSE_ERROR(
            cusparseCreateDnVec(&vecy, m_row_size, rt_arr, CUDA_R_32F));

        cusparseSetStream(m_cusparse_handle, stream);

        if (m_spmv_buffer_size == 0) {
            RXMESH_WARN(
                "Sparse matrix - Array multiplication buffer size not "
                "initialized."
                "Calculate it now.");
            arr_mul_buffer_size(in_arr, rt_arr, stream);
        }

        CUDA_ERROR(cudaMalloc(&buffer, m_spmv_buffer_size));

        CUSPARSE_ERROR(cusparseSpMV(m_cusparse_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    m_spdescr,
                                    vecx,
                                    &beta,
                                    vecy,
                                    CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT,
                                    buffer));

        CUSPARSE_ERROR(cusparseDestroyDnVec(vecx));
        CUSPARSE_ERROR(cusparseDestroyDnVec(vecy));
        CUDA_ERROR(cudaFree(buffer));
    }

    /**
     * @brief do the sparse matrix dense matrix multiplication using sparse
     * matrix array multiplication in a column wise way
     */
    void spmat_denmat_mul_cw(rxmesh::DenseMatrix<T> B_mat,
                             rxmesh::DenseMatrix<T> C_mat)
    {
        for (int i = 0; i < B_mat.m_col_size; ++i) {
            arr_mul(B_mat.col_data(i), C_mat.col_data(i));
        }
    }

    /**
     * @brief solve the Ax=b for x where x and b are all array
     */
    void spmat_linear_solve(T*              B_arr,
                            T*              X_arr,
                            rxmesh::Solver  solver,
                            rxmesh::Reorder reorder)
    {
        cusparse_linear_solver_wrapper(solver,
                                       reorder,
                                       m_cusolver_sphandle,
                                       m_descr,
                                       m_row_size,
                                       m_col_size,
                                       m_nnz,
                                       m_d_row_ptr,
                                       m_d_col_idx,
                                       m_d_val,
                                       B_arr,
                                       X_arr);
    }

    /**
     * @brief solve the AX=B for X where X and B are all dense matrix and we
     * would solve it in a column wise manner
     */
    void spmat_linear_solve(rxmesh::DenseMatrix<T> B_mat,
                            rxmesh::DenseMatrix<T> X_mat,
                            rxmesh::Solver         solver,
                            rxmesh::Reorder        reorder)
    {
        for (int i = 0; i < B_mat.m_col_size; ++i) {
            cusparse_linear_solver_wrapper(solver,
                                           reorder,
                                           m_cusolver_sphandle,
                                           m_descr,
                                           m_row_size,
                                           m_col_size,
                                           m_nnz,
                                           m_d_row_ptr,
                                           m_d_col_idx,
                                           m_d_val,
                                           B_mat.col_data(i),
                                           X_mat.col_data(i));
        }
    }

    /**
     * @brief wrap up the cusolver api for solving linear systems. This is a
     * lower level api
     */
    void cusparse_linear_solver_wrapper(const rxmesh::Solver  solver,
                                        const rxmesh::Reorder reorder,
                                        cusolverSpHandle_t    handle,
                                        cusparseMatDescr_t    descrA,
                                        int                   rowsA,
                                        int                   colsA,
                                        int                   nnzA,
                                        int*                  d_csrRowPtrA,
                                        int*                  d_csrColIndA,
                                        T*                    d_csrValA,
                                        T*                    d_b,
                                        T*                    d_x)
    {
        if constexpr ((!std::is_same_v<T, float>)&&(
                          !std::is_same_v<T, double>)) {
            RXMESH_ERROR(
                "Unsupported type for cusparse: {}"
                "Only float and double are supported",
                typeid(T).name());
        }

        double tol         = 1.e-12;
        int    singularity = 0; /* -1 if A is invertible under tol. */

        /* solve B*z = Q*b */
        if (solver == Solver::CHOL) {
            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(cusolverSpScsrlsvchol(handle,
                                                     rowsA,
                                                     nnzA,
                                                     descrA,
                                                     d_csrValA,
                                                     d_csrRowPtrA,
                                                     d_csrColIndA,
                                                     d_b,
                                                     tol,
                                                     reorder_to_int(reorder),
                                                     d_x,
                                                     &singularity));
            }

            if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_ERROR(cusolverSpDcsrlsvchol(handle,
                                                     rowsA,
                                                     nnzA,
                                                     descrA,
                                                     d_csrValA,
                                                     d_csrRowPtrA,
                                                     d_csrColIndA,
                                                     d_b,
                                                     tol,
                                                     reorder_to_int(reorder),
                                                     d_x,
                                                     &singularity));
            }

        } else if (solver == Solver::QR) {
            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(cusolverSpScsrlsvqr(handle,
                                                   rowsA,
                                                   nnzA,
                                                   descrA,
                                                   d_csrValA,
                                                   d_csrRowPtrA,
                                                   d_csrColIndA,
                                                   d_b,
                                                   tol,
                                                   reorder_to_int(reorder),
                                                   d_x,
                                                   &singularity));
            }

            if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_ERROR(cusolverSpDcsrlsvqr(handle,
                                                   rowsA,
                                                   nnzA,
                                                   descrA,
                                                   d_csrValA,
                                                   d_csrRowPtrA,
                                                   d_csrColIndA,
                                                   d_b,
                                                   tol,
                                                   reorder_to_int(reorder),
                                                   d_x,
                                                   &singularity));
            }
        } else {
            RXMESH_ERROR(
                "Only Solver::CHOL and Solver::QR is supported, use CUDA 12.x "
                "for "
                "Solver::LU");
        }
        cudaDeviceSynchronize();

        if (0 <= singularity) {
            RXMESH_WARN(
                "WARNING: the matrix is singular at row {} under tol ({})",
                singularity,
                tol);
        }
    }

    const Context        m_context;
    cusparseHandle_t     m_cusparse_handle;
    cusolverSpHandle_t   m_cusolver_sphandle;
    cusparseSpMatDescr_t m_spdescr;
    cusparseMatDescr_t   m_descr;

    size_t m_spmm_buffer_size;
    size_t m_spmv_buffer_size;

    IndexT* m_d_row_ptr;
    IndexT* m_d_col_idx;
    T*      m_d_val;
    IndexT  m_row_size;
    IndexT  m_col_size;
    IndexT  m_nnz;
};

}  // namespace rxmesh