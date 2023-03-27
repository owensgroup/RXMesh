#pragma once
#include "cusolverSp.h"
#include "cusparse.h"
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/types.h"

#include "cusolverSp_LOWLEVEL_PREVIEW.h"
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
          m_spmv_buffer_size(0),
          m_use_reorder(false)
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

    __device__ T& operator()(const IndexT x, const IndexT y)
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

    __device__ T& operator()(const IndexT x, const IndexT y) const
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

    __host__ __device__ IndexT& get_nnz() const
    {
        return m_nnz;
    }

    __device__ IndexT& get_row_ptr_at(IndexT idx) const
    {
        return m_d_row_ptr[idx];
    }

    __device__ IndexT& get_col_idx_at(IndexT idx) const
    {
        return m_d_col_idx[idx];
    }

    __device__ T& get_val_at(IndexT idx) const
    {
        return m_d_val[idx];
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

    /* ----- CUSPARSE SPMM & SPMV ----- */

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

        CUSPARSE_ERROR(cusparseSetStream(m_cusparse_handle, stream));

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

        CUSPARSE_ERROR(cusparseSetStream(m_cusparse_handle, stream));

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

        CUSPARSE_ERROR(cusparseSetStream(m_cusparse_handle, stream));

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

    /*  ----- SOLVER -----  */

    /* --- HIGH LEVEL API --- */

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
        CUDA_ERROR(cudaDeviceSynchronize());

        if (0 <= singularity) {
            RXMESH_WARN(
                "WARNING: the matrix is singular at row {} under tol ({})",
                singularity,
                tol);
        }
    }

    /* --- LOW LEVEL API --- */

    void spmat_chol_reorder(rxmesh::Reorder reorder)
    {
        if (reorder == Reorder::NONE) {
            RXMESH_INFO("None reordering is specified",
                        "Continue without reordering");
            m_use_reorder = false;
            return;
        }

        m_use_reorder = true;

        /*check on host*/
        bool on_host = true;
        if ((HOST & m_allocated) != HOST) {
            move(DEVICE, HOST);
            on_host = false;
        }

        if (reorder == Reorder::SYMRCM) {
            CUSOLVER_ERROR(cusolverSpXcsrsymrcmHost(m_cusolver_sphandle,
                                                    m_row_size,
                                                    m_nnz,
                                                    m_descr,
                                                    m_h_row_ptr,
                                                    m_h_col_idx,
                                                    m_h_permute));
        } else if (reorder == Reorder::SYMAMD) {
            CUSOLVER_ERROR(cusolverSpXcsrsymamdHost(m_cusolver_sphandle,
                                                    m_row_size,
                                                    m_nnz,
                                                    m_descr,
                                                    m_h_row_ptr,
                                                    m_h_col_idx,
                                                    m_h_permute));
        } else if (reorder == Reorder::NSTDIS) {
            CUSOLVER_ERROR(cusolverSpXcsrmetisndHost(m_cusolver_sphandle,
                                                     m_row_size,
                                                     m_nnz,
                                                     m_descr,
                                                     m_h_row_ptr,
                                                     m_h_col_idx,
                                                     NULL,
                                                     m_h_permute));
        }

        // working space for permutation: B = A*Q*A^T
        size_t size_perm       = 0;
        void*  perm_buffer_cpu = NULL;

        CUSOLVER_ERROR(cusolverSpXcsrperm_bufferSizeHost(m_cusolver_sphandle,
                                                         m_row_size,
                                                         m_col_size,
                                                         m_nnz,
                                                         m_descr,
                                                         m_h_row_ptr,
                                                         m_h_col_idx,
                                                         m_h_permute,
                                                         m_h_permute,
                                                         &size_perm));

        perm_buffer_cpu = (void*)malloc(sizeof(char) * size_perm);
        assert(NULL != perm_buffer_cpu);

        IndexT* h_mapBfromQ =
            static_cast<IndexT*>(malloc(m_nnz * sizeof(IndexT)));
        IndexT* d_mapBfromQ = CUDA_ERROR(
            cudaMalloc((void**)&m_d_col_idx, m_nnz * sizeof(IndexT)));

        // do the permutation which works only on the col and row indices
        CUSOLVER_ERROR(cusolverSpXcsrpermHost(m_cusolver_sphandle,
                                              m_row_size,
                                              m_col_size,
                                              m_nnz,
                                              m_descr,
                                              m_h_row_ptr,
                                              m_h_col_idx,
                                              m_h_permute,
                                              m_h_permute,
                                              h_mapBfromQ,
                                              perm_buffer_cpu));

        // allocate the purmutated csr and copy from the host
        CUDA_ERROR(cudaMalloc((void**)&m_d_solver_val, m_nnz * sizeof(T)));
        CUDA_ERROR(cudaMalloc((void**)&m_d_solver_row_ptr,
                              m_row_size * sizeof(IndexT)));
        CUDA_ERROR(
            cudaMalloc((void**)&m_d_solver_col_idx, m_nnz * sizeof(IndexT)));

        CUDA_ERROR(cudaMemcpyAsync(m_d_solver_val,
                                   m_h_val,
                                   m_nnz * sizeof(T),
                                   cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpyAsync(m_d_solver_row_ptr,
                                   m_h_row_ptr,
                                   m_row_size * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpyAsync(m_d_solver_col_idx,
                                   m_h_col_idx,
                                   m_nnz * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));

        // do the permutation for val indices
        cusparseDnVecDescr_t val_values;
        cusparseSpVecDescr_t val_permutation;

        CUDA_ERROR(cudaMemcpyAsync(d_mapBfromQ,
                                   h_mapBfromQ,
                                   m_nnz * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));

        CUSPARSE_ERROR(cusparseCreateSpVec(&val_permutation,
                                           m_nnz,
                                           m_nnz,
                                           d_mapBfromQ,
                                           m_d_val,
                                           CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           CUDA_R_32F));
        CUSPARSE_ERROR(cusparseCreateDnVec(
            &val_values, m_nnz, m_d_solver_val, CUDA_R_32F));
        CUSPARSE_ERROR(
            cusparseScatter(m_cusparse_handle, val_permutation, val_values));
        CUSPARSE_ERROR(cusparseDestroyDnVec(val_values));
        CUSPARSE_ERROR(cusparseDestroySpVec(val_permutation));

        free(perm_buffer_cpu);
        free(h_mapBfromQ);
        CUDA_ERROR(cudaFree(d_mapBfromQ));

        // restore the host data back to the original
        if (on_host) {
            move(DEVICE, HOST);
        } else {
            release(HOST);
        }
    }

    void spmat_chol_analysis()
    {
        if (!m_use_reorder) {
            m_d_solver_row_ptr = m_d_row_ptr;
            m_d_solver_col_idx = m_d_col_idx;
            m_d_solver_val     = m_d_val;
        }

        CUSOLVER_ERROR(cusolverSpCreateCsrcholInfo(&m_chol_info));
        m_internalDataInBytes = 0;
        m_workspaceInBytes    = 0;
        CUSOLVER_ERROR(cusolverSpXcsrcholAnalysis(m_cusolver_sphandle,
                                                  m_row_size,
                                                  m_nnz,
                                                  m_descr,
                                                  m_d_solver_row_ptr,
                                                  m_d_solver_col_idx,
                                                  m_chol_info));
    }

    void spmat_chol_buffer_alloc()
    {
        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrcholBufferInfo(m_cusolver_sphandle,
                                                        m_row_size,
                                                        m_nnz,
                                                        m_descr,
                                                        m_d_solver_val,
                                                        m_d_solver_row_ptr,
                                                        m_d_solver_col_idx,
                                                        m_chol_info,
                                                        &m_internalDataInBytes,
                                                        &m_workspaceInBytes));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrcholBufferInfo(m_cusolver_sphandle,
                                                        m_row_size,
                                                        m_nnz,
                                                        m_descr,
                                                        m_d_solver_val,
                                                        m_d_solver_row_ptr,
                                                        m_d_solver_col_idx,
                                                        m_chol_info,
                                                        &m_internalDataInBytes,
                                                        &m_workspaceInBytes));
        }

        CUDA_ERROR(cudaMalloc((void**)&m_chol_buffer, m_workspaceInBytes));
    }

    void spmat_chol_buffer_free()
    {
        CUDA_ERROR(cudaFree(m_chol_buffer));
    }

    void spmat_chol_factor()
    {
        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrcholFactor(m_cusolver_sphandle,
                                                    m_row_size,
                                                    m_nnz,
                                                    m_descr,
                                                    m_d_solver_val,
                                                    m_d_solver_row_ptr,
                                                    m_d_solver_col_idx,
                                                    m_chol_info,
                                                    m_chol_buffer));
        }
        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrcholFactor(m_cusolver_sphandle,
                                                    m_row_size,
                                                    m_nnz,
                                                    m_descr,
                                                    m_d_solver_val,
                                                    m_d_solver_row_ptr,
                                                    m_d_solver_col_idx,
                                                    m_chol_info,
                                                    m_chol_buffer));
        }

        double tol = 1.0e-8;
        int    singularity;

        CUSOLVER_ERROR(cusolverSpDcsrcholZeroPivot(
            m_cusolver_sphandle, m_chol_info, tol, &singularity));
        if (0 <= singularity) {
            RXMESH_WARN(
                "WARNING: the matrix is singular at row {} under tol ({})",
                singularity,
                tol);
        }
    }

    void spmat_chol_solve(T* d_b, T* d_x)
    {

        T* d_solver_b;
        T* d_solver_x;

        if (m_use_reorder) {
            /* purmute b and x*/
            CUDA_ERROR(cudaMalloc((void**)&d_solver_b, m_row_size * sizeof(T)));
            CUDA_ERROR(cudaMalloc((void**)&d_solver_x, m_col_size * sizeof(T)));

            cusparseDnVecDescr_t b_values;
            cusparseSpVecDescr_t b_permutation;

            CUSPARSE_ERROR(cusparseCreateSpVec(&b_permutation,
                                               m_row_size,
                                               m_row_size,
                                               m_d_permute,
                                               d_b,
                                               CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_BASE_ZERO,
                                               CUDA_R_32F));
            CUSPARSE_ERROR(
                cusparseCreateDnVec(&b_values, m_nnz, d_solver_b, CUDA_R_32F));
            CUSPARSE_ERROR(
                cusparseScatter(m_cusparse_handle, b_permutation, b_values));
            CUSPARSE_ERROR(cusparseDestroyDnVec(b_values));
            CUSPARSE_ERROR(cusparseDestroySpVec(b_permutation));

            cusparseDnVecDescr_t x_values;
            cusparseSpVecDescr_t x_permutation;

            CUSPARSE_ERROR(cusparseCreateSpVec(&x_permutation,
                                               m_col_size,
                                               m_col_size,
                                               m_d_permute,
                                               d_x,
                                               CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_BASE_ZERO,
                                               CUDA_R_32F));
            CUSPARSE_ERROR(
                cusparseCreateDnVec(&x_values, m_nnz, d_solver_x, CUDA_R_32F));
            CUSPARSE_ERROR(
                cusparseScatter(m_cusparse_handle, x_permutation, x_values));
            CUSPARSE_ERROR(cusparseDestroyDnVec(x_values));
            CUSPARSE_ERROR(cusparseDestroySpVec(x_permutation));
        } else {
            d_solver_b = d_b;
            d_solver_x = d_x;
        }

        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_ERROR(cusolverSpScsrcholSolve(m_cusolver_sphandle,
                                                   m_row_size,
                                                   d_solver_b,
                                                   d_solver_x,
                                                   m_chol_info,
                                                   m_chol_buffer));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_ERROR(cusolverSpDcsrcholSolve(m_cusolver_sphandle,
                                                   m_row_size,
                                                   d_solver_b,
                                                   d_solver_x,
                                                   m_chol_info,
                                                   m_chol_buffer));
        }
    }

    /* Host data compatibility */

    void move(locationT source, locationT target, cudaStream_t stream = NULL)
    {
        if (source == target) {
            RXMESH_WARN(
                "DenseMatrix::move() source ({}) and target ({}) "
                "are the same.",
                location_to_string(source),
                location_to_string(target));
            return;
        }

        if ((source == HOST || source == DEVICE) &&
            ((source & m_allocated) != source)) {
            RXMESH_ERROR(
                "DenseMatrix::move() moving source is not valid"
                " because it was not allocated on source i.e., {}",
                location_to_string(source));
        }

        if (((target & HOST) == HOST || (target & DEVICE) == DEVICE) &&
            ((target & m_allocated) != target)) {
            RXMESH_WARN(
                "DenseMatrix::move() allocating target before moving to {}",
                location_to_string(target));
            allocate(target);
        }

        if (source == HOST && target == DEVICE) {
            CUDA_ERROR(cudaMemcpyAsync(m_d_val,
                                       m_h_val,
                                       m_nnz * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       stream));
            CUDA_ERROR(cudaMemcpyAsync(m_d_row_ptr,
                                       m_h_row_ptr,
                                       m_row_size * sizeof(IndexT),
                                       cudaMemcpyHostToDevice,
                                       stream));
            CUDA_ERROR(cudaMemcpyAsync(m_d_col_idx,
                                       m_h_col_idx,
                                       m_nnz * sizeof(IndexT),
                                       cudaMemcpyHostToDevice,
                                       stream));
        } else if (source == DEVICE && target == HOST) {
            CUDA_ERROR(cudaMemcpyAsync(m_h_val,
                                       m_d_val,
                                       m_nnz * sizeof(T),
                                       cudaMemcpyDeviceToHost,
                                       stream));
            CUDA_ERROR(cudaMemcpyAsync(m_h_row_ptr,
                                       m_d_row_ptr,
                                       m_row_size * sizeof(IndexT),
                                       cudaMemcpyDeviceToHost,
                                       stream));
            CUDA_ERROR(cudaMemcpyAsync(m_h_col_idx,
                                       m_d_col_idx,
                                       m_nnz * sizeof(IndexT),
                                       cudaMemcpyDeviceToHost,
                                       stream));
        }
    }

    void release(locationT location = LOCATION_ALL)
    {
        if (((location & HOST) == HOST) && ((m_allocated & HOST) == HOST)) {
            free(m_h_val);
            free(m_h_row_ptr);
            free(m_h_col_idx);
            m_h_val     = nullptr;
            m_h_row_ptr = nullptr;
            m_h_col_idx = nullptr;
            m_allocated = m_allocated & (~HOST);
        }

        if (((location & DEVICE) == DEVICE) &&
            ((m_allocated & DEVICE) == DEVICE)) {
            GPU_FREE(m_d_val);
            GPU_FREE(m_h_row_ptr);
            GPU_FREE(m_h_col_idx);
            m_allocated = m_allocated & (~DEVICE);
        }
    }

    void allocate(locationT location)
    {
        if ((location & HOST) == HOST) {
            release(HOST);

            m_h_val = static_cast<T*>(malloc(m_nnz * sizeof(T)));
            m_h_row_ptr =
                static_cast<IndexT*>(malloc(m_row_size * sizeof(IndexT)));
            m_h_col_idx = static_cast<IndexT*>(malloc(m_nnz * sizeof(IndexT)));

            m_allocated = m_allocated | HOST;
        }

        if ((location & DEVICE) == DEVICE) {
            release(DEVICE);

            CUDA_ERROR(cudaMalloc((void**)&m_d_val, m_nnz * sizeof(T)));
            CUDA_ERROR(
                cudaMalloc((void**)&m_d_row_ptr, m_row_size * sizeof(IndexT)));
            CUDA_ERROR(
                cudaMalloc((void**)&m_d_col_idx, m_nnz * sizeof(IndexT)));

            m_allocated = m_allocated | DEVICE;
        }
    }


   private:
    const Context        m_context;
    cusparseHandle_t     m_cusparse_handle;
    cusolverSpHandle_t   m_cusolver_sphandle;
    cusparseSpMatDescr_t m_spdescr;
    cusparseMatDescr_t   m_descr;

    IndexT m_row_size;
    IndexT m_col_size;
    IndexT m_nnz;

    // device csr data
    IndexT* m_d_row_ptr;
    IndexT* m_d_col_idx;
    T*      m_d_val;

    // host csr data
    IndexT* m_h_row_ptr;
    IndexT* m_h_col_idx;
    T*      m_h_val;

    // susparse buffer
    size_t m_spmm_buffer_size;
    size_t m_spmv_buffer_size;

    // lower level API parameters
    csrcholInfo_t m_chol_info;
    size_t        m_internalDataInBytes;
    size_t        m_workspaceInBytes;
    void*         m_chol_buffer;

    // purmutation array
    IndexT* m_h_permute;
    IndexT* m_d_permute;

    // CSR matrix for solving only
    // equal to the original matrix if not permutated
    // only allocated as a new CSR matrix if permutated
    IndexT* m_d_solver_row_ptr;
    IndexT* m_d_solver_col_idx;
    T*      m_d_solver_val;

    // flags
    bool      m_use_reorder;
    locationT m_allocated;
};

}  // namespace rxmesh