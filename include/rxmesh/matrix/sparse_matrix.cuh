#pragma once
#include <algorithm>
#include "cusolverSp.h"
#include "cusparse.h"
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"

#include "rxmesh/launch_box.h"

#include "thrust/device_ptr.h"
#include "thrust/execution_policy.h"
#include "thrust/gather.h"
#include "thrust/scatter.h"

#include "cusolverSp_LOWLEVEL_PREVIEW.h"
#include "rxmesh/matrix/dense_matrix.cuh"
#include "rxmesh/matrix/permute_util.h"
#include "rxmesh/matrix/sparse_matrix_kernels.cuh"

#include "rxmesh/matrix/mgnd_permute.cuh"
#include "rxmesh/matrix/nd_permute.cuh"

#include "rxmesh/launch_box.h"

#include <Eigen/Sparse>

namespace rxmesh {

/**
 * @brief The enum class for choosing different solver types
 * Documentation of cuSolver low-level preview API
 * https://docs.nvidia.com/cuda/archive/8.0/cusolver/index.html#cusolver-preview-reference
 */
enum class Solver
{
    NONE = 0,
    CHOL = 1,
    LU   = 2,
    QR   = 3
};

/**
 * @brief The enum class for choosing different reorder types
 * NONE for No Reordering Applied, SYMRCM for Symmetric Reverse Cuthill-McKee
 * permutation, SYMAMD for Symmetric Approximate Minimum Degree Algorithm based
 * on Quotient Graph, NSTDIS for Nested Dissection, GPUMGND is a GPU modified
 * generalized nested dissection permutation, and GPUND is GPU nested dissection
 */
enum class PermuteMethod
{
    NONE    = 0,
    SYMRCM  = 1,
    SYMAMD  = 2,
    NSTDIS  = 3,
    GPUMGND = 4,
    GPUND   = 5
};

inline PermuteMethod string_to_permute_method(std::string prem)
{
    std::transform(prem.begin(), prem.end(), prem.begin(), [](unsigned char c) {
        return std::tolower(c);
    });

    if (prem == "symrcm") {
        return PermuteMethod::SYMRCM;
    } else if (prem == "symamd") {
        return PermuteMethod::SYMAMD;
    } else if (prem == "nstdis") {
        return PermuteMethod::NSTDIS;
    } else if (prem == "gpumgnd") {
        return PermuteMethod::GPUMGND;
    } else if (prem == "gpund") {
        return PermuteMethod::GPUND;
    } else {
        return PermuteMethod::NONE;
    }
}


inline std::string permute_method_to_string(PermuteMethod prem)
{
    if (prem == PermuteMethod::SYMRCM) {
        return "symrcm";
    } else if (prem == PermuteMethod::SYMAMD) {
        return "symamd";
    } else if (prem == PermuteMethod::NSTDIS) {
        return "nstdis";
    } else if (prem == PermuteMethod::GPUMGND) {
        return "gpumgnd";
    } else if (prem == PermuteMethod::GPUND) {
        return "gpund";
    } else {
        return "none";
    }
}

/**
 * @brief Sparse matrix that represent the VV connectivity, i.e., it
 * is a square matrix with number of rows/cols is equal to number of vertices
 * and there is non-zero values at entry (i,j) only if the vertex i is connected
 * to vertex j. The sparse matrix is stored as a CSR matrix. The matrix is
 * accessible on both host and device. The class also provides implementation
 * for matrix-vector multiplication and linear solver (using cuSolver and
 * cuSparse as a back-end.
 */
template <typename T>
struct SparseMatrix
{
    using IndexT = int;

    using EigenSparseMatrix =
        Eigen::Map<const Eigen::SparseMatrix<T, Eigen::RowMajor, IndexT>>;

    SparseMatrix()
        : m_d_row_ptr(nullptr),
          m_d_col_idx(nullptr),
          m_d_val(nullptr),
          m_h_row_ptr(nullptr),
          m_h_col_idx(nullptr),
          m_h_val(nullptr),
          m_num_rows(0),
          m_num_cols(0),
          m_nnz(0),
          m_context(Context()),
          m_cusparse_handle(NULL),
          m_descr(NULL),
          m_replicate(0),
          m_spdescr(NULL),
          m_spmm_buffer_size(0),
          m_spmv_buffer_size(0),
          m_h_permute(nullptr),
          m_d_permute(nullptr),
          m_d_solver_row_ptr(nullptr),
          m_d_solver_col_idx(nullptr),
          m_d_solver_val(nullptr),
          m_h_solver_row_ptr(nullptr),
          m_h_solver_col_idx(nullptr),
          m_h_permute_map(nullptr),
          m_d_permute_map(nullptr),
          m_use_reorder(false),
          m_reorder_allocated(false),
          m_d_cusparse_spmm_buffer(nullptr),
          m_d_cusparse_spmv_buffer(nullptr),
          m_solver_buffer(nullptr),
          m_d_solver_b(nullptr),
          m_d_solver_x(nullptr),
          m_allocated(LOCATION_NONE),
          m_current_solver(Solver::NONE)
    {
    }

    SparseMatrix(const RXMeshStatic& rx) : SparseMatrix(rx, 1) {};

   protected:
    SparseMatrix(const RXMeshStatic& rx, IndexT replicate)
        : m_d_row_ptr(nullptr),
          m_d_col_idx(nullptr),
          m_d_val(nullptr),
          m_h_row_ptr(nullptr),
          m_h_col_idx(nullptr),
          m_h_val(nullptr),
          m_num_rows(0),
          m_num_cols(0),
          m_nnz(0),
          m_context(rx.get_context()),
          m_cusparse_handle(NULL),
          m_descr(NULL),
          m_replicate(replicate),
          m_spdescr(NULL),
          m_spmm_buffer_size(0),
          m_spmv_buffer_size(0),
          m_h_permute(nullptr),
          m_d_permute(nullptr),
          m_d_solver_row_ptr(nullptr),
          m_d_solver_col_idx(nullptr),
          m_d_solver_val(nullptr),
          m_h_solver_row_ptr(nullptr),
          m_h_solver_col_idx(nullptr),
          m_h_permute_map(nullptr),
          m_d_permute_map(nullptr),
          m_use_reorder(false),
          m_reorder_allocated(false),
          m_d_cusparse_spmm_buffer(nullptr),
          m_d_cusparse_spmv_buffer(nullptr),
          m_solver_buffer(nullptr),
          m_d_solver_b(nullptr),
          m_d_solver_x(nullptr),
          m_allocated(LOCATION_NONE),
          m_current_solver(Solver::NONE)
    {
        using namespace rxmesh;
        constexpr uint32_t blockThreads = 256;

        IndexT num_patches  = rx.get_num_patches();
        IndexT num_vertices = rx.get_num_vertices();
        IndexT num_edges    = rx.get_num_edges();

        m_num_rows = num_vertices * m_replicate;
        m_num_cols = num_vertices * m_replicate;

        // row pointer allocation and init with prefix sum for CSR
        CUDA_ERROR(cudaMalloc((void**)&m_d_row_ptr,
                              (m_num_rows + 1) * sizeof(IndexT)));

        CUDA_ERROR(
            cudaMemset(m_d_row_ptr, 0, (m_num_rows + 1) * sizeof(IndexT)));

        LaunchBox<blockThreads> launch_box;
        rx.prepare_launch_box({Op::VV},
                              launch_box,
                              (void*)detail::sparse_mat_prescan<blockThreads>);

        detail::sparse_mat_prescan<blockThreads><<<launch_box.blocks,
                                                   launch_box.num_threads,
                                                   launch_box.smem_bytes_dyn>>>(
            m_context, m_d_row_ptr, m_replicate);

        // prefix sum using CUB.
        void*  d_cub_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                      temp_storage_bytes,
                                      m_d_row_ptr,
                                      m_d_row_ptr,
                                      m_num_rows + 1);
        CUDA_ERROR(cudaMalloc((void**)&d_cub_temp_storage, temp_storage_bytes));

        cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                      temp_storage_bytes,
                                      m_d_row_ptr,
                                      m_d_row_ptr,
                                      m_num_rows + 1);

        CUDA_ERROR(cudaFree(d_cub_temp_storage));

        // get nnz
        CUDA_ERROR(cudaMemcpy(&m_nnz,
                              (m_d_row_ptr + m_num_rows),
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
                m_context, m_d_row_ptr, m_d_col_idx, m_replicate);


        // allocate value ptr
        CUDA_ERROR(cudaMalloc((void**)&m_d_val, m_nnz * sizeof(T)));
        CUDA_ERROR(cudaMemset(m_d_val, 0, m_nnz * sizeof(T)));
        m_allocated = m_allocated | DEVICE;

        // create cusparse matrix
        CUSPARSE_ERROR(cusparseCreateMatDescr(&m_descr));
        CUSPARSE_ERROR(
            cusparseSetMatType(m_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        CUSPARSE_ERROR(
            cusparseSetMatDiagType(m_descr, CUSPARSE_DIAG_TYPE_NON_UNIT));
        CUSPARSE_ERROR(
            cusparseSetMatIndexBase(m_descr, CUSPARSE_INDEX_BASE_ZERO));

        CUSPARSE_ERROR(cusparseCreateCsr(&m_spdescr,
                                         m_num_rows,
                                         m_num_cols,
                                         m_nnz,
                                         m_d_row_ptr,
                                         m_d_col_idx,
                                         m_d_val,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         cuda_type<T>()));

        CUSPARSE_ERROR(cusparseCreate(&m_cusparse_handle));
        CUSOLVER_ERROR(cusolverSpCreate(&m_cusolver_sphandle));

        CUSOLVER_ERROR(cusolverSpCreateCsrcholInfo(&m_chol_info));

        CUSOLVER_ERROR(cusolverSpCreateCsrqrInfo(&m_qr_info));

        // allocate the host
        m_h_val = static_cast<T*>(malloc(m_nnz * sizeof(T)));
        m_h_row_ptr =
            static_cast<IndexT*>(malloc((m_num_rows + 1) * sizeof(IndexT)));
        m_h_col_idx = static_cast<IndexT*>(malloc(m_nnz * sizeof(IndexT)));

        CUDA_ERROR(cudaMemcpy(
            m_h_val, m_d_val, m_nnz * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(m_h_col_idx,
                              m_d_col_idx,
                              m_nnz * sizeof(IndexT),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(m_h_row_ptr,
                              m_d_row_ptr,
                              (m_num_rows + 1) * sizeof(IndexT),
                              cudaMemcpyDeviceToHost));

        m_allocated = m_allocated | HOST;

        CUSPARSE_ERROR(cusparseSetPointerMode(m_cusparse_handle,
                                              CUSPARSE_POINTER_MODE_HOST));

#ifndef NDEBUG
        // sanity check: no repeated indices in the col_id for a specific row
        for (IndexT r = 0; r < rows(); ++r) {
            IndexT start = m_h_row_ptr[r];
            IndexT stop  = m_h_row_ptr[r + 1];

            std::set<IndexT> cols;
            for (IndexT i = start; i < stop; ++i) {
                IndexT c = m_h_col_idx[i];
                if (cols.find(c) != cols.end()) {
                    RXMESH_ERROR(
                        "SparseMatrix::SparseMatrix() Error in constructing "
                        "the sparse matrix. Row {} contains repeated column "
                        "indices {}",
                        r,
                        c);
                }
                cols.insert(c);
            }
        }

#endif
    }

   public:
    /**
     * @brief export the matrix to a file that can be opened by MATLAB (i.e.,
     * 1-based indices)
     */
    __host__ void to_file(std::string file_name)
    {
        std::ofstream file(file_name);
        if (!file.is_open()) {
            RXMESH_ERROR("SparseMatrix::to_file() Can not open file {}",
                         file_name);
            return;
        }

        for_each([&](IndexT r, IndexT c, T val) {
            file << r + 1 << " " << c + 1 << " " << val << std::endl;
        });
        file.close();
    }

    /**
     * @brief set all entries in the matrix to certain value on both host and
     * device
     */
    __host__ void reset(T val, locationT location, cudaStream_t stream = NULL)
    {
        bool do_device = (location & DEVICE) == DEVICE;
        bool do_host   = (location & DEVICE) == DEVICE;

        if (do_device && do_host) {
            std::fill_n(m_h_val, m_nnz, val);
            CUDA_ERROR(cudaMemcpyAsync(m_d_val,
                                       m_h_val,
                                       m_nnz * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       stream));
        } else if (do_device) {
            const int threads = 512;
            memset<<<DIVIDE_UP(m_nnz, threads), threads, 0, stream>>>(
                m_d_val, val, m_nnz);
        } else if (do_host) {
            std::fill_n(m_h_val, m_nnz, val);
        }
    }


    /**
     * @brief return number of rows
     */
    __device__ __host__ IndexT rows() const
    {
        return m_num_rows;
    }

    /**
     * @brief return number of cols
     */
    __device__ __host__ IndexT cols() const
    {
        return m_num_cols;
    }

    /**
     * @brief return number of non-zero values
     */
    __device__ __host__ IndexT non_zeros() const
    {
        return m_nnz;
    }

    /**
     * @brief return the number of non-zero values on and below the diagonal
     */
    __host__ IndexT lower_non_zeros() const
    {
        int l_nnz = 0;

        for_each([&](IndexT r, IndexT c, T&) {
            if (c <= r) {
                l_nnz++;
            }
        });


        return l_nnz;
    }

    /**
     * @brief apply a lambda function for each entry in the sparse matrix
     * @return
     */
    template <typename FuncT>
    __host__ void for_each(FuncT fun)
    {
        for (IndexT r = 0; r < rows(); ++r) {
            IndexT start = m_h_row_ptr[r];
            IndexT stop  = m_h_row_ptr[r + 1];
            for (IndexT i = start; i < stop; ++i) {
                IndexT c = m_h_col_idx[i];
                fun(r, c, get_val_at(i));
            }
        }
    }

    /**
     * @brief access the matrix using VertexHandle
     */
    __device__ __host__ const T& operator()(const VertexHandle& row_v,
                                            const VertexHandle& col_v) const
    {
        return this->operator()(get_row_id(row_v), get_row_id(col_v));
    }

    /**
     * @brief access the matrix using VertexHandle
     */
    __device__ __host__ T& operator()(const VertexHandle& row_v,
                                      const VertexHandle& col_v)
    {
        return this->operator()(get_row_id(row_v), get_row_id(col_v));
    }

    /**
     * @brief access the matrix using row and col index
     */
    __device__ __host__ T& operator()(const IndexT x, const IndexT y)
    {
        const IndexT start = row_ptr()[x];
        const IndexT end   = row_ptr()[x + 1];

        for (IndexT i = start; i < end; ++i) {
            if (col_idx()[i] == y) {
                return get_val_at(i);
            }
        }
        assert(1 != 1);
        return get_val_at(0);
    }

    /**
     * @brief access the matrix using row and col index
     */
    __device__ __host__ const T& operator()(const IndexT x,
                                            const IndexT y) const
    {
        const IndexT start = row_ptr()[x];
        const IndexT end   = row_ptr()[x + 1];

        for (IndexT i = start; i < end; ++i) {
            if (col_idx()[i] == y) {
                return get_val_at(i);
            }
        }
        assert(1 != 1);
        return T(0);
    }

    /**
     * @brief return the row pointer of the CSR matrix
     * @return
     */
    __device__ __host__ const IndexT* row_ptr() const
    {
#ifdef __CUDA_ARCH__
        return m_d_row_ptr;
#else
        return m_h_row_ptr;
#endif
    }

    /**
     * @brief return the column index pointer of the CSR matrix
     * @return
     */
    __device__ __host__ const IndexT* col_idx() const
    {
#ifdef __CUDA_ARCH__
        return m_d_col_idx;
#else
        return m_h_col_idx;
#endif
    }

    /**
     * @brief access the value of (1D array) array that holds the nnz in the CSR
     * matrix
     */
    __device__ __host__ T& get_val_at(IndexT idx) const
    {
#ifdef __CUDA_ARCH__
        return m_d_val[idx];
#else
        return m_h_val[idx];
#endif
    }

    /**
     * @brief return the row index corresponding to specific vertex handle
     */
    __device__ __host__ uint32_t get_row_id(const VertexHandle& handle) const
    {
        auto id = handle.unpack();
        return m_context.vertex_prefix()[id.first] + id.second;
    }

    /**
     * @brief release all allocated memory
     */
    __host__ void release()
    {
        release(LOCATION_ALL);
        CUSPARSE_ERROR(cusparseDestroy(m_cusparse_handle));
        CUSPARSE_ERROR(cusparseDestroyMatDescr(m_descr));
        CUSOLVER_ERROR(cusolverSpDestroy(m_cusolver_sphandle));
        CUSOLVER_ERROR(cusolverSpDestroyCsrcholInfo(m_chol_info));
        CUSOLVER_ERROR(cusolverSpDestroyCsrqrInfo(m_qr_info));


        if (m_reorder_allocated) {
            GPU_FREE(m_d_solver_val);
            GPU_FREE(m_d_solver_row_ptr);
            GPU_FREE(m_d_solver_col_idx);
            GPU_FREE(m_d_permute);
            GPU_FREE(m_d_solver_x);
            GPU_FREE(m_d_solver_b);
            GPU_FREE(m_d_permute_map);

            free(m_h_permute);
            free(m_h_permute_map);
        }
        GPU_FREE(m_solver_buffer);
        GPU_FREE(m_d_cusparse_spmm_buffer);
        GPU_FREE(m_d_cusparse_spmv_buffer);
    }

    /**
     * @brief move the data between host an device
     */
    __host__ void move(locationT    source,
                       locationT    target,
                       cudaStream_t stream = NULL)
    {
        if (source == target) {
            RXMESH_WARN(
                "SparseMatrix::move() source ({}) and target ({}) "
                "are the same.",
                location_to_string(source),
                location_to_string(target));
            return;
        }

        if ((source == HOST || source == DEVICE) &&
            ((source & m_allocated) != source)) {
            RXMESH_ERROR(
                "SparseMatrix::move() moving source is not valid"
                " because it was not allocated on source i.e., {}",
                location_to_string(source));
            return;
        }

        if (((target & HOST) == HOST || (target & DEVICE) == DEVICE) &&
            ((target & m_allocated) != target)) {
            RXMESH_ERROR("SparseMatrix::move() target {} is not allocated!",
                         location_to_string(target));
            return;
        }

        if (source == HOST && target == DEVICE) {
            CUDA_ERROR(cudaMemcpyAsync(m_d_val,
                                       m_h_val,
                                       m_nnz * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       stream));
        } else if (source == DEVICE && target == HOST) {
            CUDA_ERROR(cudaMemcpyAsync(m_h_val,
                                       m_d_val,
                                       m_nnz * sizeof(T),
                                       cudaMemcpyDeviceToHost,
                                       stream));
        }
    }


    /**
     * @brief allocate the temp buffer needed for sparse matrix multiplication
     * by a dense matrix
     */
    __host__ void alloc_multiply_buffer(const DenseMatrix<T>& B_mat,
                                        DenseMatrix<T>&       C_mat,
                                        cudaStream_t          stream = 0)
    {
        T alpha;
        T beta;

        cusparseSpMatDescr_t matA    = m_spdescr;
        cusparseDnMatDescr_t matB    = B_mat.m_dendescr;
        cusparseDnMatDescr_t matC    = C_mat.m_dendescr;
        void*                dBuffer = NULL;

        CUSPARSE_ERROR(cusparseSetStream(m_cusparse_handle, stream));

        CUSPARSE_ERROR(cusparseSpMM_bufferSize(m_cusparse_handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               matA,
                                               matB,
                                               &beta,
                                               matC,
                                               cuda_type<T>(),
                                               CUSPARSE_SPMM_ALG_DEFAULT,
                                               &m_spmm_buffer_size));
        CUDA_ERROR(cudaMalloc(&m_d_cusparse_spmm_buffer, m_spmm_buffer_size));
    }

    /**
     * @brief multiply the sparse matrix by a dense matrix. The function
     * performs the multiplication as
     * C = A*B
     * where A is the sparse matrix, B is a dense matrix, and the result is a
     * dense matrix C.
     * This method requires extra buffer allocation for cusparse. User may want
     * to call first alloce_multiply_buffer() (with the same parameters) first
     * to do the allocation and so timing this method will reflect the timing
     * for the multiplication operation only. Otherwise, this method calls
     * alloce_multiply_buffer() if it is not called before. Note that this
     * allocation happens only once and we then reuse it
     */
    __host__ void multiply(DenseMatrix<T>& B_mat,
                           DenseMatrix<T>& C_mat,
                           cudaStream_t    stream = 0)
    {
        assert(cols() == B_mat.rows());
        assert(rows() == C_mat.rows());
        assert(B_mat.cols() == C_mat.cols());

        T alpha;
        T beta;

        if constexpr (std::is_same_v<T, cuComplex>) {
            alpha = make_cuComplex(1.f, 1.f);
            beta  = make_cuComplex(0.f, 0.f);
        }

        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            alpha = make_cuDoubleComplex(1.0, 1.0);
            beta  = make_cuDoubleComplex(0.0, 0.0);
        }

        if constexpr (!std::is_same_v<T, cuComplex> &&
                      !std::is_same_v<T, cuDoubleComplex>) {
            alpha = T(1);
            beta  = T(0);
        }

        // A_mat.create_cusparse_handle();
        cusparseSpMatDescr_t matA = m_spdescr;
        cusparseDnMatDescr_t matB = B_mat.m_dendescr;
        cusparseDnMatDescr_t matC = C_mat.m_dendescr;

        CUSPARSE_ERROR(cusparseSetStream(m_cusparse_handle, stream));

        // allocate an external buffer if needed
        if (m_d_cusparse_spmm_buffer == nullptr) {
            alloc_multiply_buffer(B_mat, C_mat, stream);
        }


        // execute SpMM
        CUSPARSE_ERROR(cusparseSpMM(m_cusparse_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    matA,
                                    matB,
                                    &beta,
                                    matC,
                                    cuda_type<T>(),
                                    CUSPARSE_SPMM_ALG_DEFAULT,
                                    m_d_cusparse_spmm_buffer));
    }

    /**
     * @brief allocate the temp buffer needed for sparse matrix multiplication
     * by a dense vector
     */
    __host__ void alloc_multiply_buffer(T*           in_arr,
                                        T*           rt_arr,
                                        cudaStream_t stream = 0)
    {
        const T alpha = 1.0;
        const T beta  = 0.0;

        cusparseDnVecDescr_t vecx = NULL;
        cusparseDnVecDescr_t vecy = NULL;

        CUSPARSE_ERROR(
            cusparseCreateDnVec(&vecx, m_num_cols, in_arr, cuda_type<T>()));
        CUSPARSE_ERROR(
            cusparseCreateDnVec(&vecy, m_num_rows, rt_arr, cuda_type<T>()));

        CUSPARSE_ERROR(cusparseSpMV_bufferSize(m_cusparse_handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               m_spdescr,
                                               vecx,
                                               &beta,
                                               vecy,
                                               cuda_type<T>(),
                                               CUSPARSE_SPMV_ALG_DEFAULT,
                                               &m_spmv_buffer_size));
        CUSPARSE_ERROR(cusparseDestroyDnVec(vecx));
        CUSPARSE_ERROR(cusparseDestroyDnVec(vecy));

        CUDA_ERROR(cudaMalloc(&m_d_cusparse_spmv_buffer, m_spmv_buffer_size));
    }


    /**
     * @brief multiply the sparse matrix by a dense vector. The function
     * performs the multiplication as
     * Y = A*X
     * where A is the sparse matrix, X is a dense vector, and the result is a
     * dense vector Y.
     * This method requires extra buffer allocation for cusparse. User may want
     * to call first alloce_multiply_buffer() (with the same parameters) first
     * to do the allocation and so timing this method will reflect the timing
     * for the multiplication operation only. Otherwise, this method calls
     * alloce_multiply_buffer() if it is not called before. Note that this
     * allocation happens only once and we then reuse it
     * TODO allow this function to take a DenseMatrix instead that represent a
     * dense vector, i.e., one column with multiple rows
     */
    __host__ void multiply(T* in_arr, T* rt_arr, cudaStream_t stream = 0)
    {
        const T alpha = 1.0;
        const T beta  = 0.0;

        cusparseDnVecDescr_t vecx = NULL;
        cusparseDnVecDescr_t vecy = NULL;

        CUSPARSE_ERROR(
            cusparseCreateDnVec(&vecx, m_num_cols, in_arr, cuda_type<T>()));
        CUSPARSE_ERROR(
            cusparseCreateDnVec(&vecy, m_num_rows, rt_arr, cuda_type<T>()));

        CUSPARSE_ERROR(cusparseSetStream(m_cusparse_handle, stream));

        if (m_d_cusparse_spmv_buffer == nullptr) {
            alloc_multiply_buffer(in_arr, rt_arr, stream);
        }


        CUSPARSE_ERROR(cusparseSpMV(m_cusparse_handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    m_spdescr,
                                    vecx,
                                    &beta,
                                    vecy,
                                    cuda_type<T>(),
                                    CUSPARSE_SPMV_ALG_DEFAULT,
                                    m_d_cusparse_spmv_buffer));

        CUSPARSE_ERROR(cusparseDestroyDnVec(vecx));
        CUSPARSE_ERROR(cusparseDestroyDnVec(vecy));
    }

    /**
     * @brief multiply the sparse matrix by a dense matrix. The function
     * performs the multiplication as
     * C = A*B
     * where A is the sparse matrix, B is a dense matrix, and the result is a
     * dense matrix C.
     * This is similar to the multiply() function above but instead of extract
     * the columns for B and multiply them separately as sparse matrix dense
     * vector multiplication
     */
    __host__ void multiply_cw(const DenseMatrix<T>& B_mat,
                              DenseMatrix<T>&       C_mat,
                              cudaStream_t          stream = 0)
    {
        assert(cols() == B_mat.cols());
        assert(rows() == C_mat.rows());
        assert(B_mat.cols() == C_mat.cols());

        for (int i = 0; i < B_mat.m_num_cols; ++i) {
            multiply(B_mat.col_data(i), C_mat.col_data(i), stream);
        }
    }


    /**
     * @brief Convert/map this sparse matrix to Eigen sparse matrix. This is a
     * zero-copy conversion so Eigen sparse matrix will point to the same memory
     * as the host-side of this SparseMatrix
     */
    __host__ EigenSparseMatrix to_eigen()
    {
        return EigenSparseMatrix(
            rows(), cols(), non_zeros(), m_h_row_ptr, m_h_col_idx, m_h_val);
    }

    /**
     * @brief copy the matrix to Eigen SparseMatirx
     */
    __host__ Eigen::SparseMatrix<T, Eigen::RowMajor, IndexT> to_eigen_copy()
    {
        using TripletT = Eigen::Triplet<T>;

        std::vector<TripletT> triplets;
        triplets.reserve(non_zeros());
        for_each(
            [&](int r, int c, T& val) { triplets.push_back({r, c, val}); });

        Eigen::SparseMatrix<T, Eigen::RowMajor, IndexT> ret(rows(), cols());

        // std::sort(
        //     triplets.begin(), triplets.end(), [](TripletT& a, TripletT& b) {
        //         return (a.row() < b.row()) ||
        //                (a.row() == b.row() && a.col() < b.col());
        //     });

        ret.setFromTriplets(triplets.begin(), triplets.end());

        return ret;
    }

    /**
     * @brief solve the AX=B for X where X and B are all dense matrix and we
     * would solve it in a column wise manner
     */
    __host__ void solve(const DenseMatrix<T>& B_mat,
                        DenseMatrix<T>&       X_mat,
                        Solver                solver,
                        PermuteMethod         reorder,
                        cudaStream_t          stream = 0)
    {
        for (int i = 0; i < B_mat.cols(); ++i) {
            cusparse_linear_solver_wrapper(
                solver,
                reorder,
                m_cusolver_sphandle,
                B_mat.col_data(i, solver == Solver::LU ? HOST : DEVICE),
                X_mat.col_data(i, solver == Solver::LU ? HOST : DEVICE),
                stream);
        }
    }

    /**
     * @brief solve the Ax=b for x
     */
    __host__ void solve(const T*      B_arr,
                        T*            X_arr,
                        Solver        solver,
                        PermuteMethod reorder,
                        cudaStream_t  stream = 0)
    {
        cusparse_linear_solver_wrapper(
            solver, reorder, m_cusolver_sphandle, B_arr, X_arr, stream);
    }


    /* --- LOW LEVEL API --- */

    /**
     * @brief return a pointer to the host memory that holds the permutation
     */
    __host__ IndexT* get_h_permute()
    {
        return m_h_permute;
    }

    /**
     * @brief allocate all temp buffers needed for the solver low-level API
     */
    __host__ void permute_alloc(PermuteMethod reorder)
    {
        if (reorder == PermuteMethod::NONE) {
            return;
        }

        if (!m_reorder_allocated) {
            m_reorder_allocated = true;
            CUDA_ERROR(cudaMalloc((void**)&m_d_solver_val, m_nnz * sizeof(T)));
            CUDA_ERROR(cudaMalloc((void**)&m_d_solver_row_ptr,
                                  (m_num_rows + 1) * sizeof(IndexT)));
            CUDA_ERROR(cudaMalloc((void**)&m_d_solver_col_idx,
                                  m_nnz * sizeof(IndexT)));

            m_h_solver_row_ptr =
                (IndexT*)malloc((m_num_rows + 1) * sizeof(IndexT));
            m_h_solver_col_idx = (IndexT*)malloc(m_nnz * sizeof(IndexT));

            m_h_permute = (IndexT*)malloc(m_num_rows * sizeof(IndexT));
            CUDA_ERROR(
                cudaMalloc((void**)&m_d_permute, m_num_rows * sizeof(IndexT)));

            m_h_permute_map =
                static_cast<IndexT*>(malloc(m_nnz * sizeof(IndexT)));

            CUDA_ERROR(
                cudaMalloc((void**)&m_d_permute_map, m_nnz * sizeof(IndexT)));

            CUDA_ERROR(
                cudaMalloc((void**)&m_d_solver_x, m_num_cols * sizeof(T)));
            CUDA_ERROR(
                cudaMalloc((void**)&m_d_solver_b, m_num_rows * sizeof(T)));
        }
        std::memcpy(
            m_h_solver_row_ptr, m_h_row_ptr, (m_num_rows + 1) * sizeof(IndexT));
        std::memcpy(m_h_solver_col_idx, m_h_col_idx, m_nnz * sizeof(IndexT));
    }

    /**
     * @brief The lower level api of reordering. Specify the reordering type or
     * simply NONE for no reordering. This should be called at the beginning of
     * the solving process. Any other function call order would be undefined.
     * @param reorder: the reorder method applied.
     */
    __host__ void permute(RXMeshStatic& rx, PermuteMethod reorder)
    {
        permute_alloc(reorder);

        if (reorder == PermuteMethod::NONE) {
            RXMESH_WARN(
                "SparseMatrix::permute() No reordering is specified. Continue "
                "without reordering!");
            m_use_reorder = false;
            return;
        }

        m_use_reorder = true;


        if (reorder == PermuteMethod::SYMRCM) {
            CUSOLVER_ERROR(cusolverSpXcsrsymrcmHost(m_cusolver_sphandle,
                                                    m_num_rows,
                                                    m_nnz,
                                                    m_descr,
                                                    m_h_solver_row_ptr,
                                                    m_h_solver_col_idx,
                                                    m_h_permute));
        } else if (reorder == PermuteMethod::SYMAMD) {
            CUSOLVER_ERROR(cusolverSpXcsrsymamdHost(m_cusolver_sphandle,
                                                    m_num_rows,
                                                    m_nnz,
                                                    m_descr,
                                                    m_h_solver_row_ptr,
                                                    m_h_solver_col_idx,
                                                    m_h_permute));
        } else if (reorder == PermuteMethod::NSTDIS) {
            CUSOLVER_ERROR(cusolverSpXcsrmetisndHost(m_cusolver_sphandle,
                                                     m_num_rows,
                                                     m_nnz,
                                                     m_descr,
                                                     m_h_solver_row_ptr,
                                                     m_h_solver_col_idx,
                                                     NULL,
                                                     m_h_permute));
        } else if (reorder == PermuteMethod::GPUMGND) {
            mgnd_permute(rx, m_h_permute);

        } else if (reorder == PermuteMethod::GPUND) {
            nd_permute(rx, m_h_permute);
        } else {
            RXMESH_ERROR("SparseMatrix::permute() incompatible reorder method");
        }


        assert(is_unique_permutation(m_num_rows, m_h_permute));

        // copy permutation to the device
        CUDA_ERROR(cudaMemcpyAsync(m_d_permute,
                                   m_h_permute,
                                   m_num_rows * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));

        // working space for permutation: B = A*Q*A^T
        // the permutation for matrix A which works only for the col and row
        // indices, the val will be done on device with the m_d_permute_map
        // only on the device since we don't need to access the permuted val on
        // the host at all
#pragma omp parallel for
        for (int j = 0; j < m_nnz; j++) {
            m_h_permute_map[j] = j;
        }

        size_t size_perm       = 0;
        void*  perm_buffer_cpu = NULL;

        CUSOLVER_ERROR(cusolverSpXcsrperm_bufferSizeHost(m_cusolver_sphandle,
                                                         m_num_rows,
                                                         m_num_cols,
                                                         m_nnz,
                                                         m_descr,
                                                         m_h_solver_row_ptr,
                                                         m_h_solver_col_idx,
                                                         m_h_permute,
                                                         m_h_permute,
                                                         &size_perm));

        perm_buffer_cpu = (void*)malloc(sizeof(char) * size_perm);

        // permute the matrix
        CUSOLVER_ERROR(cusolverSpXcsrpermHost(m_cusolver_sphandle,
                                              m_num_rows,
                                              m_num_cols,
                                              m_nnz,
                                              m_descr,
                                              m_h_solver_row_ptr,
                                              m_h_solver_col_idx,
                                              m_h_permute,
                                              m_h_permute,
                                              m_h_permute_map,
                                              perm_buffer_cpu));

        // copy the permute csr from the device
        CUDA_ERROR(cudaMemcpyAsync(m_d_solver_row_ptr,
                                   m_h_solver_row_ptr,
                                   (m_num_rows + 1) * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpyAsync(m_d_solver_col_idx,
                                   m_h_solver_col_idx,
                                   m_nnz * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));

        // do the permutation for val on device
        CUDA_ERROR(cudaMemcpyAsync(m_d_permute_map,
                                   m_h_permute_map,
                                   m_nnz * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));
        permute_gather(m_d_permute_map, m_d_val, m_d_solver_val, m_nnz);


        free(perm_buffer_cpu);
    }

    /**
     * @brief The lower level api of matrix analysis. Generating a member value
     * of type csrcholInfo_t for cucolver.
     */
    __host__ void analyze_pattern(Solver solver)
    {
        m_current_solver = solver;

        if (!m_use_reorder) {
            m_d_solver_row_ptr = m_d_row_ptr;
            m_d_solver_col_idx = m_d_col_idx;
            m_d_solver_val     = m_d_val;
        }

        if (solver == Solver::CHOL) {
            CUSOLVER_ERROR(cusolverSpXcsrcholAnalysis(m_cusolver_sphandle,
                                                      m_num_rows,
                                                      m_nnz,
                                                      m_descr,
                                                      m_d_solver_row_ptr,
                                                      m_d_solver_col_idx,
                                                      m_chol_info));
        } else if (solver == Solver::QR) {
            CUSOLVER_ERROR(cusolverSpXcsrqrAnalysis(m_cusolver_sphandle,
                                                    m_num_rows,
                                                    m_num_cols,
                                                    m_nnz,
                                                    m_descr,
                                                    m_d_solver_row_ptr,
                                                    m_d_solver_col_idx,
                                                    m_qr_info));
        } else {
            RXMESH_ERROR(
                "SparseMatrix::analyze_pattern() incompatible solver with "
                "analyze_pattern method");
        }
    }

    /**
     * @brief The lower level api of matrix factorization buffer calculation and
     * allocation. The buffer is a member variable.
     */
    __host__ void post_analyze_alloc(Solver solver)
    {
        if (solver != m_current_solver) {
            RXMESH_ERROR(
                "SparseMatrix::post_analyze_alloc() input solver is different "
                "than current solver used in analyze_pattern()");
            return;
        }

        m_internalDataInBytes = 0;
        m_workspaceInBytes    = 0;

        GPU_FREE(m_solver_buffer);

        if (solver == Solver::CHOL) {

            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(
                    cusolverSpScsrcholBufferInfo(m_cusolver_sphandle,
                                                 m_num_rows,
                                                 m_nnz,
                                                 m_descr,
                                                 m_d_solver_val,
                                                 m_d_solver_row_ptr,
                                                 m_d_solver_col_idx,
                                                 m_chol_info,
                                                 &m_internalDataInBytes,
                                                 &m_workspaceInBytes));
            }

            if constexpr (std::is_same_v<T, cuComplex>) {
                CUSOLVER_ERROR(
                    cusolverSpCcsrcholBufferInfo(m_cusolver_sphandle,
                                                 m_num_rows,
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
                CUSOLVER_ERROR(
                    cusolverSpDcsrcholBufferInfo(m_cusolver_sphandle,
                                                 m_num_rows,
                                                 m_nnz,
                                                 m_descr,
                                                 m_d_solver_val,
                                                 m_d_solver_row_ptr,
                                                 m_d_solver_col_idx,
                                                 m_chol_info,
                                                 &m_internalDataInBytes,
                                                 &m_workspaceInBytes));
            }

            if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                CUSOLVER_ERROR(
                    cusolverSpZcsrcholBufferInfo(m_cusolver_sphandle,
                                                 m_num_rows,
                                                 m_nnz,
                                                 m_descr,
                                                 m_d_solver_val,
                                                 m_d_solver_row_ptr,
                                                 m_d_solver_col_idx,
                                                 m_chol_info,
                                                 &m_internalDataInBytes,
                                                 &m_workspaceInBytes));
            }
        } else if (solver == Solver::QR) {
            if constexpr (std::is_same_v<T, float>) {
                float mu = 0.f;
                CUSOLVER_ERROR(
                    cusolverSpScsrqrBufferInfo(m_cusolver_sphandle,
                                               m_num_rows,
                                               m_num_cols,
                                               m_nnz,
                                               m_descr,
                                               m_d_solver_val,
                                               m_d_solver_row_ptr,
                                               m_d_solver_col_idx,
                                               m_qr_info,
                                               &m_internalDataInBytes,
                                               &m_workspaceInBytes));

                CUSOLVER_ERROR(cusolverSpScsrqrSetup(m_cusolver_sphandle,
                                                     m_num_rows,
                                                     m_num_cols,
                                                     m_nnz,
                                                     m_descr,
                                                     m_d_solver_val,
                                                     m_d_solver_row_ptr,
                                                     m_d_solver_col_idx,
                                                     mu,
                                                     m_qr_info));
            }

            if constexpr (std::is_same_v<T, cuComplex>) {
                cuComplex mu = make_cuComplex(0.f, 0.f);
                CUSOLVER_ERROR(
                    cusolverSpCcsrqrBufferInfo(m_cusolver_sphandle,
                                               m_num_rows,
                                               m_num_cols,
                                               m_nnz,
                                               m_descr,
                                               m_d_solver_val,
                                               m_d_solver_row_ptr,
                                               m_d_solver_col_idx,
                                               m_qr_info,
                                               &m_internalDataInBytes,
                                               &m_workspaceInBytes));

                CUSOLVER_ERROR(cusolverSpCcsrqrSetup(m_cusolver_sphandle,
                                                     m_num_rows,
                                                     m_num_cols,
                                                     m_nnz,
                                                     m_descr,
                                                     m_d_solver_val,
                                                     m_d_solver_row_ptr,
                                                     m_d_solver_col_idx,
                                                     mu,
                                                     m_qr_info));
            }

            if constexpr (std::is_same_v<T, double>) {
                double mu = 0.f;
                CUSOLVER_ERROR(
                    cusolverSpDcsrqrBufferInfo(m_cusolver_sphandle,
                                               m_num_rows,
                                               m_num_cols,
                                               m_nnz,
                                               m_descr,
                                               m_d_solver_val,
                                               m_d_solver_row_ptr,
                                               m_d_solver_col_idx,
                                               m_qr_info,
                                               &m_internalDataInBytes,
                                               &m_workspaceInBytes));

                CUSOLVER_ERROR(cusolverSpDcsrqrSetup(m_cusolver_sphandle,
                                                     m_num_rows,
                                                     m_num_cols,
                                                     m_nnz,
                                                     m_descr,
                                                     m_d_solver_val,
                                                     m_d_solver_row_ptr,
                                                     m_d_solver_col_idx,
                                                     mu,
                                                     m_qr_info));
            }

            if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                cuDoubleComplex mu = make_cuDoubleComplex(0.0, 0.0);
                CUSOLVER_ERROR(
                    cusolverSpZcsrqrBufferInfo(m_cusolver_sphandle,
                                               m_num_rows,
                                               m_num_cols,
                                               m_nnz,
                                               m_descr,
                                               m_d_solver_val,
                                               m_d_solver_row_ptr,
                                               m_d_solver_col_idx,
                                               m_qr_info,
                                               &m_internalDataInBytes,
                                               &m_workspaceInBytes));

                CUSOLVER_ERROR(cusolverSpZcsrqrSetup(m_cusolver_sphandle,
                                                     m_num_rows,
                                                     m_num_cols,
                                                     m_nnz,
                                                     m_descr,
                                                     m_d_solver_val,
                                                     m_d_solver_row_ptr,
                                                     m_d_solver_col_idx,
                                                     mu,
                                                     m_qr_info));
            }
        } else {
            RXMESH_ERROR(
                "SparseMatrix::post_analyze_alloc() incompatible solver with "
                "post_analyze_alloc method");
            return;
        }

        RXMESH_INFO(
            "post_analyze_alloc() internalDataInBytes= {}, workspaceInBytes= "
            "{}",
            m_internalDataInBytes,
            m_workspaceInBytes);
        CUDA_ERROR(cudaMalloc((void**)&m_solver_buffer, m_workspaceInBytes));
    }


    /**
     * @brief The lower level api of matrix factorization and save the
     * factorization result in to the buffer.
     */
    __host__ void factorize(Solver solver)
    {
        if (solver != m_current_solver) {
            RXMESH_ERROR(
                "SparseMatrix::post_analyze_alloc() input solver is different "
                "than current solver used in analyze_pattern()");
            return;
        }

        if (solver == Solver::CHOL) {
            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(cusolverSpScsrcholFactor(m_cusolver_sphandle,
                                                        m_num_rows,
                                                        m_nnz,
                                                        m_descr,
                                                        m_d_solver_val,
                                                        m_d_solver_row_ptr,
                                                        m_d_solver_col_idx,
                                                        m_chol_info,
                                                        m_solver_buffer));
            }

            if constexpr (std::is_same_v<T, cuComplex>) {
                CUSOLVER_ERROR(cusolverSpCcsrcholFactor(m_cusolver_sphandle,
                                                        m_num_rows,
                                                        m_nnz,
                                                        m_descr,
                                                        m_d_solver_val,
                                                        m_d_solver_row_ptr,
                                                        m_d_solver_col_idx,
                                                        m_chol_info,
                                                        m_solver_buffer));
            }
            if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_ERROR(cusolverSpDcsrcholFactor(m_cusolver_sphandle,
                                                        m_num_rows,
                                                        m_nnz,
                                                        m_descr,
                                                        m_d_solver_val,
                                                        m_d_solver_row_ptr,
                                                        m_d_solver_col_idx,
                                                        m_chol_info,
                                                        m_solver_buffer));
            }
            if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                CUSOLVER_ERROR(cusolverSpZcsrcholFactor(m_cusolver_sphandle,
                                                        m_num_rows,
                                                        m_nnz,
                                                        m_descr,
                                                        m_d_solver_val,
                                                        m_d_solver_row_ptr,
                                                        m_d_solver_col_idx,
                                                        m_chol_info,
                                                        m_solver_buffer));
            }
        } else if (solver == Solver::QR) {
            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(cusolverSpScsrqrFactor(m_cusolver_sphandle,
                                                      m_num_rows,
                                                      m_num_cols,
                                                      m_nnz,
                                                      nullptr,
                                                      nullptr,
                                                      m_qr_info,
                                                      m_solver_buffer));
            }

            if constexpr (std::is_same_v<T, cuComplex>) {
                CUSOLVER_ERROR(cusolverSpCcsrqrFactor(m_cusolver_sphandle,
                                                      m_num_rows,
                                                      m_num_cols,
                                                      m_nnz,
                                                      nullptr,
                                                      nullptr,
                                                      m_qr_info,
                                                      m_solver_buffer));
            }
            if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_ERROR(cusolverSpDcsrqrFactor(m_cusolver_sphandle,
                                                      m_num_rows,
                                                      m_num_cols,
                                                      m_nnz,
                                                      nullptr,
                                                      nullptr,
                                                      m_qr_info,
                                                      m_solver_buffer));
            }
            if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                CUSOLVER_ERROR(cusolverSpZcsrqrFactor(m_cusolver_sphandle,
                                                      m_num_rows,
                                                      m_num_cols,
                                                      m_nnz,
                                                      nullptr,
                                                      nullptr,
                                                      m_qr_info,
                                                      m_solver_buffer));
            }

        } else {
            RXMESH_ERROR(
                "SparseMatrix::factorize() incompatible solver with factorize "
                "method");
            return;
        }

        double tol = 1.0e-8;
        int    singularity;

        if (solver == Solver::CHOL) {
            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(cusolverSpScsrcholZeroPivot(
                    m_cusolver_sphandle, m_chol_info, tol, &singularity));
            }
            if constexpr (std::is_same_v<T, cuComplex>) {
                CUSOLVER_ERROR(cusolverSpCcsrcholZeroPivot(
                    m_cusolver_sphandle, m_chol_info, tol, &singularity));
            }
            if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_ERROR(cusolverSpDcsrcholZeroPivot(
                    m_cusolver_sphandle, m_chol_info, tol, &singularity));
            }
            if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                CUSOLVER_ERROR(cusolverSpZcsrcholZeroPivot(
                    m_cusolver_sphandle, m_chol_info, tol, &singularity));
            }
        } else if (solver == Solver::QR) {

            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(cusolverSpScsrqrZeroPivot(
                    m_cusolver_sphandle, m_qr_info, tol, &singularity));
            }
            if constexpr (std::is_same_v<T, cuComplex>) {
                CUSOLVER_ERROR(cusolverSpCcsrqrZeroPivot(
                    m_cusolver_sphandle, m_qr_info, tol, &singularity));
            }
            if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_ERROR(cusolverSpDcsrqrZeroPivot(
                    m_cusolver_sphandle, m_qr_info, tol, &singularity));
            }
            if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                CUSOLVER_ERROR(cusolverSpZcsrqrZeroPivot(
                    m_cusolver_sphandle, m_qr_info, tol, &singularity));
            }


        } else {
            RXMESH_ERROR(
                "SparseMatrix::factorize() incompatible solver with factorize "
                "method");
            return;
        }

        if (0 <= singularity) {
            RXMESH_WARN(
                "SparseMatrix::factorize() The matrix is singular at row {} "
                "under tol ({})",
                singularity,
                tol);
        }
    }

    /**
     * @brief Call all the necessary functions to permute and factorize the
     * sparse matrix before calling the solve() method below. After calling this
     * pre_solve(), solver() can be called with multiple right hand sides
     */
    __host__ void pre_solve(RXMeshStatic& rx,
                            Solver        solver,
                            PermuteMethod reorder = PermuteMethod::NSTDIS)
    {
        if (solver != Solver::CHOL && solver != Solver::QR) {
            RXMESH_WARN(
                "SparseMatrix::pre_solve() the low-level API only works for "
                "Cholesky and QR solvers");
            return;
        }
        m_current_solver = solver;

        permute_alloc(reorder);
        permute(rx, reorder);
        analyze_pattern(solver);
        post_analyze_alloc(solver);
        factorize(solver);
    }

    /**
     * @brief The lower level api of solving the linear system after using
     * factorization. The format follows Ax=b to solve x, where A is this sparse
     * matrix, x and b are device array. As long as A doesn't change. This
     * function could be called for many different b and x.
     * @param B_mat: right hand side
     * @param X_mat: output solution
     */
    __host__ void solve(DenseMatrix<T>& B_mat,
                        DenseMatrix<T>& X_mat,
                        cudaStream_t    stream = NULL)
    {
        CUSOLVER_ERROR(cusolverSpSetStream(m_cusolver_sphandle, stream));
        for (int i = 0; i < B_mat.cols(); ++i) {
            solve(B_mat.col_data(i), X_mat.col_data(i));
        }
    }

    /**
     * @brief The lower level api of solving the linear system after using
     * factorization. The format follows Ax=b to solve x, where A is the sparse
     * matrix, x and b are device array. As long as A doesn't change. This
     * function could be called for many different b and x.
     * @param d_b: right hand side
     * @param d_x: output solution
     */
    __host__ void solve(T* d_b, T* d_x)
    {
        T* d_solver_b;
        T* d_solver_x;

        if (m_use_reorder) {
            // purmute b and x
            d_solver_b = m_d_solver_b;
            d_solver_x = m_d_solver_x;
            permute_gather(m_d_permute, d_b, d_solver_b, m_num_rows);
            permute_gather(m_d_permute, d_x, d_solver_x, m_num_rows);
        } else {
            d_solver_b = d_b;
            d_solver_x = d_x;
        }

        if (m_current_solver == Solver::CHOL) {

            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(cusolverSpScsrcholSolve(m_cusolver_sphandle,
                                                       m_num_rows,
                                                       d_solver_b,
                                                       d_solver_x,
                                                       m_chol_info,
                                                       m_solver_buffer));
            }

            if constexpr (std::is_same_v<T, cuComplex>) {
                CUSOLVER_ERROR(cusolverSpCcsrcholSolve(m_cusolver_sphandle,
                                                       m_num_rows,
                                                       d_solver_b,
                                                       d_solver_x,
                                                       m_chol_info,
                                                       m_solver_buffer));
            }

            if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_ERROR(cusolverSpDcsrcholSolve(m_cusolver_sphandle,
                                                       m_num_rows,
                                                       d_solver_b,
                                                       d_solver_x,
                                                       m_chol_info,
                                                       m_solver_buffer));
            }
            if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                CUSOLVER_ERROR(cusolverSpZcsrcholSolve(m_cusolver_sphandle,
                                                       m_num_rows,
                                                       d_solver_b,
                                                       d_solver_x,
                                                       m_chol_info,
                                                       m_solver_buffer));
            }
        } else if (m_current_solver == Solver::QR) {

            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(cusolverSpScsrqrSolve(m_cusolver_sphandle,
                                                     m_num_rows,
                                                     m_num_cols,
                                                     d_solver_b,
                                                     d_solver_x,
                                                     m_qr_info,
                                                     m_solver_buffer));
            }

            if constexpr (std::is_same_v<T, cuComplex>) {
                CUSOLVER_ERROR(cusolverSpCcsrqrSolve(m_cusolver_sphandle,
                                                     m_num_rows,
                                                     m_num_cols,
                                                     d_solver_b,
                                                     d_solver_x,
                                                     m_qr_info,
                                                     m_solver_buffer));
            }

            if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_ERROR(cusolverSpDcsrqrSolve(m_cusolver_sphandle,
                                                     m_num_rows,
                                                     m_num_cols,
                                                     d_solver_b,
                                                     d_solver_x,
                                                     m_qr_info,
                                                     m_solver_buffer));
            }
            if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                CUSOLVER_ERROR(cusolverSpZcsrqrSolve(m_cusolver_sphandle,
                                                     m_num_rows,
                                                     m_num_cols,
                                                     d_solver_b,
                                                     d_solver_x,
                                                     m_qr_info,
                                                     m_solver_buffer));
            }


        } else {
            RXMESH_ERROR(
                "SparseMatrix::solve() the low-level API only works for "
                "Cholesky and QR solvers");
            return;
        }

        if (m_use_reorder) {
            permute_scatter(m_d_permute, d_solver_x, d_x, m_num_rows);
        }
    }


   protected:
    __host__ void release(locationT location)
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
            GPU_FREE(m_d_row_ptr);
            GPU_FREE(m_d_col_idx);
            m_allocated = m_allocated & (~DEVICE);
        }
    }

    __host__ void allocate(locationT location)
    {
        if ((location & HOST) == HOST) {
            release(HOST);

            m_h_val = static_cast<T*>(malloc(m_nnz * sizeof(T)));
            m_h_row_ptr =
                static_cast<IndexT*>(malloc((m_num_rows + 1) * sizeof(IndexT)));
            m_h_col_idx = static_cast<IndexT*>(malloc(m_nnz * sizeof(IndexT)));

            m_allocated = m_allocated | HOST;
        }

        if ((location & DEVICE) == DEVICE) {
            release(DEVICE);

            CUDA_ERROR(cudaMalloc((void**)&m_d_val, m_nnz * sizeof(T)));
            CUDA_ERROR(cudaMalloc((void**)&m_d_row_ptr,
                                  (m_num_rows + 1) * sizeof(IndexT)));
            CUDA_ERROR(
                cudaMalloc((void**)&m_d_col_idx, m_nnz * sizeof(IndexT)));

            m_allocated = m_allocated | DEVICE;
        }
    }

    /**
     * @brief wrapper for cuSolver API for solving linear systems using cuSolver
     * High-level API
     */
    __host__ void cusparse_linear_solver_wrapper(const Solver        solver,
                                                 const PermuteMethod reorder,
                                                 cusolverSpHandle_t  handle,
                                                 const T*            d_b,
                                                 T*                  d_x,
                                                 cudaStream_t        stream)
    {
        CUSOLVER_ERROR(cusolverSpSetStream(handle, stream));

        double tol = 1.e-12;

        // -1 if A is invertible under tol.
        int singularity = 0;

        if (solver == Solver::CHOL) {
            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(cusolverSpScsrlsvchol(handle,
                                                     rows(),
                                                     non_zeros(),
                                                     m_descr,
                                                     m_d_val,
                                                     m_d_row_ptr,
                                                     m_d_col_idx,
                                                     d_b,
                                                     tol,
                                                     reorder_to_int(reorder),
                                                     d_x,
                                                     &singularity));
            }

            if constexpr (std::is_same_v<T, cuComplex>) {
                CUSOLVER_ERROR(cusolverSpCcsrlsvchol(handle,
                                                     rows(),
                                                     non_zeros(),
                                                     m_descr,
                                                     m_d_val,
                                                     m_d_row_ptr,
                                                     m_d_col_idx,
                                                     d_b,
                                                     tol,
                                                     reorder_to_int(reorder),
                                                     d_x,
                                                     &singularity));
            }

            if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_ERROR(cusolverSpDcsrlsvchol(handle,
                                                     rows(),
                                                     non_zeros(),
                                                     m_descr,
                                                     m_d_val,
                                                     m_d_row_ptr,
                                                     m_d_col_idx,
                                                     d_b,
                                                     tol,
                                                     reorder_to_int(reorder),
                                                     d_x,
                                                     &singularity));
            }
            if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                CUSOLVER_ERROR(cusolverSpZcsrlsvchol(handle,
                                                     rows(),
                                                     non_zeros(),
                                                     m_d_val,
                                                     m_descr,
                                                     m_d_val,
                                                     m_d_row_ptr,
                                                     m_d_col_idx,
                                                     d_b,
                                                     tol,
                                                     reorder_to_int(reorder),
                                                     d_x,
                                                     &singularity));
            }

        } else if (solver == Solver::QR) {
            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(cusolverSpScsrlsvqr(handle,
                                                   rows(),
                                                   non_zeros(),
                                                   m_descr,
                                                   m_d_val,
                                                   m_d_row_ptr,
                                                   m_d_col_idx,
                                                   d_b,
                                                   tol,
                                                   reorder_to_int(reorder),
                                                   d_x,
                                                   &singularity));
            }

            if constexpr (std::is_same_v<T, cuComplex>) {
                CUSOLVER_ERROR(cusolverSpCcsrlsvqr(handle,
                                                   rows(),
                                                   non_zeros(),
                                                   m_descr,
                                                   m_d_val,
                                                   m_d_row_ptr,
                                                   m_d_col_idx,
                                                   d_b,
                                                   tol,
                                                   reorder_to_int(reorder),
                                                   d_x,
                                                   &singularity));
            }

            if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_ERROR(cusolverSpDcsrlsvqr(handle,
                                                   rows(),
                                                   non_zeros(),
                                                   m_descr,
                                                   m_d_val,
                                                   m_d_row_ptr,
                                                   m_d_col_idx,
                                                   d_b,
                                                   tol,
                                                   reorder_to_int(reorder),
                                                   d_x,
                                                   &singularity));
            }
            if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                CUSOLVER_ERROR(cusolverSpZcsrlsvqr(handle,
                                                   rows(),
                                                   non_zeros(),
                                                   m_descr,
                                                   m_d_val,
                                                   m_d_row_ptr,
                                                   m_d_col_idx,
                                                   d_b,
                                                   tol,
                                                   reorder_to_int(reorder),
                                                   d_x,
                                                   &singularity));
            }
        } else if (solver == Solver::LU) {
            RXMESH_WARN(
                "SparseMatrix::cusparse_linear_solver_wrapper() LU Solver is "
                "run on the host. Make sure your data resides on the host "
                "before calling the solver");

            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_ERROR(cusolverSpScsrlsvluHost(handle,
                                                       rows(),
                                                       non_zeros(),
                                                       m_descr,
                                                       m_h_val,
                                                       m_h_row_ptr,
                                                       m_h_col_idx,
                                                       d_b,
                                                       tol,
                                                       reorder_to_int(reorder),
                                                       d_x,
                                                       &singularity));
            }

            if constexpr (std::is_same_v<T, cuComplex>) {
                CUSOLVER_ERROR(cusolverSpCcsrlsvluHost(handle,
                                                       rows(),
                                                       non_zeros(),
                                                       m_descr,
                                                       m_h_val,
                                                       m_h_row_ptr,
                                                       m_h_col_idx,
                                                       d_b,
                                                       tol,
                                                       reorder_to_int(reorder),
                                                       d_x,
                                                       &singularity));
            }

            if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_ERROR(cusolverSpDcsrlsvluHost(handle,
                                                       rows(),
                                                       non_zeros(),
                                                       m_descr,
                                                       m_h_val,
                                                       m_h_row_ptr,
                                                       m_h_col_idx,
                                                       d_b,
                                                       tol,
                                                       reorder_to_int(reorder),
                                                       d_x,
                                                       &singularity));
            }
            if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                CUSOLVER_ERROR(cusolverSpZcsrlsvluHost(handle,
                                                       rows(),
                                                       non_zeros(),
                                                       m_descr,
                                                       m_h_val,
                                                       m_h_row_ptr,
                                                       m_h_col_idx,
                                                       d_b,
                                                       tol,
                                                       reorder_to_int(reorder),
                                                       d_x,
                                                       &singularity));
            }
        } else {
            RXMESH_ERROR(
                "SparseMatrix::cusparse_linear_solver_wrapper() Unsupported "
                "solver type.");
        }


        if (0 <= singularity) {
            RXMESH_WARN(
                "SparseMatrix::cusparse_linear_solver_wrapper() The matrix is "
                "singular at row {} under tol ({})",
                singularity,
                tol);
        }
    }

    int reorder_to_int(const PermuteMethod& reorder) const
    {
        switch (reorder) {
            case PermuteMethod::NONE:
                return 0;
            case PermuteMethod::SYMRCM:
                return 1;
            case PermuteMethod::SYMAMD:
                return 2;
            case PermuteMethod::NSTDIS:
                return 3;
            default: {
                RXMESH_ERROR("reorder_to_int() unknown input");
                return 0;
            }
        }
    }

    __host__ void permute_scatter(IndexT* d_p, T* d_in, T* d_out, IndexT size)
    {
        // d_out[d_p[i]] = d_in[i]
        thrust::device_ptr<IndexT> t_p(d_p);
        thrust::device_ptr<T>      t_i(d_in);
        thrust::device_ptr<T>      t_o(d_out);

        thrust::scatter(thrust::device, t_i, t_i + size, t_p, t_o);
    }

    __host__ void permute_gather(IndexT* d_p, T* d_in, T* d_out, IndexT size)
    {
        // d_out[i] = d_in[d_p[i]]
        thrust::device_ptr<IndexT> t_p(d_p);
        thrust::device_ptr<T>      t_i(d_in);
        thrust::device_ptr<T>      t_o(d_out);

        thrust::gather(thrust::device, t_p, t_p + size, t_i, t_o);
    }

    const Context        m_context;
    cusparseHandle_t     m_cusparse_handle;
    cusolverSpHandle_t   m_cusolver_sphandle;
    cusparseSpMatDescr_t m_spdescr;
    cusparseMatDescr_t   m_descr;

    int m_replicate;

    IndexT m_num_rows;
    IndexT m_num_cols;
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
    void*         m_solver_buffer;
    csrqrInfo_t   m_qr_info;

    // purmutation array
    IndexT* m_h_permute;
    IndexT* m_d_permute;

    // CSR matrix for solving only
    // equal to the original matrix if not permuted
    // only allocated as a new CSR matrix if permuted
    bool    m_reorder_allocated;
    IndexT* m_d_solver_row_ptr;
    IndexT* m_d_solver_col_idx;
    T*      m_d_solver_val;

    // caching user's solver that is used in pre_solve
    Solver m_current_solver;


    IndexT* m_h_solver_row_ptr;
    IndexT* m_h_solver_col_idx;

    IndexT* m_h_permute_map;
    IndexT* m_d_permute_map;

    T* m_d_solver_b;
    T* m_d_solver_x;

    void* m_d_cusparse_spmm_buffer;
    void* m_d_cusparse_spmv_buffer;

    // flags
    bool      m_use_reorder;
    locationT m_allocated;
};

}  // namespace rxmesh