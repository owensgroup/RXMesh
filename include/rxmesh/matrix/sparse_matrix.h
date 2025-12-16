#pragma once
#include <algorithm>
#include "cusolverSp.h"
#include "cusparse.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/block_shape.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix_kernels.cuh"


#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

namespace rxmesh {

/**
 * @brief Sparse matrix that represent the mesh connectivity, e.g., for VV, it
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
    template <typename U>
    friend class VCycle;

    template <typename U0, int U1, typename U2>
    friend struct GaussNetwtonSolver;

    using IndexT = int;

    using Type = T;

    using EigenSparseMatrix =
        Eigen::Map<Eigen::SparseMatrix<T, Eigen::RowMajor, IndexT>>;

    SparseMatrix()
        : m_d_row_ptr(nullptr),
          m_d_col_idx(nullptr),
          m_d_val(nullptr),
          m_d_row_acc(nullptr),
          m_h_row_ptr(nullptr),
          m_h_col_idx(nullptr),
          m_h_val(nullptr),
          m_num_rows(0),
          m_num_cols(0),
          m_nnz(0),
          m_context(Context()),
          m_cusparse_handle(NULL),
          m_block_shape(BlockShape(1, 1)),
          m_spdescr(NULL),
          m_spmm_buffer_size(0),
          m_spmv_buffer_size(0),
          m_d_cusparse_spmm_buffer(nullptr),
          m_d_cusparse_spmv_buffer(nullptr),
          m_allocated(LOCATION_NONE),
          m_is_user_managed(false),
          m_op(Op::INVALID),
          m_d_cub_temp_storage(nullptr),
          m_cub_temp_storage_bytes(0),
          m_capacity_factor(1.0f),
          m_extra_nnz_entires(0),
          m_trans_buffer_size(0),
          m_trans_buffer(nullptr)
    {
    }

    /**
     * @brief Constructor using specific mesh query
     */
    SparseMatrix(const RXMeshStatic& rx, Op op = Op::VV)
        : SparseMatrix(rx, 1.f, 0, op, BlockShape(1, 1), true) {};


    /**
     * @brief Construct the matrix using user-managed buffers (i.e., buffers
     * that are allocated by the user and will be freed by the user)
     * @param num_rows number of row in the matrix
     * @param num_cols number of columns in the matrix
     * @param nnz number of non-zero values in the matrix
     * @param d_row_ptr device pointer to the row pointer
     * @param d_col_idx device pointer to the column index
     * @param d_val device point to the value pointer
     * @param h_row_ptr host pointer to the row pointer
     * @param h_col_idx host pointer to the column index
     * @param h_val host point to the value pointer
     */
    SparseMatrix(IndexT  num_rows,
                 IndexT  num_cols,
                 IndexT  nnz,
                 IndexT* d_row_ptr,
                 IndexT* d_col_idx,
                 T*      d_val,
                 IndexT* h_row_ptr,
                 IndexT* h_col_idx,
                 T*      h_val)
        : SparseMatrix()
    {
        m_block_shape = BlockShape(1, 1);

        m_is_user_managed = true;

        m_d_row_ptr = d_row_ptr;
        m_d_col_idx = d_col_idx;
        m_d_val     = d_val;

        m_h_row_ptr = h_row_ptr;
        m_h_col_idx = h_col_idx;
        m_h_val     = h_val;

        m_num_rows = num_rows;
        m_num_cols = num_cols;
        m_nnz      = nnz;

        m_allocated = m_allocated | DEVICE;
        m_allocated = m_allocated | HOST;

        // create cusparse matrix
        init_cusparse(*this);
#ifndef NDEBUG
        check_repeated_indices();
#endif
        init_cudss(*this);
    }

   protected:
    SparseMatrix(const RXMeshStatic& rx,
                 const float         capacity_factor,
                 const int           extra_nnz_entries,
                 Op                  op,
                 BlockShape          block_shape,
                 bool                add_diagonal)
        : m_d_row_ptr(nullptr),
          m_d_col_idx(nullptr),
          m_d_val(nullptr),
          m_d_row_acc(nullptr),
          m_h_row_ptr(nullptr),
          m_h_col_idx(nullptr),
          m_h_val(nullptr),
          m_num_rows(0),
          m_num_cols(0),
          m_nnz(0),
          m_context(rx.get_context()),
          m_cusparse_handle(NULL),
          m_block_shape(block_shape),
          m_spdescr(NULL),
          m_spmm_buffer_size(0),
          m_spmv_buffer_size(0),
          m_d_cusparse_spmm_buffer(nullptr),
          m_d_cusparse_spmv_buffer(nullptr),
          m_allocated(LOCATION_NONE),
          m_is_user_managed(false),
          m_op(op),
          m_d_cub_temp_storage(nullptr),
          m_cub_temp_storage_bytes(0),
          m_capacity_factor(capacity_factor),
          m_extra_nnz_entires(extra_nnz_entries),
          m_trans_buffer_size(0),
          m_trans_buffer(nullptr)
    {

        if (add_diagonal && (block_shape.x != block_shape.y)) {
            RXMESH_ERROR(
                "SparseMatrix::SparseMatrix() adding diagonal blocks for "
                "non-symmetric blocks has not been tested before!");
        }

        // get num rows and cols based on the input op
        std::tie(m_num_rows, m_num_cols) =
            get_num_rows_and_cols(rx, m_op, m_block_shape);

        // row pointer allocation and init with prefix sum for CSR
        CUDA_ERROR(cudaMalloc((void**)&m_d_row_ptr,
                              (m_num_rows + 1) * sizeof(IndexT)));
        CUDA_ERROR(
            cudaMalloc((void**)&m_d_row_acc, m_num_rows * sizeof(IndexT)));

        CUDA_ERROR(
            cudaMemset(m_d_row_ptr, 0, (m_num_rows + 1) * sizeof(IndexT)));

        // count the nnz per row
        mat_prescan(rx, m_op, m_d_row_ptr, m_block_shape, add_diagonal);


        // prefix sum using CUB.
        m_d_cub_temp_storage = nullptr;

        cub::DeviceScan::ExclusiveSum(m_d_cub_temp_storage,
                                      m_cub_temp_storage_bytes,
                                      m_d_row_ptr,
                                      m_d_row_ptr,
                                      m_num_rows + 1);
        CUDA_ERROR(cudaMalloc((void**)&m_d_cub_temp_storage,
                              m_cub_temp_storage_bytes));

        cub::DeviceScan::ExclusiveSum(m_d_cub_temp_storage,
                                      m_cub_temp_storage_bytes,
                                      m_d_row_ptr,
                                      m_d_row_ptr,
                                      m_num_rows + 1);


        // get nnz
        CUDA_ERROR(cudaMemcpy(&m_nnz,
                              (m_d_row_ptr + m_num_rows),
                              sizeof(IndexT),
                              cudaMemcpyDeviceToHost));

        update_max_nnz();

        // column index allocation and init
        CUDA_ERROR(
            cudaMalloc((void**)&m_d_col_idx, m_max_nnz * sizeof(IndexT)));

        // fill in col_idx
        mat_col_fill(
            rx, op, m_d_row_ptr, m_d_col_idx, m_block_shape, add_diagonal);


        // allocate value ptr
        CUDA_ERROR(cudaMalloc((void**)&m_d_val, m_max_nnz * sizeof(T)));
        CUDA_ERROR(cudaMemset(m_d_val, 0, m_nnz * sizeof(T)));
        m_allocated = m_allocated | DEVICE;


        alloce_and_move_to_host();


        // create cusparse matrix
        init_cusparse(*this);

#ifndef NDEBUG
        check_repeated_indices();
#endif
        init_cudss(*this);
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
            file << r << " " << c << " " << val << std::endl;
        });
        file.close();
    }

    /**
     * @brief export the matrix to MatrixMarket file
     */
    __host__ void to_mtx(std::string file_name)
    {
        std::ofstream file(file_name);
        if (!file.is_open()) {
            RXMESH_ERROR("SparseMatrix::to_mtx() Can not open file {}",
                         file_name);
            return;
        }

        Eigen::saveMarket(to_eigen(), file_name);
    }

    /**
     * @brief set all entries in the matrix to certain value on both host and
     * device
     */
    __host__ void reset(T val, locationT location, cudaStream_t stream = NULL)
    {
        bool do_device = (location & DEVICE) == DEVICE;
        bool do_host   = (location & HOST) == HOST;

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


    __device__ __host__ Op get_op() const
    {
        return m_op;
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
     * @brief return number of non-zero values at certain row
     */
    __device__ __host__ IndexT non_zeros(int row_id) const
    {
        assert(row_id < rows());
        return row_ptr()[row_id + 1] - row_ptr()[row_id];
    }

    /**
     * @brief return the col index for specific row at specific nnz index.
     * Useful when iterating over nnz of a row
     */
    __device__ __host__ IndexT col_id(int row_id, int nnz_index) const
    {
        assert(row_id < rows());
        assert(nnz_index < non_zeros(row_id));

        return col_idx()[row_ptr()[row_id] + nnz_index];
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
     * @brief access the matrix using the handle as defined by the query
     * operation. This function can only be called if the memory is Not
     * user-managed
     */
    template <typename InputHandleT, typename OutputHandleT>
    __device__ __host__ const T& operator()(const InputHandleT&  row_v,
                                            const OutputHandleT& col_v) const
    {
        // TODO check on the InputHandleT and OutputHandleT if they are
        // compatible with m_op
        assert(!m_is_user_managed);
        return this->operator()(static_cast<IndexT>(get_row_id(row_v)),
                                static_cast<IndexT>(get_row_id(col_v)));
    }

    /**
     * @brief access the matrix using the handle as defined by the query
     * operation. This function can only be called if the memory is Not
     * user-managed
     */
    template <typename InputHandleT, typename OutputHandleT>
    __device__ __host__ T& operator()(const InputHandleT&  row_v,
                                      const OutputHandleT& col_v)
    {
        // TODO check on the InputHandleT and OutputHandleT if they are
        // compatible with m_op
        assert(!m_is_user_managed);
        return this->operator()(static_cast<IndexT>(get_row_id(row_v)),
                                static_cast<IndexT>(get_row_id(col_v)));
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
     * @brief check if the input entry (x,y) is a non-zero (i.e., if it
     * allocated in the CSR representation)
     */
    __device__ __host__ bool is_non_zero(const IndexT x, const IndexT y) const
    {
        const IndexT start = row_ptr()[x];
        const IndexT end   = row_ptr()[x + 1];

        for (IndexT i = start; i < end; ++i) {
            if (col_idx()[i] == y) {
                return true;
            }
        }

        return false;
    }

#ifdef USE_CUDSS
    /**
     * @brief Return cuDSS matrix
     */
    __host__ cudssMatrix_t& get_cudss_matrix()
    {
        return m_cudss_matrix;
    }
#endif

    /**
     * @brief return the row pointer of the CSR matrix
     * @return
     */
    __device__ __host__ IndexT* row_ptr() const
    {
#ifdef __CUDA_ARCH__
        return m_d_row_ptr;
#else
        return m_h_row_ptr;
#endif
    }

    /**
     * @brief return the row pointer of the CSR matrix
     * @return
     */
    __device__ __host__ IndexT* row_ptr(locationT location) const
    {
        if (location == HOST) {
            return m_h_row_ptr;
        } else if (location == DEVICE) {
            return m_d_row_ptr;
        } else {
            return nullptr;
        }
    }


    /**
     * @brief return the column index pointer of the CSR matrix
     * @return
     */
    __device__ __host__ IndexT* col_idx() const
    {
#ifdef __CUDA_ARCH__
        return m_d_col_idx;
#else
        return m_h_col_idx;
#endif
    }


    /**
     * @brief return the column index pointer of the CSR matrix
     * @return
     */
    __device__ __host__ IndexT* col_idx(locationT location) const
    {
        if (location == HOST) {
            return m_h_col_idx;
        } else if (location == DEVICE) {
            return m_d_col_idx;
        } else {
            return nullptr;
        }
    }


    /**
     * @brief return the value pointer of the CSR matrix
     * @return
     */
    __device__ __host__ T* val_ptr(locationT location) const
    {
        if (location == HOST) {
            return m_h_val;
        } else if (location == DEVICE) {
            return m_d_val;
        } else {
            return nullptr;
        }
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
    template <typename HandleT>
    __device__ __host__ uint32_t get_row_id(const HandleT& handle) const
    {
        auto id = handle.unpack();
        return m_context.template prefix<HandleT>()[id.first] + id.second;
    }

    /**
     * @brief release all allocated memory
     */
    __host__ virtual void release()
    {
        release(LOCATION_ALL);
        CUSPARSE_ERROR(cusparseDestroy(m_cusparse_handle));
        CUSPARSE_ERROR(cusparseDestroySpMat(m_spdescr));

        GPU_FREE(m_d_cusparse_spmm_buffer);
        GPU_FREE(m_d_cusparse_spmv_buffer);
        GPU_FREE(m_d_row_acc);
        if (m_cub_temp_storage_bytes > 0) {
            GPU_FREE(m_d_cub_temp_storage);
        }
#ifdef USE_CUDSS
        if (std::is_floating_point_v<T> || std::is_same_v<T, cuComplex> ||
            std::is_same_v<T, cuDoubleComplex>) {
            CUDSS_ERROR(cudssMatrixDestroy(m_cudss_matrix));
        }
#endif
    }

    /**
     * @brief insert new entries to the sparse matrix. The insertion includes
     * 1) copying the entires in the in_mat and 2) copying the entries in the
     * new entires COO format (d_rows, d_cols).
     * We only allow the sparsity to change, i.e., the number of nnz values.
     * The size of the matrix should stay the same, i.e., #rows and #cols.
     * Note: d_rows and d_cols should include only new unique entries that does
     * not exist in the in_mat
     */
    __host__ bool insert(RXMeshStatic&    rx,
                         SparseMatrix<T>& in_mat,
                         const IndexT     size,
                         const IndexT*    d_new_rows,
                         const IndexT*    d_new_cols)
    {
        if (size == 0) {
            return false;
        }

        if (in_mat.rows() != rows() || in_mat.cols() != cols()) {
            RXMESH_ERROR(
                "SparseMatrix::insert() insertion only works for matrices of "
                "the same size. This matrix size: ({}x{}), in_mat size: "
                "({}x{})",
                rows(),
                cols(),
                in_mat.rows(),
                in_mat.cols());
            return false;
        }

        IndexT* in_d_row_ptr = in_mat.m_d_row_ptr;
        IndexT* in_d_col_idx = in_mat.m_d_col_idx;

        constexpr uint32_t blockThreads = 256;

        uint32_t blocks = DIVIDE_UP(rows(), blockThreads);

        // read in_mat row sum
        for_each_item<<<DIVIDE_UP(rows(), blockThreads), blockThreads>>>(
            rows(),
            [in_d_row_ptr = in_d_row_ptr,
             m_d_row_ptr  = m_d_row_ptr] __device__(int i) mutable {
                m_d_row_ptr[i] = in_d_row_ptr[i + 1] - in_d_row_ptr[i];
            });


        // add contribution of the new entries in the row sum
        for_each_item<<<DIVIDE_UP(size, blockThreads), blockThreads>>>(
            size,
            [m_d_row_ptr = m_d_row_ptr,
             d_new_rows  = d_new_rows] __device__(int i) mutable {
                const int row = d_new_rows[i];

                ::atomicAdd(m_d_row_ptr + row, 1);
            });

        // prefix sum using CUB.
        CUDA_ERROR(
            cudaMemset(m_d_cub_temp_storage, 0, m_cub_temp_storage_bytes));
        CUDA_ERROR(cudaMemset(m_d_row_ptr + m_num_rows, 0, sizeof(IndexT)));
        cub::DeviceScan::ExclusiveSum(m_d_cub_temp_storage,
                                      m_cub_temp_storage_bytes,
                                      m_d_row_ptr,
                                      m_d_row_ptr,
                                      m_num_rows + 1);

        // get nnz
        CUDA_ERROR(cudaMemcpy(&m_nnz,
                              (m_d_row_ptr + m_num_rows),
                              sizeof(IndexT),
                              cudaMemcpyDeviceToHost));


        if (m_nnz > m_max_nnz) {
            // free memory
            RXMESH_TRACE("SparseMatrxi::insert() allocating more memory!");
            GPU_FREE(m_d_col_idx);
            GPU_FREE(m_d_val);
            free(m_h_val);
            free(m_h_col_idx);

            // update max size as a function of current nnz
            update_max_nnz();

            // if respecting the user capacity factor is not enough
            // i.e., capacity factor is 1
            while (m_nnz > m_max_nnz) {
                m_capacity_factor *= 2.f;
                update_max_nnz();
            }

            // allocate col idx and values
            CUDA_ERROR(cudaMalloc((void**)&m_d_val, m_max_nnz * sizeof(T)));
            CUDA_ERROR(
                cudaMalloc((void**)&m_d_col_idx, m_max_nnz * sizeof(IndexT)));

            m_h_val = static_cast<T*>(malloc(m_max_nnz * sizeof(T)));
            m_h_col_idx =
                static_cast<IndexT*>(malloc(m_max_nnz * sizeof(IndexT)));
        }

        // reset row accumulator so that we can keep track of the col_idx
        // of the new items
        CUDA_ERROR(cudaMemset(m_d_row_acc, 0, m_num_rows * sizeof(IndexT)));

        // fill in the col_idx with the col_idx data from in_mat
        for_each_item<<<DIVIDE_UP(rows(), blockThreads), blockThreads>>>(
            rows(),
            [in_d_row_ptr = in_d_row_ptr,
             in_d_col_idx = in_d_col_idx,
             m_d_row_ptr  = m_d_row_ptr,
             m_d_col_idx  = m_d_col_idx,
             m_d_row_acc  = m_d_row_acc] __device__(int r) mutable {
                // fill in the first stop=start chunk of this row's col_idx
                const IndexT start = in_d_row_ptr[r];

                const IndexT stop = in_d_row_ptr[r + 1];

                const IndexT new_start = m_d_row_ptr[r];

                IndexT i = 0;

                for (IndexT j = start; j < stop; ++j) {
                    IndexT col = in_d_col_idx[j];

                    m_d_col_idx[new_start + i] = col;
                    ++i;
                }

                assert(stop >= start);

                m_d_row_acc[r] = stop - start;
            });


        // fill in the col_idx with the new entries information
        for_each_item<<<DIVIDE_UP(size, blockThreads), blockThreads>>>(
            size,
            [d_new_rows  = d_new_rows,
             d_new_cols  = d_new_cols,
             m_d_row_acc = m_d_row_acc,
             m_d_row_ptr = m_d_row_ptr,
             m_d_col_idx = m_d_col_idx] __device__(const int item) {
                const IndexT r = d_new_rows[item];
                const IndexT c = d_new_cols[item];

                IndexT row_st = ::atomicAdd(m_d_row_acc + r, 1);
                row_st += m_d_row_ptr[r];

                assert(row_st <= m_d_row_ptr[r + 1]);

                m_d_col_idx[row_st] = c;
            });

        // finally update the host (could be optional)
        CUDA_ERROR(cudaMemcpy(m_h_col_idx,
                              m_d_col_idx,
                              m_nnz * sizeof(IndexT),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(m_h_row_ptr,
                              m_d_row_ptr,
                              (m_num_rows + 1) * sizeof(IndexT),
                              cudaMemcpyDeviceToHost));

        init_cusparse(*this);

        return true;
    }


    /**
     * @brief swap this matrix with another on
     */
    __host__ void swap(SparseMatrix<T>& in)
    {
        std::swap(m_context, in.m_context);
        std::swap(m_cusparse_handle, in.m_cusparse_handle);
        std::swap(m_spdescr, in.m_spdescr);
        std::swap(m_d_cub_temp_storage, in.m_d_cub_temp_storage);
        std::swap(m_cub_temp_storage_bytes, in.m_cub_temp_storage_bytes);
        std::swap(m_block_shape, in.m_block_shape);
        std::swap(m_num_rows, in.m_num_rows);
        std::swap(m_num_cols, in.m_num_cols);
        std::swap(m_nnz, in.m_nnz);
        std::swap(m_max_nnz, in.m_max_nnz);
        std::swap(m_capacity_factor, in.m_capacity_factor);
        std::swap(m_extra_nnz_entires, in.m_extra_nnz_entires);
        std::swap(m_d_row_ptr, in.m_d_row_ptr);
        std::swap(m_d_col_idx, in.m_d_col_idx);
        std::swap(m_d_val, in.m_d_val);
        std::swap(m_d_row_acc, in.m_d_row_acc);
        std::swap(m_h_row_ptr, in.m_h_row_ptr);
        std::swap(m_h_col_idx, in.m_h_col_idx);
        std::swap(m_h_val, in.m_h_val);
        std::swap(m_spmm_buffer_size, in.m_spmm_buffer_size);
        std::swap(m_spmv_buffer_size, in.m_spmv_buffer_size);
        std::swap(m_d_cusparse_spmm_buffer, in.m_d_cusparse_spmm_buffer);
        std::swap(m_d_cusparse_spmv_buffer, in.m_d_cusparse_spmv_buffer);
        std::swap(m_allocated, in.m_allocated);
        std::swap(m_is_user_managed, in.m_is_user_managed);
        std::swap(m_op, in.m_op);

#ifdef USE_CUDSS
        std::swap(m_cudss_matrix, in.m_cudss_matrix);
#endif
    }

    /**
     * @brief return another SparseMatrix that is the transpose of this
     * SparseMatrix. This function allocate memory on both host and device.
     * Thus, it is not recommended to call it during the application multiple
     * times
     */
    __host__ SparseMatrix<T> transpose()
    {
        if (m_op == Op::EVDiamond) {
            RXMESH_ERROR(
                "SparseMatrix<T>::transpose() there is not transpose for "
                "SparseMatrix with EVDiamond.");
            return *this;
        }

        SparseMatrix<T> ret;

        ret.m_num_rows        = m_num_cols;
        ret.m_num_cols        = m_num_rows;
        ret.m_nnz             = m_nnz;
        ret.m_context         = m_context;
        ret.m_block_shape.x   = m_block_shape.y;
        ret.m_block_shape.y   = m_block_shape.x;
        ret.m_allocated       = m_allocated;
        ret.m_is_user_managed = false;
        ret.m_op              = transpose_op(m_op);


        CUDA_ERROR(cudaMalloc((void**)&ret.m_d_row_ptr,
                              (m_num_cols + 1) * sizeof(IndexT)));
        CUDA_ERROR(
            cudaMalloc((void**)&ret.m_d_col_idx, m_nnz * sizeof(IndexT)));
        CUDA_ERROR(cudaMalloc((void**)&ret.m_d_val, m_nnz * sizeof(T)));

        ret.m_h_val = static_cast<T*>(malloc(m_nnz * sizeof(T)));
        ret.m_h_row_ptr =
            static_cast<IndexT*>(malloc((m_num_cols + 1) * sizeof(IndexT)));
        ret.m_h_col_idx = static_cast<IndexT*>(malloc(m_nnz * sizeof(IndexT)));

        init_cusparse(ret);
        init_cudss(ret);

        m_trans_buffer_size = 0;

        CUSPARSE_ERROR(cusparseCsr2cscEx2_bufferSize(
            ret.m_cusparse_handle,
            m_num_rows,
            m_num_cols,
            m_nnz,
            m_d_val,
            m_d_row_ptr,
            m_d_col_idx,
            ret.m_d_val,
            ret.m_d_row_ptr,
            ret.m_d_col_idx,
            cuda_type<T>(),
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG_DEFAULT,  // CUSPARSE_CSR2CSC_ALG1
            &m_trans_buffer_size));


        CUDA_ERROR(cudaMalloc((void**)&m_trans_buffer, m_trans_buffer_size));

        CUSPARSE_ERROR(cusparseCsr2cscEx2(
            ret.m_cusparse_handle,
            m_num_rows,
            m_num_cols,
            m_nnz,
            m_d_val,
            m_d_row_ptr,
            m_d_col_idx,
            ret.m_d_val,
            ret.m_d_row_ptr,
            ret.m_d_col_idx,
            cuda_type<T>(),
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG_DEFAULT,  // CUSPARSE_CSR2CSC_ALG1
            m_trans_buffer));

        ret.move(DEVICE, HOST);

        return ret;
    }


    /**
     * @brief compute the transpose and store it in ret. ret should have already
     * been allocated with the right size and nnz
     */
    __host__ void transpose(SparseMatrix<T>& ret)
    {

        if (rows() != ret.cols() || cols() != ret.rows() ||
            non_zeros() != ret.non_zeros()) {
            RXMESH_ERROR(
                "SparseMatrix<T>::transpose() size mismatch. Input: {} x {} "
                "with nnz= {}. Output: {} x {} with nnz= {}.",
                rows(),
                cols(),
                non_zeros(),
                ret.rows(),
                ret.cols(),
                ret.non_zeros());
            return;
        }

        if (m_trans_buffer_size == 0) {

            CUSPARSE_ERROR(cusparseCsr2cscEx2_bufferSize(
                ret.m_cusparse_handle,
                m_num_rows,
                m_num_cols,
                m_nnz,
                m_d_val,
                m_d_row_ptr,
                m_d_col_idx,
                ret.m_d_val,
                ret.m_d_row_ptr,
                ret.m_d_col_idx,
                cuda_type<T>(),
                CUSPARSE_ACTION_NUMERIC,
                CUSPARSE_INDEX_BASE_ZERO,
                CUSPARSE_CSR2CSC_ALG_DEFAULT,  // CUSPARSE_CSR2CSC_ALG1
                &m_trans_buffer_size));


            CUDA_ERROR(
                cudaMalloc((void**)&m_trans_buffer, m_trans_buffer_size));
        }

        CUSPARSE_ERROR(cusparseCsr2cscEx2(
            ret.m_cusparse_handle,
            m_num_rows,
            m_num_cols,
            m_nnz,
            m_d_val,
            m_d_row_ptr,
            m_d_col_idx,
            ret.m_d_val,
            ret.m_d_row_ptr,
            ret.m_d_col_idx,
            cuda_type<T>(),
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG_DEFAULT,  // CUSPARSE_CSR2CSC_ALG1
            m_trans_buffer));
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
    template <int Order>
    __host__ void alloc_multiply_buffer(const DenseMatrix<T, Order>& B_mat,
                                        DenseMatrix<T, Order>&       C_mat,
                                        bool         is_a_transpose = false,
                                        bool         is_b_transpose = false,
                                        cudaStream_t stream         = 0)
    {
        if (m_d_cusparse_spmm_buffer == nullptr) {

            T alpha;
            T beta;

            cusparseSpMatDescr_t matA    = m_spdescr;
            cusparseDnMatDescr_t matB    = B_mat.m_dendescr;
            cusparseDnMatDescr_t matC    = C_mat.m_dendescr;
            void*                dBuffer = NULL;

            cusparseOperation_t opA, opB;

            opA = (!is_a_transpose) ? CUSPARSE_OPERATION_NON_TRANSPOSE :
                                      CUSPARSE_OPERATION_TRANSPOSE;
            opB = (!is_b_transpose) ? CUSPARSE_OPERATION_NON_TRANSPOSE :
                                      CUSPARSE_OPERATION_TRANSPOSE;

            if (std::is_same_v<T, cuComplex> ||
                std::is_same_v<T, cuDoubleComplex>) {
                opA = (!is_a_transpose) ?
                          CUSPARSE_OPERATION_NON_TRANSPOSE :
                          CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
                opB = (!is_b_transpose) ?
                          CUSPARSE_OPERATION_NON_TRANSPOSE :
                          CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
            }

            CUSPARSE_ERROR(cusparseSetStream(m_cusparse_handle, stream));

            CUSPARSE_ERROR(cusparseSpMM_bufferSize(m_cusparse_handle,
                                                   opA,
                                                   opB,
                                                   &alpha,
                                                   matA,
                                                   matB,
                                                   &beta,
                                                   matC,
                                                   cuda_type<T>(),
                                                   CUSPARSE_SPMM_ALG_DEFAULT,
                                                   &m_spmm_buffer_size));
            CUDA_ERROR(
                cudaMalloc(&m_d_cusparse_spmm_buffer, m_spmm_buffer_size));
        }
    }

    /**
     * @brief multiply the sparse matrix by a dense matrix. The function
     * performs the multiplication as
     * C = alpha.op(A) * op(B) + beta.C
     * where A is the sparse matrix, B is a dense matrix, and the result is a
     * dense matrix C.
     * Op could be transpose (or conjugate transpose) set via is_a/b_transpose
     * alpha and beta are scalar.
     * This method requires extra buffer allocation for cusparse. User may want
     * to call first alloce_multiply_buffer() (with the same parameters) first
     * to do the allocation and so timing this method will reflect the timing
     * for the multiplication operation only. Otherwise, this method calls
     * alloce_multiply_buffer() if it is not called before. Note that this
     * allocation happens only once and we then reuse it
     */
    template <int Order>
    __host__ void multiply(const DenseMatrix<T, Order>& B_mat,
                           DenseMatrix<T, Order>&       C_mat,
                           bool                         is_a_transpose = false,
                           bool                         is_b_transpose = false,
                           T                            alpha          = 1.,
                           T                            beta           = 0.,
                           cudaStream_t                 stream         = 0)
    {
        if (!is_a_transpose && !is_b_transpose) {
            assert(cols() == B_mat.rows());
            assert(rows() == C_mat.rows());
            assert(B_mat.cols() == C_mat.cols());
        }

        if (is_a_transpose && !is_b_transpose) {
            assert(rows() == B_mat.rows());
            assert(cols() == C_mat.rows());
            assert(B_mat.cols() == C_mat.cols());
        }

        if (!is_a_transpose && is_b_transpose) {
            assert(cols() == B_mat.cols());
            assert(rows() == C_mat.rows());
            assert(B_mat.rows() == C_mat.cols());
        }

        if (is_a_transpose && is_b_transpose) {
            assert(rows() == B_mat.cols());
            assert(cols() == C_mat.rows());
            assert(B_mat.rows() == C_mat.cols());
        }

        cusparseOperation_t opA, opB;

        opA = (!is_a_transpose) ? CUSPARSE_OPERATION_NON_TRANSPOSE :
                                  CUSPARSE_OPERATION_TRANSPOSE;
        opB = (!is_b_transpose) ? CUSPARSE_OPERATION_NON_TRANSPOSE :
                                  CUSPARSE_OPERATION_TRANSPOSE;

        if (std::is_same_v<T, cuComplex> ||
            std::is_same_v<T, cuDoubleComplex>) {
            opA = (!is_a_transpose) ? CUSPARSE_OPERATION_NON_TRANSPOSE :
                                      CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
            opB = (!is_b_transpose) ? CUSPARSE_OPERATION_NON_TRANSPOSE :
                                      CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
        }

        // T alpha;
        // T beta;
        //
        // if constexpr (std::is_same_v<T, cuComplex>) {
        //     alpha = make_cuComplex(1.f, 1.f);
        //     beta  = make_cuComplex(0.f, 0.f);
        // }
        //
        // if constexpr (std::is_same_v<T, cuDoubleComplex>) {
        //     alpha = make_cuDoubleComplex(1.0, 1.0);
        //     beta  = make_cuDoubleComplex(0.0, 0.0);
        // }
        //
        // if constexpr (!std::is_same_v<T, cuComplex> &&
        //               !std::is_same_v<T, cuDoubleComplex>) {
        //     alpha = T(1);
        //     beta  = T(0);
        // }


        // A_mat.create_cusparse_handle();
        cusparseSpMatDescr_t matA = m_spdescr;
        cusparseDnMatDescr_t matB = B_mat.m_dendescr;
        cusparseDnMatDescr_t matC = C_mat.m_dendescr;

        CUSPARSE_ERROR(cusparseSetStream(m_cusparse_handle, stream));

        // allocate an external buffer if needed
        alloc_multiply_buffer(B_mat, C_mat, stream);


        // execute SpMM
        CUSPARSE_ERROR(cusparseSpMM(m_cusparse_handle,
                                    opA,
                                    opB,
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
     * This method requires extra buffer allocation for cuSparse. User may want
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

        std::sort(
            triplets.begin(), triplets.end(), [](TripletT& a, TripletT& b) {
                return (a.row() < b.row()) ||
                       (a.row() == b.row() && a.col() < b.col());
            });

        ret.setFromTriplets(triplets.begin(), triplets.end());

        ret.makeCompressed();

        return ret;
    }


   protected:
    __host__ void release(locationT location)
    {
        if (m_is_user_managed) {
            return;
        }
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


   protected:
    void update_max_nnz()
    {
        m_max_nnz = static_cast<IndexT>(
            std::ceil(float(m_nnz + m_extra_nnz_entires) * m_capacity_factor));
    }

    void init_cusparse(SparseMatrix<T>& mat) const
    {
        // cuSparse CSR matrix
        if (mat.m_spdescr) {
            CUSPARSE_ERROR(cusparseDestroySpMat(mat.m_spdescr));
        }
        CUSPARSE_ERROR(cusparseCreateCsr(&mat.m_spdescr,
                                         mat.m_num_rows,
                                         mat.m_num_cols,
                                         mat.m_nnz,
                                         mat.m_d_row_ptr,
                                         mat.m_d_col_idx,
                                         mat.m_d_val,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         cuda_type<T>()));

        // cuSparse handle
        if (mat.m_cusparse_handle) {
            CUSPARSE_ERROR(cusparseDestroy(m_cusparse_handle));
        }
        CUSPARSE_ERROR(cusparseCreate(&mat.m_cusparse_handle));
        CUSPARSE_ERROR(cusparseSetPointerMode(mat.m_cusparse_handle,
                                              CUSPARSE_POINTER_MODE_HOST));
    }

    void check_repeated_indices()
    {
        // sanity check: no repeated indices in the col_id for a specific row
        for (IndexT r = 0; r < rows(); ++r) {
            IndexT start = m_h_row_ptr[r];
            IndexT stop  = m_h_row_ptr[r + 1];

            std::set<IndexT> cols;
            for (IndexT i = start; i < stop; ++i) {
                IndexT c = m_h_col_idx[i];
                if (cols.find(c) != cols.end()) {
                    RXMESH_ERROR(
                        "SparseMatrix::check_repeated_indices() Error in "
                        "constructing the sparse matrix. Row {} contains "
                        "repeated column indices {}",
                        r,
                        c);
                }
                cols.insert(c);
            }
        }
    }

    /**
     * @brief initialize cuDSS
     */
    __host__ void init_cudss(SparseMatrix<T>& mat) const
    {
#ifdef USE_CUDSS
        if (std::is_floating_point_v<T> || std::is_same_v<T, cuComplex> ||
            std::is_same_v<T, cuDoubleComplex>) {
            CUDSS_ERROR(
                cudssMatrixCreateCsr(&mat.m_cudss_matrix,
                                     mat.m_num_rows,
                                     mat.m_num_cols,
                                     mat.m_nnz,
                                     mat.m_d_row_ptr,
                                     nullptr,
                                     mat.m_d_col_idx,
                                     mat.m_d_val,
                                     CUDA_R_32I,
                                     cuda_type<T>(),
                                     CUDSS_MTYPE_SYMMETRIC,  // CUDSS_MTYPE_SPD
                                     CUDSS_MVIEW_FULL,
                                     CUDSS_BASE_ZERO));
        }

#endif
    }

    /**
     * @brief call sparse_mat_prescan kernel
     */
    void mat_prescan(const RXMeshStatic& rx,
                     Op                  op,
                     IndexT*             d_row_ptr,
                     BlockShape          block_shape,
                     bool                add_diagonal)
    {
        constexpr uint32_t blockThreads = 256;

        if (op == Op::V) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_prescan<Op::V, blockThreads>,
                d_row_ptr,
                block_shape,
                add_diagonal);
        } else if (op == Op::VV) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_prescan<Op::VV, blockThreads>,
                d_row_ptr,
                block_shape,
                add_diagonal);
        } else if (op == Op::VE) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_prescan<Op::VE, blockThreads>,
                d_row_ptr,
                block_shape,
                add_diagonal);
        } else if (op == Op::VF) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_prescan<Op::VF, blockThreads>,
                d_row_ptr,
                block_shape,
                add_diagonal);
        } else if (op == Op::E) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_prescan<Op::E, blockThreads>,
                d_row_ptr,
                block_shape,
                add_diagonal);
        } else if (op == Op::EV) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_prescan<Op::EV, blockThreads>,
                d_row_ptr,
                block_shape,
                add_diagonal);
        } else if (op == Op::EF) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_prescan<Op::EF, blockThreads>,
                d_row_ptr,
                block_shape,
                add_diagonal);
        } else if (op == Op::F) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_prescan<Op::F, blockThreads>,
                d_row_ptr,
                block_shape,
                add_diagonal);
        } else if (op == Op::FV) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_prescan<Op::FV, blockThreads>,
                d_row_ptr,
                block_shape,
                add_diagonal);
        } else if (op == Op::FE) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_prescan<Op::FE, blockThreads>,
                d_row_ptr,
                block_shape,
                add_diagonal);
        } else if (op == Op::FF) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_prescan<Op::FF, blockThreads>,
                d_row_ptr,
                block_shape,
                add_diagonal);
        } else {
            RXMESH_ERROR(
                "SparseMatrix::prescan() Unsupported query operation for "
                "constructing the sparse matrix. Input operation is {}",
                op_to_string(op));
        }
    }

    /**
     * @brief call sparse_mat_col_fill
     */
    void mat_col_fill(const RXMeshStatic& rx,
                      Op                  op,
                      IndexT*             d_row_ptr,
                      IndexT*             d_col_idx,
                      BlockShape          block_shape,
                      bool                add_diagonal)
    {
        constexpr uint32_t blockThreads = 256;

        if (op == Op::V) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_col_fill<Op::V, blockThreads>,
                d_row_ptr,
                d_col_idx,
                block_shape,
                add_diagonal);
        } else if (op == Op::VV) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_col_fill<Op::VV, blockThreads>,
                d_row_ptr,
                d_col_idx,
                block_shape,
                add_diagonal);
        } else if (op == Op::VE) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_col_fill<Op::VE, blockThreads>,
                d_row_ptr,
                d_col_idx,
                block_shape,
                add_diagonal);
        } else if (op == Op::VF) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_col_fill<Op::VF, blockThreads>,
                d_row_ptr,
                d_col_idx,
                block_shape,
                add_diagonal);
        } else if (op == Op::E) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_col_fill<Op::E, blockThreads>,
                d_row_ptr,
                d_col_idx,
                block_shape,
                add_diagonal);
        } else if (op == Op::EV) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_col_fill<Op::EV, blockThreads>,
                d_row_ptr,
                d_col_idx,
                block_shape,
                add_diagonal);
        } else if (op == Op::EF) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_col_fill<Op::EF, blockThreads>,
                d_row_ptr,
                d_col_idx,
                block_shape,
                add_diagonal);
        } else if (op == Op::F) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_col_fill<Op::F, blockThreads>,
                d_row_ptr,
                d_col_idx,
                block_shape,
                add_diagonal);
        } else if (op == Op::FV) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_col_fill<Op::FV, blockThreads>,
                d_row_ptr,
                d_col_idx,
                block_shape,
                add_diagonal);
        } else if (op == Op::FE) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_col_fill<Op::FE, blockThreads>,
                d_row_ptr,
                d_col_idx,
                block_shape,
                add_diagonal);
        } else if (op == Op::FF) {
            rx.run_kernel<blockThreads>(
                {op},
                detail::sparse_mat_col_fill<Op::FF, blockThreads>,
                d_row_ptr,
                d_col_idx,
                block_shape,
                add_diagonal);
        } else {
            RXMESH_ERROR(
                "SparseMatrix::mat_col_fill() Unsupported query operation for "
                "constructing the sparse matrix. Input operation is {}",
                op_to_string(op));
        }
    }

    /**
     * @brief get the number of rows and cols based on an input Op
     */
    std::pair<uint32_t, uint32_t> get_num_rows_and_cols(const RXMeshStatic& rx,
                                                        Op                  op,
                                                        BlockShape block_shape)
    {
        int num_rows(0), num_cols(0);

        // num rows
        if (op == Op::VV || op == Op::VE || op == Op::VF || op == Op::V) {
            num_rows = rx.get_num_vertices();
        } else if (op == Op::EV || op == Op::EE || op == Op::EF ||
                   op == Op::E) {
            num_rows = rx.get_num_edges();
        } else if (op == Op::FV || op == Op::FE || op == Op::FF ||
                   op == Op::F) {
            num_rows = rx.get_num_faces();
        } else {
            RXMESH_ERROR(
                "SparseMatrix::get_num_rows_and_cols() Unsupported query "
                "operation for constructing the sparse matrix. Input operation "
                "is {}",
                op_to_string(op));
        }

        // num cols
        if (op == Op::VV || op == Op::EV || op == Op::FV || op == Op::V) {
            num_cols = rx.get_num_vertices();
        } else if (op == Op::VE || op == Op::EE || op == Op::FE ||
                   op == Op::E) {
            num_cols = rx.get_num_edges();
        } else if (op == Op::VF || op == Op::EF || op == Op::FF ||
                   op == Op::F) {
            num_cols = rx.get_num_faces();
        } else {
            RXMESH_ERROR(
                "SparseMatrix::get_num_rows_and_cols() Unsupported query "
                "operation for constructing the sparse matrix. Input operation "
                "is {}",
                op_to_string(op));
        }

        num_rows *= block_shape.x;
        num_cols *= block_shape.y;

        return {num_rows, num_cols};
    }

    /**
     * @brief allocate host memory and move csr info to the host
     */
    void alloce_and_move_to_host()
    {  // allocate the host
        m_h_val = static_cast<T*>(malloc(m_max_nnz * sizeof(T)));
        m_h_row_ptr =
            static_cast<IndexT*>(malloc((m_num_rows + 1) * sizeof(IndexT)));
        m_h_col_idx = static_cast<IndexT*>(malloc(m_max_nnz * sizeof(IndexT)));

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
    }

   
    Context              m_context;
    cusparseHandle_t     m_cusparse_handle;
    cusparseSpMatDescr_t m_spdescr;

    void*  m_d_cub_temp_storage;
    size_t m_cub_temp_storage_bytes;

    BlockShape m_block_shape;

    IndexT m_num_rows;
    IndexT m_num_cols;
    IndexT m_nnz;
    IndexT m_max_nnz;
    float  m_capacity_factor;
    int    m_extra_nnz_entires;

    // device csr data
    IndexT* m_d_row_ptr;
    IndexT* m_d_col_idx;
    T*      m_d_val;

    // accumleator for the row sum
    IndexT* m_d_row_acc;

    // host csr data
    IndexT* m_h_row_ptr;
    IndexT* m_h_col_idx;
    T*      m_h_val;

    // cuSparse multiply buffer
    size_t m_spmm_buffer_size;
    size_t m_spmv_buffer_size;

    // transpose buffer
    size_t m_trans_buffer_size;
    void*  m_trans_buffer;

    void* m_d_cusparse_spmm_buffer;
    void* m_d_cusparse_spmv_buffer;

    // flags
    locationT m_allocated;

    // indicate if memory allocation is used managed
    bool m_is_user_managed;

    Op m_op;

#ifdef USE_CUDSS
    cudssMatrix_t m_cudss_matrix;
#endif
};

}  // namespace rxmesh