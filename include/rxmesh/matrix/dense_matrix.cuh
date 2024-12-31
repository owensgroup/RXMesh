#pragma once
#include <vector>
#include "cublas_v2.h"
#include "cusparse.h"

#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/rxmesh.h"
#include "rxmesh/types.h"

#include "rxmesh/util/meta.h"

#include <Eigen/Dense>

namespace rxmesh {
/**
 * @brief dense matrix use for device and host, inside is a array.
 * The dense matrix is initialized as col major on device.
 * We would only support col major dense matrix for now since that's what
 * cusparse and cusolver wants. However, there is a limited number of operations
 * defined on row major matrices.
 * Order define the storage order of the matrix.
 */
template <typename T, int Order = Eigen::ColMajor>
struct DenseMatrix
{
    using IndexT = int;

    template <typename U>
    friend class SparseMatrix;

    using EigenDenseMatrix =
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Order>>;

    DenseMatrix()
        : m_allocated(LOCATION_NONE),
          m_num_rows(0),
          m_num_cols(0),
          m_d_val(nullptr),
          m_h_val(nullptr)
    {
    }


    DenseMatrix(const RXMesh& rx,
                IndexT        num_rows,
                IndexT        num_cols,
                locationT     location = LOCATION_ALL)
        : m_context(rx.get_context()),
          m_num_rows(num_rows),
          m_num_cols(num_cols),
          m_dendescr(NULL),
          m_h_val(nullptr),
          m_d_val(nullptr),
          m_cublas_handle(nullptr)
    {


        allocate(location);

        CUSPARSE_ERROR(cusparseCreateDnMat(&m_dendescr,
                                           m_num_rows,
                                           m_num_cols,
                                           m_num_rows,  // leading dim
                                           m_d_val,
                                           cuda_type<T>(),
                                           CUSPARSE_ORDER_COL));

        CUBLAS_ERROR(cublasCreate(&m_cublas_handle));
        CUBLAS_ERROR(
            cublasSetPointerMode(m_cublas_handle, CUBLAS_POINTER_MODE_HOST));
    }

    /**
     * @brief return the leading dimension (row by default)
     */
    __host__ __device__ IndexT lead_dim() const
    {
        if constexpr (Order == Eigen::ColMajor) {
            return m_num_rows;
        } else {
            return m_num_cols;
        }
    }

    /**
     * @brief return number of rows
     */
    __host__ __device__ IndexT rows() const
    {
        return m_num_rows;
    }

    /**
     * @brief return number of columns
     */
    __host__ __device__ IndexT cols() const
    {
        return m_num_cols;
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
            std::fill_n(m_h_val, rows() * cols(), val);
            CUDA_ERROR(cudaMemcpyAsync(m_d_val,
                                       m_h_val,
                                       rows() * cols() * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       stream));
        } else if (do_device) {
            const int    threads = 512;
            const IndexT nnz     = rows() * cols();
            memset<<<DIVIDE_UP(nnz, threads), threads, 0, stream>>>(
                m_d_val, val, nnz);
        } else if (do_host) {
            std::fill_n(m_h_val, rows() * cols(), val);
        }
    }

    /**
     * @brief fill in the matrix with random numbers on both host and device
     * @return
     */
    __host__ void fill_random(double minn = -1.0, double maxx = 1.0)
    {
        std::random_device rd;
        std::mt19937       gen(rd());

        if constexpr (std::is_same_v<T, cuComplex> ||
                      std::is_same_v<T, float>) {
            std::uniform_real_distribution<float> dis(static_cast<float>(minn),
                                                      static_cast<float>(maxx));

            for (int i = 0; i < rows() * cols(); ++i) {
                if constexpr (std::is_same_v<T, cuComplex>) {
                    m_h_val[i].x = dis(gen);
                    m_h_val[i].y = dis(gen);
                }
                if constexpr (std::is_same_v<T, float>) {
                    m_h_val[i] = dis(gen);
                }
            }
        }

        if constexpr (std::is_same_v<T, cuDoubleComplex> ||
                      std::is_same_v<T, double>) {
            std::uniform_real_distribution<double> dis(
                static_cast<double>(minn), static_cast<double>(maxx));

            for (int i = 0; i < rows() * cols(); ++i) {
                if constexpr (std::is_same_v<T, cuDoubleComplex>) {
                    m_h_val[i].x = dis(gen);
                    m_h_val[i].y = dis(gen);
                }
                if constexpr (std::is_same_v<T, double>) {
                    m_h_val[i] = dis(gen);
                }
            }
        }

        if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int32_t>) {
            std::uniform_int_distribution<int> dis(static_cast<int>(minn),
                                                   static_cast<int>(maxx));

            for (int i = 0; i < rows() * cols(); ++i) {
                m_h_val[i] = dis(gen);
            }
        }

        if constexpr (std::is_same_v<T, uint32_t>) {
            std::uniform_int_distribution<uint32_t> dis(
                static_cast<uint32_t>(minn), static_cast<uint32_t>(maxx));

            for (int i = 0; i < rows() * cols(); ++i) {
                m_h_val[i] = dis(gen);
            }
        }


        if constexpr (std::is_same_v<T, int64_t>) {
            std::uniform_int_distribution<int64_t> dis(
                static_cast<int64_t>(minn), static_cast<int64_t>(maxx));

            for (int i = 0; i < rows() * cols(); ++i) {
                m_h_val[i] = dis(gen);
            }
        }

        if constexpr (std::is_same_v<T, uint64_t>) {
            std::uniform_int_distribution<uint64_t> dis(
                static_cast<uint64_t>(minn), static_cast<uint64_t>(maxx));

            for (int i = 0; i < rows() * cols(); ++i) {
                m_h_val[i] = dis(gen);
            }
        }


        CUDA_ERROR(
            cudaMemcpy(m_d_val, m_h_val, bytes(), cudaMemcpyHostToDevice));
    }

    /**
     * @brief accessing a specific value in the matrix using the row and col
     * index. Can be used on both host and device
     */
    __host__ __device__ T& operator()(const IndexT row, const IndexT col)
    {
        assert(row < m_num_rows);
        assert(col < m_num_cols);

#ifdef __CUDA_ARCH__
        return m_d_val[get_index(row, col)];
#else
        return m_h_val[get_index(row, col)];
#endif
    }

    /**
     * @brief accessing a specific value in the matrix using the row and col
     * index. Can be used on both host and device
     */
    __host__ __device__ const T& operator()(const IndexT row,
                                            const IndexT col) const
    {
        assert(row < m_num_rows);
        assert(col < m_num_cols);

#ifdef __CUDA_ARCH__
        return m_d_val[get_index(row, col)];
#else
        return m_h_val[get_index(row, col)];
#endif
    }


    /**
     * @brief access the matrix using vertex/edge/face handle as a row index.
     */
    template <typename HandleT>
    __host__ __device__ T& operator()(const HandleT handle, const IndexT col)
    {
        return this->operator()(get_row_id(handle), col);
    }

    /**
     * @brief access the matrix using vertex/edge/face handle as a row index.
     */
    template <typename HandleT>
    __host__ __device__ const T& operator()(const HandleT  handle,
                                            const IndexT col) const
    {
        return this->operator()(get_row_id(handle), col);
    }

    /**
     * @brief compute the sum of the absolute value of all elements in the
     * matrix. For complex types (cuComples and cuDoubleComplex), we sum the
     * absolute value of the real and absolute value of the imaginary part. The
     * results are computed for the data on the device. Only float, double,
     * cuComplex, and cuDoubleComplex are supported
     * @param stream
     * @return
     */
    __host__ BaseTypeT<T> abs_sum(cudaStream_t stream = NULL)
    {
        CUBLAS_ERROR(cublasSetStream(m_cublas_handle, stream));
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double> &&
                      !std::is_same_v<T, cuComplex> &&
                      !std::is_same_v<T, cuDoubleComplex>) {
            RXMESH_ERROR(
                "DenseMatrix::abs_sum() only float, double, cuComplex, and "
                "cuDoubleComplex are supported for this function!");
            return BaseTypeT<T>(0);
        }

        BaseTypeT<T> result;
        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_ERROR(cublasSasum(
                m_cublas_handle, rows() * cols(), m_d_val, 1, &result));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUBLAS_ERROR(cublasDasum(
                m_cublas_handle, rows() * cols(), m_d_val, 1, &result));
        }

        if constexpr (std::is_same_v<T, cuComplex>) {
            CUBLAS_ERROR(cublasScasum(
                m_cublas_handle, rows() * cols(), m_d_val, 1, &result));
        }

        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            CUBLAS_ERROR(cublasDzasum(
                m_cublas_handle, rows() * cols(), m_d_val, 1, &result));
        }

        return result;
    }

    /**
     * @brief compute the following
     * Y = alpha * X + Y
     * where Y is this dense matrix, X is another dense matrix that has the same
     * dimensions as Y and alpha is a scalar. The results are computed for the
     * data on the device. Only float, double, cuComplex, and cuDoubleComplex
     * are supported
     */
    __host__ void axpy(DenseMatrix<T, Order>& X,
                       T                      alpha,
                       cudaStream_t           stream = NULL)
    {
        CUBLAS_ERROR(cublasSetStream(m_cublas_handle, stream));
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double> &&
                      !std::is_same_v<T, cuComplex> &&
                      !std::is_same_v<T, cuDoubleComplex>) {
            RXMESH_ERROR(
                "DenseMatrix::axpy() only float, double, cuComplex, and "
                "cuDoubleComplex are supported for this function!");
            return;
        }


        if (rows() != X.rows() || cols() != X.cols()) {
            RXMESH_ERROR(
                "DenseMatrix::axpy() The input matrices size does not match. "
                "This matrix size is {},{} while X size is {},{}",
                rows(),
                cols(),
                X.rows(),
                X.cols());
            return;
        }

        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_ERROR(cublasSaxpy(m_cublas_handle,
                                     rows() * cols(),
                                     &alpha,
                                     X.m_d_val,
                                     1,
                                     m_d_val,
                                     1));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUBLAS_ERROR(cublasDaxpy(m_cublas_handle,
                                     rows() * cols(),
                                     &alpha,
                                     X.m_d_val,
                                     1,
                                     m_d_val,
                                     1));
        }

        if constexpr (std::is_same_v<T, cuComplex>) {
            CUBLAS_ERROR(cublasCaxpy(m_cublas_handle,
                                     rows() * cols(),
                                     &alpha,
                                     X.m_d_val,
                                     1,
                                     m_d_val,
                                     1));
        }

        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            CUBLAS_ERROR(cublasZaxpy(m_cublas_handle,
                                     rows() * cols(),
                                     &alpha,
                                     X.m_d_val,
                                     1,
                                     m_d_val,
                                     1));
        }
    }

    /**
     * @brief compute the dot produce with another dense matrix. If the matrix
     * is a 1D vector, it is the inner product. If the matrix represents a 2D
     * matrix, then it is the sum of the element-wise multiplication. The
     * results are computed for the data on the device. Only float, double,
     * cuComplex, and cuDoubleComplex are supported. For complex matrices
     * (cuComplex and cuDoubleComplex), it is optional to use the conjugate of
     * x.
     */
    __host__ T dot(const DenseMatrix<T, Order>& x,
                   bool                         use_conjugate = false,
                   cudaStream_t                 stream        = NULL) const
    {
        CUBLAS_ERROR(cublasSetStream(m_cublas_handle, stream));
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double> &&
                      !std::is_same_v<T, cuComplex> &&
                      !std::is_same_v<T, cuDoubleComplex>) {
            RXMESH_ERROR(
                "DenseMatrix::dot() only float, double, cuComplex, and "
                "cuDoubleComplex are supported for this function!");
            return T(0);
        }

        T result;
        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_ERROR(cublasSdot(m_cublas_handle,
                                    rows() * cols(),
                                    x.m_d_val,
                                    1,
                                    m_d_val,
                                    1,
                                    &result));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUBLAS_ERROR(cublasDdot(m_cublas_handle,
                                    rows() * cols(),
                                    x.m_d_val,
                                    1,
                                    m_d_val,
                                    1,
                                    &result));
        }

        if constexpr (std::is_same_v<T, cuComplex>) {
            if (use_conjugate) {
                CUBLAS_ERROR(cublasCdotc(m_cublas_handle,
                                         rows() * cols(),
                                         x.m_d_val,
                                         1,
                                         m_d_val,
                                         1,
                                         &result));
            } else {
                CUBLAS_ERROR(cublasCdotu(m_cublas_handle,
                                         rows() * cols(),
                                         x.m_d_val,
                                         1,
                                         m_d_val,
                                         1,
                                         &result));
            }
        }

        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            if (use_conjugate) {
                CUBLAS_ERROR(cublasZdotc(m_cublas_handle,
                                         rows() * cols(),
                                         x.m_d_val,
                                         1,
                                         m_d_val,
                                         1,
                                         &result));
            } else {
                CUBLAS_ERROR(cublasZdotu(m_cublas_handle,
                                         rows() * cols(),
                                         x.m_d_val,
                                         1,
                                         m_d_val,
                                         1,
                                         &result));
            }
        }

        return result;
    }


    /**
     * @brief compute the norm of a dense matrix which is computed as
     * sqrt(\sum (x[i]*x[i]**H) for i = 0,...,n*m where n is number of rows and
     * m is number of columns, and **H denotes the conjugate if x is complex
     * number. The results are computed for the data on the device. Only float,
     * double, cuComplex, and cuDoubleComplex are supported.
     */
    __host__ BaseTypeT<T> norm2(cudaStream_t stream = NULL)
    {
        CUBLAS_ERROR(cublasSetStream(m_cublas_handle, stream));
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double> &&
                      !std::is_same_v<T, cuComplex> &&
                      !std::is_same_v<T, cuDoubleComplex>) {
            RXMESH_ERROR(
                "DenseMatrix::norm2() only float, double, cuComplex, and "
                "cuDoubleComplex are supported for this function!");
            return BaseTypeT<T>(0);
        }

        BaseTypeT<T> result;
        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_ERROR(cublasSnrm2(
                m_cublas_handle, rows() * cols(), m_d_val, 1, &result));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUBLAS_ERROR(cublasDnrm2(
                m_cublas_handle, rows() * cols(), m_d_val, 1, &result));
        }

        if constexpr (std::is_same_v<T, cuComplex>) {
            CUBLAS_ERROR(cublasScnrm2(
                m_cublas_handle, rows() * cols(), m_d_val, 1, &result));
        }

        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            CUBLAS_ERROR(cublasDznrm2(
                m_cublas_handle, rows() * cols(), m_d_val, 1, &result));
        }

        return result;
    }


    /**
     * @brief multiply all entries in the dense matrix by a scalar (i.e.,
     * scaling). For complex number, the scalar could be either a complex or
     * real number. The results are computed for the data on the device. Only
     * float, double, cuComplex, and cuDoubleComplex are supported.
     */
    template <typename U>
    __host__ void multiply(U scalar, cudaStream_t stream = NULL)
    {
        CUBLAS_ERROR(cublasSetStream(m_cublas_handle, stream));
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double> &&
                      !std::is_same_v<T, cuComplex> &&
                      !std::is_same_v<T, cuDoubleComplex>) {
            RXMESH_ERROR(
                "DenseMatrix::multiply() only float, double, cuComplex, and "
                "cuDoubleComplex are supported for this function!");
            return T(0);
        }


        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_ERROR(cublasSscal(
                m_cublas_handle, rows() * cols(), &scalar, m_d_val, 1));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUBLAS_ERROR(cublasDscal(
                m_cublas_handle, rows() * cols(), &scalar, m_d_val, 1));
        }

        if constexpr (std::is_same_v<T, cuComplex>) {
            if constexpr (std::is_same_v<U, cuComplex>) {
                CUBLAS_ERROR(cublasCscal(
                    m_cublas_handle, rows() * cols(), &scalar, m_d_val, 1));
            } else {
                CUBLAS_ERROR(cublasCsscal(
                    m_cublas_handle, rows() * cols(), &scalar, m_d_val, 1));
            }
        }

        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            if constexpr (std::is_same_v<U, cuDoubleComplex>) {
                CUBLAS_ERROR(cublasZscal(
                    m_cublas_handle, rows() * cols(), &scalar, m_d_val, 1));
            } else {
                CUBLAS_ERROR(cublasZdscal(
                    m_cublas_handle, rows() * cols(), &scalar, m_d_val, 1));
            }
        }
    }

    /**
     * @brief Swap the content of this dense matrix with another dense matrix.
     * The results are computed for the data on the device. Only float, double,
     * cuComplex, and cuDoubleComplex are supported.
     */
    __host__ void swap(DenseMatrix<T, Order>& X, cudaStream_t stream = NULL)
    {
        CUBLAS_ERROR(cublasSetStream(m_cublas_handle, stream));
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double> &&
                      !std::is_same_v<T, cuComplex> &&
                      !std::is_same_v<T, cuDoubleComplex>) {
            RXMESH_ERROR(
                "DenseMatrix::swap() only float, double, cuComplex, and "
                "cuDoubleComplex are supported for this function!");
            return;
        }

        if (rows() != X.rows() || cols() != X.cols()) {
            RXMESH_ERROR(
                "DenseMatrix::swap() The input matrices size does not match. "
                "This matrix size is {},{} while X size is {},{}",
                rows(),
                cols(),
                X.rows(),
                X.cols());
            return;
        }


        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_ERROR(cublasSswap(
                m_cublas_handle, rows() * cols(), m_d_val, 1, X.m_d_val, 1));
        }

        if constexpr (std::is_same_v<T, double>) {
            CUBLAS_ERROR(cublasDswap(
                m_cublas_handle, rows() * cols(), m_d_val, 1, X.m_d_val, 1));
        }

        if constexpr (std::is_same_v<T, cuComplex>) {
            CUBLAS_ERROR(cublasCswap(
                m_cublas_handle, rows() * cols(), m_d_val, 1, X.m_d_val, 1));
        }

        if constexpr (std::is_same_v<T, cuDoubleComplex>) {
            CUBLAS_ERROR(cublasZswap(
                m_cublas_handle, rows() * cols(), m_d_val, 1, X.m_d_val, 1));
        }
    }

    /**
     * @brief return the row index corresponding to specific vertex/edge/face
     * handle
     */
    template <typename HandleT>
    __host__ __device__ IndexT get_row_id(const HandleT handle) const
    {
        auto id = handle.unpack();

        IndexT row;

        if constexpr (std::is_same_v<HandleT, VertexHandle>) {
            row = m_context.vertex_prefix()[id.first] + id.second;
        }

        if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
            row = m_context.edge_prefix()[id.first] + id.second;
        }

        if constexpr (std::is_same_v<HandleT, FaceHandle>) {
            row = m_context.face_prefix()[id.first] + id.second;
        }

        return row;
    }

    /**
     * @brief return the raw pointer based on the specified location (host vs.
     * device)
     */
    __host__ __device__ T* data(locationT location = DEVICE) const
    {
        if ((location & HOST) == HOST) {
            return m_h_val;
        }

        if ((location & DEVICE) == DEVICE) {
            return m_d_val;
        }

        assert(1 != 1);
        return nullptr;
    }

    /**
     * @brief return the raw pointer pf a column.
     */
    __host__ const T* col_data(const IndexT ld_idx,
                               locationT      location = DEVICE) const
    {
        if ((location & HOST) == HOST) {
            return m_h_val + ld_idx * m_num_rows;
        }

        if ((location & DEVICE) == DEVICE) {
            return m_d_val + ld_idx * m_num_rows;
        }

        if ((location & m_allocated) != location) {
            RXMESH_ERROR(
                "DenseMatrix::col_data() Requested data not allocated on {}",
                location_to_string(location));
        }

        return nullptr;
    }

    /**
     * @brief return the raw pointer pf a column.
     */
    __host__ T* col_data(const IndexT ld_idx, locationT location = DEVICE)
    {
        if ((location & HOST) == HOST) {
            return m_h_val + ld_idx * m_num_rows;
        }

        if ((location & DEVICE) == DEVICE) {
            return m_d_val + ld_idx * m_num_rows;
        }

        if ((location & m_allocated) != location) {
            RXMESH_ERROR(
                "DenseMatrix::col_data() Requested data not allocated on {}",
                location_to_string(location));
        }

        return nullptr;
    }

    /**
     * @brief return the total number bytes used to allocate the matrix
     */
    __host__ __device__ IndexT bytes() const
    {
        return m_num_rows * m_num_cols * sizeof(T);
    }

    /**
     * @brief move the data between host and device
     */
    __host__ void move(locationT    source,
                       locationT    target,
                       cudaStream_t stream = NULL)
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
            CUDA_ERROR(cudaMemcpyAsync(
                m_d_val, m_h_val, bytes(), cudaMemcpyHostToDevice, stream));
        } else if (source == DEVICE && target == HOST) {
            CUDA_ERROR(cudaMemcpyAsync(
                m_h_val, m_d_val, bytes(), cudaMemcpyDeviceToHost, stream));
        }
    }


    /**
     * @brief Deep copy from a source matrix. If source_flag and target_flag are
     * both set to LOCATION_ALL, then we copy what is on host to host, and what
     * on device to device. If sourc_flag is set to HOST (or DEVICE) and
     * target_flag is set to LOCATION_ALL, then we copy source's HOST (or
     * DEVICE) to both HOST and DEVICE. Setting source_flag to
     * LOCATION_ALL while target_flag is NOT set to LOCATION_ALL is invalid
     * because we don't know which source to copy from
     * @param source matrix to copy from
     * @param source_flag defines where we will copy from
     * @param target_flag defines where we will copy to
     * @param stream used to launch kernel/memcpy
     */
    __host__ void copy_from(DenseMatrix<T, Order>& source,
                            locationT              source_flag = LOCATION_ALL,
                            locationT              target_flag = LOCATION_ALL,
                            cudaStream_t           stream      = NULL)
    {
        if (rows() != source.rows() || cols() != source.cols()) {
            RXMESH_ERROR(
                "DenseMatrix::copy_from() the number of rows/cols is "
                "different!");
            return;
        }

        if ((source_flag & LOCATION_ALL) == LOCATION_ALL &&
            (target_flag & LOCATION_ALL) != LOCATION_ALL) {
            RXMESH_ERROR("DenseMatrix::copy_from() Invalid configuration!");
            return;
        }

        if ((source_flag & m_allocated) != source_flag) {
            RXMESH_ERROR(
                "DenseMatrix::copy_from() copying source is not valid"
                " because it was not allocated on source i.e., {}",
                location_to_string(source_flag));
            return;
        }

        if ((target_flag & m_allocated) != target_flag) {
            RXMESH_WARN(
                "DenseMatrix::copy_from() allocating target before moving to "
                "{}",
                location_to_string(target_flag));
            allocate(target_flag);
        }
        // 1) copy from HOST to HOST
        if ((source_flag & HOST) == HOST && (target_flag & HOST) == HOST) {
            std::memcpy(m_h_val, source.m_h_val, bytes());
        }

        // 2) copy from DEVICE to DEVICE
        if ((source_flag & DEVICE) == DEVICE &&
            (target_flag & DEVICE) == DEVICE) {
            CUDA_ERROR(cudaMemcpyAsync(m_d_val,
                                       source.m_d_val,
                                       bytes(),
                                       cudaMemcpyDeviceToDevice,
                                       stream));
        }

        // 3) copy from DEVICE to HOST
        if ((source_flag & DEVICE) == DEVICE && (target_flag & HOST) == HOST) {
            CUDA_ERROR(cudaMemcpyAsync(m_h_val,
                                       source.m_d_val,
                                       bytes(),
                                       cudaMemcpyDeviceToHost,
                                       stream));
        }


        // 4) copy from HOST to DEVICE
        if ((source_flag & HOST) == HOST && (target_flag & DEVICE) == DEVICE) {
            CUDA_ERROR(cudaMemcpyAsync(m_d_val,
                                       source.m_h_val,
                                       bytes(),
                                       cudaMemcpyHostToDevice,
                                       stream));
        }
    }

    /**
     * @brief Convert/map the dense matrix to Eigen dense matrix. This is a
     * zero-copy conversion so Eigen dense matrix will point to the same memory
     * as the host-side of this DenseMatrix
     */
    __host__ EigenDenseMatrix to_eigen()
    {
        return EigenDenseMatrix(m_h_val, rows(), cols());
    }

    /**
     * @brief release the data on host or device
     */
    __host__ void release(locationT location = LOCATION_ALL)
    {
        if (((location & HOST) == HOST) && ((m_allocated & HOST) == HOST)) {
            free(m_h_val);
            m_h_val     = nullptr;
            m_allocated = m_allocated & (~HOST);
        }

        if (((location & DEVICE) == DEVICE) &&
            ((m_allocated & DEVICE) == DEVICE)) {
            GPU_FREE(m_d_val);
            m_allocated = m_allocated & (~DEVICE);
        }
        if ((location & LOCATION_ALL) == LOCATION_ALL) {
            CUSPARSE_ERROR(cusparseDestroyDnMat(m_dendescr));
        }
    }

   private:
    /**
     * @brief allocate the data on host or device
     */
    void allocate(locationT location)
    {
        if ((location & HOST) == HOST) {
            release(HOST);

            m_h_val = static_cast<T*>(malloc(bytes()));

            m_allocated = m_allocated | HOST;
        }

        if ((location & DEVICE) == DEVICE) {
            release(DEVICE);

            CUDA_ERROR(cudaMalloc((void**)&m_d_val, bytes()));

            m_allocated = m_allocated | DEVICE;
        }
    }


    /**
     * @brief return the 1d indext given the row and column id
     */
    __host__ __device__ __inline__ int get_index(IndexT row,
                                                 IndexT col) const
    {
        if constexpr (Order == Eigen::ColMajor) {
            return col * m_num_rows + row;
        } else {
            return col + row * m_num_cols;
        }
    }


    const Context        m_context;
    cusparseDnMatDescr_t m_dendescr;
    cublasHandle_t       m_cublas_handle;
    locationT            m_allocated;
    IndexT               m_num_rows;
    IndexT               m_num_cols;
    T*                   m_d_val;
    T*                   m_h_val;
};

}  // namespace rxmesh
