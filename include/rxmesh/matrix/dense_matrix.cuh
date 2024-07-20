#pragma once
#include <vector>
#include "cusparse.h"
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/rxmesh.h"
#include "rxmesh/types.h"

namespace rxmesh {

/**
 * @brief dense matrix use for device and host, inside is a array.
 * The dense matrix is initialized as col major on device.
 * We would only support col major dense matrix for now since that's what
 * cusparse and cusolver wants.
 */
template <typename T, typename IndexT = int, unsigned int MemAlignSize = 0>
struct DenseMatrix
{
    template <typename U, typename IndexU>
    friend class SparseMatrix;

    DenseMatrix()
        : m_allocated(LOCATION_NONE),
          m_num_rows(0),
          m_num_cols(0),
          m_d_val(nullptr),
          m_h_val(nullptr),
          m_col_pad_bytes(0),
          m_col_pad_idx(0)
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
          m_col_pad_bytes(0),
          m_col_pad_idx(0)
    {

        IndexT col_data_bytes = m_num_rows * sizeof(T);
        if (MemAlignSize != 0 && col_data_bytes % MemAlignSize != 0) {
            m_col_pad_bytes = MemAlignSize - (col_data_bytes % MemAlignSize);
            m_col_pad_idx   = m_col_pad_bytes / sizeof(T);
        }

        allocate(location);

        CUSPARSE_ERROR(cusparseCreateDnMat(&m_dendescr,
                                           m_num_rows,
                                           m_num_cols,
                                           m_num_rows,  // leading dim
                                           m_d_val,
                                           CUDA_R_32F,
                                           CUSPARSE_ORDER_COL));
    }

    /**
     * @brief return the leading dimension (row by default)
     */
    __host__ __device__ IndexT lead_dim() const
    {
        return m_num_rows;
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
     * @brief set all entries in the matrix to zeros on both host and device
     */
    __host__ void set_zeros()
    {
        std::memset(m_h_val, 0, bytes());

        CUDA_ERROR(cudaMemset(m_d_val, 0, bytes()));
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
    __host__ __device__ T& operator()(const uint32_t row, const uint32_t col)
    {
        assert(row < m_num_rows);
        assert(col < m_num_cols);

#ifdef __CUDA_ARCH__
        return m_d_val[col * (m_num_rows + m_col_pad_idx) + row];
#else
        return m_h_val[col * (m_num_rows + m_col_pad_idx) + row];
#endif
    }

    /**
     * @brief accessing a specific value in the matrix using the row and col
     * index. Can be used on both host and device
     */
    __host__ __device__ const T& operator()(const uint32_t row,
                                            const uint32_t col) const
    {
        assert(row < m_num_rows);
        assert(col < m_num_cols);

#ifdef __CUDA_ARCH__
        return m_d_val[col * (m_num_rows + m_col_pad_idx) + row];
#else
        return m_h_val[col * (m_num_rows + m_col_pad_idx) + row];
#endif
    }


    /**
     * @brief access the matrix using vertex/edge/face handle as a row index.
     */
    template <typename HandleT>
    __host__ __device__ T& operator()(const HandleT handle, const uint32_t col)
    {
        return this->operator()(get_row_id(handle), col);
    }

    /**
     * @brief access the matrix using vertex/edge/face handle as a row index.
     */
    template <typename HandleT>
    __host__ __device__ const T& operator()(const HandleT  handle,
                                            const uint32_t col) const
    {
        return this->operator()(get_row_id(handle), col);
    }

    /**
     * @brief return the row index corresponding to specific vertex/edge/face
     * handle
     */
    template <typename HandleT>
    __host__ __device__ const uint32_t get_row_id(const HandleT handle) const
    {
        auto id = handle.unpack();

        uint32_t row;

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
    __host__ const T* col_data(const uint32_t ld_idx,
                               locationT      location = DEVICE) const
    {
        if ((location & HOST) == HOST) {
            return m_h_val + ld_idx * (m_num_rows + m_col_pad_idx);
        }

        if ((location & DEVICE) == DEVICE) {
            return m_d_val + ld_idx * (m_num_rows + m_col_pad_idx);
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
    __host__ T* col_data(const uint32_t ld_idx, locationT location = DEVICE)
    {
        if ((location & HOST) == HOST) {
            return m_h_val + ld_idx * (m_num_rows + m_col_pad_idx);
        }

        if ((location & DEVICE) == DEVICE) {
            return m_d_val + ld_idx * (m_num_rows + m_col_pad_idx);
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
        return (m_num_rows + m_col_pad_idx) * m_num_cols * sizeof(T);
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


    const Context        m_context;
    cusparseDnMatDescr_t m_dendescr;
    locationT            m_allocated;
    IndexT               m_num_rows;
    IndexT               m_num_cols;
    T*                   m_d_val;
    T*                   m_h_val;

    IndexT m_col_pad_bytes;
    IndexT m_col_pad_idx;
};

}  // namespace rxmesh
