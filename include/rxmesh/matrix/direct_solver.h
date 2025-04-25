#pragma once
#include "rxmesh/matrix/solver_base.h"

#include "rxmesh/matrix/permute_method.h"
#include "rxmesh/matrix/permute_util.h"

#include "rxmesh/matrix/mgnd_permute.cuh"
#include "rxmesh/matrix/nd_permute.cuh"

#include "thrust/device_ptr.h"
#include "thrust/execution_policy.h"
#include "thrust/gather.h"
#include "thrust/scatter.h"

namespace rxmesh {

/**
 * @brief abstract class for direct solvers--mainly to implement the permute
 * method that is common among all direct solvers
 */
template <typename SpMatT, int DenseMatOrder = Eigen::ColMajor>
struct DirectSolver : public SolverBase<SpMatT, DenseMatOrder>
{
    using IndexT = typename SpMatT::IndexT;
    using Type   = typename SpMatT::Type;

    DirectSolver(SpMatT* mat, PermuteMethod perm)
        : SolverBase<SpMatT, DenseMatOrder>(mat),
          m_perm(perm),
          m_perm_allocated(false),
          m_use_permute(false)
    {
        // cuSparse matrix descriptor
        CUSPARSE_ERROR(cusparseCreateMatDescr(&m_descr));
        CUSPARSE_ERROR(
            cusparseSetMatType(m_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        CUSPARSE_ERROR(
            cusparseSetMatDiagType(m_descr, CUSPARSE_DIAG_TYPE_NON_UNIT));
        CUSPARSE_ERROR(
            cusparseSetMatIndexBase(m_descr, CUSPARSE_INDEX_BASE_ZERO));

        // cuSparse sp handle
        CUSOLVER_ERROR(cusolverSpCreate(&m_cusolver_sphandle));
    }

    virtual ~DirectSolver()
    {

        if (m_perm_allocated) {
            GPU_FREE(m_d_solver_val);
            GPU_FREE(m_d_solver_row_ptr);
            GPU_FREE(m_d_solver_col_idx);
            GPU_FREE(m_d_permute);
            GPU_FREE(m_d_solver_x);
            GPU_FREE(m_d_solver_b);
            GPU_FREE(m_d_permute_map);

            free(m_h_permute);
            free(m_h_permute_map);
            free(m_h_solver_row_ptr);
            free(m_h_solver_col_idx);
        }
        CUSPARSE_ERROR(cusparseDestroyMatDescr(m_descr));
        CUSOLVER_ERROR(cusolverSpDestroy(m_cusolver_sphandle));
    }

    /**
     * @brief The lower level API of matrix permutation. Specify the reordering
     * type or simply NONE for no permutation. This should be called at the
     * beginning of the solving process. Any other function call order would be
     * undefined.
     */
    virtual void permute(RXMeshStatic& rx)
    {
        permute_alloc();

        if (m_perm == PermuteMethod::NONE) {
            RXMESH_WARN(
                "DirectSolver::permute() No permutation is specified. Continue "
                "without permutation!");
            m_use_permute = false;


            m_d_solver_row_ptr = m_mat->row_ptr(DEVICE);
            m_d_solver_col_idx = m_mat->col_idx(DEVICE);
            m_d_solver_val     = m_mat->val_ptr(DEVICE);

            return;
        }

        m_use_permute = true;


        if (m_perm == PermuteMethod::SYMRCM) {
            CUSOLVER_ERROR(cusolverSpXcsrsymrcmHost(m_cusolver_sphandle,
                                                    m_mat->rows(),
                                                    m_mat->non_zeros(),
                                                    m_descr,
                                                    m_h_solver_row_ptr,
                                                    m_h_solver_col_idx,
                                                    m_h_permute));
        } else if (m_perm == PermuteMethod::SYMAMD) {
            CUSOLVER_ERROR(cusolverSpXcsrsymamdHost(m_cusolver_sphandle,
                                                    m_mat->rows(),
                                                    m_mat->non_zeros(),
                                                    m_descr,
                                                    m_h_solver_row_ptr,
                                                    m_h_solver_col_idx,
                                                    m_h_permute));
        } else if (m_perm == PermuteMethod::NSTDIS) {
            CUSOLVER_ERROR(cusolverSpXcsrmetisndHost(m_cusolver_sphandle,
                                                     m_mat->rows(),
                                                     m_mat->non_zeros(),
                                                     m_descr,
                                                     m_h_solver_row_ptr,
                                                     m_h_solver_col_idx,
                                                     NULL,
                                                     m_h_permute));
        } else if (m_perm == PermuteMethod::GPUMGND) {
            mgnd_permute(rx, m_h_permute);

        } else if (m_perm == PermuteMethod::GPUND) {
            nd_permute(rx, m_h_permute);
        } else {
            RXMESH_ERROR("DirectSolver::permute() incompatible permute method");
        }


        assert(is_unique_permutation(m_mat->rows(), m_h_permute));

        // copy permutation to the device
        CUDA_ERROR(cudaMemcpyAsync(m_d_permute,
                                   m_h_permute,
                                   m_mat->rows() * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));

        // working space for permutation: B = A*Q*A^T
        // the permutation for matrix A which works only for the col and row
        // indices, the val will be done on device with the m_d_permute_map
        // only on the device since we don't need to access the permuted val on
        // the host at all
#pragma omp parallel for
        for (int j = 0; j < m_mat->non_zeros(); j++) {
            m_h_permute_map[j] = j;
        }

        size_t size_perm       = 0;
        void*  perm_buffer_cpu = NULL;

        CUSOLVER_ERROR(cusolverSpXcsrperm_bufferSizeHost(m_cusolver_sphandle,
                                                         m_mat->rows(),
                                                         m_mat->cols(),
                                                         m_mat->non_zeros(),
                                                         m_descr,
                                                         m_h_solver_row_ptr,
                                                         m_h_solver_col_idx,
                                                         m_h_permute,
                                                         m_h_permute,
                                                         &size_perm));

        perm_buffer_cpu = (void*)malloc(sizeof(char) * size_perm);

        // permute the matrix
        CUSOLVER_ERROR(cusolverSpXcsrpermHost(m_cusolver_sphandle,
                                              m_mat->rows(),
                                              m_mat->cols(),
                                              m_mat->non_zeros(),
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
                                   (m_mat->rows() + 1) * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpyAsync(m_d_solver_col_idx,
                                   m_h_solver_col_idx,
                                   m_mat->non_zeros() * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));

        // do the permutation for val on device
        CUDA_ERROR(cudaMemcpyAsync(m_d_permute_map,
                                   m_h_permute_map,
                                   m_mat->non_zeros() * sizeof(IndexT),
                                   cudaMemcpyHostToDevice));

        free(perm_buffer_cpu);
    }

    /**
     * @brief In permute, we permute the row_id and col_id. here we permute
     * the value pointer. We separate the logic here since we usually want to
     * to permute the rows/cols id once but we might want to permute the
     * value pointer many times.
     */
    void premute_value_ptr()
    {

        permute_gather(m_d_permute_map,
                       m_mat->val_ptr(DEVICE),
                       m_d_solver_val,
                       m_mat->non_zeros());
    }


    /**
     * @brief return a pointer to the host memory that holds the permutation
     */
    IndexT* get_h_permute()
    {
        return m_h_permute;
    }

    /**
     * @brief allocate all temp buffers needed for the solver low-level API
     */
    void permute_alloc()
    {
        if (m_perm == PermuteMethod::NONE) {
            return;
        }

        if (!m_perm_allocated) {
            m_perm_allocated = true;
            CUDA_ERROR(cudaMalloc((void**)&m_d_solver_val,
                                  m_mat->non_zeros() * sizeof(Type)));
            CUDA_ERROR(cudaMalloc((void**)&m_d_solver_row_ptr,
                                  (m_mat->rows() + 1) * sizeof(IndexT)));
            CUDA_ERROR(cudaMalloc((void**)&m_d_solver_col_idx,
                                  m_mat->non_zeros() * sizeof(IndexT)));

            m_h_solver_row_ptr =
                (IndexT*)malloc((m_mat->rows() + 1) * sizeof(IndexT));
            m_h_solver_col_idx =
                (IndexT*)malloc(m_mat->non_zeros() * sizeof(IndexT));

            m_h_permute = (IndexT*)malloc(m_mat->rows() * sizeof(IndexT));
            CUDA_ERROR(cudaMalloc((void**)&m_d_permute,
                                  m_mat->rows() * sizeof(IndexT)));

            m_h_permute_map = static_cast<IndexT*>(
                malloc(m_mat->non_zeros() * sizeof(IndexT)));

            CUDA_ERROR(cudaMalloc((void**)&m_d_permute_map,
                                  m_mat->non_zeros() * sizeof(IndexT)));

            CUDA_ERROR(cudaMalloc((void**)&m_d_solver_x,
                                  m_mat->cols() * sizeof(Type)));
            CUDA_ERROR(cudaMalloc((void**)&m_d_solver_b,
                                  m_mat->rows() * sizeof(Type)));
        }
        std::memcpy(m_h_solver_row_ptr,
                    m_mat->row_ptr(),
                    (m_mat->rows() + 1) * sizeof(IndexT));
        std::memcpy(m_h_solver_col_idx,
                    m_mat->col_idx(),
                    m_mat->non_zeros() * sizeof(IndexT));
    }


   protected:
    int permute_to_int() const
    {
        switch (m_perm) {
            case PermuteMethod::NONE:
                return 0;
            case PermuteMethod::SYMRCM:
                return 1;
            case PermuteMethod::SYMAMD:
                return 2;
            case PermuteMethod::NSTDIS:
                return 3;
            default: {
                RXMESH_ERROR("DirectSolver::permute_to_int() unknown input");
                return 0;
            }
        }
    }

    __host__ void permute_scatter(IndexT* d_p,
                                  Type*   d_in,
                                  Type*   d_out,
                                  IndexT  size)
    {
        // d_out[d_p[i]] = d_in[i]
        thrust::device_ptr<IndexT> t_p(d_p);
        thrust::device_ptr<Type>   t_i(d_in);
        thrust::device_ptr<Type>   t_o(d_out);

        thrust::scatter(thrust::device, t_i, t_i + size, t_p, t_o);
    }

    __host__ void permute_gather(IndexT* d_p,
                                 Type*   d_in,
                                 Type*   d_out,
                                 IndexT  size)
    {
        // d_out[i] = d_in[d_p[i]]
        thrust::device_ptr<IndexT> t_p(d_p);
        thrust::device_ptr<Type>   t_i(d_in);
        thrust::device_ptr<Type>   t_o(d_out);

        thrust::gather(thrust::device, t_p, t_p + size, t_i, t_o);
    }


    PermuteMethod m_perm;

    bool m_perm_allocated;
    bool m_use_permute;

    cusparseMatDescr_t m_descr;
    cusolverSpHandle_t m_cusolver_sphandle;

    // permutation array
    IndexT* m_h_permute;
    IndexT* m_d_permute;
    IndexT* m_h_permute_map;
    IndexT* m_d_permute_map;

    // permuted CSR matrix
    IndexT* m_d_solver_row_ptr;
    IndexT* m_d_solver_col_idx;
    Type*   m_d_solver_val;

    IndexT* m_h_solver_row_ptr;
    IndexT* m_h_solver_col_idx;

    // permuted lhs and rhs
    Type* m_d_solver_b;
    Type* m_d_solver_x;
};

}  // namespace rxmesh