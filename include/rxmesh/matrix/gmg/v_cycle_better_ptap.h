#pragma once
#include "v_cycle.h"

using namespace rxmesh;


template <typename T>
struct VCycle_Better : public rxmesh::VCycle<T>
{
    using rxmesh::VCycle<T>::VCycle;  // Inherit constructors

    std::vector<CoarseA<T>> m_verification_a;  // levels


    int*   m_d_entries;
    int*   m_d_row_ptr;
    void*  m_d_temp_storage     = nullptr;
    size_t m_temp_storage_bytes = 0;


    VCycle_Better(GMG<T>&               gmg,
                  RXMeshStatic&         rx,
                  SparseMatrix<T>&      A,
                  const DenseMatrix<T>& rhs,
                  CoarseSolver          coarse_solver  = CoarseSolver::Jacobi,
                  int                   num_pre_relax  = 2,
                  int                   num_post_relax = 2)
        : VCycle<T>(gmg,
                    rx,
                    A,
                    rhs,
                    coarse_solver,
                    num_pre_relax,
                    num_post_relax)
    {

        CUDA_ERROR(cudaMalloc(&m_d_entries,
                             sizeof(int) * gmg.m_prolong_op[0].cols()));
        CUDA_ERROR(cudaMalloc(&m_d_row_ptr,
                   sizeof(int) * (gmg.m_prolong_op[0].cols() + 1)));


        cub::DeviceScan::ExclusiveSum(m_d_temp_storage,
                                      m_temp_storage_bytes,
                                      m_d_entries,
                                      m_d_row_ptr,
                                      gmg.m_prolong_op[0].cols() + 1);

        CUDA_ERROR(cudaMalloc(&m_d_temp_storage, m_temp_storage_bytes));
    }

    template <int MAX_UNIQUE = 2500>
    void new_ptap(SparseMatrix<T>  p,
                  SparseMatrix<T>  p_t,
                  SparseMatrix<T>& new_a,
                  SparseMatrix<T>  old_a)
    {
        constexpr uint32_t blockThreads = 256;
        uint32_t           blocks_new   = DIVIDE_UP(p.cols(), blockThreads);
        uint32_t           blocks_old   = DIVIDE_UP(old_a.rows(), blockThreads);


        int*   d_entries     = m_d_entries;
        int*   d_row_ptr_tmp = m_d_row_ptr;
        void*  d_temp        = m_d_temp_storage;
        size_t temp_bytes    = m_temp_storage_bytes;

        for_each_item<<<blocks_new, blockThreads>>>(
            p.cols(), [d_entries, old_a, p_t, p] __device__(int i) mutable {
                // maybe we add shared memory later
                // const int MAX_UNIQUE = 64;
                int vals[MAX_UNIQUE];  // register
                int count = 0;

                for (int p_iter = p_t.row_ptr()[i];
                     p_iter < p_t.row_ptr()[i + 1];
                     ++p_iter) {
                    int p_t_col = p_t.col_idx()[p_iter];

                    for (int q = old_a.row_ptr()[p_t_col];
                         q < old_a.row_ptr()[p_t_col + 1];
                         ++q) {
                        int a_col = old_a.col_idx()[q];
                        for (int k = p.row_ptr()[a_col];
                             k < p.row_ptr()[a_col + 1];
                             ++k) {
                            int p_col = p.col_idx()[k];

                            bool already_seen = false;
                            for (int j = 0; j < count; ++j) {
                                if (vals[j] == p_col) {
                                    already_seen = true;
                                    break;
                                }
                            }
                            // Add if unique
                            if (!already_seen && count < MAX_UNIQUE) {
                                vals[count++] = p_col;
                            }
                        }
                    }
                }
                // printf("\nThread %d found %d unique entries: ", i, count);
                d_entries[i] = count;
            });


        int num_rows = p.cols();

        cub::DeviceScan::ExclusiveSum(
            d_temp, temp_bytes, d_entries, d_row_ptr_tmp, num_rows + 1);

        int h_nnz;
        CUDA_ERROR(cudaMemcpy(&h_nnz,
                   d_row_ptr_tmp + num_rows,
                   sizeof(int),
                   cudaMemcpyDeviceToHost));

        int* d_row_ptr;
        CUDA_ERROR(cudaMalloc(&d_row_ptr, sizeof(int) * (num_rows + 1)));
        CUDA_ERROR(cudaMemcpy(d_row_ptr,
                   d_row_ptr_tmp,
                   sizeof(int) * (num_rows + 1),
                   cudaMemcpyDeviceToDevice));

        int* d_col_idx;
        T*   d_val;
        CUDA_ERROR(cudaMalloc(&d_col_idx, h_nnz * sizeof(int)));
        CUDA_ERROR(cudaMalloc(&d_val, h_nnz * sizeof(T)));

        // multiply
        for_each_item<<<blocks_new, blockThreads>>>(
            p.cols(),
            [old_a, p_t, p, new_a, d_row_ptr, d_col_idx, d_val] __device__(
                int i) mutable {
                int offset = d_row_ptr[i];
                int local  = 0;

                int vals[MAX_UNIQUE];
                int pcol_to_idx[MAX_UNIQUE];
                int count = 0;

                for (int p_iter = p_t.row_ptr()[i];
                     p_iter < p_t.row_ptr()[i + 1];
                     ++p_iter) {
                    int   m    = p_t.col_idx()[p_iter];
                    float P_mi = p_t.get_val_at(p_iter);  // P_{m i}

                    for (int a_iter = old_a.row_ptr()[m];
                         a_iter < old_a.row_ptr()[m + 1];
                         ++a_iter) {
                        int   n    = old_a.col_idx()[a_iter];
                        float A_mn = old_a.get_val_at(a_iter);  // A_{m n}

                        for (int p_iter2 = p.row_ptr()[n];
                             p_iter2 < p.row_ptr()[n + 1];
                             ++p_iter2) {
                            int   j    = p.col_idx()[p_iter2];
                            float P_nj = p.get_val_at(p_iter2);  // P_{n j}

                            float contrib = P_mi * A_mn * P_nj;

                            // Accumulate
                            int match_idx = -1;
                            for (int k = 0; k < count; ++k) {
                                if (vals[k] == j) {
                                    match_idx = k;
                                    break;
                                }
                            }

                            if (match_idx == -1) {
                                vals[count]               = j;
                                pcol_to_idx[count]        = local;
                                d_col_idx[offset + local] = j;
                                d_val[offset + local]     = contrib;
                                ++local;
                                ++count;
                            } else {
                                int idx = pcol_to_idx[match_idx];
                                d_val[offset + idx] += contrib;
                            }
                        }
                    }
                }

            });


        int* h_row_ptr = new int[num_rows + 1];
        int* h_col_idx = new int[h_nnz];
        T*   h_val     = new T[h_nnz];

        CUDA_ERROR(cudaMemcpy(h_row_ptr,
                   d_row_ptr,
                   sizeof(int) * (num_rows + 1),
                   cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(
            h_col_idx, d_col_idx, sizeof(int) * h_nnz, cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(
            h_val, d_val, sizeof(T) * h_nnz, cudaMemcpyDeviceToHost));

        SparseMatrix<T> a(num_rows,
                          num_rows,
                          h_nnz,
                          d_row_ptr,
                          d_col_idx,
                          d_val,
                          h_row_ptr,
                          h_col_idx,
                          h_val);
        new_a = a;
    }

    void verify_laplacians(GMG<T>& gmg, SparseMatrix<T>& A) override
    {
        const double error_threshold = 1e-3;
        m_verification_a.resize(gmg.m_num_levels - 1);
        pt_A_p(gmg.m_prolong_op[0], A, m_verification_a[0]);
        for (int l = 1; l < gmg.m_num_levels - 1; ++l) {
            pt_A_p(gmg.m_prolong_op[l],
                   m_verification_a[l - 1].a,
                   m_verification_a[l]);
        }
        constexpr uint32_t blockThreads = 256;

        for (int i = 0; i < gmg.m_num_levels - 1; ++i) {
            auto our_a     = m_a[i].a;
            auto correct_a = m_verification_a[i].a;
            if (!our_a.non_zeros() == correct_a.non_zeros()) 
            {
                printf("\nERROR: NUMBER OF NONZERO VALUES ARE NOT MATCHING");
            }

            for_each_item<<<our_a.rows(), blockThreads>>>(
                our_a.rows(),
                [our_a, correct_a, error_threshold] __device__(int i) mutable {
                    for (int iter = correct_a.row_ptr()[i];
                         iter < correct_a.row_ptr()[i + 1];
                         ++iter) {
                        int col1 = correct_a.col_idx()[iter];
                        T   val1 = correct_a.get_val_at(iter);

                        for (int iter2 = our_a.row_ptr()[i];
                             iter2 <= our_a.row_ptr()[i + 1];
                             ++iter2) {
                            if (iter2 == our_a.row_ptr()[i + 1]) {
                                printf(
                                    "\nERROR: UNABLE TO FIND COLUMN ENTRY %d "
                                    "IN ROW %d",
                                    col1,
                                    i);
                                break;
                            }

                            int col2 = our_a.col_idx()[iter2];
                            if (col2 == col1) {
                                T val2 = our_a.get_val_at(iter2);

                                if (abs(val1 - val2) < error_threshold) {
                                    // good case
                                } else {
                                    printf(
                                        "\nERROR VALUES INCORRECT AT ROW %d: "
                                        "%f != %f",
                                        i,
                                        val1,
                                        val2);
                                }
                                break;
                            }
                        }
                    }
                });
        }
    }

    void get_intermediate_laplacians(GMG<T>& gmg, SparseMatrix<T>& A) override
    {
        for (int i = 0; i < gmg.m_num_levels - 1; i++) {
            SparseMatrixConstantNNZRow<float, 3> p_const = gmg.m_prolong_op[i];
            auto                                 p_t     = p_const.transpose();
            if (i == 0) {
                new_ptap(p_const, p_t, m_a[i].a, A);
            } else {
                new_ptap(p_const, p_t, m_a[i].a, m_a[i - 1].a);
            }
        }
    }
};
