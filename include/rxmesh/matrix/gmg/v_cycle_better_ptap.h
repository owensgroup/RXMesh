#pragma once
#include "v_cycle.h"

using namespace rxmesh;

template <typename T>
struct VCycle_Better : public rxmesh::VCycle<T>
{
    using rxmesh::VCycle<T>::VCycle;  // Inherit constructors

    // Override the new_ptap method
    void new_ptap(SparseMatrix<T> p,
                  SparseMatrix<T> p_t,
                  SparseMatrix<T> new_a,
                  SparseMatrix<T> old_a)
    {
        constexpr uint32_t blockThreads = 256;
        uint32_t           blocks_new   = DIVIDE_UP(p.cols(), blockThreads);
        uint32_t           blocks_old   = DIVIDE_UP(old_a.rows(), blockThreads);
        for_each_item<<<blocks_new, blockThreads>>>(
            p.cols(), [old_a, p_t, p, new_a] __device__(int i) mutable {

                printf("%d", i);

                const int MAX_UNIQUE = 64;
                int       vals[MAX_UNIQUE];  // register
                int       count = 0;

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
                            if (!already_seen && count < MAX_UNIQUE) 
                            {
                                vals[count++] = p_col;
                                printf("\n%d is a non-sparse entry in row %d",
                                       p_col,
                                       i);
                            }
                        }
                    }
                }
                printf("\nThread %d found %d unique entries: ", i, count);
            });


    }

    void get_intermediate_laplacians(GMG<T>& gmg, SparseMatrix<T>& A) override
    {
        SparseMatrix<T> p_t = gmg.m_prolong_op[0].transpose();
        new_ptap(gmg.m_prolong_op[0], p_t, m_a[0].a, A);

        //base class function call so that everything functions temporarily
        rxmesh::VCycle<T>::get_intermediate_laplacians(gmg,A);
    }
};
