#pragma once

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/dense_matrix.cuh"
#include "rxmesh/matrix/sparse_matrix.cuh"

namespace rxmesh {

template <typename T>
struct JacobiSolver
{
    DenseMatrix<T> m_x_new;

    __host__ JacobiSolver(int num_rows, int num_cols)
        : m_x_new(num_rows, num_cols)
    {
    }


    template <int numCol>
    __host__ void solve(const SparseMatrix<T>& A,
                        const DenseMatrix<T>&  b,
                        DenseMatrix<T>&        x,
                        int                    num_iter)
    {
        constexpr uint32_t blockThreads = 256;

        uint32_t blocks = DIVIDE_UP(A.rows(), blockThreads);


        for (int iter = 0; iter < num_iter; ++iter) {

            for_each_item<<<blocks, blockThreads>>>(

                A.rows(), [=] __device__(int row) mutable {
                    T    diag     = 0.0f;
                    bool has_diag = false;

                    T sum[numCol];

                    for (int c = 0; c < numCol; ++c) {
                        sum[c] = 0;
                    }

                    int start = A.row_ptr()[row];
                    int stop  = A.row_ptr()[row + 1];

                    for (int j = start; j < stop; ++j) {

                        int col = A.col_idx()[j];

                        T val = A.get_val_at(j);

                        if (col == row) {
                            // If it's a diagonal element, store its value
                            diag     = val;
                            has_diag = true;
                        } else {
                            // Sum non-diagonal elements
                            for (int c = 0; c < numCol; ++c) {
                                sum[c] += val * x(col, c);
                            }
                        }
                    }

                    assert(has_diag);

                    if (has_diag && abs(diag) > 10e-8f) {
                        for (int c = 0; c < numCol; ++c) {
                            m_x_new(row, c) = (b(row, c) - sum[c]) / diag;
                        }

                    } else {
                        for (int c = 0; c < numCol; ++c) {
                            m_x_new(row, c) = x(row, c);
                        }
                    }
                });

            x.swap(m_x_new);
        }
    }
};
}  // namespace rxmesh