#pragma once

#include "rxmesh/matrix/sparse_matrix.cuh"

/**
 * @brief compute the number of nnz that will result if we compute Cholesky
 * decomposition on an input matrix. Taken from
 * Eigen::SimplicialCholeskyBase::analyzePattern_preordered
 */
template <typename T>
int compute_chol_nnz(const rxmesh::SparseMatrix<T>& mat)
{
    const int size = mat.rows();

    std::vector<int> parent(size);
    std::vector<int> nonZerosPerCol(size);
    std::vector<int> tags(size);
    int              nnz = 0;

    for (int r = 0; r < size; ++r) {
        /* L(r,:) pattern: all nodes reachable in etree from nz in A(0:r-1,r) */
        parent[r]         = -1; /* parent of r is not yet known */
        tags[r]           = r;  /* mark node r as visited */
        nonZerosPerCol[r] = 0;  /* count of nonzeros in column r of L */

        int start = mat.row_ptr()[r];
        int end   = mat.row_ptr()[r + 1];

        for (int i = start; i < end; ++i) {
            int c = mat.col_idx()[i];

            if (c < r) {
                /* follow path from c to root of etree, stop at flagged node */
                for (; tags[c] != r; c = parent[c]) {
                    /* find parent of c if not yet determined */
                    if (parent[c] == -1)
                        parent[c] = r;
                    nonZerosPerCol[c]++; /* L (r,c) is nonzero */
                    nnz++;
                    tags[c] = r; /* mark c as visited */
                }
            }
        }
    }

    return nnz;
}