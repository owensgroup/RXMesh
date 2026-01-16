#pragma once 

#include <thrust/sequence.h>

/**
 * @brief Simple COO matrix structure for sparsity pattern comparison
 */
struct COOMatrix
{
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    int              num_rows;
    int              num_cols;
    int              nnz;
};

/**
 * CuSPARSE-based Hessian update implementation for benchmarking
 * This provides a baseline comparison against the custom optimized
 * implementation
 */
template <typename ProblemT, typename T>
COOMatrix update_hessian_cusparse(ProblemT&     problem,
                                  RXMeshStatic& rx,
                                  const bool    return_matrix)
{
    if (!problem.hess) {
        // printf("[CuSPARSE] Hessian not initialized, returning\n");
        return COOMatrix{};
    }

    // For a fair comparison, we replicate the same logic as update_hessian()
    // but use CuSPARSE primitives where applicable

    int vv_prv_num_index = problem.vv_pairs.num_index();
    int vv_prv_num_pairs = problem.vv_pairs.num_pairs();
    // printf("[CuSPARSE] Initial vv_pairs: num_index=%d, num_pairs=%d\n",
    //        vv_prv_num_index, vv_prv_num_pairs);

    // Expand VF pairs to VV pairs (same as custom implementation)
    if (problem.face_interact_vertex) {
        // printf("[CuSPARSE] Expanding VF pairs to VV pairs\n");
        detail::add_vf_pairs_to_vv_pairs(rx,
                                         problem,
                                         problem.vf_pairs,
                                         problem.vv_pairs,
                                         *problem.face_interact_vertex);
    }

    // Get current pair data
    int  num_new_pairs = problem.vv_pairs.num_index();
    auto d_new_rows    = problem.vv_pairs.m_pairs_id.col_data(0);
    auto d_new_cols    = problem.vv_pairs.m_pairs_id.col_data(1);
    // printf("[CuSPARSE] After expansion: num_new_pairs=%d\n", num_new_pairs);

    // Convert existing Hessian from CSR to COO format using CuSPARSE
    auto& hess     = *problem.hess;
    int   num_rows = hess.rows();
    int   nnz      = hess.non_zeros();
    // printf("[CuSPARSE] Hessian: rows=%d, nnz=%d\n", num_rows, nnz);

    // Get CSR pointers from Hessian using accessor methods
    const int* d_csr_row_ptr = hess.row_ptr(DEVICE);
    const int* d_csr_col_idx = hess.col_idx(DEVICE);
    const T*   d_csr_val     = hess.val_ptr(DEVICE);

    // Create cusparse handle for this function
    cusparseHandle_t cusparse_handle;
    CUSPARSE_ERROR(cusparseCreate(&cusparse_handle));
    // printf("[CuSPARSE] Created cusparse handle\n");

    // Preallocate COO arrays with capacity for existing + new entries
    // This avoids reallocation and reduces memory traffic
    int  total_nnz = nnz + num_new_pairs;
    int* d_coo_row = nullptr;
    int* d_coo_col = nullptr;
    T*   d_coo_val = nullptr;

    // printf("[CuSPARSE] Allocating COO arrays: total_nnz=%d\n", total_nnz);
    CUDA_ERROR(cudaMalloc(&d_coo_row, total_nnz * sizeof(int)));
    CUDA_ERROR(cudaMalloc(&d_coo_col, total_nnz * sizeof(int)));
    CUDA_ERROR(cudaMalloc(&d_coo_val, total_nnz * sizeof(T)));

    // Convert existing CSR to COO directly into preallocated arrays
    // printf("[CuSPARSE] Converting CSR to COO\n");
    CUSPARSE_ERROR(
        cusparseXcsr2coo(cusparse_handle,
                         d_csr_row_ptr,
                         nnz,
                         num_rows,
                         d_coo_row,  // Write directly to first nnz entries
                         CUSPARSE_INDEX_BASE_ZERO));

    // Copy column indices and values for existing entries
    // printf("[CuSPARSE] Copying column indices and values\n");
    CUDA_ERROR(cudaMemcpy(
        d_coo_col, d_csr_col_idx, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaMemcpy(
        d_coo_val, d_csr_val, nnz * sizeof(T), cudaMemcpyDeviceToDevice));

    // Append new contact pairs to preallocated arrays (initialized with zero
    // values) printf("[CuSPARSE] Appending %d new contact pairs\n",
    // num_new_pairs);
    CUDA_ERROR(cudaMemcpy(d_coo_row + nnz,
                          d_new_rows,
                          num_new_pairs * sizeof(int),
                          cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaMemcpy(d_coo_col + nnz,
                          d_new_cols,
                          num_new_pairs * sizeof(int),
                          cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaMemset(d_coo_val + nnz, 1, num_new_pairs * sizeof(T)));

    // Sort the COO arrays by (row, col) using cusparseXcoosort
    size_t buffer_size   = 0;
    void*  d_temp_buffer = nullptr;
    int*   d_permutation = nullptr;

    CUDA_ERROR(cudaMalloc(&d_permutation, total_nnz * sizeof(int)));

    // // Get buffer size for sorting
    CUSPARSE_ERROR(cusparseXcoosort_bufferSizeExt(cusparse_handle,
                                                  num_rows,
                                                  num_rows,
                                                  total_nnz,
                                                  d_coo_row,
                                                  d_coo_col,
                                                  &buffer_size));

    CUDA_ERROR(cudaMalloc(&d_temp_buffer, buffer_size));

    // // Create identity permutation
    // CUSPARSE_ERROR(cusparseCreateIdentityPermutation(cusparse_handle,
    // total_nnz, d_permutation));
    thrust::sequence(thrust::device_ptr<int>(d_permutation),
                     thrust::device_ptr<int>(d_permutation + total_nnz));

    // // Sort COO by row (and column within row)
    CUSPARSE_ERROR(cusparseXcoosortByRow(cusparse_handle,
                                         num_rows,
                                         num_rows,
                                         total_nnz,
                                         d_coo_row,
                                         d_coo_col,
                                         d_permutation,
                                         d_temp_buffer));

    // // Apply permutation to values
    // if constexpr (std::is_same_v<T, float>) {
    //     CUSPARSE_ERROR(cusparseSgthr(cusparse_handle, total_nnz, d_coo_val,
    //                                  d_coo_val, d_permutation,
    //                                  CUSPARSE_INDEX_BASE_ZERO));
    // } else if constexpr (std::is_same_v<T, double>) {
    //     CUSPARSE_ERROR(cusparseDgthr(cusparse_handle, total_nnz, d_coo_val,
    //                                  d_coo_val, d_permutation,
    //                                  CUSPARSE_INDEX_BASE_ZERO));
    // }

    // Now convert sorted COO back to CSR using cusparseXcoo2csr
    // printf("[CuSPARSE] Converting COO back to CSR\n");
    int* d_new_csr_row_ptr = nullptr;
    CUDA_ERROR(cudaMalloc(&d_new_csr_row_ptr, (num_rows + 1) * sizeof(int)));

    CUSPARSE_ERROR(cusparseXcoo2csr(cusparse_handle,
                                    d_coo_row,
                                    total_nnz,
                                    num_rows,
                                    d_new_csr_row_ptr,
                                    CUSPARSE_INDEX_BASE_ZERO));

    // Now we have the updated Hessian in CSR format:
    // - d_new_csr_row_ptr: compressed row pointer
    // - d_coo_col: column indices (sorted)
    // - d_coo_val: values (sorted, with zeros for new pairs)

    // Update the Hessian with the new CSR representation
    // Note: This would require updating hess_new with the merged data
    // For now, we still use the custom insert as a fallback
    // printf("[CuSPARSE] Converted to CSR format\n");

    // Convert CSR back to COO for verification
    COOMatrix result;
    if (return_matrix) {
        // printf("[CuSPARSE] Converting final CSR back to COO for
        // verification\n");

        // Allocate device arrays for COO format
        int* d_final_coo_row = nullptr;
        int* d_final_coo_col = nullptr;
        CUDA_ERROR(cudaMalloc(&d_final_coo_row, total_nnz * sizeof(int)));
        CUDA_ERROR(cudaMalloc(&d_final_coo_col, total_nnz * sizeof(int)));

        // Convert CSR to COO
        CUSPARSE_ERROR(cusparseXcsr2coo(cusparse_handle,
                                        d_new_csr_row_ptr,
                                        total_nnz,
                                        num_rows,
                                        d_final_coo_row,
                                        CUSPARSE_INDEX_BASE_ZERO));

        // Copy column indices (they're already sorted in d_coo_col)
        CUDA_ERROR(cudaMemcpy(d_final_coo_col,
                              d_coo_col,
                              total_nnz * sizeof(int),
                              cudaMemcpyDeviceToDevice));

        // printf("[CuSPARSE] Converted CSR to COO\n");

        // Copy COO data to host
        result.num_rows = num_rows;
        result.num_cols = num_rows;  // Square matrix
        result.nnz      = total_nnz;
        result.row_idx.resize(total_nnz);
        result.col_idx.resize(total_nnz);

        CUDA_ERROR(cudaMemcpy(result.row_idx.data(),
                              d_final_coo_row,
                              total_nnz * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(result.col_idx.data(),
                              d_final_coo_col,
                              total_nnz * sizeof(int),
                              cudaMemcpyDeviceToHost));
        // printf("[CuSPARSE] Copied COO row_idx and col_idx to host\n");

        // Cleanup COO arrays
        GPU_FREE(d_final_coo_row);
        GPU_FREE(d_final_coo_col);
    }

    // Cleanup temporary arrays
    // printf("[CuSPARSE] Cleaning up temporary arrays\n");
    GPU_FREE(d_coo_row);
    GPU_FREE(d_coo_col);
    GPU_FREE(d_coo_val);
    GPU_FREE(d_permutation);
    GPU_FREE(d_temp_buffer);
    GPU_FREE(d_new_csr_row_ptr);

    // Cleanup cusparse handle
    CUSPARSE_ERROR(cusparseDestroy(cusparse_handle));
    // printf("[CuSPARSE] Cleanup complete\n");

    // Reset vv_pairs to previous state (same as custom implementation)
    if (problem.face_interact_vertex) {
        problem.vv_pairs.reset(vv_prv_num_pairs, vv_prv_num_index);
        // printf("[CuSPARSE] Reset vv_pairs to previous state\n");
    }

    return result;
}

/**
 * @brief Verify that two COO sparsity patterns match
 * @param hess_custom The custom Hessian matrix (on device, in CSR format)
 * @param hess_cusparse_coo The CuSPARSE-generated COO matrix (on host)
 * @return true if sparsity patterns match exactly, false otherwise
 */
template <typename T>
bool verify_sparsity_patterns(const HessianSparseMatrix<T, 3>& hess_custom,
                              const COOMatrix& hess_cusparse_coo)
{
    // printf("\n[VERIFY] Starting sparsity pattern comparison in COO
    // format\n");

    // Get dimensions from custom Hessian
    int custom_rows = hess_custom.rows();
    int custom_cols = hess_custom.cols();
    int custom_nnz  = hess_custom.non_zeros();

    // printf("[VERIFY] Custom Hessian: %d x %d, nnz = %d\n",
    //        custom_rows, custom_cols, custom_nnz);
    // printf("[VERIFY] CuSPARSE Hessian: %d x %d, nnz = %d\n",
    //        hess_cusparse_coo.num_rows, hess_cusparse_coo.num_cols,
    //        hess_cusparse_coo.nnz);

    // Check dimensions
    bool dimensions_match = true;
    if (custom_rows != hess_cusparse_coo.num_rows) {
        printf("[VERIFY] ERROR: Row count mismatch! Custom=%d, CuSPARSE=%d\n",
               custom_rows,
               hess_cusparse_coo.num_rows);
        dimensions_match = false;
    }
    if (custom_cols != hess_cusparse_coo.num_cols) {
        printf(
            "[VERIFY] ERROR: Column count mismatch! Custom=%d, CuSPARSE=%d\n",
            custom_cols,
            hess_cusparse_coo.num_cols);
        dimensions_match = false;
    }
    if (custom_nnz != hess_cusparse_coo.nnz) {
        printf("[VERIFY] ERROR: NNZ count mismatch! Custom=%d, CuSPARSE=%d\n",
               custom_nnz,
               hess_cusparse_coo.nnz);
        dimensions_match = false;
    }

    if (!dimensions_match) {
        return false;
    }

    // Convert custom Hessian from CSR to COO
    // printf("[VERIFY] Converting custom Hessian from CSR to COO\n");

    // Create cusparse handle
    cusparseHandle_t cusparse_handle;
    CUSPARSE_ERROR(cusparseCreate(&cusparse_handle));

    // Allocate device COO arrays for custom Hessian
    int* d_custom_coo_row = nullptr;
    int* d_custom_coo_col = nullptr;
    CUDA_ERROR(cudaMalloc(&d_custom_coo_row, custom_nnz * sizeof(int)));
    CUDA_ERROR(cudaMalloc(&d_custom_coo_col, custom_nnz * sizeof(int)));

    // Get custom Hessian CSR pointers
    const int* d_custom_row_ptr = hess_custom.row_ptr(DEVICE);
    const int* d_custom_col_idx = hess_custom.col_idx(DEVICE);

    // Convert CSR to COO
    CUSPARSE_ERROR(cusparseXcsr2coo(cusparse_handle,
                                    d_custom_row_ptr,
                                    custom_nnz,
                                    custom_rows,
                                    d_custom_coo_row,
                                    CUSPARSE_INDEX_BASE_ZERO));

    // Copy column indices
    CUDA_ERROR(cudaMemcpy(d_custom_coo_col,
                          d_custom_col_idx,
                          custom_nnz * sizeof(int),
                          cudaMemcpyDeviceToDevice));

    // Copy custom COO to host
    std::vector<int> custom_coo_row(custom_nnz);
    std::vector<int> custom_coo_col(custom_nnz);

    CUDA_ERROR(cudaMemcpy(custom_coo_row.data(),
                          d_custom_coo_row,
                          custom_nnz * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(custom_coo_col.data(),
                          d_custom_coo_col,
                          custom_nnz * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // printf("[VERIFY] Converted custom Hessian to COO and copied to host\n");

    // Cleanup device COO arrays and cusparse handle
    GPU_FREE(d_custom_coo_row);
    GPU_FREE(d_custom_coo_col);
    CUSPARSE_ERROR(cusparseDestroy(cusparse_handle));

    // Create sorted (row, col) pairs for both matrices
    std::vector<std::pair<int, int>> custom_entries;
    std::vector<std::pair<int, int>> cusparse_entries;

    custom_entries.reserve(custom_nnz);
    cusparse_entries.reserve(custom_nnz);

    for (int i = 0; i < custom_nnz; i++) {
        custom_entries.push_back({custom_coo_row[i], custom_coo_col[i]});
        cusparse_entries.push_back(
            {hess_cusparse_coo.row_idx[i], hess_cusparse_coo.col_idx[i]});
    }

    // Sort both entry lists
    // printf("[VERIFY] Sorting COO entries for comparison\n");
    std::sort(custom_entries.begin(), custom_entries.end());
    std::sort(cusparse_entries.begin(), cusparse_entries.end());

    // Check for duplicates using std::unique
    // auto custom_unique_end = std::unique(custom_entries.begin(),
    // custom_entries.end()); auto cusparse_unique_end =
    // std::unique(cusparse_entries.begin(), cusparse_entries.end());

    // int custom_unique_count = std::distance(custom_entries.begin(),
    // custom_unique_end); int cusparse_unique_count =
    // std::distance(cusparse_entries.begin(), cusparse_unique_end);

    // int custom_duplicates = custom_nnz - custom_unique_count;
    // int cusparse_duplicates = hess_cusparse_coo.nnz - cusparse_unique_count;

    // if (custom_duplicates > 0) {
    //     printf("[VERIFY] WARNING: Custom Hessian has %d duplicate
    //     entries!\n", custom_duplicates);
    // }
    // if (cusparse_duplicates > 0) {
    //     printf("[VERIFY] WARNING: CuSPARSE Hessian has %d duplicate
    //     entries!\n", cusparse_duplicates);
    // }

    // Compare sorted entries
    bool entries_match          = true;
    int  num_mismatches         = 0;
    int  first_duplicate_row_id = -1;
    for (int i = 0; i < custom_nnz; i++) {
        if (i > 0) {
            if (custom_entries[i] == custom_entries[i - 1]) {
                printf("[VERIFY] Custom Hessian has duplicate entry: %d %d\n",
                       custom_entries[i].first,
                       custom_entries[i].second);
                first_duplicate_row_id = custom_entries[i].first;
                break;
            }
            if (cusparse_entries[i] == cusparse_entries[i - 1]) {
                printf("[VERIFY] cusparse Hessian has duplicate entry: %d %d\n",
                       cusparse_entries[i].first,
                       cusparse_entries[i].second);
            }
        }

        if (custom_entries[i] != cusparse_entries[i]) {
            if (num_mismatches == 0) {
                printf("[VERIFY] ERROR: First entry mismatch at index %d:\n",
                       i);
                printf("[VERIFY]   Custom: (%d, %d), CuSPARSE: (%d, %d)\n",
                       custom_entries[i].first,
                       custom_entries[i].second,
                       cusparse_entries[i].first,
                       cusparse_entries[i].second);
            }
            entries_match = false;
            num_mismatches++;
            if (num_mismatches < 10) {
                printf(
                    "[VERIFY]   Mismatch %d: Custom=(%d,%d), "
                    "CuSPARSE=(%d,%d)\n",
                    num_mismatches,
                    custom_entries[i].first,
                    custom_entries[i].second,
                    cusparse_entries[i].first,
                    cusparse_entries[i].second);
            }
        }
    }

    // If we found a duplicate, inspect that row in detail
    if (first_duplicate_row_id != -1) {
        printf("[VERIFY] Inspecting row %d in custom Hessian (CSR format):\n",
               first_duplicate_row_id);

        const int* h_custom_row_ptr = hess_custom.row_ptr(HOST);
        const int* h_custom_col_idx = hess_custom.col_idx(HOST);

        int row_start = h_custom_row_ptr[first_duplicate_row_id];
        int row_end   = h_custom_row_ptr[first_duplicate_row_id + 1];
        int row_nnz   = row_end - row_start;

        printf(
            "[VERIFY] Row %d has %d entries in CSR format (indices %d to "
            "%d):\n",
            first_duplicate_row_id,
            row_nnz,
            row_start,
            row_end - 1);

        // Use std::set to track seen columns and identify duplicates
        std::set<int>    seen_cols;
        std::vector<int> duplicate_cols;

        for (int i = row_start; i < row_end; i++) {
            int col = h_custom_col_idx[i];
            if (seen_cols.find(col) != seen_cols.end()) {
                // This column already exists - it's a duplicate
                duplicate_cols.push_back(col);
            } else {
                seen_cols.insert(col);
            }
        }

        printf(
            "[VERIFY] Row has %zu unique columns and %zu duplicate entries\n",
            seen_cols.size(),
            duplicate_cols.size());

        if (!duplicate_cols.empty()) {
            printf("[VERIFY] Duplicate column indices: ");
            for (int col : duplicate_cols) {
                printf("%d ", col);
            }
            printf("\n");
        }

        printf("[VERIFY] All column indices (first 30): ");
        for (int i = row_start; i < row_end && i - row_start < 30; i++) {
            printf("%d ", h_custom_col_idx[i]);
        }
        if (row_end - row_start > 30) {
            printf("... (%d more)", row_end - row_start - 30);
        }
        printf("\n");
    }

    if (entries_match) {
        // printf("[VERIFY] COO entries match perfectly ✓\n");
    } else {
        printf("[VERIFY] ERROR: COO entries mismatch ✗\n");
        printf("[VERIFY] Total mismatches: %d out of %d entries\n",
               num_mismatches,
               custom_nnz);
    }

    // printf("[VERIFY] Overall result: %s\n\n",
    //        entries_match ? "MATCH ✓" : "MISMATCH ✗");

    return entries_match;
}