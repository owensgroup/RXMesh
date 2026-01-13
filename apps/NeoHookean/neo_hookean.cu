#include "rxmesh/rxmesh_static.h"

#include "rxmesh/geometry_factory.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"

#include "barrier_energy.h"
#include "bending_energy.h"
#include "boundary_condition.h"
#include "draw.h"
#include "friction_energy.h"
#include "gravity_energy.h"
#include "inertial_energy.h"
#include "init.h"
#include "neo_hookean_energy.h"
#include "scene_config.h"
#include "spring_energy.h"

#include <Eigen/Core>
#include <unordered_set>
#include <thrust/sequence.h>

using namespace rxmesh;

using T = float;

/**
 * @brief Simple COO matrix structure for sparsity pattern comparison
 */
struct COOMatrix {
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    int num_rows;
    int num_cols;
    int nnz;
};

/**
 * CuSPARSE-based Hessian update implementation for benchmarking
 * This provides a baseline comparison against the custom optimized implementation
 */
template <typename ProblemT>
COOMatrix update_hessian_cusparse(ProblemT& problem, RXMeshStatic& rx, const bool return_matrix)
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
        detail::add_vf_pairs_to_vv_pairs(
            rx, problem, problem.vf_pairs, problem.vv_pairs, *problem.face_interact_vertex);
    }

    // Get current pair data
    int num_new_pairs = problem.vv_pairs.num_index();
    auto d_new_rows = problem.vv_pairs.m_pairs_id.col_data(0);
    auto d_new_cols = problem.vv_pairs.m_pairs_id.col_data(1);
    // printf("[CuSPARSE] After expansion: num_new_pairs=%d\n", num_new_pairs);

    // Convert existing Hessian from CSR to COO format using CuSPARSE
    auto& hess = *problem.hess;
    int num_rows = hess.rows();
    int nnz = hess.non_zeros();
    // printf("[CuSPARSE] Hessian: rows=%d, nnz=%d\n", num_rows, nnz);

    // Get CSR pointers from Hessian using accessor methods
    const int* d_csr_row_ptr = hess.row_ptr(DEVICE);
    const int* d_csr_col_idx = hess.col_idx(DEVICE);
    const T*   d_csr_val = hess.val_ptr(DEVICE);

    // Create cusparse handle for this function
    cusparseHandle_t cusparse_handle;
    CUSPARSE_ERROR(cusparseCreate(&cusparse_handle));
    // printf("[CuSPARSE] Created cusparse handle\n");

    // Preallocate COO arrays with capacity for existing + new entries
    // This avoids reallocation and reduces memory traffic
    int total_nnz = nnz + num_new_pairs;
    int* d_coo_row = nullptr;
    int* d_coo_col = nullptr;
    T*   d_coo_val = nullptr;

    // printf("[CuSPARSE] Allocating COO arrays: total_nnz=%d\n", total_nnz);
    CUDA_ERROR(cudaMalloc(&d_coo_row, total_nnz * sizeof(int)));
    CUDA_ERROR(cudaMalloc(&d_coo_col, total_nnz * sizeof(int)));
    CUDA_ERROR(cudaMalloc(&d_coo_val, total_nnz * sizeof(T)));

    // Convert existing CSR to COO directly into preallocated arrays
    // printf("[CuSPARSE] Converting CSR to COO\n");
    CUSPARSE_ERROR(cusparseXcsr2coo(
        cusparse_handle,
        d_csr_row_ptr,
        nnz,
        num_rows,
        d_coo_row,  // Write directly to first nnz entries
        CUSPARSE_INDEX_BASE_ZERO));

    // Copy column indices and values for existing entries
    // printf("[CuSPARSE] Copying column indices and values\n");
    CUDA_ERROR(cudaMemcpy(d_coo_col, d_csr_col_idx, nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaMemcpy(d_coo_val, d_csr_val, nnz * sizeof(T), cudaMemcpyDeviceToDevice));

    // Append new contact pairs to preallocated arrays (initialized with zero values)
    // printf("[CuSPARSE] Appending %d new contact pairs\n", num_new_pairs);
    CUDA_ERROR(cudaMemcpy(d_coo_row + nnz, d_new_rows, num_new_pairs * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaMemcpy(d_coo_col + nnz, d_new_cols, num_new_pairs * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_ERROR(cudaMemset(d_coo_val + nnz, 1, num_new_pairs * sizeof(T)));

    // Sort the COO arrays by (row, col) using cusparseXcoosort
    size_t buffer_size = 0;
    void* d_temp_buffer = nullptr;
    int* d_permutation = nullptr;

    CUDA_ERROR(cudaMalloc(&d_permutation, total_nnz * sizeof(int)));

    // // Get buffer size for sorting
    CUSPARSE_ERROR(cusparseXcoosort_bufferSizeExt(
        cusparse_handle, num_rows, num_rows, total_nnz,
        d_coo_row, d_coo_col, &buffer_size));

    CUDA_ERROR(cudaMalloc(&d_temp_buffer, buffer_size));

    // // Create identity permutation
    // CUSPARSE_ERROR(cusparseCreateIdentityPermutation(cusparse_handle, total_nnz, d_permutation));
    thrust::sequence(thrust::device_ptr<int>(d_permutation), thrust::device_ptr<int>(d_permutation + total_nnz));

    // // Sort COO by row (and column within row)
    CUSPARSE_ERROR(cusparseXcoosortByRow(
        cusparse_handle, num_rows, num_rows, total_nnz,
        d_coo_row, d_coo_col, d_permutation, d_temp_buffer));

    // // Apply permutation to values
    // if constexpr (std::is_same_v<T, float>) {
    //     CUSPARSE_ERROR(cusparseSgthr(cusparse_handle, total_nnz, d_coo_val,
    //                                  d_coo_val, d_permutation, CUSPARSE_INDEX_BASE_ZERO));
    // } else if constexpr (std::is_same_v<T, double>) {
    //     CUSPARSE_ERROR(cusparseDgthr(cusparse_handle, total_nnz, d_coo_val,
    //                                  d_coo_val, d_permutation, CUSPARSE_INDEX_BASE_ZERO));
    // }

    // Now convert sorted COO back to CSR using cusparseXcoo2csr
    // printf("[CuSPARSE] Converting COO back to CSR\n");
    int* d_new_csr_row_ptr = nullptr;
    CUDA_ERROR(cudaMalloc(&d_new_csr_row_ptr, (num_rows + 1) * sizeof(int)));

    CUSPARSE_ERROR(cusparseXcoo2csr(
        cusparse_handle, d_coo_row, total_nnz, num_rows,
        d_new_csr_row_ptr, CUSPARSE_INDEX_BASE_ZERO));

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
        // printf("[CuSPARSE] Converting final CSR back to COO for verification\n");

        // Allocate device arrays for COO format
        int* d_final_coo_row = nullptr;
        int* d_final_coo_col = nullptr;
        CUDA_ERROR(cudaMalloc(&d_final_coo_row, total_nnz * sizeof(int)));
        CUDA_ERROR(cudaMalloc(&d_final_coo_col, total_nnz * sizeof(int)));

        // Convert CSR to COO
        CUSPARSE_ERROR(cusparseXcsr2coo(
            cusparse_handle,
            d_new_csr_row_ptr,
            total_nnz,
            num_rows,
            d_final_coo_row,
            CUSPARSE_INDEX_BASE_ZERO));

        // Copy column indices (they're already sorted in d_coo_col)
        CUDA_ERROR(cudaMemcpy(d_final_coo_col, d_coo_col,
                             total_nnz * sizeof(int), cudaMemcpyDeviceToDevice));

        // printf("[CuSPARSE] Converted CSR to COO\n");

        // Copy COO data to host
        result.num_rows = num_rows;
        result.num_cols = num_rows;  // Square matrix
        result.nnz = total_nnz;
        result.row_idx.resize(total_nnz);
        result.col_idx.resize(total_nnz);

        CUDA_ERROR(cudaMemcpy(result.row_idx.data(), d_final_coo_row,
                             total_nnz * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(result.col_idx.data(), d_final_coo_col,
                             total_nnz * sizeof(int), cudaMemcpyDeviceToHost));
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
    // printf("\n[VERIFY] Starting sparsity pattern comparison in COO format\n");

    // Get dimensions from custom Hessian
    int custom_rows = hess_custom.rows();
    int custom_cols = hess_custom.cols();
    int custom_nnz = hess_custom.non_zeros();

    // printf("[VERIFY] Custom Hessian: %d x %d, nnz = %d\n",
    //        custom_rows, custom_cols, custom_nnz);
    // printf("[VERIFY] CuSPARSE Hessian: %d x %d, nnz = %d\n",
    //        hess_cusparse_coo.num_rows, hess_cusparse_coo.num_cols, hess_cusparse_coo.nnz);

    // Check dimensions
    bool dimensions_match = true;
    if (custom_rows != hess_cusparse_coo.num_rows) {
        printf("[VERIFY] ERROR: Row count mismatch! Custom=%d, CuSPARSE=%d\n",
               custom_rows, hess_cusparse_coo.num_rows);
        dimensions_match = false;
    }
    if (custom_cols != hess_cusparse_coo.num_cols) {
        printf("[VERIFY] ERROR: Column count mismatch! Custom=%d, CuSPARSE=%d\n",
               custom_cols, hess_cusparse_coo.num_cols);
        dimensions_match = false;
    }
    if (custom_nnz != hess_cusparse_coo.nnz) {
        printf("[VERIFY] ERROR: NNZ count mismatch! Custom=%d, CuSPARSE=%d\n",
               custom_nnz, hess_cusparse_coo.nnz);
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
    CUSPARSE_ERROR(cusparseXcsr2coo(
        cusparse_handle,
        d_custom_row_ptr,
        custom_nnz,
        custom_rows,
        d_custom_coo_row,
        CUSPARSE_INDEX_BASE_ZERO));

    // Copy column indices
    CUDA_ERROR(cudaMemcpy(d_custom_coo_col, d_custom_col_idx,
                         custom_nnz * sizeof(int), cudaMemcpyDeviceToDevice));

    // Copy custom COO to host
    std::vector<int> custom_coo_row(custom_nnz);
    std::vector<int> custom_coo_col(custom_nnz);

    CUDA_ERROR(cudaMemcpy(custom_coo_row.data(), d_custom_coo_row,
                         custom_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(custom_coo_col.data(), d_custom_coo_col,
                         custom_nnz * sizeof(int), cudaMemcpyDeviceToHost));

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
        cusparse_entries.push_back({hess_cusparse_coo.row_idx[i], hess_cusparse_coo.col_idx[i]});
    }

    // Sort both entry lists
    // printf("[VERIFY] Sorting COO entries for comparison\n");
    std::sort(custom_entries.begin(), custom_entries.end());
    std::sort(cusparse_entries.begin(), cusparse_entries.end());

    // Check for duplicates using std::unique
    // auto custom_unique_end = std::unique(custom_entries.begin(), custom_entries.end());
    // auto cusparse_unique_end = std::unique(cusparse_entries.begin(), cusparse_entries.end());

    // int custom_unique_count = std::distance(custom_entries.begin(), custom_unique_end);
    // int cusparse_unique_count = std::distance(cusparse_entries.begin(), cusparse_unique_end);

    // int custom_duplicates = custom_nnz - custom_unique_count;
    // int cusparse_duplicates = hess_cusparse_coo.nnz - cusparse_unique_count;

    // if (custom_duplicates > 0) {
    //     printf("[VERIFY] WARNING: Custom Hessian has %d duplicate entries!\n", custom_duplicates);
    // }
    // if (cusparse_duplicates > 0) {
    //     printf("[VERIFY] WARNING: CuSPARSE Hessian has %d duplicate entries!\n", cusparse_duplicates);
    // }

    // Compare sorted entries
    bool entries_match = true;
    int num_mismatches = 0;
    int first_duplicate_row_id = -1;
    for (int i = 0; i < custom_nnz; i++) {
        if (i > 0) {
            if (custom_entries[i] == custom_entries[i - 1]) {
                printf("[VERIFY] Custom Hessian has duplicate entry: %d %d\n", custom_entries[i].first, custom_entries[i].second);
                first_duplicate_row_id = custom_entries[i].first;
                break;
            }
            if (cusparse_entries[i] == cusparse_entries[i - 1]) {
                printf("[VERIFY] cusparse Hessian has duplicate entry: %d %d\n", cusparse_entries[i].first, cusparse_entries[i].second);
            }
        }

        if (custom_entries[i] != cusparse_entries[i]) {
            if (num_mismatches == 0) {
                printf("[VERIFY] ERROR: First entry mismatch at index %d:\n",
                       i);
                printf("[VERIFY]   Custom: (%d, %d), CuSPARSE: (%d, %d)\n",
                       custom_entries[i].first, custom_entries[i].second,
                       cusparse_entries[i].first, cusparse_entries[i].second);
            }
            entries_match = false;
            num_mismatches++;
            if (num_mismatches < 10) {
                printf("[VERIFY]   Mismatch %d: Custom=(%d,%d), CuSPARSE=(%d,%d)\n",
                       num_mismatches,
                       custom_entries[i].first, custom_entries[i].second,
                       cusparse_entries[i].first, cusparse_entries[i].second);
            }
        }
    }

    // If we found a duplicate, inspect that row in detail
    if (first_duplicate_row_id != -1) {
        printf("[VERIFY] Inspecting row %d in custom Hessian (CSR format):\n", first_duplicate_row_id);

        const int* h_custom_row_ptr = hess_custom.row_ptr(HOST);
        const int* h_custom_col_idx = hess_custom.col_idx(HOST);

        int row_start = h_custom_row_ptr[first_duplicate_row_id];
        int row_end = h_custom_row_ptr[first_duplicate_row_id + 1];
        int row_nnz = row_end - row_start;

        printf("[VERIFY] Row %d has %d entries in CSR format (indices %d to %d):\n",
               first_duplicate_row_id, row_nnz, row_start, row_end - 1);

        // Use std::set to track seen columns and identify duplicates
        std::set<int> seen_cols;
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

        printf("[VERIFY] Row has %zu unique columns and %zu duplicate entries\n",
               seen_cols.size(), duplicate_cols.size());

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
               num_mismatches, custom_nnz);
    }

    // printf("[VERIFY] Overall result: %s\n\n",
    //        entries_match ? "MATCH ✓" : "MISMATCH ✗");

    return entries_match;
}

struct PhysicsParams {
    T   density        = 1000;   // rho
    T   young_mod      = 1e5;    // E
    T   poisson_ratio  = 0.4;    // nu
    T   time_step      = 0.01;   // h
    T   fricition_coef = 0.11;   // mu
    T   stiffness_coef = 4e4;
    T   tol            = 0.01;
    T   dhat           = 0.1;
    T   kappa          = 1e5;
    T   bending_stiff  = 1e8;    // k_b
    std::vector<int> export_steps;  // List of step IDs to export as OBJ
    int num_steps      = 5;      // Number of simulation steps
};

void neo_hookean(RXMeshStatic& rx, T dx, const PhysicsParams& params)
{
    // printf("neo_hookean: Starting function\n");

    // Toggle to enable CuSPARSE benchmark comparison
    constexpr bool BENCHMARK_CUSPARSE = true;
    constexpr bool VERIFY_SPARSITY = true;

    constexpr int VariableDim = 3;

    constexpr uint32_t blockThreads = 256;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    using HessMatT = typename ProblemT::HessMatT;

    // Problem parameters
    const int max_vv_candidate_pairs = 5000;
    const int max_vf_candidate_pairs = 5000;

    const T        density        = params.density;
    const T        young_mod      = params.young_mod;
    const T        poisson_ratio  = params.poisson_ratio;
    const T        time_step      = params.time_step;
    const T        fricition_coef = params.fricition_coef;
    const T        stiffness_coef = params.stiffness_coef;
    const T        tol            = params.tol;
    const T        inv_time_step  = T(1) / time_step;
    const T dhat          = params.dhat;
    const T kappa         = params.kappa;
    const T bending_stiff = params.bending_stiff;

    // TODO the limits and velocity should be different for different Dirichlet
    // nodes
    const vec3<T> ground_o(0.0f, -1.0f, 0.0f);  // a point on the slope
    const vec3<T> ground_n =
        glm::normalize(vec3<T>(0.0f, 1.0f, 0.0f));  // normal of the slope


    // Derived parameters
    const T mu_lame = 0.5 * young_mod / (1 + poisson_ratio);
    const T lam     = young_mod * poisson_ratio /
                  ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));

    glm::vec3 bb_lower(0), bb_upper(0);
    rx.bounding_box(bb_lower, bb_upper);
    glm::vec3 bb = bb_upper - bb_lower;

    // mass per vertex = rho * volume /num_vertices
    T total_volume = bb[0] * bb[1] * bb[2];
    T mass = density * total_volume /
             (rx.get_num_vertices());  // m

    // printf("neo_hookean: Setting up attributes\n");

    // Attributes
    auto velocity = *rx.add_vertex_attribute<T>("Velocity", 3);  // v
    velocity.reset(0, DEVICE);

    auto volume = *rx.add_face_attribute<T>("Volume", 1);  // vol
    volume.reset(0, DEVICE);

    auto mu_lambda = *rx.add_vertex_attribute<T>("mu_lambda", 1);  // mu_lambda
    mu_lambda.reset(0, DEVICE);

    auto inv_b =
        *rx.add_face_attribute<Eigen::Matrix<T, 3, 3>>("InvB", 1);  // IB

    auto contact_area = *rx.add_vertex_attribute<T>("ContactArea", 1);
    contact_area.reset(dx, DEVICE);  // perimeter split to each vertex

    // Bending energy attributes
    auto rest_angle = *rx.add_edge_attribute<T>("RestAngle", 1);
    rest_angle.reset(0, DEVICE);

    auto edge_area = *rx.add_edge_attribute<T>("EdgeArea", 1);
    edge_area.reset(0, DEVICE);

    // Get region labels for multiple meshes
    auto vertex_region_label = *rx.get_vertex_region_label();
    auto face_region_label = *rx.get_face_region_label();

    // Store vertex handles for each face (3 vertices per face)
    auto face_vertices = *rx.add_face_attribute<uint64_t>("FaceVertices", 3);

    // Diff problem and solvers
    ProblemT problem(rx, true, max_vv_candidate_pairs, max_vf_candidate_pairs);

    // Pre-allocate BVH bounding boxes buffer for contact detection
    BVHBuffers<T> vertex_bvh_buffers(rx.get_num_vertices());
    BVHBuffers<T> face_bvh_buffers(rx.get_num_faces());

    // CGSolver<T, ProblemT::DenseMatT::OrderT> solver(*problem.hess, 1, 1000);
    PCGSolver<T, ProblemT::DenseMatT::OrderT> solver(*problem.hess, 1, 1000);

    // CholeskySolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(
    //     problem.hess.get());

    NetwtonSolver newton_solver(problem, &solver);

    auto& x = *problem.objective;
    x.copy_from(*rx.get_input_vertex_coordinates(), DEVICE, DEVICE);

    auto x_n     = *rx.add_vertex_attribute_like("x_n", x);
    auto x_tilde = *rx.add_vertex_attribute_like("x_tilde", x);


    // printf("neo_hookean: Running initializations\n");

    // Initializations
    init_volume_inverse_b(rx, x, volume, inv_b);
    // printf("neo_hookean: Finished init_volume_inverse_b\n");

    init_bending(rx, x, rest_angle, edge_area);
    // printf("neo_hookean: Finished init_bending\n");

    // Initialize face_vertices: store the 3 vertex handles for each face
    init_face_vertices(rx, face_vertices);

    // // Debug: print bending initialization stats
    // rest_angle.move(DEVICE, HOST);
    // edge_area.move(DEVICE, HOST);

    // T min_angle = std::numeric_limits<T>::max();
    // T max_angle = std::numeric_limits<T>::lowest();
    // T min_area = std::numeric_limits<T>::max();
    // T max_area = std::numeric_limits<T>::lowest();
    // int num_internal_edges = 0;

    // rx.for_each_edge(HOST, [&](EdgeHandle eh) {
    //     T angle = rest_angle(eh);
    //     T area = edge_area(eh);
    //     if (area > 0) {  // internal edge
    //         num_internal_edges++;
    //         min_angle = std::min(min_angle, angle);
    //         max_angle = std::max(max_angle, angle);
    //         min_area = std::min(min_area, area);
    //         max_area = std::max(max_area, area);
    //     }
    // });

    // RXMESH_INFO("Bending initialization:");
    // RXMESH_INFO("  Internal edges: {}", num_internal_edges);
    // RXMESH_INFO("  Rest angle range: [{}, {}] radians", min_angle, max_angle);
    // RXMESH_INFO("  Edge area range: [{}, {}]", min_area, max_area);

    // rest_angle.move(HOST, DEVICE);
    // edge_area.move(HOST, DEVICE);


    typename ProblemT::DenseMatT alpha(
        rx, std::max(rx.get_num_vertices(), rx.get_num_faces()), 1, DEVICE);

#if USE_POLYSCOPE
    // add volume to polyscope
    volume.move(DEVICE, HOST);
    rx.get_polyscope_mesh()->addFaceScalarQuantity("Volume", volume);
#endif

    // printf("neo_hookean: Adding energy terms\n");

    // add inertial energy term
    inertial_energy(problem, x_tilde, mass);
    // printf("neo_hookean: Added inertial energy\n");

    // add gravity energy
    gravity_energy(problem, time_step, mass);
    // printf("neo_hookean: Added gravity energy\n");

    // add barrier energy
    floor_barrier_energy(problem,
                         contact_area,
                         time_step,
                         ground_n,
                         ground_o,
                         dhat,
                         kappa);

    // printf("neo_hookean: Added floor barrier energy\n");

    vv_contact_energy(problem, contact_area, time_step, dhat, kappa);
    // printf("neo_hookean: Added vv contact energy\n");

    vf_contact_energy(problem, contact_area, time_step, dhat, kappa);
    // printf("neo_hookean: Added vf contact energy\n");

    DenseMatrix<T, Eigen::RowMajor> dir(
        rx, problem.grad.rows(), problem.grad.cols(), LOCATION_ALL);

    DenseMatrix<T, Eigen::RowMajor> grad(
        rx, problem.grad.rows(), problem.grad.cols(), LOCATION_ALL);

    T line_search_init_step = 0;

    // add friction energy
    // TODO alpha should change during different runs (e.g., in the line search)
    // friction_energy(problem,
    //                x_n,
    //                newton_solver.dir,
    //                line_search_init_step,
    //                mu_lambda,
    //                time_step,
    //                ground_n);

    // add neo hooken energy
    neo_hookean_energy(problem, volume, inv_b, mu_lame, time_step, lam);
    // printf("neo_hookean: Added neo-hookean energy\n");

    // add bending energy
    bending_energy(problem, rest_angle, edge_area, bending_stiff, time_step);
    // printf("neo_hookean: Added bending energy\n");


    int steps = 0;

    Timers<GPUTimer> timer;
    timer.add("Step");
    timer.add("ContactDetection_Explicit");
    timer.add("ContactDetection_LineSearch");
    timer.add("ContactDetection_PostLineSearch");
    timer.add("ContactDetection");
    timer.add("EnergyEval");
    timer.add("LinearSolver");
    timer.add("LineSearch");
    timer.add("StepSize");
    timer.add("UpdateHessian");
    timer.add("UpdateHessian_CuSPARSE");

    auto step_forward = [&]() {
        // printf("neo_hookean: step_forward() - Starting step %d\n", steps);

        // x_tilde = x + v*h
        timer.start("Step");
        rx.for_each_vertex(DEVICE, [=] __device__(VertexHandle vh) mutable {
            for (int i = 0; i < 3; ++i) {
                x_tilde(vh, i) = x(vh, i) + time_step * velocity(vh, i);
            }
        });

        // copy current position
        x_n.copy_from(x, DEVICE, DEVICE);

        // compute mu * lambda for each node using x_n
        /*compute_mu_lambda(rx,
                          fricition_coef,
                          dhat,
                          kappa,
                          ground_n,
                          ground_o,
                          x,
                          contact_area,
                          mu_lambda);*/


        // evaluate energy
        // printf("neo_hookean: step_forward() - Adding contact\n");
        timer.start("ContactDetection_Explicit");
        add_contact(problem,
                    rx,
                    problem.vv_pairs,
                    problem.vf_pairs,
                    vertex_bvh_buffers,
                    face_bvh_buffers,
                    x,
                    contact_area,
                    time_step,
                    dhat,
                    kappa,
                    vertex_region_label,
                    face_region_label,
                    face_vertices);
        timer.stop("ContactDetection_Explicit");

        // printf("neo_hookean: step_forward() - Updating hessian\n");
        timer.start("EnergyEval");
        problem.update_hessian();
        // printf("neo_hookean: step_forward() - Evaluating terms\n");
        problem.eval_terms();
        timer.stop("EnergyEval");
        // printf("neo_hookean: step_forward() - Finished evaluating terms\n");

        grad.copy_from(problem.grad, DEVICE, DEVICE);

        // get newton direction
        // printf("neo_hookean: step_forward() - Computing newton direction\n");
        timer.start("LinearSolver");
        newton_solver.compute_direction();
        timer.stop("LinearSolver");
        // printf("neo_hookean: step_forward() - Finished computing newton direction\n");

        dir.copy_from(newton_solver.dir, DEVICE, DEVICE);
        // residual is abs_max(newton_dir)/ h
        T residual = newton_solver.dir.abs_max() / time_step;
        // printf("neo_hookean: step_forward() - Initial residual: %f\n", residual);

        T f = problem.get_current_loss();
        RXMESH_INFO(
            "*******Step: {}, Energy: {}, Residual: {}",
            steps,
            f,
            residual);

        int iter = 0;
        // printf("neo_hookean: step_forward() - Entering iteration loop\n");
        while (residual > tol) {
            // printf("neo_hookean: step_forward() - Iteration %d, residual: %f, num_satisfied: %d\n", iter, residual, num_satisfied);
            // printf("neo_hookean: step_forward() - Iteration %d, residual: %f \n", iter, residual);

            // printf("neo_hookean: step_forward() - Computing neo_hookean_step_size\n");
            timer.start("StepSize");
            T nh_step = neo_hookean_step_size(rx, x, newton_solver.dir, alpha);
            // printf("neo_hookean: step_forward() - nh_step: %f\n", nh_step);

            // printf("neo_hookean: step_forward() - Computing barrier_step_size\n");
            T bar_step = barrier_step_size(rx,
                                           newton_solver.dir,
                                           alpha,
                                           x,
                                           ground_n,
                                           ground_o);
            // printf("neo_hookean: step_forward() - bar_step: %f\n", bar_step);

            line_search_init_step = std::min(nh_step, bar_step);
            timer.stop("StepSize");
            // printf("neo_hookean: step_forward() - line_search_init_step: %f\n", line_search_init_step);

            // TODO: line search should pass the step to the friction energy
            // printf("neo_hookean: step_forward() - Starting line search\n");
            timer.start("LineSearch");
            bool ls_success = newton_solver.line_search(
                line_search_init_step, 0.5, 64, 0.0, [&](auto temp_x) {
                    timer.start("ContactDetection_LineSearch");
                    add_contact(problem,
                                rx,
                                problem.vv_pairs,
                                problem.vf_pairs,
                                vertex_bvh_buffers,
                                face_bvh_buffers,
                                temp_x,
                                contact_area,
                                time_step,
                                dhat,
                                kappa,
                                vertex_region_label,
                                face_region_label,
                                face_vertices);
                    timer.stop("ContactDetection_LineSearch");
                });
            timer.stop("LineSearch");
            // printf("neo_hookean: step_forward() - Finished line search, success: %d\n", ls_success);

            if (!ls_success) {
                RXMESH_WARN("Line search failed!");
            }

            // evaluate energy
            // printf("neo_hookean: step_forward() - Re-evaluating energy after line search\n");
            timer.start("ContactDetection_PostLineSearch");
            add_contact(problem,
                        rx,
                        problem.vv_pairs,
                        problem.vf_pairs,
                        vertex_bvh_buffers,
                        face_bvh_buffers,
                        x,
                        contact_area,
                        time_step,
                        dhat,
                        kappa,
                        vertex_region_label,
                        face_region_label,
                        face_vertices);
            timer.stop("ContactDetection_PostLineSearch");

            // printf("neo_hookean: step_forward() - Updating hessian after line search\n");
            timer.start("UpdateHessian");
            problem.update_hessian();
            timer.stop("UpdateHessian");

            // CuSPARSE baseline comparison (for benchmarking)
            if constexpr (BENCHMARK_CUSPARSE) {
                timer.start("UpdateHessian_CuSPARSE");
                COOMatrix cusparse_hess_coo = update_hessian_cusparse(problem, rx, VERIFY_SPARSITY);
                timer.stop("UpdateHessian_CuSPARSE");

                // Verify sparsity patterns match
                if (VERIFY_SPARSITY) {
                    bool patterns_match = verify_sparsity_patterns(*problem.hess, cusparse_hess_coo);
                    if (patterns_match) {
                        printf("[VERIFY] Sparsity patterns MATCH ✓\n");
                    } else {
                        printf("[VERIFY] Sparsity patterns MISMATCH ✗\n");
                    }
                }
            }

            // printf("neo_hookean: step_forward() - Evaluating terms after line search\n");
            timer.start("EnergyEval");
            problem.eval_terms();
            timer.stop("EnergyEval");

            T f = problem.get_current_loss();
            // printf("neo_hookean: step_forward() - Current loss: %f\n", f);

            // get newton direction
            // printf("neo_hookean: step_forward() - Computing newton direction (iteration %d)\n", iter);
            timer.start("LinearSolver");
            newton_solver.compute_direction();
            timer.stop("LinearSolver");

            // residual is abs_max(newton_dir)/ h
            residual = newton_solver.dir.abs_max() / time_step;
            // printf("neo_hookean: step_forward() - New residual: %f\n", residual);

            RXMESH_INFO(
                "  Subsetp: {}, F: {}, R: {}, line_search_init_step={}, ",
                iter,
                f,
                residual,
                line_search_init_step);

            iter++;

            if (iter > 10) {
                break;
            }
        }

        RXMESH_INFO("===================");

        //  update velocity
        rx.for_each_vertex(
            DEVICE,
            [x, x_n, velocity, inv_time_step = 1.0 / time_step] __device__(
                VertexHandle vh) mutable {
                for (int i = 0; i < 3; ++i) {
                    velocity(vh, i) = inv_time_step * (x(vh, i) - x_n(vh, i));

                    // x(vh, i) = x_tilde(vh, i);
                }
            });

        steps++;
        timer.stop("Step");
    };

    // printf("declared everything. starting simulation.\n");
// #if USE_POLYSCOPE
//     draw(rx, x, velocity, step_forward, dir, grad, steps);
// #else
    // Convert export_steps vector to set for O(1) lookup
    std::unordered_set<int> export_set(params.export_steps.begin(), params.export_steps.end());

    while (steps < params.num_steps) {
        step_forward();

        // Check if we should export at this step
        if (export_set.count(steps - 1) > 0) {  // steps is already incremented in step_forward()
            // Move vertex positions to HOST
            x.move(DEVICE, HOST);

            // Create filename with step number
            std::string filename = STRINGIFY(OUTPUT_DIR) + std::string("scene_step_") +
                                   std::to_string(steps - 1) + ".obj";

            RXMESH_INFO("Exporting mesh at step {} to {}", steps - 1, filename);
            rx.export_obj(filename, x);

            // Move back to DEVICE for next iteration
            x.move(HOST, DEVICE);
        }
    }
// #endif


    // Print comprehensive timing summary
    RXMESH_INFO("=== TIMING SUMMARY ===");
    RXMESH_INFO("Number of steps: {}", steps);
    RXMESH_INFO("Total Step Time:        {:.2f} ms ({:.2f} ms/iter)",
                timer.elapsed_millis("Step"),
                timer.elapsed_millis("Step") / float(steps));

    // Contact Detection broken down by context
    T total_contact = timer.elapsed_millis("ContactDetection_Explicit") +
                      timer.elapsed_millis("ContactDetection_LineSearch") +
                      timer.elapsed_millis("ContactDetection_PostLineSearch");
    RXMESH_INFO("  Contact Detection (Total): {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                total_contact,
                100.0 * total_contact / timer.elapsed_millis("Step"),
                total_contact / float(steps));
    RXMESH_INFO("    - Explicit:          {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("ContactDetection_Explicit"),
                100.0 * timer.elapsed_millis("ContactDetection_Explicit") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("ContactDetection_Explicit") / float(steps));
    RXMESH_INFO("    - LineSearch:        {:.2f} ms  ({:.1f}%)",
                timer.elapsed_millis("ContactDetection_LineSearch"),
                100.0 * timer.elapsed_millis("ContactDetection_LineSearch") / timer.elapsed_millis("Step"));
    RXMESH_INFO("    - PostLineSearch:    {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("ContactDetection_PostLineSearch"),
                100.0 * timer.elapsed_millis("ContactDetection_PostLineSearch") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("ContactDetection_PostLineSearch") / float(steps));

    RXMESH_INFO("  Energy Evaluation:    {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("EnergyEval"),
                100.0 * timer.elapsed_millis("EnergyEval") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("EnergyEval") / float(steps));
    RXMESH_INFO("  Linear Solver:        {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("LinearSolver"),
                100.0 * timer.elapsed_millis("LinearSolver") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("LinearSolver") / float(steps));
    RXMESH_INFO("  Line Search:          {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("LineSearch"),
                100.0 * timer.elapsed_millis("LineSearch") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("LineSearch") / float(steps));
    RXMESH_INFO("  Step Size Compute:    {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("StepSize"),
                100.0 * timer.elapsed_millis("StepSize") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("StepSize") / float(steps));
    RXMESH_INFO("  Update Hessian Compute:    {:.2f} ms  ({:.1f}%) [{:.2f} ms/iter]",
                timer.elapsed_millis("UpdateHessian"),
                100.0 * timer.elapsed_millis("UpdateHessian") / timer.elapsed_millis("Step"),
                timer.elapsed_millis("UpdateHessian") / float(steps));

    // Print CuSPARSE comparison if enabled
    if constexpr (BENCHMARK_CUSPARSE) {
        T cusparse_time = timer.elapsed_millis("UpdateHessian_CuSPARSE");
        T custom_time = timer.elapsed_millis("UpdateHessian");
        T speedup = cusparse_time / custom_time;
        RXMESH_INFO("");
        RXMESH_INFO("=== Hessian Update Benchmark ===");
        RXMESH_INFO("Custom implementation:  {:.2f} ms ({:.2f} ms/iter)",
                    custom_time, custom_time / float(steps));
        RXMESH_INFO("CuSPARSE baseline:     {:.2f} ms ({:.2f} ms/iter)",
                    cusparse_time, cusparse_time / float(steps));
        RXMESH_INFO("Speedup: {:.2f}x", speedup);
        RXMESH_INFO("================================");
    }

    RXMESH_INFO("======================");
}

/**
 * Struct to store per-instance transformation data
 */
template <typename T>
struct InstanceTransform {
    T tx, ty, tz;  // Translation
    T scale;        // Scale
    T vx, vy, vz;  // Initial velocity
};

/**
 * Generate mesh instances from scene configuration
 * Returns: vector of mesh filepaths (with duplicates for instances)
 *          vector of transforms (one per instance)
 */
template <typename T>
void generate_instances(const SceneConfig<T>& config,
                        std::vector<std::string>& mesh_paths,
                        std::vector<InstanceTransform<T>>& transforms)
{
    mesh_paths.clear();
    transforms.clear();

    for (const auto& mesh_cfg : config.meshes) {
        int total_instances = mesh_cfg.num_instances_x *
                              mesh_cfg.num_instances_y *
                              mesh_cfg.num_instances_z;

        RXMESH_INFO("Generating {} instances of {}", total_instances, mesh_cfg.filepath);

        // Generate instances in 3D grid
        for (int iz = 0; iz < mesh_cfg.num_instances_z; ++iz) {
            for (int iy = 0; iy < mesh_cfg.num_instances_y; ++iy) {
                for (int ix = 0; ix < mesh_cfg.num_instances_x; ++ix) {
                    // Add mesh filepath
                    mesh_paths.push_back(mesh_cfg.filepath);

                    // Compute instance transform
                    InstanceTransform<T> transform;
                    transform.tx = mesh_cfg.offset_x + ix * mesh_cfg.spacing_x;
                    transform.ty = mesh_cfg.offset_y + iy * mesh_cfg.spacing_y;
                    transform.tz = mesh_cfg.offset_z + iz * mesh_cfg.spacing_z;
                    transform.scale = mesh_cfg.scale;
                    transform.vx = mesh_cfg.velocity_x;
                    transform.vy = mesh_cfg.velocity_y;
                    transform.vz = mesh_cfg.velocity_z;

                    transforms.push_back(transform);
                }
            }
        }
    }

    RXMESH_INFO("Total instances generated: {}", mesh_paths.size());
}

int main(int argc, char** argv)
{
    rx_init(0, spdlog::level::info);

    // Check for config file argument
    std::string config_file;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
            break;
        }
    }

    // Parse command line arguments for physics parameters
    PhysicsParams params;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config") {
            i++;  // Skip the config file path
            continue;
        } else if (arg == "--density" && i + 1 < argc) {
            params.density = std::atof(argv[++i]);
        } else if (arg == "--young" && i + 1 < argc) {
            params.young_mod = std::atof(argv[++i]);
        } else if (arg == "--poisson" && i + 1 < argc) {
            params.poisson_ratio = std::atof(argv[++i]);
        } else if (arg == "--timestep" && i + 1 < argc) {
            params.time_step = std::atof(argv[++i]);
        } else if (arg == "--friction" && i + 1 < argc) {
            params.fricition_coef = std::atof(argv[++i]);
        } else if (arg == "--stiffness" && i + 1 < argc) {
            params.stiffness_coef = std::atof(argv[++i]);
        } else if (arg == "--tol" && i + 1 < argc) {
            params.tol = std::atof(argv[++i]);
        } else if (arg == "--dhat" && i + 1 < argc) {
            params.dhat = std::atof(argv[++i]);
        } else if (arg == "--kappa" && i + 1 < argc) {
            params.kappa = std::atof(argv[++i]);
        } else if (arg == "--bending" && i + 1 < argc) {
            params.bending_stiff = std::atof(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            params.num_steps = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --config <file>    Load scene from configuration file\n");
            printf("  --density <val>    Density (default: 1000)\n");
            printf("  --young <val>      Young's modulus (default: 1e5)\n");
            printf("  --poisson <val>    Poisson ratio (default: 0.4)\n");
            printf("  --timestep <val>   Time step (default: 0.01)\n");
            printf("  --friction <val>   Friction coefficient (default: 0.11)\n");
            printf("  --stiffness <val>  Stiffness coefficient (default: 4e4)\n");
            printf("  --tol <val>        Tolerance (default: 0.01)\n");
            printf("  --dhat <val>       Contact distance threshold (default: 0.1)\n");
            printf("  --kappa <val>      Contact stiffness (default: 1e5)\n");
            printf("  --bending <val>    Bending stiffness (default: 1e3)\n");
            printf("  --steps <val>      Number of simulation steps (default: 5)\n");
            printf("  --help, -h         Show this help message\n");
            return 0;
        }
    }

    // Load meshes and apply transforms
    std::vector<std::string> inputs;
    std::vector<InstanceTransform<T>> transforms;

    if (!config_file.empty()) {
        // Parse config file
        RXMESH_INFO("Loading scene from config file: {}", config_file);
        SceneConfig<T> scene_config;
        if (!SceneConfigParser<T>::parse(config_file, scene_config)) {
            RXMESH_ERROR("Failed to parse config file");
            return 1;
        }

        // Override params with simulation config
        params.time_step = scene_config.simulation.time_step;
        params.fricition_coef = scene_config.simulation.fricition_coef;
        params.stiffness_coef = scene_config.simulation.stiffness_coef;
        params.tol = scene_config.simulation.tol;
        params.dhat = scene_config.simulation.dhat;
        params.kappa = scene_config.simulation.kappa;
        params.num_steps = scene_config.simulation.num_steps;
        params.export_steps = scene_config.simulation.export_steps;

        // Generate instances
        generate_instances(scene_config, inputs, transforms);
    } else {
        // Default: load two spheres (backward compatible)
        inputs = {
            STRINGIFY(INPUT_DIR) "el_topo_sphere_1280.obj",
            STRINGIFY(INPUT_DIR) "el_topo_sphere_1280.obj"
        };

        // Create default transforms
        InstanceTransform<T> t1{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
        InstanceTransform<T> t2{0.0, 2.5, 0.0, 1.0, 0.0, 0.0, 0.0};
        transforms = {t1, t2};
    }

    RXMESH_INFO("Physics Parameters:");
    RXMESH_INFO("  Density: {}", params.density);
    RXMESH_INFO("  Young's modulus: {}", params.young_mod);
    RXMESH_INFO("  Poisson ratio: {}", params.poisson_ratio);
    RXMESH_INFO("  Time step: {}", params.time_step);
    RXMESH_INFO("  Friction coefficient: {}", params.fricition_coef);
    RXMESH_INFO("  Stiffness coefficient: {}", params.stiffness_coef);
    RXMESH_INFO("  Tolerance: {}", params.tol);
    RXMESH_INFO("  dhat: {}", params.dhat);
    RXMESH_INFO("  kappa: {}", params.kappa);
    RXMESH_INFO("  Bending stiffness: {}", params.bending_stiff);
    RXMESH_INFO("  Number of steps: {}", params.num_steps);

    // Load meshes
    RXMeshStatic rx(inputs);
    RXMESH_INFO("#Faces: {}, #Vertices: {}", rx.get_num_faces(), rx.get_num_vertices());

    T dx = 0.1f;  // mesh spacing for contact area
    auto x = *rx.get_input_vertex_coordinates();
    auto velocity = *rx.add_vertex_attribute<T>("Velocity", 3);
    velocity.reset(0, DEVICE);

    // Apply transformations per instance
    auto vertex_region_label = *rx.get_vertex_region_label();
    x.move(DEVICE, HOST);
    rx.for_each_vertex(
        HOST,
        [=] __host__ (VertexHandle vh) mutable {
            int region = vertex_region_label(vh);
            if (region >= 0 && region < static_cast<int>(transforms.size())) {
                const auto& t = transforms[region];

                // Apply scale
                if (t.scale != T(1.0)) {
                    x(vh, 0) *= t.scale;
                    x(vh, 1) *= t.scale;
                    x(vh, 2) *= t.scale;
                }

                // Apply translation
                x(vh, 0) += t.tx;
                x(vh, 1) += t.ty;
                x(vh, 2) += t.tz;

                // Set initial velocity
                velocity(vh, 0) = t.vx;
                velocity(vh, 1) = t.vy;
                velocity(vh, 2) = t.vz;
            }
        },
        NULL,
        false
    );

    x.move(HOST, DEVICE);
#if USE_POLYSCOPE
    rx.get_polyscope_mesh()->updateVertexPositions(x);
#endif

    neo_hookean(rx, dx, params);
}
