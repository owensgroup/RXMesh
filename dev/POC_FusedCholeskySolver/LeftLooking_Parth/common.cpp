//
// Created by behrooz on 25/10/22.
//
#include "LeftLooking_Parth.h"
#include <unsupported/Eigen/SparseExtra>

namespace PARTH {

ParthSolver::ParthSolver(void)
{
    ///@brief Init options to a good default value
    opt.setReorderingType(ReorderingType::METIS);
    opt.setAnalyzeType(AnalyzeType::SimulationAware);
    opt.setNumericReuseType(NumericReuseType::NUMERIC_REUSE_BASE);
    opt.setVerbose(true);
    opt.setComputeResidual(true);
    opt.setNumRegions(8);
    ///@bierf Hessian related variables
    Ap = nullptr;
    Ai = nullptr;
    Ax = nullptr;
    A_nnz = 0;
    A_n = 0;
    cholmod_start(&A_cm);
    chol_A = nullptr;
    chol_Z = nullptr;
    chol_Y = nullptr;
    chol_E = nullptr;
    chol_R = nullptr;
    chol_A_prev = nullptr;

    chol_L = nullptr;
    chol_L_prev = nullptr;

    chol_b = nullptr;
    chol_sol = nullptr;
    chol_Ap_tmp = nullptr;
    chol_Ai_tmp = nullptr;
    chol_Ax_tmp = nullptr;
    chol_b_tmp = nullptr;
    ///@bierf Mesh related variables
    Mp = nullptr;
    Mi = nullptr;
    M_n = 0;
    M_n_prev = 0;
    M_nnz_prev = 0;

    ///@bierf Error Checking Variables
    mat_allocated = false;
    mesh_init = false;
}

ParthSolver::~ParthSolver(void)
{
    if (chol_A) {
        chol_A->p = chol_Ap_tmp;
        chol_A->i = chol_Ai_tmp;
        chol_A->x = chol_Ax_tmp;
        cholmod_free_sparse(&chol_A, &A_cm);
    }

    if (chol_A_prev) {
        cholmod_free_sparse(&chol_A_prev, &A_cm);
    }

    if (chol_L == chol_L_prev) {
        cholmod_free_factor(&chol_L, &A_cm);
        chol_L_prev = nullptr;
    }
    else {
        cholmod_free_factor(&chol_L, &A_cm);
        cholmod_free_factor(&chol_L_prev, &A_cm);
    }

    if (chol_b) {
        chol_b->x = chol_b_tmp;
        cholmod_free_dense(&chol_b, &A_cm);
    }

    if (chol_sol) {
        cholmod_free_dense(&chol_sol, &A_cm);
    }

    if (chol_Z) {
        cholmod_free_dense(&chol_Z, &A_cm);
    }

    if (chol_Y) {
        cholmod_free_dense(&chol_Y, &A_cm);
    }

    if (chol_E) {
        cholmod_free_dense(&chol_E, &A_cm);
    }

    if (chol_R) {
        cholmod_free_dense(&chol_R, &A_cm);
    }

    cholmod_finish(&A_cm);
}

void ParthSolver::outputA(const std::string& filePath)
{
    assert(Ap != nullptr);
    assert(Ai != nullptr);
    assert(Ax != nullptr);
    assert(A_n != 0);
    assert(A_nnz != 0);
    if (this->mat_allocated) {
        Eigen::saveMarket(A, filePath);
    }
    else {
        Eigen::SparseMatrix<double> tmp;
        tmp.conservativeResize(A_n, A_n);
        tmp.reserve(A_nnz);
        std::copy(Ap, Ap + A_n + 1, tmp.outerIndexPtr());
        std::copy(Ai, Ai + A_nnz, tmp.innerIndexPtr());
        std::copy(Ax, Ax + A_nnz, tmp.valuePtr());
        Eigen::saveMarket(tmp, filePath);
    }
}

void ParthSolver::outputRegions(const std::string& filePath)
{
    //  int cnt = 0;
    //  for (auto &iter : regions.regions_stack) {
    //    if (iter.M_n == 0) {
    //      cnt++;
    //      continue;
    //    }
    //    Eigen::SparseMatrix<double> tmp;
    //    tmp.conservativeResize(iter.M_n, iter.M_n);
    //    tmp.reserve(iter.M_nnz);
    //    std::copy(iter.Mp.data(), iter.Mp.data() + iter.M_n + 1,
    //              tmp.outerIndexPtr());
    //    std::copy(iter.Mi.data(), iter.Mi.data() + iter.M_nnz,
    //    tmp.innerIndexPtr()); std::fill(tmp.valuePtr(), tmp.valuePtr() +
    //    iter.M_nnz, 1); Eigen::saveMarket(tmp, filePath + std::to_string(cnt) +
    //    ".mtx"); cnt++;
    //  }
    std::cerr << "Implement this for ND Regions" << std::endl;
}

///--------------------------------------------------------------------------
/// permuteMatrix - get permuted hessian
///--------------------------------------------------------------------------
void ParthSolver::permuteMatrix(int n, int* p, int* i, double* x,
    std::vector<int>& perm,
    Eigen::SparseMatrix<double>& permuted_A)
{
    assert(n != 0);
    assert(p != nullptr);
    assert(i != nullptr);
    assert(x != nullptr);

    void* tmp_p;
    void* tmp_i;
    void* tmp_x;

    cholmod_common common;
    cholmod_start(&common);
    int N = n;
    int NNZ = p[n];
    auto input = cholmod_allocate_sparse(N, N, NNZ, true, true, -1, CHOLMOD_REAL, &common);
    tmp_p = input->p;
    tmp_i = input->i;
    tmp_x = input->x;

    input->p = p;
    input->i = i;
    input->x = x;
    cholmod_sparse* A2 = cholmod_ptranspose(input, 2, perm.data(), NULL, 0, &common);

    cholmod_sparse* A1 = cholmod_ptranspose(A2, 2, NULL, NULL, 0, &common);

    // TODO DELETE LATER:
    int* Ap = (int*)A1->p;
    int* Ai = (int*)A1->i;
    double* Ax = (double*)A1->x;
    int A_n = A1->nrow;
    int A_nnz = A1->nzmax;
    assert(Ap != nullptr);
    assert(Ai != nullptr);
    assert(Ax != nullptr);
    assert(A_n != 0);
    assert(A_nnz != 0);

    Eigen::Map<Eigen::SparseMatrix<double>> spMap(A_n, A_n, A_nnz, Ap, Ai, Ax,
        nullptr);
    permuted_A = spMap.eval();

    cholmod_free_sparse(&A1, &common);
    cholmod_free_sparse(&A2, &common);

    input->p = tmp_p;
    input->i = tmp_i;
    input->x = tmp_x;
    cholmod_free_sparse(&input, &common);
    cholmod_finish(&common);
}

void ParthSolver::setFrameIter(int frame, int iter, std::string name)
{
    this->frame = frame;
    this->iter = iter;
    this->sim_name = name;
}

double ParthSolver::getSymbolicReuse() { return regions.getReuseRatio(); }

double ParthSolver::getNumericReuse()
{
    double cached = 0;
    for (auto& iter : super_is_cached) {
        if (iter == 1) {
            cached++;
        }
    }
    return cached / total_num_of_super;
}

///--------------------------------------------------------------------------
/// computeCachedSupernodes - Compute Cached supernodes based on residual
/// distribution computed in initScoreVector with residual values
///--------------------------------------------------------------------------
void ParthSolver::computeInverseEliminationTree(
    int n, ///<[in] number of nodes inside the elimination tree
    const int* parent_vector, ///<[in] parent vector of the tree
    std::vector<int>& tree_ptr, ///<[out] pointer array in CSC format
    std::vector<int>& tree_set ///<[out] index array in CSC format
)
{
    if (tree_is_computed) {
        return;
    }
    // Creating the inverse elemination tree
    std::vector<int> number_of_child(n, 0);
    for (int i = 0; i < n; i++) {
        if (parent_vector[i] != -1) {
            number_of_child[parent_vector[i]]++;
        }
    }

    tree_ptr.clear();
    tree_ptr.resize(n + 1, 0);
    for (int i = 0; i < n; i++) {
        tree_ptr[i + 1] = number_of_child[i] + tree_ptr[i];
    }

    std::vector<int> child_cnt(n, 0);
    tree_set.resize(tree_ptr[n]);
    for (int s = 0; s < n; s++) {
        int parent = parent_vector[s];
        if (parent != -1) {
            int start_idx = tree_ptr[parent];
            int start_child_idx = child_cnt[parent];
            tree_set[start_idx + start_child_idx] = s;
            child_cnt[parent]++;
        }
    }
    tree_is_computed = true;

#ifndef NDEBUG
    for (int s = 0; s < n; s++) {
        assert(child_cnt[s] == number_of_child[s]);
        int parent = parent_vector[s];
        if (parent != -1) {
            auto start = tree_set.begin() + tree_ptr[parent];
            auto end = tree_set.begin() + tree_ptr[parent + 1];
            assert(std::find(start, end, s) != end);
        }
    }

#endif
}

///--------------------------------------------------------------------------
/// permuteMatrix - Permute a lower triangular matrix in CSC format
/// NOTE: Allocate Ap_perm, Ai_perm, and Ax_perm beforehand (it has the same
/// size as the input matrix)
///--------------------------------------------------------------------------
void ParthSolver::permuteMatrix(
    int* perm, ///<[in] The permutation array
    int N, ///<[in] Number of cols/rows
    int* p, ///<[in] The pointer array in CSC format
    int* i, ///<[in] The index array in CSC format
    double* x, ///<[in] The value array in CSC format
    int* p_perm, ///<[out] The pointer array in CSC format
    int* i_perm, ///<[out] The index array in CSC format
    double* x_perm ///<[out] The value array in CSC format
)
{
    assert(N != 0);
    assert(p != nullptr);
    assert(i != nullptr);
    assert(x != nullptr);

    void* tmp_p;
    void* tmp_i;
    void* tmp_x;

    cholmod_common common;
    cholmod_start(&common);
    int NNZ = p[N];
    auto input = cholmod_allocate_sparse(N, N, NNZ, true, true, -1, CHOLMOD_REAL, &common);
    tmp_p = input->p;
    tmp_i = input->i;
    tmp_x = input->x;

    input->p = p;
    input->i = i;
    input->x = x;
    cholmod_sparse* A2 = cholmod_ptranspose(input, 2, perm, NULL, 0, &common);

    cholmod_sparse* A1 = cholmod_ptranspose(A2, 2, NULL, NULL, 0, &common);

    // TODO DELETE LATER:
    int* Ap = (int*)A1->p;
    int* Ai = (int*)A1->i;
    double* Ax = (double*)A1->x;
    int A_n = A1->nrow;
    int A_nnz = A1->nzmax;
    assert(Ap != nullptr);
    assert(Ai != nullptr);
    assert(Ax != nullptr);
    assert(A_n != 0);
    assert(A_nnz != 0);

    std::copy(Ap, Ap + N + 1, p_perm);
    std::copy(Ai, Ai + NNZ, i_perm);
    std::copy(Ax, Ax + NNZ, x_perm);

    // Freeing memory
    cholmod_free_sparse(&A1, &common);
    cholmod_free_sparse(&A2, &common);

    input->p = tmp_p;
    input->i = tmp_i;
    input->x = tmp_x;
    cholmod_free_sparse(&input, &common);
    cholmod_finish(&common);
}

void ParthSolver::TakeDiff(cholmod_factor* full, int adjuster_cnt)
{
    // For each supernode, check whether their names are identical
    if (chol_L_prev == nullptr) {
        return;
    }
    assert(chol_b != nullptr);
    assert(chol_sol != nullptr);
    auto Super_prev = (int*)chol_L_prev->super;
    int nsuper_prev = chol_L_prev->nsuper;

    auto Super_curr = (int*)chol_L->super;
    int nsuper_curr = chol_L->nsuper;

    auto Lpi_curr = (int*)chol_L->pi;
    auto Ls_curr = (int*)chol_L->s;
    auto Lpx_curr = (int*)chol_L->px;
    auto Lx_curr = (double*)chol_L->x;

    auto Lx_curr_accurate = (double*)full->x;

    auto Lpi_prev = (int*)chol_L_prev->pi;
    auto Ls_prev = (int*)chol_L_prev->s;
    auto Lpx_prev = (int*)chol_L_prev->px;
    auto Lx_prev = (double*)chol_L_prev->x;

    col_to_super_curr.clear();
    col_to_super_curr.resize(chol_L->n, -1);
    //  #pragma omp parallel for//TODO: make it parallel and remove it from this
    //  function
    for (int s = 0; s < nsuper_curr; s++) {
        int col_start = Super_curr[s];
        int col_end = Super_curr[s + 1];
        for (int i = col_start; i < col_end; i++) {
            col_to_super_curr[i] = s;
        }
    }

    std::vector<int> number_of_child(total_num_of_super, 0);
    for (int i = 0; i < total_num_of_super; i++) {
        if (super_parents[i] != -1) {
            number_of_child[super_parents[i]]++;
        }
    }

    std::vector<std::string> Runtime_headers;
    Runtime_headers.emplace_back("id");
    Runtime_headers.emplace_back("A_curr-A_prev");
    Runtime_headers.emplace_back("L_accurate-L_prev");
    Runtime_headers.emplace_back("L_approx-L_accurate");
    Runtime_headers.emplace_back("L_approx-L_prev");
    Runtime_headers.emplace_back("sol_diff");
    Runtime_headers.emplace_back("Cached");
    Runtime_headers.emplace_back("Leaf");
    Runtime_headers.emplace_back("Reuse");

    profiling_utils::CSVManager runtime_csv(
        "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/" + std::to_string(frame) + "_" + std::to_string(iter) + "_" + std::to_string(adjuster_cnt),
        "some address", Runtime_headers, false);

    std::vector<double> supernode_A_score(nsuper_curr, -1);
    std::vector<double> supernode_A_MaxCoeff(nsuper_curr, -1);
    //  for (auto &s_s : supernode_reuse_score) {
    //    int super = std::get<0>(s_s);
    //    double score = std::get<1>(s_s);
    //    supernode_A_score[super] = score;
    //  }

    assert(chol_L != nullptr);

    double one[2] = { 1, 0 }, m1[2] = { -1, 0 };
    cholmod_dense* r = cholmod_copy_dense(chol_b, &A_cm);
    cholmod_sdmult(chol_A, 0, m1, one, chol_sol, r, &A_cm);

    auto sol = (double*)r->x;
    auto rhs = (double*)chol_b->x;

    // Unmark supernodes with different fill-ins and structure
    // #pragma omp parallel for//TODO: make it parallel and remove it from this
    for (int s_curr = 0; s_curr < nsuper_curr; s_curr++) {
        double L2_accurate_prev = 0;
        double L2_approx_prev = 0;
        double L2_accurate_approximate = 0;

        int isLeaf = 0;
        if (number_of_child[s_curr] == 0) {
            isLeaf = 1;
        }
        int k1_curr = Super_curr[s_curr]; /* s contains columns k1 to k2-1 of L */
        int k2_curr = Super_curr[s_curr + 1];

        int s_prev = col_to_super_prev[k1_curr];
        int k1_prev = Super_prev[s_prev]; /* s contains columns k1 to k2-1 of L */
        int k2_prev = Super_prev[s_prev + 1];

        int ncol = k2_prev - k1_prev;

        double sol_diff = 0;
        for (int i = k1_curr; i < k2_curr; i++) {
            if (sol_diff < std::abs(sol[i])) {
                sol_diff = std::abs(sol[i]);
            }
        }

        // The supernode start and end columns are not the same
        if (k1_curr != k1_prev || k2_curr != k2_prev) {
            runtime_csv.addElementToRecord(s_curr, "id");
            runtime_csv.addElementToRecord(1000, "L_approx-L_prev");
            runtime_csv.addElementToRecord(1000, "L_accurate-L_prev");
            runtime_csv.addElementToRecord(1000, "L_approx-L_accurate");
            runtime_csv.addElementToRecord(1000, "A_curr-A_prev");
            runtime_csv.addElementToRecord(sol_diff, "sol_diff");
            runtime_csv.addElementToRecord(super_is_cached[s_curr], "Cached");
            runtime_csv.addElementToRecord(isLeaf, "Leaf");
            runtime_csv.addElementToRecord(getNumericReuse(), "Reuse");
            runtime_csv.addRecord();
            continue;
        }

        int psi_curr = Lpi_curr[s_curr]; /* pointer to first row of s in Ls */
        int psend_curr = Lpi_curr[s_curr + 1]; /* pointer just past last row of s in Ls */
        int nsrow_curr = psend_curr - psi_curr; /* # of rows in all of s */

        int psi_prev = Lpi_prev[s_prev]; /* pointer to first row of s in Ls */
        int psend_prev = Lpi_prev[s_prev + 1]; /* pointer just past last row of s in Ls */
        int nsrow_prev = psend_prev - psi_prev; /* # of rows in all of s */

        // TODO: Make the cachability more fine-grain
        // Spupernodes should have the same number of rows
        if (nsrow_curr != nsrow_prev) {
            runtime_csv.addElementToRecord(s_curr, "id");
            runtime_csv.addElementToRecord(1000, "L_approx-L_prev");
            runtime_csv.addElementToRecord(1000, "L_accurate-L_prev");
            runtime_csv.addElementToRecord(1000, "L_approx-L_accurate");
            runtime_csv.addElementToRecord(1000, "A_curr-A_prev");
            runtime_csv.addElementToRecord(sol_diff, "sol_diff");
            runtime_csv.addElementToRecord(super_is_cached[s_curr], "Cached");
            runtime_csv.addElementToRecord(isLeaf, "Leaf");
            runtime_csv.addElementToRecord(getNumericReuse(), "Reuse");
            runtime_csv.addRecord();
            continue;
        }

        // Check whether the corresponding supernodes have exact same pattern
        for (int k = 0; k < nsrow_prev; k++) {
            if (Ls_prev[psi_prev + k] != Ls_curr[psi_curr + k]) {
                runtime_csv.addElementToRecord(s_curr, "id");
                runtime_csv.addElementToRecord(1000, "L_approx-L_prev");
                runtime_csv.addElementToRecord(1000, "L_accurate-L_prev");
                runtime_csv.addElementToRecord(1000, "L_approx-L_accurate");
                runtime_csv.addElementToRecord(1000, "A_curr-A_prev");
                runtime_csv.addElementToRecord(sol_diff, "sol_diff");
                runtime_csv.addElementToRecord(super_is_cached[s_curr], "Cached");
                runtime_csv.addElementToRecord(isLeaf, "Leaf");
                runtime_csv.addElementToRecord(getNumericReuse(), "Reuse");
                runtime_csv.addRecord();
                continue;
            }
        }

        // If supernodes have same structure, they can be cached -> Take the diff
        psi_prev = Lpi_prev[s_prev]; /* pointer to first row of s in Ls */
        psend_prev = Lpi_prev[s_prev + 1];
        nsrow_prev = psend_prev - psi_prev;

        auto nscol_prev = k2_prev - k1_prev;
        auto nscol_curr = k2_curr - k1_curr;

        assert(nsrow_curr == nsrow_prev);
        assert(nscol_prev == nscol_curr);

        auto psx_prev = Lpx_prev[s_prev];
        auto pend_prev = psx_prev + nsrow_prev * nscol_prev;

        auto psx_curr = Lpx_curr[s_curr];
        auto pend_curr = psx_curr + nsrow_curr * nscol_curr;
        assert(pend_curr - psx_curr == pend_prev - psx_prev);
        // #pragma omp parallel for
        for (int cnt = 0; cnt < pend_curr - psx_curr; cnt++) {
            L2_approx_prev += (Lx_curr[cnt + psx_curr] - Lx_prev[psx_prev + cnt]) * (Lx_curr[cnt + psx_curr] - Lx_prev[psx_prev + cnt]);

            L2_accurate_prev += (Lx_curr_accurate[cnt + psx_curr] - Lx_prev[psx_prev + cnt]) * (Lx_curr_accurate[cnt + psx_curr] - Lx_prev[psx_prev + cnt]);

            L2_accurate_approximate += (Lx_curr[cnt + psx_curr] - Lx_curr_accurate[psx_curr + cnt]) * (Lx_curr[cnt + psx_curr] - Lx_curr_accurate[psx_curr + cnt]);
        }
        runtime_csv.addElementToRecord(s_curr, "id");
        runtime_csv.addElementToRecord(-1, "A_curr-A_prev");
        runtime_csv.addElementToRecord(L2_approx_prev / ncol, "L_approx-L_prev");
        runtime_csv.addElementToRecord(L2_accurate_prev / ncol,
            "L_accurate-L_prev");
        runtime_csv.addElementToRecord(L2_accurate_approximate / ncol,
            "L_approx-L_accurate");
        runtime_csv.addElementToRecord(sol_diff, "sol_diff");
        runtime_csv.addElementToRecord(super_is_cached[s_curr], "Cached");
        runtime_csv.addElementToRecord(isLeaf, "Leaf");
        runtime_csv.addElementToRecord(getNumericReuse(), "Reuse");
        runtime_csv.addRecord();
    }
    cholmod_free_dense(&r, &A_cm);
}

///--------------------------------------------------------------------------
/// savePrevMatrixForReuse - It  saves the chol_A into chol_A_prev;
/// chol_A is still valid after the operation
///--------------------------------------------------------------------------
void ParthSolver::savePrevMatrixForReuse(int* Perm)
{
    // TODO: Save the permuted matrix
    if (chol_A_prev) {
        cholmod_free_sparse(&chol_A_prev, &A_cm);
    }
    assert(Perm != nullptr);

    auto tmp = (int*)chol_A->p;
    assert(A_nnz == tmp[A_n]);

    auto p = (int*)chol_A->p;
    auto i = (int*)chol_A->i;
    auto x = (double*)chol_A->x;
    int N = chol_A->ncol;
    int NNZ = p[N];
    std::vector<int> Ap(N + 1);
    std::vector<int> Ai(NNZ);
    std::vector<double> Ax(NNZ);

    chol_A_prev = cholmod_allocate_sparse(A_n, A_n, A_nnz, true, true, -1,
        CHOLMOD_REAL, &A_cm);
    permuteMatrix(Perm, N, p, i, x, (int*)chol_A_prev->p, (int*)chol_A_prev->i,
        (double*)chol_A_prev->x);
}

double ParthSolver::getAnalyzeTime(bool reset)
{
    double time = analyze_total_time;
    if (reset) {
        analyze_total_time = 0;
    }
    return time;
}

double ParthSolver::getFactorTime(bool reset)
{
    double time = factorize_total_time;
    if (reset) {
        factorize_total_time = 0;
    }
    return time;
}

double ParthSolver::getSolveTime(bool reset)
{
    double time = solve_total_time;
    if (reset) {
        solve_total_time = 0;
    }
    return time;
}

} // namespace PARTH
