//
// Created by behrooz on 28/09/22.
//

#include "LeftLooking_Parth.h"
#include "csv_utils.h"
#include <algorithm>
#include <functional>
#include <iostream>

namespace PARTH {

bool ParthSolver::factorize(void) {
  bool output;
  if (Options().getNumericReuseType() == NumericReuseType::NUMERIC_REUSE_BASE) {
    if (Options().getAnalyzeType() != AnalyzeType::SimulationAware) {
      std::cerr
          << "The analysis type should be SimulationAware in order for "
             "Numeric reuse to be used -> Changing the reuse type to no reuse"
          << std::endl;
      Options().setNumericReuseType(NumericReuseType::NUMERIC_NO_REUSE);
    }
  }

  switch (Options().getNumericReuseType()) {
  case NumericReuseType::NUMERIC_NO_REUSE:
    if (Options().getVerbose()) {
      std::cout << "+++ PARTH: Cholmod default Factorization +++" << std::endl;
    }
    output = this->cholmodFactoization();
    break;
  case NumericReuseType::NUMERIC_REUSE_BASE:
    if (Options().getVerbose()) {
      std::cout << "+++ PARTH: ANALYZE REUSE Factorization +++" << std::endl;
    }
    output = this->analyzeReuseFactorization();
    break;
  case NumericReuseType::NUMERIC_NO_REUSE_PARALLEL:
    if (Options().getVerbose()) {
      std::cout
          << "+++ PARTH: No REUSE, SuperNode Parallelism Factorization +++"
          << std::endl;
    }
    output = this->parallelFactoization();
    break;
  case NumericReuseType::NUMERIC_REUSE_PARALLEL:
    if (Options().getVerbose()) {
      std::cout << "+++ PARTH: REUSE + Parallel Factorization +++" << std::endl;
    }
    output = this->ParallelReuseFactorization();
    break;
  default:
    std::cerr << "This numeric reuse type is not define" << std::endl;
    output = false;
  }
  return output;
}

bool ParthSolver::cholmodFactoization() {
  double start = omp_get_wtime();
  cholmod_factorize(chol_A, chol_L, &A_cm);
  start = omp_get_wtime() - start;
  if (Options().getVerbose()) {
    std::cout << "Factorization Time: " << start << std::endl;
  }
  assert(chol_L != nullptr);
  return A_cm.status != CHOLMOD_NOT_POSDEF;
}

bool ParthSolver::analyzeReuseFactorization() {
  factorize_total_time = 0;
  if (chol_L_prev == nullptr) {
    assert(analyze_is_requested);
    analyze(true);
    double start = omp_get_wtime();
    cholmod_factorize(chol_A, chol_L, &A_cm);
    savePrevFactorForReuse();
    factorize_total_time += omp_get_wtime() - start;

    return A_cm.status != CHOLMOD_NOT_POSDEF;
  } else {
    return true;
  }
}

bool ParthSolver::ParallelReuseFactorization() {
  factorize_total_time = 0;
  if (chol_L_prev == nullptr) {
    assert(analyze_is_requested);
    analyze(true);
    double start = omp_get_wtime();
    super_is_cached.clear();
    super_is_cached.resize(total_num_of_super, 0);
    copy_prev_factor = false;
    cholmod_factorize_custom(chol_A, chol_L, &A_cm);
    savePrevFactorForReuse();
    factorize_total_time += omp_get_wtime() - start;

    return A_cm.status != CHOLMOD_NOT_POSDEF;
  } else {
    return true;
  }
}

bool ParthSolver::reuseFactoization() {
  std::cerr << "This function is useless" << std::endl;
  return A_cm.status != CHOLMOD_NOT_POSDEF;
}

bool ParthSolver::parallelFactoization() {
  double start = omp_get_wtime();
  assert(chol_L != nullptr);
  cholmod_factorize_custom(chol_A, chol_L, &A_cm);
  factorize_total_time = omp_get_wtime() - start;
  if (Options().getVerbose()) {
    std::cout << "Factorization Time: " << start << std::endl;
  }
  assert(chol_L != nullptr);

  return A_cm.status != CHOLMOD_NOT_POSDEF;
}

bool ParthSolver::outputFactorization(const std::string &filePath) {
  if (chol_L) {
    cholmod_sparse *spm = cholmod_factor_to_sparse(chol_L, &A_cm);

    FILE *out = fopen(filePath.c_str(), "w");
    assert(out);

    cholmod_write_sparse(out, spm, NULL, "", &A_cm);
    cholmod_free_sparse(&spm, &A_cm);
    fclose(out);
    return true;
  } else {
    if (Options().getVerbose()) {
      std::cout << "The factor does not exist" << std::endl;
    }
    return false;
  }
}

///--------------------------------------------------------------------------
/// getHessianPerm - get Hessian permutation vector
///--------------------------------------------------------------------------
std::vector<int> ParthSolver::getHessianPerm() { return hessian_perm; }
std::vector<int> ParthSolver::getMeshPerm() { return mesh_perm; }

int ParthSolver::getFactorNonzeros(void) { return A_cm.lnz * 2 - A_n; }

double ParthSolver::getResidual() { return residual; }

void ParthSolver::initScoreVector(double *r) {
  res_vec_per_elem.clear();
  res_vec_per_supernode.clear();

  res_vec_per_supernode.resize(total_num_of_super, 0);
  res_vec_per_elem.resize(M_n, 0);
  std::vector<int> new_label_inv(M_n, 0);

  int nsuper = chol_L->nsuper;
  auto super = (int *)chol_L->super;
  auto Perm = (int *)chol_L->Perm;

#pragma omp parallel
  {
#pragma omp for
    for (int elem = 0; elem < M_n; elem++) {
      double elem_max_res = 0;
      for (int i = elem * 3; i < (elem + 1) * 3; i++) {
        if (elem_max_res < std::abs(r[i])) {
          elem_max_res = std::abs(r[i]);
        }
      }
      res_vec_per_elem[elem] = elem_max_res;
    }

    // original elem to permuted column C_org[i] = i_permuted
#pragma omp for
    for (int j = 0; j < M_n; j++) {
      int idx = Perm[j * 3] / 3;
      assert(idx < M_n);
      new_label_inv[j] = idx;
    }

#pragma omp for
    for (int s = 0; s < total_num_of_super; s++) {
      int start_elem = super[s] / 3;
      int end_elem = super[s + 1] / 3;
      for (int elem = start_elem; elem < end_elem; elem++) {
        assert(elem < M_n);
        double elem_score = res_vec_per_elem[new_label_inv[elem]];
        if (res_vec_per_supernode[s] < elem_score) {
          res_vec_per_supernode[s] = elem_score;
        }
      }
    }
  };
}

std::vector<double>
ParthSolver::convertSuperScoreToNodeScore(std::vector<double> &super_score) {
  auto Perm = (int *)chol_L->Perm;
  auto super = (int *)chol_L->super;
  assert(super_score.size() == total_num_of_super);
  // original elem to permuted column C_org[i] = i_permuted
  std::vector<int> new_labels(M_n, 0);
  for (int j = 0; j < M_n; j++) {
    int idx = Perm[j * 3] / 3;
    assert(idx < M_n);
    new_labels[idx] = j;
  }
  //
  std::vector<int> new_label_inv(M_n, 0);
  // C_permuted[i] = i_org
  for (int j = 0; j < M_n; j++) {
    assert(new_labels[j] < M_n);
    new_label_inv[new_labels[j]] = j;
  }

  // Convert super score to element score

  std::vector<double> elem_score(M_n, 0);
  for (int s = 0; s < total_num_of_super; s++) {
    int start_elem = super[s] / 3;
    int end_elem = super[s + 1] / 3;
    for (int elem = start_elem; elem < end_elem; elem++) {
      assert(elem < M_n);
      elem_score[new_label_inv[elem]] = super_score[s];
    }
  }
  return elem_score;
}

std::vector<int>
ParthSolver::convertSuperFlagToNodeFlag(std::vector<int> &super_flag) {
  auto Perm = (int *)chol_L->Perm;
  assert(super_flag.size() == total_num_of_super);
  // original col to permuted column C_org[i] = i_permuted
  std::vector<int> new_labels(chol_L->n, 0);
  for (int j = 0; j < chol_L->n; j++) {
    new_labels[Perm[j]] = j;
  }

  std::vector<int> new_label_inv(chol_L->n, 0);
  // C_permuted[i] = i_org
  for (int j = 0; j < chol_L->n; j++) {
    new_label_inv[new_labels[j]] = j;
  }

  // Convert super is cached to column is cached
  std::vector<int> col_is_cached(chol_L->n, 0);
  int nsuper = chol_L->nsuper;
  auto super = (int *)chol_L->super;
  for (int s = 0; s < nsuper; s++) {
    int start_col = super[s];
    int end_col = super[s + 1];
    for (int col = start_col; col < end_col; col++) {
      if (super_flag[s] == 1) {
        col_is_cached[col] = 1;
      } else {
        col_is_cached[col] = 0;
      }
    }
  }

  // Convert col is cached to original col is cached
  std::vector<int> org_col_is_cached(chol_L->n, 0);
  for (int i = 0; i < chol_L->n; i++) {
    org_col_is_cached[new_label_inv[i]] = col_is_cached[i];
  }

  // Convert the column cached to element cached
  int n_elem = chol_L->n / 3;
  std::vector<int> elem_is_cached(n_elem, 0);
  for (int e = 0; e < n_elem; e++) {
    elem_is_cached[e] = 1;
    for (int i = 0; i < 3; i++) {
      if (org_col_is_cached[e * 3 + i] == 0) {
        elem_is_cached[e] = 0;
        break;
      }
    }
  }
  return elem_is_cached;
}

///--------------------------------------------------------------------------
/// savePrevFactorForReuse - make chol_L equal to nullptr and create
/// chol_L_prev. It should be called after solve so we use a valid chol_L
/// for solve part.
///--------------------------------------------------------------------------
void ParthSolver::savePrevFactorForReuse() {
  if (chol_L == nullptr) {
    return;
  }

  if (chol_L_prev != chol_L) {
    cholmod_free_factor(&chol_L_prev, &A_cm);
  }

  //    chol_L_prev = cholmod_copy_factor(chol_L, &A_cm);
  chol_L_prev = chol_L;
  chol_L = nullptr;

  int *Super = (int *)chol_L_prev->super;
  int nsuper = chol_L_prev->nsuper;
  this->col_to_super_prev.resize(M_n * 3, -1);
  //  #pragma omp parallel for//TODO: make it parallel and remove it from this
  //  function
  for (int s = 0; s < nsuper; s++) {
    int col_start = Super[s];
    int col_end = Super[s + 1];
    for (int i = col_start; i < col_end; i++) {
      col_to_super_prev[i] = s;
    }
  }

  //  savePrevMatrixForReuse((int *)chol_L_prev->Perm);

  // Assign col to super
  if (!chol_L_prev->is_super) {
    std::cerr << "This simulation uses simplical factorization instead of "
                 "supernode."
              << std::endl;
    return;
  }
}

void ParthSolver::OracleCachingMechanism(double *residual_vec) {
  // For each supernode, check whether their names are identical
  assert(chol_L_prev != nullptr);
  initScoreVector(residual_vec);

  super_is_cached.clear();
  super_is_cached.resize(total_num_of_super, 1);

  int nsuper = chol_L->nsuper;
  auto super = (int *)chol_L->super;
  int n_elem = chol_L->n / 3;
  assert(chol_L->n % 3 == 0);

  auto perm_prev = (int *)chol_L_prev->Perm;
  auto perm_curr = (int *)chol_L->Perm;

  assert(chol_L->n == chol_L_prev->n);

  // For each supernode, check whether their structure are identical
  auto Super_prev = (int *)chol_L_prev->super;
  int nsuper_prev = chol_L_prev->nsuper;

  auto Super_curr = (int *)chol_L->super;
  int nsuper_curr = chol_L->nsuper;

  auto Lpi_curr = (int *)chol_L->pi;
  auto Ls_curr = (int *)chol_L->s;
  auto Lpx_curr = (int *)chol_L->px;

  auto Lpi_prev = (int *)chol_L_prev->pi;
  auto Ls_prev = (int *)chol_L_prev->s;
  auto Lpx_prev = (int *)chol_L_prev->px;

  // Unmark supernodes with different fill-ins and structure

  if (chol_L != chol_L_prev) {
#pragma omp parallel for
    for (int s_curr = 0; s_curr < nsuper_curr; s_curr++) {
      double InfNorm = 0;
      int k1_curr = Super_curr[s_curr]; /* s contains columns k1 to k2-1 of L */
      int k2_curr = Super_curr[s_curr + 1];

      int s_prev = col_to_super_prev[k1_curr];
      int k1_prev = Super_prev[s_prev]; /* s contains columns k1 to k2-1 of L */
      int k2_prev = Super_prev[s_prev + 1];

      // The supernode start and end columns are not the same
      if (k1_curr != k1_prev || k2_curr != k2_prev) {
        super_is_cached[s_curr] = 0;
        continue;
      }

      int nscol = k2_curr - k1_curr;

      int psi_curr = Lpi_curr[s_curr]; /* pointer to first row of s in Ls */
      int psend_curr =
          Lpi_curr[s_curr + 1]; /* pointer just past last row of s in Ls */
      int nsrow_curr = psend_curr - psi_curr; /* # of rows in all of s */

      int psi_prev = Lpi_prev[s_prev]; /* pointer to first row of s in Ls */
      int psend_prev =
          Lpi_prev[s_prev + 1]; /* pointer just past last row of s in Ls */
      int nsrow_prev = psend_prev - psi_prev; /* # of rows in all of s */

      // TODO: Make the cachability more fine-grain
      // Supernodes should have the same number of rows
      if (nsrow_curr != nsrow_prev) {
        super_is_cached[s_curr] = 0;
        continue;
      }

      // Check whether the corresponding supernodes have exact same pattern
      for (int k = 0; k < nsrow_prev; k++) {
        if (Ls_prev[psi_prev + k] != Ls_curr[psi_curr + k]) {
          super_is_cached[s_curr] = 0;
          continue;
        }
      }
    }
  }

  computeInverseEliminationTree(total_num_of_super, super_parents.data(),
                                tree_ptr, tree_set);
  double max_error = 0;
  for (int s = 0; s < total_num_of_super; s++) {
    if (res_vec_per_supernode[s] > max_error) {
      max_error = res_vec_per_supernode[s];
    }
  }

  std::cout << "+++ PARTH: Current MAX RESIDUAL IS: " << max_error
            << " - Acceptable residual is: " << solve_acceptable_error << "\t"
            << " - Acceptable preconditioner residual is: "
            << preconditioner_acceptable_error << std::endl;

  std::vector<bool> marked(total_num_of_super, false);
  for (int s = total_num_of_super - 1; s >= 0; s--) {
    if (res_vec_per_supernode[s] > preconditioner_acceptable_error ||
        super_is_cached[s] == 0) {
      if (marked[s]) {
        continue;
      }
      // Mark all the descendent
      std::queue<double> r_queue;
      r_queue.push(s);
      while (!r_queue.empty()) {
        int node = r_queue.front();
        super_is_cached[node] = 0;
        marked[node] = true;
        r_queue.pop();
        for (int child_ptr = tree_ptr[node]; child_ptr < tree_ptr[node + 1];
             child_ptr++) {
          int child = tree_set[child_ptr];
          r_queue.push(child);
        }
      }
    }
  }

#ifndef NDEBUG
  std::vector<double> caching_d(total_num_of_super, 0);
  for (int s = 0; s < total_num_of_super; s++) {
    caching_d[s] = super_is_cached[s];
  }
  score_view.emplace_back(res_vec_per_elem);
  score_name.emplace_back("ResVec_element");
  score_view.emplace_back(convertSuperScoreToNodeScore(res_vec_per_supernode));
  score_name.emplace_back("ResVec_Supernode");
  score_view.emplace_back(convertSuperScoreToNodeScore(caching_d));
  score_name.emplace_back("caching_flag");
#endif
}
void ParthSolver::setInputAddress(std::string address) {
  input_address = address;
}

} // namespace PARTH
