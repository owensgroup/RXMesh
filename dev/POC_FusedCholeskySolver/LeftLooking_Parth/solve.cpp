//
// Created by behrooz on 25/10/22.
//

#include "LeftLooking_Parth.h"
#include "mkl.h"
#include <iostream>

namespace PARTH {

void ParthSolver::solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) {
  switch (Options().getNumericReuseType()) {
  case NumericReuseType::NUMERIC_NO_REUSE:
    if (Options().getVerbose()) {
      std::cout << "+++ PARTH: Cholmod default Solve +++" << std::endl;
    }
    this->cholmodSolve(rhs, result);
    break;
  case NumericReuseType::NUMERIC_NO_REUSE_PARALLEL:
    if (Options().getVerbose()) {
      std::cout << "+++ PARTH: Parallel PARTH Solve +++" << std::endl;
    }
    this->cholmodSolve(rhs, result);
    break;
  case NumericReuseType::NUMERIC_REUSE_BASE:
    if (Options().getVerbose()) {
      std::cout << "+++ PARTH: Reuse PARTH Solve +++" << std::endl;
    }
    this->analyzeReuseSolve(rhs, result);
    break;
  case NumericReuseType::NUMERIC_REUSE_PARALLEL:
    if (Options().getVerbose()) {
      std::cout << "+++ PARTH: Parallel + Reuse Solve +++" << std::endl;
    }
    this->ParallelReuseSolve(rhs, result);
    break;
  default:
    std::cerr << "This numeric reuse type is not defined and the corresponding "
                 "solve does not exist"
              << std::endl;
    return;
  }
}

void ParthSolver::cholmodSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) {
  solve_total_time = omp_get_wtime();
  // TODO: directly point to rhs?
  if (!chol_b) {
    chol_b = cholmod_allocate_dense(A_n, 1, A_n, CHOLMOD_REAL, &A_cm);
    chol_b_tmp = chol_b->x;
  }
  chol_b->x = rhs.data();

  if (chol_sol) {
    cholmod_free_dense(&chol_sol, &A_cm);
  }

  chol_sol = cholmod_solve(CHOLMOD_A, chol_L, chol_b, &A_cm);

  if (Options().getComputeResidual()) {
    if (Options().getVerbose()) {
      std::cout << "+++ PARTH: Computing residual +++" << std::endl;
    }
    double one[2] = {1, 0}, m1[2] = {-1, 0};
    cholmod_dense *r = cholmod_copy_dense(chol_b, &A_cm);
    cholmod_sdmult(chol_A, 0, m1, one, chol_sol, r, &A_cm);
    residual = cholmod_norm_dense(r, 0, &A_cm);
    cholmod_free_dense(&r, &A_cm);
    if (Options().getVerbose()) {
      std::cout << "+++ PARTH: The Residual Is: " << residual << std::endl;
    }
  }

  result.conservativeResize(rhs.size());
  memcpy(result.data(), chol_sol->x, result.size() * sizeof(result[0]));
  solve_total_time = omp_get_wtime() - solve_total_time;
}

void ParthSolver::analyzeReuseSolve(Eigen::VectorXd &rhs,
                                    Eigen::VectorXd &result) {

  SpMV_total_time = 0;
  solve_kernel_total_time = 0;
  num_steps_to_get_result = 1;
  num_refinement_iteration = 0;
#ifndef NDEBUG
  score_name.clear();
  score_view.clear();
#endif

  assert(total_num_of_super != 0);
  super_is_cached.resize(total_num_of_super);
  std::fill(super_is_cached.begin(), super_is_cached.end(), 1);

  solve_total_time = 0;

  // TODO: directly point to rhs?
  double start = omp_get_wtime();
  //======== STEP 1: Compute the residual based on previous hessian ===========
  if (!chol_b) {
    chol_b = cholmod_allocate_dense(A_n, 1, A_n, CHOLMOD_REAL, &A_cm);
    chol_b_tmp = chol_b->x;
  }

  std::vector<double> res_vec;
  //  iterativeRefinement(num_max_refinement_iteration,
  //                      preconditioner_acceptable_error, rhs.size(), res_vec,
  //                      rhs.data(), chol_L_prev, result, residual);
  applyPCG(num_max_refinement_iteration, preconditioner_acceptable_error,
           rhs.size(), res_vec, rhs.data(), chol_L_prev, result, residual);

  solve_total_time += omp_get_wtime() - start;
  // L2 Norm
  if (residual < solve_acceptable_error) {
#ifndef NDEBUG
    score_view.emplace_back(std::vector<double>(M_n, 0));
    score_name.emplace_back("Step1Return");
#endif
    return;
  }

  num_steps_to_get_result = 2;
  //======== STEP 2: Check whether the hessian is acceptable ===========
  // If it was not acceptable to a partial factorization and measure the error
  // Use the oracle caching process to compute the cached supernodes
  copy_prev_factor = true;
  if (chol_L == nullptr && !analyze_is_requested) {
    chol_L = chol_L_prev;
    copy_prev_factor = false;
  } else {
    // Analyze is requested before
    analyze(true);
  }
  assert(chol_L != NULL);

  start = omp_get_wtime();
  OracleCachingMechanism(res_vec.data());
  double reuse = getNumericReuse();
  if (reuse > 0.2) {
    double reused_factor_time = omp_get_wtime();
    cholmod_factorize_custom(chol_A, chol_L, &A_cm);
    reused_factor_time = omp_get_wtime() - reused_factor_time;
    std::cout << "+++PARTH: REUSED TIME: " << reused_factor_time
              << " with the reuse of: " << reuse << std::endl;
    factorize_total_time += omp_get_wtime() - start;
  } else {
    std::cout << "+++PARTH: Doing the reuse of " << reuse
              << " is pointless. Doing full factorization" << std::endl;
    std::fill(super_is_cached.begin(), super_is_cached.end(), 0);
    copy_prev_factor = false;
    cholmod_factorize_custom(chol_A, chol_L, &A_cm);
    factorize_total_time += omp_get_wtime() - start;

    start = omp_get_wtime();
    cholmodSolve(rhs, result);
    solve_total_time += omp_get_wtime() - start;

    start = omp_get_wtime();
    savePrevFactorForReuse();
    factorize_total_time += omp_get_wtime() - start;
    return;
  }

  // If it was not acceptable to a partial factorization and measure the error
  start = omp_get_wtime();
  applyPCG(num_max_PCG_iteration, solve_acceptable_error, rhs.size(), res_vec,
           rhs.data(), chol_L, result, residual);
  solve_total_time += omp_get_wtime() - start;

  // Save the current full hessian
  if (residual < solve_acceptable_error) {
    start = omp_get_wtime();
    savePrevFactorForReuse();
    factorize_total_time += omp_get_wtime() - start;
    return;
  }
  num_steps_to_get_result = 3;

  //=== STEP 3: If the approximation didn't work, do a full factorization ===
  start = omp_get_wtime();
  // Compute the rest of the supernodes
  copy_prev_factor = false;
  for (int s = 0; s < total_num_of_super; s++) {
    if (super_is_cached[s] == 1) {
      super_is_cached[s] = 0;
    } else if (super_is_cached[s] == 0) {
      super_is_cached[s] = 1;
    } else {
      std::cerr << "+++PARTH: super_is_cached has invalid value" << std::endl;
    }
  }
  cholmod_factorize_custom(chol_A, chol_L, &A_cm);
  std::fill(super_is_cached.begin(), super_is_cached.end(), 0);
  factorize_total_time += omp_get_wtime() - start;

  start = omp_get_wtime();
  cholmodSolve(rhs, result);
  solve_total_time += omp_get_wtime() - start;

  start = omp_get_wtime();
  savePrevFactorForReuse();
  factorize_total_time += omp_get_wtime() - start;
}

void ParthSolver::ParallelReuseSolve(Eigen::VectorXd &rhs,
                                     Eigen::VectorXd &result) {

  SpMV_total_time = 0;
  solve_kernel_total_time = 0;
  num_steps_to_get_result = 1;
  num_refinement_iteration = 0;
#ifndef NDEBUG
  score_name.clear();
  score_view.clear();
#endif

  assert(total_num_of_super != 0);
  super_is_cached.resize(total_num_of_super);
  std::fill(super_is_cached.begin(), super_is_cached.end(), 1);

  solve_total_time = 0;

  // TODO: directly point to rhs?
  double start = omp_get_wtime();
  //======== STEP 1: Compute the residual based on previous hessian ===========
  if (!chol_b) {
    chol_b = cholmod_allocate_dense(A_n, 1, A_n, CHOLMOD_REAL, &A_cm);
    chol_b_tmp = chol_b->x;
  }

  std::vector<double> res_vec;
  //  iterativeRefinement(num_max_refinement_iteration,
  //                      preconditioner_acceptable_error, rhs.size(), res_vec,
  //                      rhs.data(), chol_L_prev, result, residual);
  applyPCG(num_max_refinement_iteration, preconditioner_acceptable_error,
           rhs.size(), res_vec, rhs.data(), chol_L_prev, result, residual);

  solve_total_time += omp_get_wtime() - start;
  // L2 Norm
  if (residual < solve_acceptable_error) {
#ifndef NDEBUG
    score_view.emplace_back(std::vector<double>(M_n, 0));
    score_name.emplace_back("Step1Return");
#endif
    return;
  }

  num_steps_to_get_result = 2;
  //======== STEP 2: Check whether the hessian is acceptable ===========
  // If it was not acceptable to a partial factorization and measure the error
  // Use the oracle caching process to compute the cached supernodes
  copy_prev_factor = true;
  if (chol_L == nullptr && !analyze_is_requested) {
    chol_L = chol_L_prev;
    copy_prev_factor = false;
  } else {
    // Analyze is requested before
    analyze(true);
  }
  assert(chol_L != NULL);

  start = omp_get_wtime();
  OracleCachingMechanism(res_vec.data());
  double reuse = getNumericReuse();
  if (reuse > 0.2) {
    double reused_factor_time = omp_get_wtime();
    if (reuse > 0.8) {
      Options().setNumericReuseType(NumericReuseType::NUMERIC_REUSE_BASE);
    }
    cholmod_factorize_custom(chol_A, chol_L, &A_cm);
    reused_factor_time = omp_get_wtime() - reused_factor_time;
    std::cout << "+++PARTH: REUSED TIME: " << reused_factor_time
              << " with the reuse of: " << reuse << std::endl;
    factorize_total_time += omp_get_wtime() - start;
    Options().setNumericReuseType(NumericReuseType::NUMERIC_REUSE_PARALLEL);
  } else {
    std::cout << "+++PARTH: Doing the reuse of " << reuse
              << " is pointless. Doing full factorization" << std::endl;
    std::fill(super_is_cached.begin(), super_is_cached.end(), 0);
    copy_prev_factor = false;
    cholmod_factorize_custom(chol_A, chol_L, &A_cm);
    factorize_total_time += omp_get_wtime() - start;

    start = omp_get_wtime();
    cholmodSolve(rhs, result);
    solve_total_time += omp_get_wtime() - start;

    start = omp_get_wtime();
    savePrevFactorForReuse();
    factorize_total_time += omp_get_wtime() - start;
    return;
  }

  // TODO: Make it a debug thing or delete it
  //  MeasureTheSuperNodeDifference();

  // If it was not acceptable to a partial factorization and measure the error
  start = omp_get_wtime();
  applyPCG(num_max_PCG_iteration, solve_acceptable_error, rhs.size(), res_vec,
           rhs.data(), chol_L, result, residual);
  solve_total_time += omp_get_wtime() - start;

  // Save the current full hessian
  if (residual < solve_acceptable_error) {
    start = omp_get_wtime();
    savePrevFactorForReuse();
    factorize_total_time += omp_get_wtime() - start;
    return;
  }
  num_steps_to_get_result = 3;

  //=== STEP 3: If the approximation didn't work, do a full factorization ===
  start = omp_get_wtime();
  // Compute the rest of the supernodes

  for (int s = 0; s < total_num_of_super; s++) {
    if (super_is_cached[s] == 1) {
      super_is_cached[s] = 0;
    } else if (super_is_cached[s] == 0) {
      super_is_cached[s] = 1;
    } else {
      std::cerr << "+++PARTH: super_is_cached has invalid value" << std::endl;
    }
  }
  copy_prev_factor = false;
  cholmod_factorize_custom(chol_A, chol_L, &A_cm);
  std::fill(super_is_cached.begin(), super_is_cached.end(), 0);
  factorize_total_time += omp_get_wtime() - start;

  start = omp_get_wtime();
  cholmodSolve(rhs, result);
  solve_total_time += omp_get_wtime() - start;

  start = omp_get_wtime();
  savePrevFactorForReuse();
  factorize_total_time += omp_get_wtime() - start;
}

// TODO: You really need to optimize these codes
///--------------------------------------------------------------------------
/// iterativeRefinement - Apply an iterative refinement
/// NOTE - remember to free the array r and pass it as a nullptr
///--------------------------------------------------------------------------
void ParthSolver::iterativeRefinement(
    int ref_iter_num,             ///<[in] number of refinement iteration
    double acceptable_error,      ///<[in] number of refinement iteration
    int rhs_size,                 ///<[in] size of the rhs array
    std::vector<double> &res_vec, ///<[in] residual vector
    double *rhs,                  ///<[in] the pointer to rhs array
    cholmod_factor *L,       ///<[in] The factor that will be used for solve
    Eigen::VectorXd &result, ///<[out] result vector
    double &residual         ///<[out] residual of the current result
) {

  // Saving the first residual for later reuse computation
  res_vec.resize(rhs_size);

  char transa = 'N'; // don't transpose the matrix
  char matdescra[6] = {
      'S', 'L', 'N',
      'C', ' ', ' '}; // symmetric matrix in 0-based column-major format

  double Minus = -1;
  double ONE = 1;

  assert(rhs_size == L->n);
  result.resize(rhs_size);
  cholmod_dense *r = cholmod_allocate_dense(A_n, 1, A_n, CHOLMOD_REAL, &A_cm);

  // Compute Ax = b
  if (chol_sol) {
    cholmod_free_dense(&chol_sol, &A_cm);
  }
  memcpy(r->x, rhs, rhs_size * sizeof(result[0]));
  chol_sol = cholmod_allocate_dense(A_n, 1, A_n, CHOLMOD_REAL, &A_cm);

  // chol_sol = x0 - first result
  cholmod_dense *Y = NULL;
  cholmod_dense *E = NULL;

  int solve_is_ok =
      cholmod_solve2(CHOLMOD_A, L, r, NULL, &chol_sol, NULL, &Y, &E, &A_cm);
  if (!solve_is_ok) {
    std::cerr << "The solve in PCG has problem" << std::endl;
  }

  mkl_dcscmv(&transa, &A_n, &A_n, &Minus, matdescra, Ax, Ai, &Ap[0], &Ap[1],
             (double *)chol_sol->x, &ONE,
             (double *)r->x); // perform the multiplication

  residual = cholmod_norm_dense(r, 0, &A_cm);
  std::cout << "+++ PARTH: Refinement iter: " << -1
            << " - InfNorm Residual: " << residual << std::endl;
  if (residual < acceptable_error) {
    memcpy(res_vec.data(), r->x, rhs_size * sizeof(result[0]));
    cholmod_free_dense(&r, &A_cm);
    cholmod_free_dense(&Y, &A_cm);
    cholmod_free_dense(&E, &A_cm);
    return;
  }

  // Saving the first result
  memcpy(result.data(), chol_sol->x, result.size() * sizeof(result[0]));

  auto sol_x = (double *)chol_sol->x;
  double start_residual = residual;
  int bad_convergence_cnt = 0;

  // Apply iterative refinement
  for (int i = 0; i < ref_iter_num; i++) {
    // Compute r = Ax - b
    // Inf Norm
    memcpy(r->x, rhs, rhs_size * sizeof(rhs[0]));
    mkl_dcscmv(&transa, &A_n, &A_n, &Minus, matdescra, Ax, Ai, &Ap[0], &Ap[1],
               (double *)chol_sol->x, &ONE,
               (double *)r->x); // perform the multiplication
    num_refinement_iteration++;

#ifndef NDEBUG
    std::vector<std::string> Runtime_headers;
    Runtime_headers.emplace_back("id");
    Runtime_headers.emplace_back("residual");

    profiling_utils::CSVManager runtime_csv(
        "/home/behrooz/Desktop/IPC_Project/SolverTestBench/output/res" +
            std::to_string(frame) + "_" + std::to_string(iter) + "_" +
            std::to_string(i),
        "some address", Runtime_headers, false);

    for (int c = 0; c < rhs_size; c++) {
      runtime_csv.addElementToRecord(c, "id");
      runtime_csv.addElementToRecord(((double *)r->x)[c], "residual");
      runtime_csv.addRecord();
    }
#endif

    residual = cholmod_norm_dense(r, 0, &A_cm);
    std::cout << "+++ PARTH: Refinement iter: " << i
              << " - InfNorm Residual: " << residual << std::endl;
    if (residual < acceptable_error) {
      break;
    }

    if (start_residual < residual) {
      bad_convergence_cnt++;
      if (bad_convergence_cnt == num_non_convergent_iteration) {
        break;
      }
    }

    // Compute Ad = r
    if (chol_sol) {
      cholmod_free_dense(&chol_sol, &A_cm);
    }

    int solve_is_ok =
        cholmod_solve2(CHOLMOD_A, L, r, NULL, &chol_sol, NULL, &Y, &E, &A_cm);
    if (!solve_is_ok) {
      std::cerr << "The solve in PCG has problem" << std::endl;
    }

    // x = x - d
    for (int c = 0; c < rhs_size; c++) {
      result[c] = result[c] + sol_x[c];
    }
    memcpy(chol_sol->x, result.data(), rhs_size * sizeof(result[0]));
  }

  memcpy(res_vec.data(), r->x, rhs_size * sizeof(result[0]));
  cholmod_free_dense(&r, &A_cm);
  cholmod_free_dense(&Y, &A_cm);
  cholmod_free_dense(&E, &A_cm);
}

// TODO: You really need to optimize these codes
///--------------------------------------------------------------------------
/// applyPCG - Apply the PCG algorithm using the current approximated Factor L
/// as a preconditioner
///--------------------------------------------------------------------------
void ParthSolver::applyPCG(
    int pcg_iter_num,             ///<[in] number of refinement iteration
    double acceptable_error,      ///<[in] number of refinement iteration
    int rhs_size,                 ///<[in] size of the rhs array
    std::vector<double> &res_vec, ///<[in] residual vector
    double *rhs,                  ///<[in] the pointer to rhs array
    cholmod_factor *L, ///<[in] The factor that will be used for solve
    Eigen::VectorXd
        &result,     ///<[in/out] result vector containing the first guess
    double &residual ///<[out] residual of the current result
) {

  // Data for using the MKL SpMV
  char transa = 'N'; // don't transpose the matrix
  char matdescra[6] = {
      'S', 'L', 'N',
      'C', ' ', ' '}; // symmetric matrix in 0-based column-major format

  double Minus = -1;
  double ONE = 1;
  double ZERO = 0;
  //-----------------------------------------

  chol_b->x = rhs;

  result.resize(rhs_size);

  if (!chol_Z) {
    chol_Z = cholmod_allocate_dense(A_n, 1, A_n, CHOLMOD_REAL, &A_cm);
  }

  if (!chol_R) {
    chol_R = cholmod_allocate_dense(A_n, 1, A_n, CHOLMOD_REAL, &A_cm);
  }

  // chol_sol = x0 - first result
  double start = omp_get_wtime();
  int solve_is_ok = cholmod_solve2(CHOLMOD_A, L, chol_b, NULL, &chol_Z, NULL,
                                   &chol_Y, &chol_E, &A_cm);
  solve_kernel_total_time += omp_get_wtime() - start;

  if (!solve_is_ok) {
    std::cerr << "The solve in PCG has problem" << std::endl;
  }
  memcpy(result.data(), chol_Z->x, result.size() * sizeof(result[0]));

  // r = b - Ax0
  memcpy(chol_R->x, chol_b->x, result.size() * sizeof(result[0]));
  start = omp_get_wtime();
  mkl_dcscmv(&transa, &A_n, &A_n, &Minus, matdescra, Ax, Ai, &Ap[0], &Ap[1],
             (double *)chol_Z->x, &ONE,
             (double *)chol_R->x); // perform the multiplication
  SpMV_total_time += omp_get_wtime() - start;

  residual = cholmod_norm_dense(chol_R, 0, &A_cm);
  if (Options().getVerbose()) {
    std::cout << "+++ PARTH: PCG iter: " << -1
              << " - InfNorm Residual: " << residual << std::endl;
  }

  if (residual < acceptable_error) {
    res_vec.resize(rhs_size);
    memcpy(res_vec.data(), chol_R->x, rhs_size * sizeof(result[0]));
    return;
  }

  // z0 = M^(-1) * r0
  solve_is_ok = cholmod_solve2(CHOLMOD_A, L, chol_R, NULL, &chol_Z, NULL,
                               &chol_Y, &chol_E, &A_cm);
  if (!solve_is_ok) {
    std::cerr << "The solve in PCG has problem" << std::endl;
  }
  // p0 = z0
  cholmod_dense *p = cholmod_copy_dense(chol_Z, &A_cm);

  double prev_residual = residual;
  int non_convergence_cnt = 0;
  double alpha, beta, tmp;

  for (int i = 0; i < pcg_iter_num; i++) {
    num_refinement_iteration++;
    // alpha(i) = r(i)^t * z(i) / (p^t * A * p(i))
    // r(i)^t * z(i)
    tmp = dotProduct((double *)chol_R->x, (double *)chol_Z->x, rhs_size);

    start = omp_get_wtime();
    mkl_dcscmv(&transa, &A_n, &A_n, &ONE, matdescra, Ax, Ai, &Ap[0], &Ap[1],
               (double *)p->x, &ZERO,
               (double *)chol_Z->x); // perform the multiplication
    SpMV_total_time += omp_get_wtime() - start;

    alpha = tmp / dotProduct((double *)p->x, (double *)chol_Z->x, rhs_size);
    //        alpha = 1;
    // x(i + 1) = x(i) + alpha(i) * p(i)
    vectorSum(result.data(), 1, (double *)p->x, alpha, rhs_size, result.data());

    // r(i + 1) = r(i) - alpha(i) * A * p(i)
    vectorSum((double *)chol_R->x, 1, (double *)chol_Z->x, -1 * alpha, rhs_size,
              (double *)chol_R->x);

    residual = cholmod_norm_dense(chol_R, 0, &A_cm);
    if (Options().getVerbose()) {
      std::cout << "+++ PARTH: PCG iter: " << i
                << " - InfNorm Residual: " << residual << std::endl;
    }

    if (residual < acceptable_error) {
      break;
    }

    if (prev_residual < residual) {
      non_convergence_cnt++;
      if (non_convergence_cnt == num_non_convergent_iteration) {
        break;
      }
    }

    double start = omp_get_wtime();
    solve_is_ok = cholmod_solve2(CHOLMOD_A, L, chol_R, NULL, &chol_Z, NULL,
                                 &chol_Y, &chol_E, &A_cm);
    if (!solve_is_ok) {
      std::cerr << "The solve in PCG has problem" << std::endl;
    }
    solve_kernel_total_time += omp_get_wtime() - start;

    // Beta = r(i + 1)^t * z(i + 1) / (r(i)^t * z(i))
    beta = dotProduct((double *)chol_R->x, (double *)chol_Z->x, rhs_size) / tmp;
    //        beta = 1;
    // p(i+1) = z(i+1) + beta * p(i)
    vectorSum((double *)p->x, beta, (double *)chol_Z->x, 1, rhs_size,
              (double *)p->x);
  }

  res_vec.resize(rhs_size);
  memcpy(res_vec.data(), chol_R->x, rhs_size * sizeof(result[0]));
  cholmod_free_dense(&p, &A_cm);
}

} // namespace PARTH