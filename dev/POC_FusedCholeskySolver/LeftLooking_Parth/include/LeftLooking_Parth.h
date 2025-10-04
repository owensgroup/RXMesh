//
// Created by behrooz on 28/09/22.
//

#ifndef PARTH_SOLVER_LEFTLOOKING_PARTH_H
#define PARTH_SOLVER_LEFTLOOKING_PARTH_H

#include <Eigen/SparseCore>

#include "ND_Regions.h"
#include "Regions.h"
#include "cholmod.h"
#include "cholmod_core.h"

#include <fstream>
#include <iostream>
#include <omp.h>
#include <queue>
#include <set>
#include <string>
#include <vector>

namespace PARTH {

class ParthSolver {

public:
  // visibility will default to private unless you specify it
  enum ReorderingType { METIS, AMD, AUTO };
  enum AnalyzeType { SimulationAware, Mesh, Hessian };
  enum SymbolicReuseType {
    SYMBOLIC_NO_REUSE,
    SYMBOLIC_REUSE_BASE,
    SYMBOLIC_REUSE_AGGRESSIVE
  };
  enum NumericReuseType {
    NUMERIC_NO_REUSE,
    NUMERIC_REUSE_BASE,
    NUMERIC_NO_REUSE_PARALLEL,
    NUMERIC_REUSE_PARALLEL
  };

  struct SolverConfig {
    // specify members here;
  private:
    ReorderingType reorder_type;
    AnalyzeType analyze_type;
    SymbolicReuseType symbolic_reuse_type;
    NumericReuseType numeric_reuse_type;
    int num_cores = 10;
    int num_sockets = 2;

    bool verbose = true;
    bool compute_residual = true;
    int num_regions = 0;
    std::string csv_export_address;

  public:
    ///--------------------------------------------------------------------------
    /// setReorderingType - set the reordering type from METIS, AMD, AUTO
    ///--------------------------------------------------------------------------
    void setReorderingType(ReorderingType type);
    ReorderingType getReorderingType() const;

    ///--------------------------------------------------------------------------
    /// setAnalyzeType - set the Analysis type from Mesh, Hessian, or Simulation
    /// Aware
    ///--------------------------------------------------------------------------
    void setAnalyzeType(AnalyzeType type);
    AnalyzeType getAnalyzeType() const;

    ///--------------------------------------------------------------------------
    /// setVerbose - if true, report will be printed
    ///--------------------------------------------------------------------------
    void setVerbose(bool verbose);
    bool getVerbose() const;

    ///--------------------------------------------------------------------------
    /// setComputeResidual - if true, the residual will be computed after each
    /// solve
    ///--------------------------------------------------------------------------
    void setComputeResidual(bool flag);
    bool getComputeResidual() const;

    ///--------------------------------------------------------------------------
    /// setNumSubMesh - Set the number of regions that the mesh will be
    /// decomposed too
    ///--------------------------------------------------------------------------
    void setNumRegions(int num);
    int getNumRegions() const;

    ///--------------------------------------------------------------------------
    /// setSymbolicReuse - Set the level of reuse in symbolic factorization
    ///--------------------------------------------------------------------------
    void setSymbolicReuseType(SymbolicReuseType type);
    SymbolicReuseType getSymbolicReuseType() const;

    ///--------------------------------------------------------------------------
    /// setNumericReuse - Set the level of reuse in numeric factorization
    ///--------------------------------------------------------------------------
    void setNumericReuseType(NumericReuseType type);
    NumericReuseType getNumericReuseType() const;

    ///--------------------------------------------------------------------------
    /// setNumberOfCores - Set the number of cores for region parallelism
    ///--------------------------------------------------------------------------
    void setNumberOfCores(int num_cores);
    int getNumberOfCores() const;

    ///--------------------------------------------------------------------------
    /// setNumberOfCores - Set the number of sockets or processors node for MPI
    /// parallelism
    ///--------------------------------------------------------------------------
    void setNumberOfSockets(int num_sockets);
    int getNumberOfSockets() const;

    ///--------------------------------------------------------------------------
    /// setNumSubMesh - Set the number of regions that the mesh will be
    /// decomposed too
    ///--------------------------------------------------------------------------
    void setCSVExportAddress(std::string csv_export_address);
    std::string getCSVExportAddress();
  };

protected:
  ///@brief Basic Storages for A
  Eigen::SparseMatrix<double> A;
  int *Ap, *Ai;
  double *Ax;
  int A_nnz, A_n;
  ///@bierf Hessian related variables
  cholmod_common A_cm;
  cholmod_sparse *chol_A;
  cholmod_sparse *chol_A_prev;

  cholmod_factor *chol_L;
  cholmod_dense *chol_b, *chol_sol;
  cholmod_dense *chol_Z, *chol_E, *chol_Y, *chol_R;

  void *chol_Ap_tmp, *chol_Ai_tmp, *chol_Ax_tmp, *chol_b_tmp;

  ///@bierf Config Properties
  SolverConfig opt;

  ///@bierf Error Checking Variables
  bool mat_allocated;
  bool mesh_init;

  ///@bierf Profiling Variables
  double residual;

  ///@brief Contact aware simulations variables
  int *Mp;
  int *Mi;
  int M_n;

  std::vector<int> Mp_prev;
  std::vector<int> Mi_prev;
  int M_n_prev;
  int M_nnz_prev;

  std::vector<int> mesh_perm;
  std::vector<int> mesh_perm_prev;
  std::vector<int> hessian_perm;

  int frame = -1;
  int iter = -1;
  std::string sim_name;
  std::string input_address;

  ///@brief timing variables
  double analyze_total_time = 0;
  double factorize_total_time = 0;
  double solve_total_time = 0;

public:
  double SpMV_total_time = 0;
  double solve_kernel_total_time = 0;

  ///@brief if one of the element change its region,
  /// the from and to region will be dirty
  //  Regions regions;
  ND_Regions regions;

  ///@brief supernode caching variables
  std::vector<int> col_to_super_prev;
  std::vector<int> col_to_super_curr;
  std::vector<int> first_nonzero_per_row;
  std::vector<int> node_cost;

  bool copy_prev_factor = false;
  std::vector<int> super_is_cached;
  std::vector<double> res_vec_per_supernode;
  std::vector<double> res_vec_per_elem;
  std::vector<double> scaling_factor_vec;

  int total_num_of_super = 0;
  int num_steps_to_get_result = 0;
  bool tree_is_computed = 0;
  bool analyze_is_requested = false;

  // supernode Etree variables
  std::vector<int> super_parents;
  std::vector<int> tree_ptr;
  std::vector<int> tree_set;

  ///@brief User defined acceptable error in infinite norm
  double solve_acceptable_error = 1e-3;
  double preconditioner_acceptable_error = 1e-3;
  int num_refinement_iteration = 0;
  int num_max_refinement_iteration = 2;
  int num_max_PCG_iteration = 5;
  int num_non_convergent_iteration = 2;

  ///@brief
  cholmod_factor *chol_L_prev;

  // TODO:Delete or clear these variables
  std::vector<std::vector<double>> score_view;
  std::vector<std::string> score_name;

  ///@breif Parallelization variables
  int parallel_levels = 0;
  std::vector<int> level_ptr;
  std::vector<int> part_ptr;
  std::vector<int> supernode_idx;
  std::vector<int> super_nz;

  ParthSolver(void);
  ~ParthSolver(void);
  void outputA(const std::string &filePath);
  void outputRegions(const std::string &filePath);

  //===========================================================================
  //====================== Initializing Matrices Fuctions =====================
  //===========================================================================
  ///--------------------------------------------------------------------------
  /// setMatrix - allocate the matrices for Parth usage
  ///--------------------------------------------------------------------------
  void setMatrix(const int *Ap,    ///<[in] pointer array
                 const int *Ai,    ///<[in] index array
                 const double *Ax, /// [in] value array
                 const int n,      /// [in] Number of rows/columns
                 const int nnz     /// [in] Number of nonzeros
  );

  ///--------------------------------------------------------------------------
  /// setMatrix - allocate the
  /// matrices for Parth usage
  ///--------------------------------------------------------------------------
  void setMatrix(const Eigen::SparseMatrix<double> &A); // NOTE: mtr must be SPD

  ///--------------------------------------------------------------------------
  /// setMatrixPointers - it only set the pointers to the appropriate used
  /// defined allocated matrices\n NOTE: The correctness of memory allocation is
  /// on the user and Parth may change the arrays
  ///--------------------------------------------------------------------------
  void setMatrixPointers(int *Ap,    ///<[in] pointer array
                         int *Ai,    ///<[in] index array
                         double *Ax, /// [in] value array
                         int n,      /// [in] Number of rows/columns
                         int nnz     /// [in] Number of nonzeros
  );

  ///--------------------------------------------------------------------------
  /// setMatrixPointers - it only set the pointers to the appropriate used
  /// defined allocated matrices\n NOTE: The correctness of memory allocation is
  /// on the user and Parth may change the arrays
  ///--------------------------------------------------------------------------
  void setMatrixPointers(Eigen::SparseMatrix<double> &A);

  ///--------------------------------------------------------------------------
  /// setMatrixPointers - it only set the pointers to the appropriate used
  /// defined allocated matrices\n NOTE: The correctness of memory allocation is
  /// on the user and Parth may change the arrays
  ///--------------------------------------------------------------------------
  void setMeshPointers(int n,   /// [in] Number of rows/columns
                       int *Mp, ///<[in] pointer array
                       int *Mi  ///<[in] index array
  );

  //===========================================================================
  //========================== Analysis Functions  ============================
  //===========================================================================

  ///--------------------------------------------------------------------------
  /// analyze - the entry point for matrix inspection before factorization
  ///--------------------------------------------------------------------------
  void analyze(bool do_analysis = false ///<[in] force the analysis to happen
  );

  ///--------------------------------------------------------------------------
  /// hessianAnalyze - the entry point for analysis of the hessian
  ///--------------------------------------------------------------------------
  void hessianAnalyze();

  ///--------------------------------------------------------------------------
  /// meshAnalyze - the entry point for applying the analysis
  /// to hessian using mesh information
  ///--------------------------------------------------------------------------
  void meshAnalyze();

  ///--------------------------------------------------------------------------
  /// SimulationAwareAnalyze - the entry point for applying the analysis
  /// to hessian using mesh and simulation's information
  ///--------------------------------------------------------------------------
  void SimulationAwareAnalyze();

  ///--------------------------------------------------------------------------
  /// getElemRegions - return a vector with size M_n \n
  /// that shows the region of each node
  ///--------------------------------------------------------------------------
  std::vector<int> &getElemRegions();

  ///--------------------------------------------------------------------------
  /// computeContactPoints - Based on the nodes neighbor across to iterations,\n
  /// computes the contact points. If it is the first time of calling the
  /// function \n it returns all the points as contact
  ///--------------------------------------------------------------------------
  void computeContacts(
      std::vector<int> &contact_points ///<[out] points with changed neighbors
  );

  /// --------------------------------------------------------------------------
  /// cholmod_analyze_custom:  order and analyze (simplicial or supernodal) */
  /// --------------------------------------------------------------------------
  /// Orders and analyzes A, AA', PAP', or PAA'P' and returns a symbolic factor
  /// that can later be passed to cholmod_factorize.
  cholmod_factor *cholmod_analyze_custom(
      /* ---- input ---- */
      cholmod_sparse *A, /* matrix to order and analyze */
      /* --------------- */
      cholmod_common *Common);

  /// ==========================================================================
  /// === cholmod_analyze_p ====================================================
  /// ==========================================================================
  /// */ Orders and analyzes A, AA', PAP', PAA'P', FF', or PFF'P and returns a
  /// symbolic factor that can later be passed to cholmod_factorize, where
  /// F = A(:,fset) if fset is not NULL and A->stype is zero.
  /// UserPerm is tried if non-NULL.
  cholmod_factor *cholmod_analyze_p_custom(
      /* ---- input ---- */
      cholmod_sparse *A, /* matrix to order and analyze */
      int *UserPerm,     /* user-provided permutation, size A->nrow */
      int *fset,         /* subset of 0:(A->ncol)-1 */
      size_t fsize,      /* size of fset */
      /* --------------- */
      cholmod_common *Common);

  /// --------------------------------------------------------------------------
  /// cholmod_analyze_p2:  analyze for sparse Cholesky or sparse QR */
  /// --------------------------------------------------------------------------
  cholmod_factor *cholmod_analyze_p2_custom(
      /* ---- input ---- */
      int for_whom,      /* FOR_SPQR     (0): for SPQR but not GPU-accelerated
                         FOR_CHOLESKY (1): for Cholesky (GPU or not)
                         FOR_SPQRGPU  (2): for SPQR with GPU acceleration */
      cholmod_sparse *A, /* matrix to order and analyze */
      int *UserPerm,     /* user-provided permutation, size A->nrow */
      int *fset,         /* subset of 0:(A->ncol)-1 */
      size_t fsize,      /* size of fset */
      /* --------------- */
      cholmod_common *Common);

  int cholmod_analyze_ordering_custom(
      /* ---- input ---- */
      cholmod_sparse *A, /* matrix to analyze */
      int ordering,      /* ordering method used */
      int *Perm,         /* size n, fill-reducing permutation to analyze */
      int *fset,         /* subset of 0:(A->ncol)-1 */
      size_t fsize,      /* size of fset */
      /* ---- output --- */
      int *Parent,   /* size n, elimination tree */
      int *Post,     /* size n, postordering of elimination tree */
      int *ColCount, /* size n, nnz in each column of L */
      /* ---- workspace  */
      int *First, /* size n workspace for cholmod_postorder */
      int *Level, /* size n workspace for cholmod_postorder */
      /* --------------- */
      cholmod_common *Common);

  /* --------------------------------------------------------------------------
   */
  /* cholmod_super_symbolic */
  /* --------------------------------------------------------------------------
   */

  /* Analyzes A, AA', or A(:,f)*A(:,f)' in preparation for a supernodal numeric
   * factorization.  The user need not call this directly; cholmod_analyze is
   * a "simple" wrapper for this routine.
   */

  int cholmod_super_symbolic_custom(
      /* ---- input ---- */
      cholmod_sparse *A, /* matrix to analyze */
      cholmod_sparse *F, /* F = A' or A(:,f)' */
      int *Parent,       /* elimination tree */
      /* ---- in/out --- */
      cholmod_factor *L, /* simplicial symbolic on input,
                          * supernodal symbolic on output */
      /* --------------- */
      cholmod_common *Common);

  /* --------------------------------------------------------------------------
   */
  /* cholmod_super_symbolic2 */
  /* --------------------------------------------------------------------------
   */

  /* Analyze for supernodal Cholesky or multifrontal QR */

  int cholmod_super_symbolic2_custom(
      /* ---- input ---- */
      int for_whom,      /* FOR_SPQR     (0): for SPQR but not GPU-accelerated
                                 FOR_CHOLESKY (1): for Cholesky (GPU or not)
                                 FOR_SPQRGPU  (2): for SPQR with GPU acceleration */
      cholmod_sparse *A, /* matrix to analyze */
      cholmod_sparse *F, /* F = A' or A(:,f)' */
      int *Parent,       /* elimination tree */
      /* ---- in/out --- */
      cholmod_factor *L, /* simplicial symbolic on input,
                          * supernodal symbolic on output */
      /* --------------- */
      cholmod_common *Common);

  void computeFirstchild(int n, int *Ap, int *Ai);

  void createParallelSchedule();

  void createSpTRSVParallelSchedule();

  void createReuseParallelSchedule();

  int computeETreeCost(int *super, int *tree_ptr, int *tree_set,
                       int current_node);

  //===========================================================================
  //======================= Factorization Functions  ==========================
  //===========================================================================
  ///--------------------------------------------------------------------------
  /// factorize - the entry point for matrix factorization
  ///--------------------------------------------------------------------------
  bool factorize();

  bool cholmodFactoization();

  bool analyzeReuseFactorization();

  bool ParallelReuseFactorization();
  bool reuseFactoization();

  bool parallelFactoization();
  ///--------------------------------------------------------------------------
  /// outputFactorization - save the factor into filePath
  ///--------------------------------------------------------------------------
  bool outputFactorization(
      const std::string &filePath ///[in] save location + name of the matrix
  );
  ///--------------------------------------------------------------------------
  /// getFactorNonzeros - return the number of nnz in the factor
  ///--------------------------------------------------------------------------
  int getFactorNonzeros();

  ///---------------------------------------------------------------------
  /// These are main supernodal functions adapted from cholmod
  ///---------------------------------------------------------------------
  int cholmod_factorize_custom(
      cholmod_sparse *A,     ///<[in] matrix to factorize
      cholmod_factor *L,     ///<[out] resulting factorization
      cholmod_common *Common ///<[in/out]
  );

  ///---------------------------------------------------------------------
  /// These are main supernodal functions adapted from cholmod
  ///---------------------------------------------------------------------
  int cholmod_factorize_p_custom(
      cholmod_sparse *A,     ///<[in] matrix to factorize */
      double beta[2],        ///<[in] factorize beta*I+A or beta*I+A'*A */
      int *fset,             ///<[in] subset of 0:(A->ncol)-1 */
      size_t fsize,          ///<[in] size of fset */
      cholmod_factor *L,     ///<[in/out] resulting factorization */
      cholmod_common *Common ///<[in/out]
  );

  int cholmod_super_numeric_custom(
      cholmod_sparse *A, ///<[in] matrix to factorize */
      cholmod_sparse *F, ///<[in] F = A' or A(:,f)' */
      double
          beta[2], ///<[in] beta*I is added to diagonal of matrix to factorize
      cholmod_factor *L,     ///<[in/out]factorization */
      cholmod_common *Common ///<[in/out]
  );

  ///--------------------------------------------------------------------------
  /// initScoreVector - Compute residual distribution per supernode
  ///--------------------------------------------------------------------------
  void initScoreVector(double *r);

  ///--------------------------------------------------------------------------
  /// computeInverseEliminationTree - Compute the elimination tree inverse
  /// - from node to its ancestors - given parent_vector array (see the Tim's
  /// book for parent_vector nature)
  ///--------------------------------------------------------------------------
  void computeInverseEliminationTree(
      int n, ///<[in] number of nodes inside the elimination tree
      const int *parent_vector,   ///<[in] parent vector of the tree
      std::vector<int> &tree_ptr, ///<[out] pointer array in CSC format
      std::vector<int> &tree_set  ///<[out] index array in CSC format
  );

  ///--------------------------------------------------------------------------
  /// savePrevMatrixForReuse - It  saves the chol_A into chol_A_prev;
  /// chol_A is still valid after the operation
  ///--------------------------------------------------------------------------
  void savePrevMatrixForReuse(int *Perm);

  ///--------------------------------------------------------------------------
  /// savePrevFactorForReuse - make chol_L equal to chol_L_prev.
  /// NOTE: Be careful about freeing chol_L if you want to use chol_L_prev
  ///--------------------------------------------------------------------------
  void savePrevFactorForReuse();

  ///--------------------------------------------------------------------------
  /// permuteMatrix - Permute a lower triangular matrix in CSC format
  /// NOTE: Allocate Ap_perm, Ai_perm, and Ax_perm beforehand (it has the same
  /// size as the input matrix)
  ///--------------------------------------------------------------------------
  void permuteMatrix(int *perm,     ///<[in] The permutation array
                     int N,         ///<[in] Number of cols/rows
                     int *p,        ///<[in] The pointer array in CSC format
                     int *i,        ///<[in] The index array in CSC format
                     double *x,     ///<[in] The value array in CSC format
                     int *p_perm,   ///<[out] The pointer array in CSC format
                     int *i_perm,   ///<[out] The index array in CSC format
                     double *x_perm ///<[out] The value array in CSC format
  );
  //===========================================================================
  //============================ Solve Functions  =============================
  //===========================================================================

  cholmod_dense *cholmod_solve_custom(
      /* ---- input ---- */
      int sys,           /* system to solve */
      cholmod_factor *L, /* factorization to use */
      cholmod_dense *B,  /* right-hand-side */
      /* --------------- */
      cholmod_common *Common);

  int cholmod_solve2_custom /* returns TRUE on success, FALSE on failure */
      (
          /* ---- input ---- */
          int sys,           /* system to solve */
          cholmod_factor *L, /* factorization to use */
          cholmod_dense *B,  /* right-hand-side */
          cholmod_sparse *Bset,
          /* ---- output --- */
          cholmod_dense **X_Handle, /* solution, allocated if need be */
          cholmod_sparse **Xset_Handle,
          /* ---- workspace  */
          cholmod_dense **Y_Handle, /* workspace, or NULL */
          cholmod_dense **E_Handle, /* workspace, or NULL */
          /* --------------- */
          cholmod_common *Common);

  int cholmod_super_ltsolve_custom /* TRUE if OK, FALSE if BLAS overflow occured
                                    */
      (
          /* ---- input ---- */
          cholmod_factor *L, /* factor to use for the backsolve */
          /* ---- output ---- */
          cholmod_dense *X, /* b on input, solution to L'x=b on output */
          /* ---- workspace ---- */
          cholmod_dense *E, /* workspace of size nrhs*(L->maxesize) */
          /* --------------- */
          cholmod_common *Common);

  int cholmod_super_lsolve_custom /* TRUE if OK, FALSE if BLAS overflow occured
                                   */
      (
          /* ---- input ---- */
          cholmod_factor *L, /* factor to use for the forward solve */
          /* ---- output ---- */
          cholmod_dense *X, /* b on input, solution to Lx=b on output */
          /* ---- workspace ---- */
          cholmod_dense *E, /* workspace of size nrhs*(L->maxesize) */
          /* --------------- */
          cholmod_common *Common);

  ///--------------------------------------------------------------------------
  /// solve - Entry point for solve function
  ///--------------------------------------------------------------------------
  void solve(Eigen::VectorXd &rhs,   ///<[in] right hand side of Ax=b
             Eigen::VectorXd &result ///<[out] the unknown x in Ax=b
  );

  ///--------------------------------------------------------------------------
  /// cholmodSolve - basic cholmod solve
  ///--------------------------------------------------------------------------
  void cholmodSolve(Eigen::VectorXd &rhs,   ///<[in] right hand side of Ax=b
                    Eigen::VectorXd &result ///<[out] the unknown x in Ax=b
  );

  ///--------------------------------------------------------------------------
  /// analyzeReuseSolve - solve with refinement and PCG
  ///--------------------------------------------------------------------------
  void analyzeReuseSolve(Eigen::VectorXd &rhs, ///<[in] right hand side of Ax=b
                         Eigen::VectorXd &result ///<[out] the unknown x in Ax=b
  );

  ///--------------------------------------------------------------------------
  /// analyzeReuseSolve - solve with refinement and PCG
  ///--------------------------------------------------------------------------
  void
  ParallelReuseSolve(Eigen::VectorXd &rhs,   ///<[in] right hand side of Ax=b
                     Eigen::VectorXd &result ///<[out] the unknown x in Ax=b
  );

  double getResidual();

  //===========================================================================
  //============================ Options Functions ============================
  //===========================================================================
  SolverConfig &Options();

  //===========================================================================
  //============================ Auxiliary Functions
  //============================
  //===========================================================================
  ///--------------------------------------------------------------------------
  /// getHessianPerm - get Hessian permutation vector
  ///--------------------------------------------------------------------------
  std::vector<int> getHessianPerm();

  ///--------------------------------------------------------------------------
  /// getHessianPerm - get Hessian permutation vector
  ///--------------------------------------------------------------------------
  std::vector<int> getMeshPerm();

  ///--------------------------------------------------------------------------
  /// permuteMatrix - get permuted hessian
  ///--------------------------------------------------------------------------
  void permuteMatrix(int An, int *Ap, int *Ai, double *Ax,
                     std::vector<int> &perm,
                     Eigen::SparseMatrix<double> &permuted_A);

  ///--------------------------------------------------------------------------
  /// setFrameIter - Set the current frame and newton iteration that
  /// the solver is called from
  ///--------------------------------------------------------------------------
  void setFrameIter(int frame, int iter, std::string name);

  ///--------------------------------------------------------------------------
  /// getSymbolicReuse - Get the ratio of columns that are not re-permuted
  /// to the total number of columns in hessian
  ///--------------------------------------------------------------------------
  double getSymbolicReuse();

  ///--------------------------------------------------------------------------
  /// getNumericReuse - This function relies on the assumption that if
  /// a supernode is already marked, all of its ancestors are also marked
  /// with zero.
  ///--------------------------------------------------------------------------
  double getNumericReuse();

  ///--------------------------------------------------------------------------
  /// TakeDiff - A useful debugging function that export information
  /// about reusing opportunities
  ///--------------------------------------------------------------------------
  void TakeDiff(cholmod_factor *full, int adjuster_cnt);

  ///--------------------------------------------------------------------------
  /// OracleCachingMechanism - given a residual vector, the function
  /// will mark the hotspots that the factor is inexact for those regions
  ///--------------------------------------------------------------------------
  void OracleCachingMechanism(double *residual_vec);

  ///--------------------------------------------------------------------------
  /// convertSuperFlagToNodeFlag - Function is used for debugging purposes
  ///--------------------------------------------------------------------------
  std::vector<int> convertSuperFlagToNodeFlag(std::vector<int> &super_flag);

  ///--------------------------------------------------------------------------
  /// convertSuperScoreToNodeScore - Function is used for debugging purposes
  ///--------------------------------------------------------------------------
  std::vector<double>
  convertSuperScoreToNodeScore(std::vector<double> &super_flag);

  void setInputAddress(std::string address);

  ///--------------------------------------------------------------------------
  /// iterativeRefinement - Apply an iterative refinement
  /// NOTE - remember to free the array r and pass it as a nullptr
  ///--------------------------------------------------------------------------
  void iterativeRefinement(
      int ref_iter_num,             ///<[in] number of refinement iteration
      double acceptable_error,      ///<[in] number of refinement iteration
      int rhs_size,                 ///<[in] size of the rhs array
      std::vector<double> &res_vec, ///<[in] residual vector
      double *rhs,                  ///<[in] the pointer to rhs array
      cholmod_factor *L,       ///<[in] The factor that will be used for solve
      Eigen::VectorXd &result, ///<[out] result vector
      double &residual         ///<[out] residual of the current result
  );

  ///--------------------------------------------------------------------------
  /// applyPCG - Apply the PCG algorithm using the current approximated Factor L
  /// as a preconditioner
  ///@return the scaled residual
  ///--------------------------------------------------------------------------
  void
  applyPCG(int pcg_iter_num,             ///<[in] number of refinement iteration
           double acceptable_error,      ///<[in] number of refinement iteration
           int rhs_size,                 ///<[in] size of the rhs array
           std::vector<double> &res_vec, ///<[in] residual vector
           double *rhs,                  ///<[in] the pointer to rhs array
           cholmod_factor *L, ///<[in] The factor that will be used for solve
           Eigen::VectorXd &result, ///<[in/out] result vector
           double &residual         ///<[out] residual of the current result
  );

  ///--------------------------------------------------------------------------
  /// getAnalyzeTime - return the total symbolic analysis time
  ///--------------------------------------------------------------------------
  double getAnalyzeTime(
      bool reset = true ///<[in] make the timing variable zero if equal to true
  );

  ///--------------------------------------------------------------------------
  /// iterativeRefinement - Apply an iterative refinement
  /// NOTE - remember to free the array r and pass it as a nullptr
  ///--------------------------------------------------------------------------
  double getFactorTime(
      bool reset = true ///<[in] make the timing variable zero if equal to true
  );

  ///--------------------------------------------------------------------------
  /// iterativeRefinement - Apply an iterative refinement
  /// NOTE - remember to free the array r and pass it as a nullptr
  ///--------------------------------------------------------------------------
  double getSolveTime(
      bool reset = true ///<[in] make the timing variable zero if equal to true
  );

  ///--------------------------------------------------------------------------
  /// getPermutationWithReuse - By saving the matrix and mesh in each
  /// permutation, it return the permutation matrix with reuse with respect to
  /// contacts
  ///--------------------------------------------------------------------------
  void getPermutationWithReuse(std::vector<int> &perm);
};

} // namespace PARTH

#endif
