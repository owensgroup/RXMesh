//
//  CHOLMODSolver.cpp
//  IPC
//
//  Created by Minchen Li on 6/22/18.
//

#ifdef USE_SUITESPARSE

#include "CHOLMODSolver.hpp"
#include <cassert>
#include <iostream>
#include "omp.h"
#include <spdlog/spdlog.h>

namespace RXMESH_SOLVER {
CHOLMODSolver::~CHOLMODSolver()
{
    if (use_gpu) {
        // GPU mode cleanup with int64 functions
        if (A) {
            A->i = Ai;
            A->p = Ap;
            A->x = Ax;
            cholmod_l_free_sparse(&A, &cm);
        }

        if (L) {
            cholmod_l_free_factor(&L, &cm);
        }

        if (b) {
            b->x = bx;
            cholmod_l_free_dense(&b, &cm);
        }

        if (x_solve) {
            cholmod_l_free_dense(&x_solve, &cm);
        }

        cholmod_l_finish(&cm);
    } else {
        // CPU mode cleanup with regular functions
        if (A) {
            A->i = Ai;
            A->p = Ap;
            A->x = Ax;
            cholmod_free_sparse(&A, &cm);
        }

        if (L) {
            cholmod_free_factor(&L, &cm);
        }

        if (b) {
            b->x = bx;
            cholmod_free_dense(&b, &cm);
        }

        if (x_solve) {
            cholmod_free_dense(&x_solve, &cm);
        }

        cholmod_finish(&cm);
    }
}

CHOLMODSolver::CHOLMODSolver()
{
    // Use int64 version for GPU mode, regular for CPU mode
    if (use_gpu) {
        cholmod_l_start(&cm);
        spdlog::info("CHOLMOD initialized in GPU mode (int64)");
    } else {
        cholmod_start(&cm);
        spdlog::info("CHOLMOD initialized in CPU-only mode");
    }

    bx      = NULL;
    A       = NULL;
    L       = NULL;
    b       = NULL;
    x_solve = NULL;
    Ai = Ap = Ax = NULL;
    
    //If GPU mode requested, initialize and probe GPU
    if (use_gpu) {
        initializeGPU();
        if (!probeGPU()) {
            // GPU requested but not available - throw error (ungraceful)
            spdlog::error("GPU mode requested but GPU not available");
            throw std::runtime_error("GPU not available for CHOLMOD - GPU mode was requested but GPU is not available");
        }
    } else {
        spdlog::info("CPU-only mode - GPU support not required");
    }
}

void CHOLMODSolver::initializeGPU()
{
    // Enable GPU usage for CHOLMOD
    cm.useGPU = 1;
    // Force supernodal factorization for GPU (required for GPU acceleration)
    cm.supernodal = CHOLMOD_SUPERNODAL;
    // Set additional GPU-friendly parameters
    cm.print = 3;  // Print more diagnostic info
    
    // Set numerical tolerance parameters to help with potential precision issues
    // GPU computations can be more sensitive to numerical issues
    cm.final_ll = 1;  // Force LL' factorization (more stable than LDL')
    
    spdlog::info("CHOLMOD GPU mode enabled with supernodal LL' factorization");
}

bool CHOLMODSolver::probeGPU()
{
    // Probe for GPU availability
    #if defined(CHOLMOD_HAS_CUDA)
    int gpu_status = cholmod_l_gpu_probe(&cm);
    if (gpu_status == 1) {
        gpu_available = true;
        spdlog::info("CHOLMOD GPU detected and available");
        // Allocate GPU memory
        cholmod_l_gpu_allocate(&cm);
        spdlog::info("CHOLMOD GPU memory allocated");
        return true;
    } else {
        spdlog::error("CHOLMOD GPU not available (probe returned {})", gpu_status);
        return false;
    }
    #else
    spdlog::error("CHOLMOD was not compiled with CUDA support");
    return false;
    #endif
}

void CHOLMODSolver::cholmod_clean_memory()
{
    if (use_gpu) {
        // GPU mode cleanup with int64 functions
        if (A) {
            A->i = Ai;
            A->p = Ap;
            A->x = Ax;
            cholmod_l_free_sparse(&A, &cm);
        }

        if (b) {
            b->x = bx;
            cholmod_l_free_dense(&b, &cm);
        }

        if (x_solve) {
            cholmod_l_free_dense(&x_solve, &cm);
        }
    } else {
        // CPU mode cleanup with regular functions
        if (A) {
            A->i = Ai;
            A->p = Ap;
            A->x = Ax;
            cholmod_free_sparse(&A, &cm);
        }

        if (b) {
            b->x = bx;
            cholmod_free_dense(&b, &cm);
        }

        if (x_solve) {
            cholmod_free_dense(&x_solve, &cm);
        }
    }

    A       = NULL;
    b       = NULL;
    x_solve = NULL;
    Ai = Ap = Ax = NULL;
}

void CHOLMODSolver::setMatrix(int*              p,
                              int*              i,
                              double*           x,
                              int               A_N,
                              int               NNZ)
{
    assert(p[A_N] == NNZ);
    this->N   = A_N;
    this->NNZ = NNZ;

    this->cholmod_clean_memory();

    if (!A) {
        if (use_gpu) {
            A = cholmod_l_allocate_sparse(
                N, N, NNZ, true, true, -1, CHOLMOD_REAL, &cm);
            //Convert the values in p and i to long int (int64_t)
            p_long.resize(N + 1);
            i_long.resize(NNZ);
            
            for (int idx = 0; idx < N + 1; idx++) {
                p_long[idx] = static_cast<long int>(p[idx]);
            }
            
            for (int idx = 0; idx < NNZ; idx++) {
                i_long[idx] = static_cast<long int>(i[idx]);
            }

            this->Ap = A->p;
            this->Ax = A->x;
            this->Ai = A->i;

            A->p = p_long.data();
            A->i = i_long.data();
            A->x = x;
            
            spdlog::info("Matrix allocated for GPU mode: {}x{}, NNZ={}, stype=-1 (lower triangular)", N, N, NNZ);
            
            // Verify matrix is properly sorted (required for CHOLMOD)
            int status = cholmod_l_check_sparse(A, &cm);
            if (status == 0) {
                spdlog::error("Matrix check failed! Matrix may not be properly formed");
            } else {
                spdlog::info("Matrix check passed (GPU mode)");
            }
        } else {
            A = cholmod_allocate_sparse(
                N, N, NNZ, true, true, -1, CHOLMOD_REAL, &cm);

            this->Ap = A->p;
            this->Ax = A->x;
            this->Ai = A->i;

            A->p = p;
            A->i = i;
            A->x = x;
            
            spdlog::info("Matrix allocated for CPU mode: {}x{}, NNZ={}", N, N, NNZ);
        }

        // -1: upper right part will be ignored during computation (stype = -1 means lower triangular stored)
    }


}

void CHOLMODSolver::innerAnalyze_pattern(std::vector<int>& user_defined_perm)
{
    if (use_gpu) {
        cholmod_l_free_factor(&L, &cm);
        std::vector<long int> long_user_defined_perm(user_defined_perm.size());
        for (int i = 0; i < user_defined_perm.size(); i++) {
            long_user_defined_perm[i] = user_defined_perm[i];
        }
        
        // Verify GPU is available before analysis
        if (!gpu_available) {
            spdlog::error("GPU mode enabled but GPU not available for CHOLMOD analysis");
            throw std::runtime_error("GPU not available for CHOLMOD - GPU mode was requested but GPU is not available");
        }
        
        // Ensure GPU and supernodal settings are still active
        cm.useGPU = 1;
        cm.supernodal = CHOLMOD_SUPERNODAL;
        
        spdlog::info("Matrix size: {}, NNZ: {} (GPU mode)", N, NNZ);
        
        if (user_defined_perm.size() == N) {
            spdlog::info("Using user provided permutation (GPU mode)");
            cm.nmethods           = 1;
            cm.method[0].ordering = CHOLMOD_GIVEN;
            L                     = cholmod_l_analyze_p(A, long_user_defined_perm.data(), NULL, 0, &cm);
        } else {
            spdlog::info("Using METIS permutation (GPU mode)");
            cm.nmethods           = 1;
            cm.method[0].ordering = CHOLMOD_METIS;
            L                     = cholmod_l_analyze(A, &cm);
        }
        
        spdlog::info("CHOLMOD symbolic analysis complete with GPU enabled");
    } else {
        cholmod_free_factor(&L, &cm);
        
        cm.useGPU = 0;
        cm.supernodal = CHOLMOD_SUPERNODAL;
        
        if (user_defined_perm.size() == N) {
            spdlog::info("Using user provided permutation (CPU mode)");
            cm.nmethods           = 1;
            cm.method[0].ordering = CHOLMOD_GIVEN;
            L                     = cholmod_analyze_p(A, user_defined_perm.data(), NULL, 0, &cm);
        } else {
            spdlog::info("Using METIS permutation (CPU mode)");
            cm.nmethods           = 1;
            cm.method[0].ordering = CHOLMOD_METIS;
            L                     = cholmod_analyze(A, &cm);
        }
        
        spdlog::info("CHOLMOD symbolic analysis complete in CPU mode");
    }
    
    assert(L != nullptr);
    if (L == nullptr) {
        std::cerr << "ERROR during symbolic factorization:" << std::endl;
        throw std::runtime_error("Symbolic factorization failed");
    }
    L_NNZ = cm.lnz * 2 - N;
}

void CHOLMODSolver::innerFactorize(void)
{
    if (use_gpu) {
        cholmod_l_factorize(A, L, &cm);
    } else {
        cholmod_factorize(A, L, &cm);
    }
    
    if (cm.status == CHOLMOD_NOT_POSDEF) {
        std::cerr << "ERROR during numerical factorization - matrix not positive definite"
                  << std::endl;
        std::cerr << "Matrix size: " << N << ", NNZ: " << NNZ << std::endl;
        if (use_gpu) {
            std::cerr << "GPU mode was enabled" << std::endl;
            #if defined(CHOLMOD_HAS_CUDA)
            spdlog::error("Printing GPU statistics:");
            cholmod_l_gpu_stats(&cm);
            #endif
        }
        throw std::runtime_error("Numerical factorization failed - matrix not positive definite");
    }
    
    //A bit of debugging flags
    if (L->is_super) {
        if (use_gpu) {
            spdlog::info("CHOLMOD Choose Supernodal computation (GPU accelerated)");
            #if defined(CHOLMOD_HAS_CUDA)
            spdlog::info("Printing GPU statistics:");
            cholmod_l_gpu_stats(&cm);
            #endif
        } else {
            spdlog::info("CHOLMOD Choose Supernodal computation (CPU)");
        }
    } else {
        spdlog::info("CHOLMOD Choose simplicial computation");
    }
}

void CHOLMODSolver::innerSolve(Eigen::VectorXd& rhs, Eigen::VectorXd& result)
{
    if (use_gpu) {
        if (!b) {
            b  = cholmod_l_allocate_dense(N, 1, N, CHOLMOD_REAL, &cm);
            bx = b->x;
        }
        b->x = rhs.data();

        if (x_solve) {
            cholmod_l_free_dense(&x_solve, &cm);
        }

        x_solve = cholmod_l_solve(CHOLMOD_A, L, b, &cm);
    } else {
        if (!b) {
            b  = cholmod_allocate_dense(N, 1, N, CHOLMOD_REAL, &cm);
            bx = b->x;
        }
        b->x = rhs.data();

        if (x_solve) {
            cholmod_free_dense(&x_solve, &cm);
        }

        x_solve = cholmod_solve(CHOLMOD_A, L, b, &cm);
    }
    
    result.conservativeResize(rhs.size());
    memcpy(result.data(), x_solve->x, result.rows() * sizeof(result[0]));
    // save_factor("/home/behrooz/Desktop/Last_Project/RXMesh-dev/output/factor_" + ordering_name + ".mtx");
}


void CHOLMODSolver::resetSolver()
{
    cholmod_clean_memory();

    A       = NULL;
    L       = NULL;
    b       = NULL;
    x_solve = NULL;
    Ai = Ap = Ax = NULL;
    bx           = NULL;
}

void CHOLMODSolver::save_factor(
    const std::string &filePath) {
    cholmod_sparse *spm;
    
    if (use_gpu) {
        spm = cholmod_l_factor_to_sparse(L, &cm);
    } else {
        spm = cholmod_factor_to_sparse(L, &cm);
    }

    FILE *out = fopen(filePath.c_str(), "w");
    assert(out);

    if (use_gpu) {
        cholmod_l_write_sparse(out, spm, NULL, "", &cm);
    } else {
        cholmod_write_sparse(out, spm, NULL, "", &cm);
    }

    fclose(out);
}

LinSysSolverType CHOLMODSolver::type() const
{
    return LinSysSolverType::CPU_CHOLMOD;
};

}  // namespace RXMESH_SOLVER

#endif
