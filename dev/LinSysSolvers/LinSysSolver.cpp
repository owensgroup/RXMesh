#include "LinSysSolver.hpp"
#include <iostream>

#ifdef USE_SUITESPARSE
#include "CHOLMODSolver.hpp"
#endif

#ifdef USE_CUDSS
#include "CUDSSSolver.hpp"
#endif

#ifdef USE_PARTH
#include "ParthSolver.hpp"
#endif


#ifdef USE_STRUMPACK
#include <STRUMPACKSolver.hpp>
#endif


namespace RXMESH_SOLVER {

    LinSysSolver *LinSysSolver::create(const LinSysSolverType type) {
        switch (type) {


#ifdef USE_SUITESPARSE
            case LinSysSolverType::CPU_CHOLMOD:
                return new CHOLMODSolver();
#endif

#ifdef USE_CUDSS
            case LinSysSolverType::GPU_CUDSS:
                return new CUDSSSolver();
#endif

#ifdef USE_PARTH
            case LinSysSolverType::PARTH_SOLVER:
                return new ParthSolver();
#endif

#ifdef USE_STRUMPACK
            case LinSysSolverType::GPU_STRUMPACK:
                return new STRUMPACKSolver();
#endif
            default:
                std::cerr << "Uknown linear system solver type" << std::endl;
                return nullptr;
        }
    }


} // namespace RXMESH_SOLVER
