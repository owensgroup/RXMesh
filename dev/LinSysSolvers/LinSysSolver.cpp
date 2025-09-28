#include "LinSysSolver.hpp"
#include <iostream>

#ifdef RXMESH_WITH_SUITESPARSE

#include "CHOLMODSolver.hpp"

#endif

namespace RXMESH_SOLVER {

    LinSysSolver *LinSysSolver::create(const LinSysSolverType type) {
        switch (type) {


#ifdef RXMESH_WITH_SUITESPARSE
            case LinSysSolverType::CPU_CHOLMOD:
                return new CHOLMODSolver();
#endif

            default:
                std::cerr << "Uknown linear system solver type" << std::endl;
                return nullptr;
        }
    }


} // namespace IPC
