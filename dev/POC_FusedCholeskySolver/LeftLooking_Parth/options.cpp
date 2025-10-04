//
// Created by behrooz on 23/10/22.
//

#include "LeftLooking_Parth.h"

namespace PARTH {
//======================== Options Functions ========================
void ParthSolver::SolverConfig::setReorderingType(
    ParthSolver::ReorderingType type)
{
    this->reorder_type = type;
}
ParthSolver::ReorderingType
ParthSolver::SolverConfig::getReorderingType() const
{
    return reorder_type;
}

void ParthSolver::SolverConfig::setAnalyzeType(ParthSolver::AnalyzeType type)
{
    this->analyze_type = type;
}
ParthSolver::AnalyzeType ParthSolver::SolverConfig::getAnalyzeType() const
{
    return analyze_type;
}

void ParthSolver::SolverConfig::setVerbose(bool verbose)
{
    this->verbose = verbose;
}
bool ParthSolver::SolverConfig::getVerbose() const { return verbose; }

void ParthSolver::SolverConfig::setComputeResidual(bool flag)
{
    this->compute_residual = flag;
}
bool ParthSolver::SolverConfig::getComputeResidual() const
{
    return compute_residual;
}

void ParthSolver::SolverConfig::setNumRegions(int num)
{
    this->num_regions = num;
}
int ParthSolver::SolverConfig::getNumRegions() const
{
    return this->num_regions;
}

///--------------------------------------------------------------------------
/// setNumberOfCores - Set the number of cores for region parallelism
///--------------------------------------------------------------------------
void ParthSolver::SolverConfig::setNumberOfCores(int num_cores)
{
    this->num_cores = num_cores;
}
int ParthSolver::SolverConfig::getNumberOfCores() const
{
    return this->num_cores;
}

///--------------------------------------------------------------------------
/// setNumberOfCores - Set the number of sockets or processors node for MPI
/// parallelism
///--------------------------------------------------------------------------
void ParthSolver::SolverConfig::setNumberOfSockets(int num_sockets)
{
    this->num_sockets = num_sockets;
}
int ParthSolver::SolverConfig::getNumberOfSockets() const
{
    return this->num_sockets;
}

void ParthSolver::SolverConfig::setCSVExportAddress(
    std::string csv_export_address)
{
    this->csv_export_address = csv_export_address;
}

std::string ParthSolver::SolverConfig::getCSVExportAddress()
{
    return this->csv_export_address;
}

ParthSolver::SolverConfig& ParthSolver::Options() { return opt; }

///--------------------------------------------------------------------------
/// setSymbolicReuse - Set the level of reuse in symbolic factorization
///--------------------------------------------------------------------------
void ParthSolver::SolverConfig::setSymbolicReuseType(SymbolicReuseType type)
{
    this->symbolic_reuse_type = type;
}
ParthSolver::SymbolicReuseType
ParthSolver::SolverConfig::getSymbolicReuseType() const
{
    return this->symbolic_reuse_type;
}

///--------------------------------------------------------------------------
/// setNumericReuse - Set the level of reuse in numeric factorization
///--------------------------------------------------------------------------
void ParthSolver::SolverConfig::setNumericReuseType(NumericReuseType type)
{
    this->numeric_reuse_type = type;
}

ParthSolver::NumericReuseType
ParthSolver::SolverConfig::getNumericReuseType() const
{
    return this->numeric_reuse_type;
}

} // namespace PARTH