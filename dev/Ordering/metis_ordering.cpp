//
// Created by behrooz on 2025-09-29.
//

#include "metis_ordering.h"


#include <cassert>
#include <iostream>
#include <metis.h>
#include "ordering.h"
#include "csv_utils.h"

namespace RXMESH_SOLVER {

MetisOrdering::~MetisOrdering()
{
}

void MetisOrdering::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    this->Gp = Gp;
    this->Gi = Gi;
    this->G_N = G_N;
    this->G_NNZ = NNZ;
}

void MetisOrdering::compute_permutation(std::vector<int>& perm)
{
    idx_t N = G_N;
    idx_t NNZ = Gp[G_N];
    perm.resize(G_N);
    if (NNZ == 0) {
        assert(G_N != 0);
        for (int i = 0; i < G_N; i++) {
#ifndef NDEBUG
            //      std::cout << "WARNING: This decomposition does not have edges"
            //                << std::endl;
#endif
            perm[i] = i;
        }
        return;
    }
    // TODO add memory allocation protection later like CHOLMOD

    std::vector<int> tmp(G_N);
    METIS_NodeND(&N, Gp, Gi, NULL, NULL, perm.data(), tmp.data());
}


RXMESH_Ordering_Type MetisOrdering::type() const
{
    return  RXMESH_Ordering_Type::METIS;
}
std::string MetisOrdering::typeStr() const
{
    return "METIS";
}
void MetisOrdering::add_record(std::string save_address, std::map<std::string, double> extra_info, std::string mesh_name)
{
    std::string csv_name = save_address + "/sep_runtime_analysis";
    std::vector<std::string> header;
    header.emplace_back("mesh_name");
    header.emplace_back("G_N");
    header.emplace_back("G_NNZ");

    header.emplace_back("ordering_type");
    header.emplace_back("local_permute_method");
    header.emplace_back("separator_finding_method");
    header.emplace_back("separator_refinement_method");
    header.emplace_back("separator_ratio");
    header.emplace_back("fill-ratio");

    PARTH::CSVManager runtime_csv(csv_name, "some address", header,
                                  false);
    runtime_csv.addElementToRecord(mesh_name, "mesh_name");
    runtime_csv.addElementToRecord(G_N, "G_N");
    runtime_csv.addElementToRecord(G_NNZ, "G_NNZ");
    runtime_csv.addElementToRecord(typeStr(), "ordering_type");
    runtime_csv.addElementToRecord("", "local_permute_method");
    runtime_csv.addElementToRecord("", "separator_finding_method");
    runtime_csv.addElementToRecord("", "separator_refinement_method");
    runtime_csv.addElementToRecord("", "separator_ratio");
    runtime_csv.addElementToRecord(extra_info.at("fill-ratio"), "fill-ratio");
    runtime_csv.addRecord();
}

}