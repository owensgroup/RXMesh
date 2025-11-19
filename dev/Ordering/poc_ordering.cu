//
// Created by behrooz on 2025-09-29.
//

#include "poc_ordering.h"
#include "csv_utils.h"

namespace RXMESH_SOLVER {

POCOrdering::~POCOrdering()
{
}

void POCOrdering::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    this->Gp = Gp;
    this->Gi = Gi;
    this->G_N = G_N;
    this->G_NNZ = NNZ;
    // Also set the graph in the base GPUOrdering class
    gpu_order.setGraph(Gp, Gi, G_N, NNZ);
}

void POCOrdering::setMesh(const double* V_data, int V_rows, int V_cols,
                          const int* F_data, int F_rows, int F_cols)
{
    m_has_mesh = true;
    gpu_order.setMesh(V_data, V_rows, V_cols, F_data, F_rows, F_cols);
    gpu_order.init_patches();
}

bool POCOrdering::needsMesh() const
{
    return true;
}

void POCOrdering::compute_permutation(std::vector<int>& perm)
{
    assert(m_has_mesh);
    gpu_order.compute_permutation(perm);
}


RXMESH_Ordering_Type POCOrdering::type() const
{
    return RXMESH_Ordering_Type::POC_ND;
}

std::string POCOrdering::typeStr() const
{
    return "POC_ND";
}

void POCOrdering::setOptions(const std::map<std::string, std::string>& options)
{
    if (options.find("local_permute_method") != options.end()) {
        this->gpu_order.local_permute_method = options.at("local_permute_method");
    } else {
        this->gpu_order.local_permute_method = "metis";
    }

    if (options.find("use_gpu") != options.end()) {
        this->gpu_order._use_gpu = std::stoi(options.at("use_gpu"));
    } else {
        this->gpu_order._use_gpu = false;
    }

    // if(options.find("separator_finding_method") != options.end()) {
    //     this->gpu_order.separator_finding_method = options.at("separator_finding_method");
    // } else {
    //     this->gpu_order.separator_finding_method = "max_degree";
    // }
    //
    // if (options.find("separator_refinement_method") != options.end()) {
    //     this->gpu_order.separator_refinement_method = options.at("separator_refinement_method");
    // } else {
    //     this->gpu_order.separator_refinement_method = "nothing";
    // }
}


void POCOrdering::add_record(std::string save_address, std::map<std::string, double> extra_info, std::string mesh_name)
{
    std::string csv_name = save_address + "/sep_runtime_analysis";
    std::vector<std::string> header;
    header.emplace_back("mesh_name");
    header.emplace_back("G_N");
    header.emplace_back("G_NNZ");

    header.emplace_back("ordering_type");
    header.emplace_back("local_permute_method");
    // header.emplace_back("separator_finding_method");
    // header.emplace_back("separator_refinement_method");
    header.emplace_back("separator_ratio");
    header.emplace_back("fill-ratio");



    PARTH::CSVManager runtime_csv(csv_name, "some address", header,
                                  false);
    runtime_csv.addElementToRecord(mesh_name, "mesh_name");
    runtime_csv.addElementToRecord(G_N, "G_N");
    runtime_csv.addElementToRecord(G_NNZ, "G_NNZ");
    runtime_csv.addElementToRecord(typeStr(), "ordering_type");
    runtime_csv.addElementToRecord(gpu_order.local_permute_method, "local_permute_method");
    // runtime_csv.addElementToRecord(gpu_order.separator_finding_method, "separator_finding_method");
    // runtime_csv.addElementToRecord(gpu_order.separator_refinement_method, "separator_refinement_method");
    runtime_csv.addElementToRecord(gpu_order._separator_ratio, "separator_ratio");
    runtime_csv.addElementToRecord(extra_info.at("fill-ratio"), "fill-ratio");
    runtime_csv.addRecord();
}

}