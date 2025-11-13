//
// Created by Behrooz on 2025-09-10.
//


#pragma once

#include <vector>
#include <unordered_map>
#include <limits>
//Return the factor's nnz using CHOLMOD analysis
namespace RXMESH_SOLVER {
    class MinVertexCoverBipartite {
    public:
        int M_n;
        int* M_p;
        int* M_i;
        std::vector<int> pair_u;
        std::vector<int> pair_v;
        std::vector<int> u_nodes;
        std::vector<int> v_nodes;
        std::vector<int> level;
        std::vector<int> u_node_to_index; // Map from node ID to index in u_nodes
        std::vector<int> v_node_to_index; // Map from node ID to index in v_nodes
        int inf = std::numeric_limits<int>::max();
        int NIL = -1;
    public:
        MinVertexCoverBipartite(int M_n, int* M_p, int* M_i, std::vector<int>& node_to_partition);
        ~MinVertexCoverBipartite();
        //Compute the max matching using Hopcroft–Karp
        int compute_max_matching_hopcroft_karp();


        bool bfs();
        bool dfs(int u);


        //Compute the min vertex cover using the max matching
        std::vector<int> compute_min_vertex_cover();
    };
}

