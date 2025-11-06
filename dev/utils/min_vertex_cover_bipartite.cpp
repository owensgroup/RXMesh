//
// Created by Behrooz on 2025-09-10.
//

#include "min_vertex_cover_bipartite.h"
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <spdlog/spdlog.h>

#include <boost/mpl/assert.hpp>

namespace RXMESH_SOLVER {
    MinVertexCoverBipartite::MinVertexCoverBipartite(int M_n, int* M_p, int* M_i, std::vector<int>& node_to_partition) {
        this->M_n = M_n;
        this->M_p = M_p;
        this->M_i = M_i;

        
        // Initialize the pairU and pairV
        int n_left = 0;
        int n_right = 0;
        for(int i = 0; i < M_n; i++) {
            if(node_to_partition[i] == 0) {
                u_nodes.push_back(i);
            } else {
                v_nodes.push_back(i);
            }
        }
        // Make the u nodes the smaller set
        if(u_nodes.size() > v_nodes.size()) {
            std::swap(u_nodes, v_nodes);
        }
        // Set NIL to M_n (sentinel value for unmatched nodes)
        NIL = std::max(u_nodes.size(), v_nodes.size());

    }
    MinVertexCoverBipartite::~MinVertexCoverBipartite() {
    }

    bool MinVertexCoverBipartite::bfs() {
        std::queue<int> Q;
        
        // First layer of vertices (set distance as 0)
        for(int u_idx = 0; u_idx < u_nodes.size(); u_idx++) {
            // If this is a free vertex, add it to queue
            if(pair_u[u_idx] == NIL) {
                // u is not matched
                level[u_idx] = 0;
                Q.push(u_idx);
            } else {
                // Else set distance as infinite so that this vertex
                // is considered next time
                level[u_idx] = inf;
            }
        }
        
        // Initialize distance to NIL as infinite
        level[NIL] = inf; // NIL index is at u_nodes.size()
        
        // Q is going to contain u_node indices only
        while(!Q.empty()) {
            // Dequeue a vertex
            int u_idx = Q.front();
            Q.pop();
            
            // If this node is not NIL and can provide a shorter path to NIL
            if(level[u_idx] < level[NIL]) {
                // Get the actual node ID
                int u_node = u_nodes[u_idx];
                
                // Get all adjacent vertices of the dequeued vertex u
                // Using CSR format: M_p[u_node] to M_p[u_node + 1]
                for(int edge_idx = M_p[u_node]; edge_idx < M_p[u_node + 1]; edge_idx++) {
                    int v_node = M_i[edge_idx];
                    int v_idx = v_node_to_index[v_node];
                    assert(v_idx != -1);
                    int paired_u_idx = pair_v[v_idx];
                    if(level[paired_u_idx] == inf) {
                        // Consider the pair and add it to queue
                        level[paired_u_idx] = level[u_idx] + 1;
                        Q.push(paired_u_idx);
                    }
                }
            }
        }
        
        // If we could come back to NIL using alternating path of distinct
        // vertices then there is an augmenting path
        return level[NIL] != inf;
    }

    bool MinVertexCoverBipartite::dfs(int u_idx) {
        if(u_idx != NIL) { // If not NIL
            // Get the actual node ID
            int u_node = u_nodes[u_idx];
            
            // Get all adjacent vertices of u
            // Using CSR format: M_p[u_node] to M_p[u_node + 1]
            for(int edge_idx = M_p[u_node]; edge_idx < M_p[u_node + 1]; edge_idx++) {
                int v_node = M_i[edge_idx];
                
                // Check if v_node is in v_nodes
                int v_idx = v_node_to_index[v_node];
                assert(v_idx != -1);
                    
                // Get the paired u index
                int paired_u_idx = pair_v[v_idx];
                
                // Follow the distances set by BFS
                if(level[paired_u_idx] == level[u_idx] + 1) {
                    // If dfs for pair of v also returns true
                    if(dfs(paired_u_idx)) {
                        pair_v[v_idx] = u_idx; // Store index
                        pair_u[u_idx] = v_idx; // Store index
                        return true;
                    }
                }
            }
            level[u_idx] = inf;
            return false;
        }
        return true;
    }

    int MinVertexCoverBipartite::compute_max_matching_hopcroft_karp() {
        // Build mapping from node_id to index in u_nodes and v_nodes
        v_node_to_index.clear();
        u_node_to_index.clear();
        u_node_to_index.resize(M_n, -1);
        v_node_to_index.resize(M_n, -1);
        for(int i = 0; i < u_nodes.size(); i++) {
            u_node_to_index[u_nodes[i]] = i;
        }
        for(int i = 0; i < v_nodes.size(); i++) {
            v_node_to_index[v_nodes[i]] = i;
        }
        
        pair_u.resize(u_nodes.size() + 1, NIL);
        pair_v.resize(v_nodes.size() + 1, NIL);
        level.resize(NIL + 1, 0);
        int max_matching = 0;
        // Keep updating the result while there is an augmenting path
        while(bfs()) {
            // Find a free vertex
            for(int u_idx = 0; u_idx < u_nodes.size(); u_idx++) {
                // If current vertex is free and there is
                // an augmenting path from current vertex
                if(pair_u[u_idx] == NIL && dfs(u_idx)) {
                    max_matching++;
                }
            }
        }
        return max_matching;
    }

    std::vector<int> MinVertexCoverBipartite::compute_min_vertex_cover() {
        //Initialize the max matching using Hopcroft-Karp
        int max_match_size = compute_max_matching_hopcroft_karp();
        //Initialize the visited vectors
        std::vector<bool> left_visited(u_nodes.size(), false);
        std::vector<bool> right_visited(v_nodes.size(), false);
        //Find all the unpared us
        std::queue<int> Q;
        for (int u_idx = 0; u_idx < u_nodes.size(); u_idx++) {
            if (pair_u[u_idx] == NIL) {
                Q.push(u_idx); // Each initial node is unmatched
                left_visited[u_idx] = true;
            }
        }

        //Apply bfs
        while (!Q.empty()) {
            int u_idx = Q.front();
            Q.pop();
            int u_node = u_nodes[u_idx];
            for (int nbr_ptr = M_p[u_node]; nbr_ptr < M_p[u_node + 1]; nbr_ptr++) {
                int v_node = M_i[nbr_ptr];
                int v_idx = v_node_to_index[v_node];
                assert(v_idx != -1);
                int paired_u_idx = pair_v[v_idx];
                
                // Skip if this is the matched edge from u to v
                if(paired_u_idx == u_idx) continue;
                
                // Skip if v has already been visited
                if (right_visited[v_idx]) continue;
                
                // Mark v as visited (traversing unmatched edge u->v)
                right_visited[v_idx] = true;
                
                // If v is matched, follow the matched edge back to its pair
                if (paired_u_idx != NIL) {
                    if (!left_visited[paired_u_idx]) {
                        left_visited[paired_u_idx] = true;
                        Q.push(paired_u_idx);
                    }
                }
            }
        }


        std::vector<int> min_vertex_cover;
        //Those vertex in left that are not visited are in the min vertex cover
        for(int u_idx = 0; u_idx < u_nodes.size(); u_idx++) {
            if(!left_visited[u_idx]) {
                min_vertex_cover.push_back(u_nodes[u_idx]);
            }
        }
        //Those vertex in right that are visited are in the min vertex cover
        for(int v_idx = 0; v_idx < v_nodes.size(); v_idx++) {
            if(right_visited[v_idx]) {
                min_vertex_cover.push_back(v_nodes[v_idx]);
            }
        }
        assert(min_vertex_cover.size() == max_match_size);
#ifndef NDEBUG
        //Check whether the min_vertex_cover cover all the edges
        std::sort(min_vertex_cover.begin(), min_vertex_cover.end());
        spdlog::info("Checking the correctness of min_vertex_cover");
        for (int i = 0; i < M_n; i++) {
            for (int nbr_ptr = M_p[i]; nbr_ptr < M_p[i + 1]; nbr_ptr++) {
                int nbr = M_i[nbr_ptr];
                //Check whether i or nbr exist in min_vertex_cover
                if (std::find(min_vertex_cover.begin(), min_vertex_cover.end(), i) == min_vertex_cover.end() &&
                    std::find(min_vertex_cover.begin(), min_vertex_cover.end(), nbr) == min_vertex_cover.end()) {
                    spdlog::error("The min_vertex_cover does not cover all the edges");
                }

            }
        }
        //Make sure there is no repetative node
        std::vector<bool> visited(M_n, false);
        for (int i = 0; i < min_vertex_cover.size(); i++) {
            assert(visited[min_vertex_cover[i]] == false);
            visited[min_vertex_cover[i]] = true;
        }
#endif
        return min_vertex_cover;
    }


}
