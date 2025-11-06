//
// Debug test for MinVertexCoverBipartite using Hopcroft-Karp algorithm
// Test cases from Rosetta Code
//

#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include "spdlog/spdlog.h"
#include "min_vertex_cover_bipartite.h"

struct Edge {
    int from;
    int to;
};

// Helper function to convert Rosetta Code test format to our 0-indexed format
// and build CSR matrix using Eigen
bool test_hopcroft_karp(int test_num, int m, int n, const std::vector<Edge>& edges, int expected_result)
{
    // Total nodes: m nodes in U partition (0 to m-1), n nodes in V partition (m to m+n-1)
    int total_nodes = m + n;
    
    // Create partition vector: 0 for U nodes, 1 for V nodes
    std::vector<int> node_to_partition(total_nodes);
    for (int i = 0; i < m; i++) {
        node_to_partition[i] = 0;  // U partition
    }
    for (int i = m; i < total_nodes; i++) {
        node_to_partition[i] = 1;  // V partition
    }
    
    // Build adjacency matrix using Eigen triplets
    std::vector<Eigen::Triplet<double>> triplets;
    
    for (const auto& edge : edges) {
        // Convert from 1-indexed separate partitions to 0-indexed unified nodes
        // U partition: 1-indexed u becomes 0-indexed (u-1)
        // V partition: 1-indexed v becomes 0-indexed (m-1+v)
        int u_node = edge.from - 1;           // U node in range [0, m-1]
        int v_node = edge.to - 1;         // V node in range [m, m+n-1]
        
        // Add edge (u -> v) to adjacency matrix
        triplets.push_back(Eigen::Triplet<double>(u_node, v_node, 1.0));
        triplets.push_back(Eigen::Triplet<double>(v_node, u_node, 1.0));
    }
    
    // Build sparse matrix in row-major format (CSR)
    Eigen::SparseMatrix<double, Eigen::RowMajor> adj_matrix(total_nodes, total_nodes);
    adj_matrix.setFromTriplets(triplets.begin(), triplets.end());
    adj_matrix.makeCompressed();
    
    // Get CSR format pointers
    int* M_p = const_cast<int*>(adj_matrix.outerIndexPtr());
    int* M_i = const_cast<int*>(adj_matrix.innerIndexPtr());
    
    // Create the MinVertexCoverBipartite object
    RXMESH_SOLVER::MinVertexCoverBipartite solver(total_nodes, M_p, M_i, node_to_partition);
    
    // Compute maximum matching using Hopcroft-Karp
    int result = solver.compute_max_matching_hopcroft_karp();
    
    // Check result
    std::string condition;
    if (result == expected_result) {
        condition = " ✓ PASS";
    } else {
        condition = " ✗ FAIL";
    }
    spdlog::info( "Test {}: Result = {}, Expected = {}" + condition, test_num, result, expected_result);
    return result == expected_result;
}

// Helper function to test minimum vertex cover computation
bool test_min_vertex_cover_helper(int test_num, int m, int n, const std::vector<Edge>& edges, int expected_cover_size)
{
    // Total nodes: m nodes in U partition (0 to m-1), n nodes in V partition (m to m+n-1)
    int total_nodes = m + n;
    
    // Create partition vector: 0 for U nodes, 1 for V nodes
    std::vector<int> node_to_partition(total_nodes);
    for (int i = 0; i < m; i++) {
        node_to_partition[i] = 0;  // U partition
    }
    for (int i = m; i < total_nodes; i++) {
        node_to_partition[i] = 1;  // V partition
    }
    
    // Build adjacency matrix using Eigen triplets
    std::vector<Eigen::Triplet<double>> triplets;
    
    for (const auto& edge : edges) {
        // Convert from 1-indexed separate partitions to 0-indexed unified nodes
        int u_node = edge.from - 1;           // U node in range [0, m-1]
        int v_node = edge.to - 1;         // V node in range [m, m+n-1]
        
        // Add edge (u -> v) to adjacency matrix
        triplets.push_back(Eigen::Triplet<double>(u_node, v_node, 1.0));
        triplets.push_back(Eigen::Triplet<double>(v_node, u_node, 1.0));
    }
    
    // Build sparse matrix in row-major format (CSR)
    Eigen::SparseMatrix<double, Eigen::RowMajor> adj_matrix(total_nodes, total_nodes);
    adj_matrix.setFromTriplets(triplets.begin(), triplets.end());
    adj_matrix.makeCompressed();
    
    // Get CSR format pointers
    int* M_p = const_cast<int*>(adj_matrix.outerIndexPtr());
    int* M_i = const_cast<int*>(adj_matrix.innerIndexPtr());
    
    // Create the MinVertexCoverBipartite object
    RXMESH_SOLVER::MinVertexCoverBipartite solver(total_nodes, M_p, M_i, node_to_partition);
    
    // Compute minimum vertex cover
    std::vector<int> vertex_cover = solver.compute_min_vertex_cover();
    
    // Print results
    std::cout << "Test " << test_num << ":" << std::endl;
    std::cout << "  Vertex Cover Size: " << vertex_cover.size() << std::endl;
    std::cout << "  Separator Nodes: { ";
    for (size_t i = 0; i < vertex_cover.size(); i++) {
        std::cout << vertex_cover[i];
        if (i < vertex_cover.size() - 1) std::cout << ", ";
    }
    std::cout << " }" << std::endl;
    
    // Check result
    bool passed = (vertex_cover.size() == expected_cover_size);
    std::string condition = passed ? " ✓ PASS" : " ✗ FAIL";
    spdlog::info("Test {}: Cover Size = {}, Expected = {}{}", test_num, vertex_cover.size(), expected_cover_size, condition);
    
    return passed;
}

bool max_matching_test()
{
    std::cout << "Running Hopcroft-Karp Algorithm Tests:" << std::endl;
    std::cout << "========================================" << std::endl;

    int passed = 0;
    int total = 5;

    // Test Case 1: Simple single edge
    // m=3, n=5, edge(1,4)
    // U: {1,2,3}, V: {1,2,3,4,5}
    // Edge: 1->4
    std::vector<Edge> edges1 = {
        {1, 3},
        {1, 4},
        {2, 3}
    };
    if (test_hopcroft_karp(1, 2, 2, edges1, 2)) passed++;

    // Test Case 2: Three edges
    // m=6, n=6
    // Edges: 1->4, 1->5, 5->1
    std::vector<Edge> edges2 = {
        {1, 5},
        {1, 8},
        {2, 5},
        {2, 6},
        {3, 6},
        {3, 7},
        {4, 6},
        {4, 8}
    };
    if (test_hopcroft_karp(2, 4, 4, edges2, 4)) passed++;

    // Test Case 3: Complete Bipartite Graph K(3,3)
    // m=3, n=3
    // All edges from U to V
    std::vector<Edge> edges3;
    for (int i = 1; i <= 3; i++) {
        for (int j = 4; j <= 6; j++) {
            edges3.push_back({i, j});
        }
    }
    if (test_hopcroft_karp(3, 3, 3, edges3, 3)) passed++;

    // Test Case 4: No edges
    // m=2, n=2
    std::vector<Edge> edges4;  // Empty
    if (test_hopcroft_karp(4, 2, 2, edges4, 0)) passed++;

    // Test Case 5: Complex graph
    // m=4, n=4
    // Edges: 1->1, 1->3, 2->3, 3->4, 4->3, 4->2
    std::vector<Edge> edges5 = {
        {1, 5},
        {1, 7},
        {2, 7},
        {3, 8},
        {4, 7},
        {4, 6}
    };
    if (test_hopcroft_karp(5, 4, 4, edges5, 4)) passed++;

    // Summary
    std::cout << "========================================" << std::endl;
    std::cout << "Tests passed: " << passed << "/" << total << std::endl;

    if (passed == total) {
        std::cout << "All tests passed! ✓" << std::endl;
        return true;
    } else {
        std::cout << "Some tests failed. ✗" << std::endl;
        return false;
    }
}

bool test_min_vertex_cover()
{
    std::cout << std::endl;
    std::cout << "Running Minimum Vertex Cover Tests:" << std::endl;
    std::cout << "====================================" << std::endl;

    int passed = 0;
    int total = 6;

    // Test Case 1: 2x2 bipartite with 3 edges
    // By König's theorem, min vertex cover size = max matching size = 2
    std::vector<Edge> edges1 = {
        {1, 3},
        {1, 4},
        {2, 3}
    };
    if (test_min_vertex_cover_helper(1, 2, 2, edges1, 2)) passed++;

    // Test Case 2: 4x4 bipartite with 8 edges
    // Expected min vertex cover size = max matching size = 4
    std::vector<Edge> edges2 = {
        {1, 5},
        {1, 8},
        {2, 5},
        {2, 6},
        {3, 6},
        {3, 7},
        {4, 6},
        {4, 8}
    };
    if (test_min_vertex_cover_helper(2, 4, 4, edges2, 4)) passed++;

    // Test Case 3: Complete Bipartite Graph K(3,3)
    // Expected min vertex cover size = max matching size = 3
    std::vector<Edge> edges3;
    for (int i = 1; i <= 3; i++) {
        for (int j = 4; j <= 6; j++) {
            edges3.push_back({i, j});
        }
    }
    if (test_min_vertex_cover_helper(3, 3, 3, edges3, 3)) passed++;

    // Test Case 4: No edges
    // Expected min vertex cover size = 0
    std::vector<Edge> edges4;  // Empty
    if (test_min_vertex_cover_helper(4, 2, 2, edges4, 0)) passed++;

    // Test Case 5: Complex 4x4 graph
    // Expected min vertex cover size = max matching size = 4
    std::vector<Edge> edges5 = {
        {1, 5},
        {1, 7},
        {2, 7},
        {3, 8},
        {4, 7},
        {4, 6}
    };
    if (test_min_vertex_cover_helper(5, 4, 4, edges5, 4)) passed++;

    // Test Case 6: Complex 4x4 graph
    // Expected min vertex cover size = max matching size = 4
    std::vector<Edge> edges6 = {
        {1,6},
        {2,6},
        {2,7},
        {3,7},
        {4,6},
        {4,8},
        {4,9},
        {5,8},
        {5,9}
    };
    if (test_min_vertex_cover_helper(6, 5, 4, edges6, 4)) passed++;

    // Summary
    std::cout << "====================================" << std::endl;
    std::cout << "Tests passed: " << passed << "/" << total << std::endl;

    if (passed == total) {
        std::cout << "All tests passed! ✓" << std::endl;
        return true;
    } else {
        std::cout << "Some tests failed. ✗" << std::endl;
        return false;
    }
}

int main() {
    // if (!max_matching_test()) {
    //     spdlog::error("Max matching tests did not pass.");
    // } else {
    //     spdlog::info("Max matching tests passed.");
    // }

    // Min vertex cover test
    if (!test_min_vertex_cover()) {
        spdlog::error("Min vertex cover tests did not pass.");
        return 1;
    } else {
        spdlog::info("Min vertex cover tests passed.");
    }

    return 0;
}

