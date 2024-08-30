#pragma once

#include <unordered_set>
#include <vector>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/permute_util.h"
#include "rxmesh/matrix/separator_tree.h"

#include "metis.h"

namespace rxmesh {

template <typename T>
struct Graph
{
    // number of vertices in the graph
    T n;

    // xadj is of length n+1 marking the start of the adjancy list of each
    // vertex in adjncy.
    std::vector<T> xadj;

    // adjncy stores the adjacency lists of the vertices. The adjnacy list of a
    // vertex should not contain the vertex itself.
    std::vector<T> adjncy;

    void print()
    {
        std::cout << "\n*** graph ***\n";
        for (int i = 0; i < n; ++i) {
            // std::cout << "Row " << i << "\n ";
            for (int j = xadj[i]; j < xadj[i + 1]; ++j) {
                if (i <= adjncy[j]) {
                    std::cout << i << " -> " << adjncy[j] << ";\n";
                }
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        std::cout << "\n*** graph ***\n";
        for (int i = 0; i < n; ++i) {
            // std::cout << "Row " << i << "\n ";
            for (int j = xadj[i]; j < xadj[i + 1]; ++j) {
                std::cout << i << "," << adjncy[j] << ";";
            }
            // std::cout << "\n";
        }
        std::cout << "\n";
    }
};

template <typename T>
void construct_a_simple_chordal_graph(Graph<T>& graph)
{
    graph.n                            = 20;
    std::vector<std::pair<T, T>> edges = {
        {0, 1},   {1, 2},   {2, 3},   {3, 4},   {4, 5},   {5, 6},
        {6, 7},   {7, 8},   {8, 9},   {0, 2},   {1, 3},   {3, 5},
        {4, 6},   {5, 7},   {6, 8},   {7, 9},   {9, 10},  {10, 11},
        {11, 12}, {12, 13}, {13, 14}, {14, 15}, {15, 16}, {16, 17},
        {17, 18}, {18, 19}, {8, 10},  {9, 11},  {10, 12}, {11, 13},
        {12, 14}, {13, 15}, {14, 16}, {15, 17}, {16, 18}, {17, 19}};

    // graph.n = 40;
    // std::vector<std::pair<int, int>> edges = {
    //     {0, 1},   {1, 2},   {2, 3},   {3, 4},   {4, 5},   {5, 6},   {6, 7},
    //     {7, 8},   {8, 9},   {9, 10},  {10, 11}, {11, 12}, {12, 13}, {13, 14},
    //     {14, 15}, {15, 16}, {16, 17}, {17, 18}, {18, 19}, {19, 20}, {20, 21},
    //     {21, 22}, {22, 23}, {23, 24}, {24, 25}, {25, 26}, {26, 27}, {27, 28},
    //     {28, 29}, {29, 30}, {30, 31}, {31, 32}, {32, 33}, {33, 34}, {34, 35},
    //     {35, 36}, {36, 37}, {37, 38}, {38, 39}, {0, 2},   {1, 3},   {3, 5},
    //     {4, 6},   {5, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {10, 12},
    //     {11, 13}, {12, 14}, {13, 15}, {14, 16}, {15, 17}, {16, 18}, {17, 19},
    //     {18, 20}, {19, 21}, {20, 22}, {21, 23}, {23, 25}, {24, 26}, {25, 27},
    //     {26, 28}, {27, 29}, {28, 30}, {29, 31}, {30, 32}, {31, 33}, {32, 34},
    //     {33, 35}, {34, 36}, {35, 37}, {36, 38}, {37, 39}};


    graph.xadj.resize(graph.n + 1, 0);


    for (const auto& edge : edges) {
        graph.xadj[edge.first]++;
        graph.xadj[edge.second]++;
    }

    int before = 0;
    for (int i = 0; i <= graph.n; ++i) {
        int c         = graph.xadj[i];
        graph.xadj[i] = before;
        before += c;
    }

    graph.adjncy.resize(graph.xadj.back());


    std::vector<T> current_pos(graph.xadj.size(), 0);
    for (auto& edge : edges) {
        int row = edge.first;
        int idx = current_pos[row] + graph.xadj[row];
        current_pos[row]++;
        graph.adjncy[idx] = edge.second;

        std::swap(edge.first, edge.second);

        row = edge.first;
        idx = current_pos[row] + graph.xadj[row];
        current_pos[row]++;
        graph.adjncy[idx] = edge.second;
    }
}

/**
 * @brief construct patch neighbor graph which is the same as what we store in
 * PatchStash but in format acceptable by metis
 */
void construct_patches_neighbor_graph(RXMeshStatic& rx,
                                      Graph<idx_t>& patches_graph)
{
    uint32_t num_patches = rx.get_num_patches();

    patches_graph.n = num_patches;
    patches_graph.xadj.resize(num_patches + 1);
    patches_graph.adjncy.reserve(8 * num_patches);

    patches_graph.xadj[0] = 0;
    for (uint32_t p = 0; p < num_patches; ++p) {
        const PatchInfo& pi = rx.get_patch(p);

        for (uint8_t i = 0; i < PatchStash::stash_size; ++i) {
            uint32_t n = pi.patch_stash.get_patch(i);
            if (n != INVALID32 && n != p) {
                patches_graph.adjncy.push_back(n);
            }
        }
        patches_graph.xadj[p + 1] = patches_graph.adjncy.size();
    }
}

/**
 * @brief run metis on the patch neighbor graph
 */
void run_metis(Graph<idx_t>&       patches_graph,
               std::vector<idx_t>& h_permute,
               std::vector<idx_t>& h_iperm)
{
    // Metis options
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    // options[METIS_OPTION_DBGLVL]    = 1;

    int ret = METIS_NodeND(&patches_graph.n,
                           patches_graph.xadj.data(),
                           patches_graph.adjncy.data(),
                           NULL,
                           options,
                           h_permute.data(),
                           h_iperm.data());
    if (ret != 1) {
        RXMESH_ERROR("run_metis() METIS failed.");
    }
}

template <typename integer_t>
void print_graph_vertices_at_each_level(
    const SeparatorTree<integer_t>& tree,
    const std::vector<integer_t>&   inv_permute,
    integer_t                       node,
    integer_t                       level)
{
    if (node == -1)
        return;

    // Extract the range of vertices in the permuted graph
    integer_t start = tree.sizes[node];
    integer_t end   = tree.sizes[node + 1];

    std::cout << "Level " << level << ": Separator node " << node
              << " separates original vertices: ";

    // Map the permuted indices back to the original graph vertices
    for (integer_t i = start; i < end; i++) {
        std::cout << inv_permute[i] << " ";
    }
    std::cout << std::endl;

    // Recursively traverse the left and right children
    if (tree.lch[node] != -1) {
        print_graph_vertices_at_each_level(
            tree, inv_permute, tree.lch[node], level + 1);
    }
    if (tree.rch[node] != -1) {
        print_graph_vertices_at_each_level(
            tree, inv_permute, tree.rch[node], level + 1);
    }
}

void nd_permute_metis_sep_tree(RXMeshStatic& rx, std::vector<int>& h_permute)
{
    // a graph representing the patch connectivity
    Graph<idx_t> p_graph;

    // construct_patches_neighbor_graph(rx, p_graph);
    construct_a_simple_chordal_graph(p_graph);


    p_graph.print();

    std::vector<idx_t> p_graph_permute(p_graph.n);
    std::vector<idx_t> p_graph_inv_permute(p_graph.n);

    // fill_with_sequential_numbers(p_graph_permute.data(),
    //                              p_graph_permute.size());
    //
    // fill_with_sequential_numbers(p_graph_inv_permute.data(),
    //                              p_graph_inv_permute.size());
    run_metis(p_graph, p_graph_permute, p_graph_inv_permute);

    if (!is_unique_permutation(p_graph_permute.size(),
                               p_graph_permute.data())) {
        RXMESH_ERROR("nd_permute() METIS failed to give unique permutation");
    }

    SeparatorTree<idx_t> sep_tree =
        build_sep_tree_from_perm(p_graph.xadj.data(),
                                 p_graph.adjncy.data(),
                                 p_graph_permute,
                                 p_graph_inv_permute);

    sep_tree.print();
    sep_tree.printm("sep_tree");
    sep_tree.check();

    print_graph_vertices_at_each_level(
        sep_tree, p_graph_inv_permute, sep_tree.root(), idx_t(0));


    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();

    polyscope::show();
}


template <typename integer_t>
class Node
{
   public:
    Node(integer_t parent, integer_t left_child, integer_t right_child)
        : pa(parent), lch(left_child), rch(right_child)
    {
    }
    integer_t pa, lch, rch;
};

template <typename integer_t>
struct Level
{
    // a  list nodes that defines a level
    std::vector<Node<integer_t>> nodes;
};

template <typename integer_t>
struct MaxMatchTree
{
    // The tree contain many levels. Level 0 is the finest level and nodes in
    // this level have children corresponds to the vertices in the graph.
    // The root of this tree is the last level at levels.size() -1
    std::vector<Level<integer_t>> levels;
};

template <typename integer_t>
void random_max_matching(const Graph<integer_t>&  graph,
                         MaxMatchTree<integer_t>& max_match_tree)
{
    if (graph.n <= 1) {
        return;
    }
    std::vector<bool> matched(graph.n, false);

    Level l;
    l.nodes.reserve(DIVIDE_UP(graph.n, 2));

    // we store the parent of each node which we use to create the coarse graph
    // i.e., next level graph
    std::vector<integer_t> parents;
    parents.resize(graph.n, integer_t(-1));

    // TODO traverse the graph in a random order
    for (int v = 0; v < graph.n; ++v) {
        // the neighbor with which we will match v
        integer_t matched_neighbour = integer_t(-1);

        for (int i = graph.xadj[v]; i < graph.xadj[v + 1]; ++i) {
            integer_t n = graph.adjncy[i];
            if (!matched[n]) {
                matched_neighbour = n;
                break;
            }
        }
        if (matched_neighbour != integer_t(-1)) {
            int pa = l.nodes.size();

            l.nodes.push_back(Node(pa, v, matched_neighbour));
            matched[v]                 = true;
            matched[matched_neighbour] = true;
            parents[v]                 = pa;
            parents[matched_neighbour] = pa;
        }
    }

    // create a node for unmatched vertices
    for (int v = 0; v < graph.n; ++v) {
        if (!matched[v]) {
            int pa = l.nodes.size();
            l.nodes.push_back(Node(pa, v, v));
            parents[v] = pa;
        }
    }


    // create a coarse graph and update the nodes parent
    // since we incrementally (and serially) build the graph, we can build the
    // adjacency list right way (not need to create edges and then create
    // adjacency list)
    Graph<integer_t> c_graph;
    c_graph.n = l.nodes.size();
    c_graph.xadj.resize(c_graph.n + 1, 0);
    c_graph.adjncy.reserve(c_graph.n * 3);  // just a guess


    for (int i = 0; i < l.nodes.size(); ++i) {
        const auto& node = l.nodes[i];
        // the neighbors to this node is the union of neighbors of node.lch and
        // node.rch
        std::unordered_set<integer_t> u_neighbour;

        auto get_neighbour = [&](integer_t x, integer_t y) {
            for (integer_t n = graph.xadj[x]; n < graph.xadj[x + 1]; ++n) {
                if (graph.adjncy[n] != y) {
                    u_neighbour.insert(graph.adjncy[n]);
                }
            }
        };

        get_neighbour(node.lch, node.rch);
        get_neighbour(node.rch, node.lch);

        for (const auto& u : u_neighbour) {
            // actually what we insert in the coarse graph is the parents
            c_graph.adjncy.push_back(parents[u]);
            c_graph.xadj[i + 1] = c_graph.xadj[i] + 1;
        }
    }

    // Now that we have this level finished, we can insert in the tree
    max_match_tree.levels.push_back(l);

    // recurse to the next level
    random_max_matching(c_graph, max_match_tree);
}

void nd_permute(RXMeshStatic& rx, std::vector<int>& h_permute)
{
    MaxMatchTree<int> max_match_tree;

    // a graph representing the patch connectivity
    Graph<int> p_graph;

    // construct_patches_neighbor_graph(rx, p_graph);
    construct_a_simple_chordal_graph(p_graph);

    // create max match tree
    random_max_matching(p_graph, max_match_tree);
}
}  // namespace rxmesh