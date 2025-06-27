#pragma once
#include <stdint.h>
#include <functional>
#include <queue>
#include <set>
#include <vector>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/mgnd_permute.cuh"
#include "rxmesh/matrix/patch_permute.cuh"
#include "rxmesh/matrix/permute_util.h"

#include "metis.h"
// if we should calc and use vertex weight in max match
#define USE_V_WEIGHTS

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

    // stores the weights for each edge. this vector is indexed the same way
    // adjncy is indexed
    std::vector<T> e_weights;

    // Mark the edges by the level of partitioning
    std::vector<T> e_partition;
#ifdef USE_V_WEIGHTS
    // store the vertex weights
    std::vector<T> v_weights;
#endif


    /**
     * @brief return the weight between x and y. If there is no edge between
     * them return the default weight
     */
    T get_weight(T x, T y, T default_val = -1) const
    {
        for (T i = xadj[x]; i < xadj[x + 1]; ++i) {
            if (adjncy[i] == y) {
                return e_weights[i];
            }
        }
        return default_val;
    }

    void print() const
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

/**
 * @brief construct patch neighbor graph which is the same as what we store in
 * PatchStash but in a different format
 */
template <typename T>
void construct_patches_neighbor_graph(
    const RXMeshStatic&     rx,
    Graph<T>&               patches_graph,
    const std::vector<int>& h_patch_graph_edge_weight,
    const int*              d_patch_graph_vertex_weight)
{
    uint32_t num_patches = rx.get_num_patches();

    patches_graph.n = num_patches;
    patches_graph.xadj.resize(num_patches + 1);
    patches_graph.adjncy.reserve(8 * num_patches);

#ifdef USE_V_WEIGHTS
    patches_graph.v_weights.resize(num_patches);
    CUDA_ERROR(cudaMemcpy(patches_graph.v_weights.data(),
                          d_patch_graph_vertex_weight,
                          sizeof(int) * num_patches,
                          cudaMemcpyDeviceToHost));
#endif


    patches_graph.xadj[0] = 0;
    for (uint32_t p = 0; p < num_patches; ++p) {
        const PatchInfo& pi = rx.get_patch(p);

        for (uint8_t i = 0; i < PatchStash::stash_size; ++i) {
            uint32_t n = pi.patch_stash.get_patch(i);
            if (n != INVALID32 && n != p) {
                patches_graph.adjncy.push_back(n);
                patches_graph.e_weights.push_back(
                    h_patch_graph_edge_weight[PatchStash::stash_size * p + i]);
                patches_graph.e_partition.push_back(INVALID32);
            }
        }
        patches_graph.xadj[p + 1] = patches_graph.adjncy.size();
    }
}


template <typename integer_t>
class Node
{
   public:
    Node() : lch(-1), rch(-1), pa(integer_t(-1))
    {
    }

    Node(integer_t left_child, integer_t right_child)
        : lch(left_child), rch(right_child), pa(integer_t(-1))
    {
    }
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

    // we cache the node in this level that branches off to certain patch
    // i.e., the node in this level where a patch is projected
    std::vector<int> patch_proj;
};

template <typename integer_t>
struct MaxMatchTree
{
    // The tree contain many levels. Level 0 is the finest level and nodes in
    // this level have children corresponds to the vertices in the graph.
    // The root of this tree is the last level at levels.size() -1
    std::vector<Level<integer_t>> levels;

    void print() const
    {
        for (int l = levels.size() - 1; l >= 0; --l) {
            const auto& level = levels[l];
            for (int n = 0; n < level.nodes.size(); ++n) {
                const auto& node = level.nodes[n];
                if (l == 0) {
                    std::cout << "L" << l << "_" << n << " -> " << node.lch
                              << ";\n";

                    std::cout << "L" << l << "_" << n << " -> " << node.rch
                              << ";\n";
                } else {
                    std::cout << "L" << l << "_" << n << " -> "
                              << "L" << l - 1 << "_" << node.lch << ";\n";

                    std::cout << "L" << l << "_" << n << " -> "
                              << "L" << l - 1 << "_" << node.rch << ";\n";
                }
            }
        }
    }

    template <typename FuncT>
    void print(FuncT label) const
    {
        for (int l = levels.size() - 1; l >= 0; --l) {
            const auto& level = levels[l];
            for (int n = 0; n < level.nodes.size(); ++n) {
                const auto& node = level.nodes[n];
                if (l == 0) {
                    std::cout << "L" << label(l, n) << " -> "
                              << "N" << label(l - 1, node.lch) << ";\n";

                    std::cout << "L" << label(l, n) << " -> "
                              << "N" << label(l - 1, node.rch) << ";\n";
                } else {
                    std::cout << "L" << label(l, n) << " -> "
                              << "L" << label(l - 1, node.lch) << ";\n";

                    std::cout << "L" << label(l, n) << " -> "
                              << "L" << label(l - 1, node.rch) << ";\n";
                }
            }
        }
    }
};

template <typename integer_t>
void metis_bipartition(Graph<integer_t>& graph, std::vector<int32_t>& part)
{
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
    options[METIS_OPTION_OBJTYPE] =
        METIS_OBJTYPE_VOL;  // Total communication volume minimization.
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_CONTIG]    = 0;
    options[METIS_OPTION_COMPRESS]  = 0;
    options[METIS_OPTION_DBGLVL]    = 0;

    idx_t   nvtxs  = graph.n;
    idx_t   ncon   = 1;
    idx_t*  vwgt   = NULL;
    idx_t*  vsize  = NULL;
    idx_t   nparts = 2;
    real_t* tpwgts = NULL;
    real_t* ubvec  = NULL;
    idx_t   objval = 0;

    part.resize(graph.n, 0);

    int metis_status = METIS_PartGraphKway(&nvtxs,
                                           &ncon,
                                           graph.xadj.data(),
                                           graph.adjncy.data(),
                                           graph.v_weights.data(),
                                           vsize,
                                           graph.e_weights.data(),
                                           &nparts,
                                           tpwgts,
                                           ubvec,
                                           options,
                                           &objval,
                                           part.data());

    if (metis_status == METIS_ERROR_INPUT) {
        RXMESH_ERROR("METIS ERROR INPUT");
        exit(EXIT_FAILURE);
    } else if (metis_status == METIS_ERROR_MEMORY) {
        RXMESH_ERROR("\n METIS ERROR MEMORY \n");
        exit(EXIT_FAILURE);
    } else if (metis_status == METIS_ERROR) {
        RXMESH_ERROR("\n METIS ERROR\n");
        exit(EXIT_FAILURE);
    }

    std::vector<int> part_size(nparts, 0);
    for (int i = 0; i < part.size(); ++i) {
        part_size[part[i]]++;
    }

    // RXMESH_INFO(" Metis parts size: ");
    // for (int i = 0; i < part_size.size(); ++i) {
    //     RXMESH_INFO("   Parts {}= {}", i, part_size[i]);
    // }
}


template <typename integer_t>
void hierarchical_patch_graph_partitioning_recurse(
    const int                level_id,
    const int                parent_id,
    Graph<integer_t>&        graph,
    std::vector<int>         patch_node_mapping,
    std::vector<int32_t>&    part,
    MaxMatchTree<integer_t>& max_match_tree)
{
    assert(graph.n > 0);

    if (graph.n <= 2) {
        if (max_match_tree.levels.size() <= level_id) {
            Level<integer_t> l;
            max_match_tree.levels.push_back(l);
        }
        if (graph.n == 1) {
            int ch = std::max(patch_node_mapping[0], patch_node_mapping[1]);
            max_match_tree.levels[level_id].nodes.push_back(
                Node(parent_id, ch, ch));
        } else {
            max_match_tree.levels[level_id].nodes.push_back(
                Node(parent_id, patch_node_mapping[0], patch_node_mapping[1]));
        }
        return;
    }

    //*** Partition
    metis_bipartition(graph, part);

    //*** Create subgraphs
    // map nodes in the new graph to their index in the input graph
    std::vector<integer_t> nodes_left, nodes_right;
    nodes_left.reserve(graph.n);
    nodes_right.reserve(graph.n);
    for (integer_t i = 0; i < graph.n; ++i) {
        if (part[i] == 0) {
            nodes_left.push_back(i);
        } else {
            nodes_right.push_back(i);
        }
    }

    std::unordered_map<integer_t, integer_t> old_to_new;

    auto build_subgraph = [&](std::vector<integer_t>& nodes_map,
                              Graph<integer_t>&       subgraph) {
        // TODO fix the weights

        old_to_new.clear();
        for (size_t i = 0; i < nodes_map.size(); ++i) {
            old_to_new[nodes_map[i]] = static_cast<integer_t>(i);
        }

        subgraph.n = static_cast<integer_t>(nodes_map.size());
        subgraph.xadj.push_back(0);

        for (auto u : nodes_map) {
            integer_t deg = 0;
            for (integer_t j = graph.xadj[u]; j < graph.xadj[u + 1]; ++j) {
                integer_t v = graph.adjncy[j];
                if (old_to_new.count(v)) {
                    subgraph.adjncy.push_back(old_to_new[v]);
                    subgraph.e_weights.push_back(graph.e_weights[j]);
                    ++deg;
                }
            }
            subgraph.xadj.push_back(subgraph.xadj.back() + deg);
            subgraph.v_weights.push_back(graph.v_weights[u]);
        }
    };

    Graph<integer_t> left_subgraph, right_subgraph;

    build_subgraph(nodes_left, left_subgraph);
    build_subgraph(nodes_right, right_subgraph);

    //*** Update max match tree levels
    if (max_match_tree.levels.size() <= level_id) {
        Level<integer_t> l;
        max_match_tree.levels.push_back(l);
    }

    int left_child_id, right_child_id;
    if (max_match_tree.levels.size() <= level_id + 1) {
        // we have not created this level yet,
        left_child_id = 0;
    } else {
        // we are appending new nodes to this next level
        left_child_id = max_match_tree.levels[level_id + 1].nodes.size();
    }
    right_child_id = left_child_id + 1;

    max_match_tree.levels[level_id].nodes.push_back(
        Node(parent_id, left_child_id, right_child_id));

    //*** Recurse subgraphs
    int id = max_match_tree.levels[level_id].nodes.size() - 1;

    std::vector<int> leaf_patch_node_mapping(
        std::max(left_subgraph.n, right_subgraph.n));

    auto gen_patch_node_map = [&](std::vector<integer_t>& nodes_map,
                                  Graph<integer_t>&       subgraph) {
        std::fill(
            leaf_patch_node_mapping.begin(), leaf_patch_node_mapping.end(), -1);

        for (int i = 0; i < subgraph.n; ++i) {
            leaf_patch_node_mapping[i] = patch_node_mapping[nodes_map[i]];
        }
    };

    gen_patch_node_map(nodes_left, left_subgraph);
    hierarchical_patch_graph_partitioning_recurse(level_id + 1,
                                                  id,
                                                  left_subgraph,
                                                  leaf_patch_node_mapping,
                                                  part,
                                                  max_match_tree);

    gen_patch_node_map(nodes_right, right_subgraph);
    hierarchical_patch_graph_partitioning_recurse(level_id + 1,
                                                  id,
                                                  right_subgraph,
                                                  leaf_patch_node_mapping,
                                                  part,
                                                  max_match_tree);
}

template <typename integer_t>
void pad_shallow_leaves(MaxMatchTree<integer_t>& tree)
{
    const int tree_depth = tree.levels.size();

    auto is_leaf = [&](const Node<integer_t>& node, int level) {
        if (level == 0) {
            return true;
        }
        const auto& child_level = tree.levels[level - 1].nodes;
        return node.lch >= static_cast<integer_t>(child_level.size()) &&
               node.rch >= static_cast<integer_t>(child_level.size());
    };

    for (int level = 1; level < tree_depth; ++level) {

        auto& nodes = tree.levels[level].nodes;

        for (integer_t node_id = 0; node_id < nodes.size(); ++node_id) {
            Node<integer_t>& node = nodes[node_id];

            if (!is_leaf(node, level)) {
                continue;
            }

            Node<integer_t> dummy_r(node_id, node.rch, node.rch);
            Node<integer_t> dummy_l(node_id, node.lch, node.lch);

            integer_t dummy_r_id = tree.levels[level - 1].nodes.size();
            integer_t dummy_l_id = dummy_r_id + 1;

            tree.levels[level - 1].nodes.push_back(dummy_r);
            tree.levels[level - 1].nodes.push_back(dummy_l);

            node.rch = dummy_r_id;
            node.lch = dummy_l_id;


            // tree.levels[level - 1].nodes[node.rch].rch = dummy_r_id;
            // tree.levels[level - 1].nodes[node.rch].lch = dummy_r_id;
            //
            // tree.levels[level - 1].nodes[node.lch].rch = dummy_l_id;
            // tree.levels[level - 1].nodes[node.lch].lch = dummy_l_id;

            ///
            // integer_t curr_node_id = node_id;
            //
            // int curr_level = level;
            //
            // while (curr_level >= 0) {
            //     --curr_level;
            //
            //     auto& new_child_level = tree.levels[curr_level].nodes;
            //
            //     integer_t new_node_id =
            //         static_cast<integer_t>(new_child_level.size());
            //
            //     Node<integer_t> dummy;
            //     dummy.lch = curr_node_id;
            //     dummy.rch = curr_node_id;
            //     dummy.pa  = node_id;
            //
            //     new_child_level.push_back(dummy);
            //
            //     tree.levels[curr_level - 1].nodes[curr_node_id].pa =
            //         new_node_id;
            //
            //     curr_node_id = new_node_id;
            // }
            // break;
        }
        break;
    }
}


template <typename integer_t>
void hierarchical_patch_graph_partitioning(
    RXMeshStatic&            rx,
    Graph<integer_t>&        graph,
    MaxMatchTree<integer_t>& max_match_tree)
{
    std::vector<int32_t> part(graph.n);
    std::vector<int>     patch_node_mapping(graph.n);

    fill_with_sequential_numbers(patch_node_mapping.data(),
                                 patch_node_mapping.size());

    hierarchical_patch_graph_partitioning_recurse(
        0, -1, graph, patch_node_mapping, part, max_match_tree);

    // since we are building the tree top down, we have to reverse the order of
    // the levels
    std::reverse(max_match_tree.levels.begin(), max_match_tree.levels.end());

    // add a fix up so that all leaves nodes are at the same level
    pad_shallow_leaves(max_match_tree);

    max_match_tree.print();

    for (auto& l : max_match_tree.levels) {
        l.patch_proj.resize(rx.get_num_patches());
    }
}


template <typename integer_t>
void heavy_max_matching(const RXMeshStatic&      rx,
                        const Graph<integer_t>&  graph,
                        MaxMatchTree<integer_t>& max_match_tree,
                        std::vector<int>&        sizes)
{
    // Kuhn's algorithm

#ifdef USE_V_WEIGHTS
#ifndef NDEBUG
    int v_weight_sum = 0;
    for (size_t i = 0; i < graph.n; ++i) {
        v_weight_sum += graph.v_weights[i];
    }
    if (v_weight_sum != rx.get_num_vertices()) {
        RXMESH_ERROR(
            "Unexpected behavior in heavy_max_matching as the sum of the patch "
            "graph's vertex weight ({}) does not match the number of vertices "
            "in the mesh({}).",
            v_weight_sum,
            rx.get_num_vertices());
    }

#endif
#endif

    // TODO workaround isolated island (comes from disconnected component input
    // mesh)
    if (graph.n <= 1) {
        return;
    }

    // graph.print();

    std::vector<bool> matched(graph.n, false);

    Level<integer_t> l;
    l.nodes.reserve(DIVIDE_UP(graph.n, 2));
    l.patch_proj.resize(rx.get_num_patches());

    // we store the parent of each node which we use to create the coarse graph
    // i.e., next level graph
    std::vector<integer_t> parents;
    parents.resize(graph.n, integer_t(-1));

    std::vector<integer_t> rands(graph.n);
    fill_with_random_numbers(rands.data(), rands.size());
    // fill_with_sequential_numbers(rands.data(), rands.size());


    // for (int k = 0; k < graph.n; ++k) {
    //     int v = rands[k];
    //     if (!matched[v]) {
    //
    //         // the neighbor with which we will match v
    //         integer_t matched_neighbour = integer_t(-1);
    //         integer_t max_w             = 0;
    //
    //         for (int i = graph.xadj[v]; i < graph.xadj[v + 1]; ++i) {
    //             integer_t n = graph.adjncy[i];
    //             integer_t w = graph.e_weights[i];
    //
    //             if (!matched[n]) {
    //                 if (w > max_w) {
    //                     matched_neighbour = n;
    //                     max_w             = w;
    //                 }
    //             }
    //         }
    //         if (matched_neighbour != integer_t(-1)) {
    //             int node_id = l.nodes.size();
    //
    //             l.nodes.push_back(Node(v, matched_neighbour));
    //             matched[v]                 = true;
    //             matched[matched_neighbour] = true;
    //
    //             parents[v]                 = node_id;
    //             parents[matched_neighbour] = node_id;
    //
    //             // update the parent of the previous level
    //             if (!max_match_tree.levels.empty()) {
    //                 max_match_tree.levels.back().nodes[v].pa = node_id;
    //                 max_match_tree.levels.back().nodes[matched_neighbour].pa
    //                 =
    //                     node_id;
    //             }
    //         }
    //     }
    // }


    for (int k = 0; k < graph.n; ++k) {
        int v = rands[k];
        if (!matched[v]) {

            // the neighbor with which we will match v
            integer_t matched_neighbour = integer_t(-1);

            double max_score = -1.0;


            for (int i = graph.xadj[v]; i < graph.xadj[v + 1]; ++i) {
                integer_t n = graph.adjncy[i];
                integer_t w = graph.e_weights[i];

                if (!matched[n]) {

                    // Balanced score
                    int    s1    = sizes[v];
                    int    s2    = sizes[n];
                    double score = double(w) / (1.0 + std::abs(s1 - s2));

                    if (score > max_score) {
                        matched_neighbour = n;
                        max_score         = score;
                    }
                }
            }
            if (matched_neighbour != integer_t(-1)) {
                int node_id = l.nodes.size();

                l.nodes.push_back(Node(v, matched_neighbour));
                matched[v]                 = true;
                matched[matched_neighbour] = true;

                parents[v]                 = node_id;
                parents[matched_neighbour] = node_id;

                // update the parent of the previous level
                if (!max_match_tree.levels.empty()) {
                    max_match_tree.levels.back().nodes[v].pa = node_id;
                    max_match_tree.levels.back().nodes[matched_neighbour].pa =
                        node_id;
                }

                // New node size is the sum of its children
                sizes[node_id] = sizes[v] + sizes[matched_neighbour];
            }
        }
    }

    // create a node for unmatched vertices
    for (int v = 0; v < graph.n; ++v) {
        if (!matched[v]) {
            int node_id = l.nodes.size();
            l.nodes.push_back(Node(v, v));
            parents[v] = node_id;

            // update the parent of the previous level
            if (!max_match_tree.levels.empty()) {
                max_match_tree.levels.back().nodes[v].pa = node_id;
            }

            sizes[node_id] = sizes[v];  // preserve size
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
    c_graph.e_weights.reserve(c_graph.n * 3);
#ifdef USE_V_WEIGHTS
    c_graph.v_weights.resize(c_graph.n);
#endif


    for (size_t i = 0; i < l.nodes.size(); ++i) {
        const auto& node = l.nodes[i];
        // the neighbors to this node is the union of neighbors of node.lch and
        // node.rch. We don't store node.lcu/node.rch, but instead we store
        // their parent (because this is the coarse graph)
        using NeighboutT = std::pair<integer_t, integer_t>;

        struct NeighboutTLess
        {
            constexpr bool operator()(const NeighboutT& lhs,
                                      const NeighboutT& rhs) const
            {
                return lhs.first < rhs.first;
            }
        };


        std::set<NeighboutT, NeighboutTLess> u_neighbour;

        auto get_neighbour = [&](integer_t x, integer_t y) {
            for (integer_t i = graph.xadj[x]; i < graph.xadj[x + 1]; ++i) {
                integer_t n = graph.adjncy[i];
                if (n != y) {
                    assert(parents[n] != integer_t(-1));

                    // the new edge weight is
                    // 1) the weight of the edge between n and x
                    // 2) plus the weight of the edge between n and y (only if n
                    // is connect to y)
                    integer_t new_weight = graph.e_weights[i];

                    // check if the n is connect to y as well
                    // if so, the weight
                    integer_t wy = graph.get_weight(y, n);

                    if (wy != -1) {
                        new_weight += wy;
                    }

                    u_neighbour.insert({
                        parents[n],
                        new_weight,
                    });
                }
            }
        };

        get_neighbour(node.lch, node.rch);
        get_neighbour(node.rch, node.lch);

#ifdef USE_V_WEIGHTS
        c_graph.v_weights[i] = graph.v_weights[node.lch];
        if (node.lch != node.rch) {
            c_graph.v_weights[i] += graph.v_weights[node.rch];
        }
#endif


        c_graph.xadj[i + 1] = c_graph.xadj[i];
        for (const auto& u : u_neighbour) {
            // actually what we insert in the coarse graph is the parents
            c_graph.adjncy.push_back(u.first);
            c_graph.e_weights.push_back(u.second);
            c_graph.xadj[i + 1]++;
        }
    }

    // Now that we have this level finished, we can insert in the tree
    max_match_tree.levels.push_back(l);


    std::vector<int> coarse_sizes(l.nodes.size(), 0);
    for (int i = 0; i < l.nodes.size(); ++i) {
        auto& node      = l.nodes[i];
        coarse_sizes[i] = sizes[node.lch];
        if (node.lch != node.rch)
            coarse_sizes[i] += sizes[node.rch];
    }


    // recurse to the next level
    heavy_max_matching(rx, c_graph, max_match_tree, coarse_sizes);
}

template <typename integer_t>
void recurse_compute_projection(int                      root,
                                int                      root_level,
                                int                      parent,
                                int                      parent_l,
                                MaxMatchTree<integer_t>& max_match_tree)
{
    int lch = max_match_tree.levels[parent_l].nodes[parent].lch;
    int rch = max_match_tree.levels[parent_l].nodes[parent].rch;
    if (parent_l == 0) {
        max_match_tree.levels[root_level].patch_proj[lch] = root;
        max_match_tree.levels[root_level].patch_proj[rch] = root;
    } else {
        recurse_compute_projection(
            root, root_level, lch, parent_l - 1, max_match_tree);
        recurse_compute_projection(
            root, root_level, rch, parent_l - 1, max_match_tree);
    }
}


template <typename integer_t>
void compute_projection(MaxMatchTree<integer_t>& max_match_tree)
{
    int num_levels = max_match_tree.levels.size();
    for (int l = 0; l < num_levels; ++l) {
        auto& level = max_match_tree.levels[l];
        for (int n = 0; n < level.nodes.size(); ++n) {
            recurse_compute_projection(n, l, n, l, max_match_tree);
        }
    }
}

namespace detail {

template <uint32_t blockThreads>
__global__ static void compute_patch_graph_edge_weight(const Context context,
                                                       int* d_edge_weight,
                                                       int* d_vertex_weight)
{
    __shared__ int s_owned_vertex_sum;
    if (threadIdx.x == 0) {
        s_owned_vertex_sum = 0;
    }

    auto weights = [&](EdgeHandle e_id, VertexIterator& ev) {
        VertexHandle v0          = ev[0];
        uint32_t     v0_patch_id = v0.patch_id();

        VertexHandle v1          = ev[1];
        uint32_t     v1_patch_id = v1.patch_id();

        // find the boundary edges
        if (v0_patch_id != v1_patch_id) {
            PatchStash& v0_patch_stash =
                context.m_patches_info[v0_patch_id].patch_stash;

            PatchStash& v1_patch_stash =
                context.m_patches_info[v1_patch_id].patch_stash;

            // update edge weight for both patches
            uint8_t v0_stash_idx = v0_patch_stash.find_patch_index(v1_patch_id);
            ::atomicAdd(&d_edge_weight[PatchStash::stash_size * v0_patch_id +
                                       v0_stash_idx],
                        1);


            uint8_t v1_stash_idx = v1_patch_stash.find_patch_index(v0_patch_id);
            ::atomicAdd(&d_edge_weight[PatchStash::stash_size * v1_patch_id +
                                       v1_stash_idx],
                        1);
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, weights);

#ifdef USE_V_WEIGHTS
    // sum the owned vertices in the patch in shared memory and then write
    // it to global memory
    for_each_vertex(query.get_patch_info(), [&](const VertexHandle& vh) {
        ::atomicAdd(&s_owned_vertex_sum, int(1));
    });

    block.sync();
    if (threadIdx.x == 0) {
        d_vertex_weight[query.get_patch_id()] = s_owned_vertex_sum;
    }
#endif
}

__inline__ __device__ bool is_v_on_grand_separator(const VertexHandle v_id,
                                                   uint32_t           v_gp,
                                                   const int* d_patch_proj,
                                                   const VertexIterator& iter)
{
    // on a certain level of the max match, return the sibling to a certain
    // mesh vertex on the tree (only if this sibling is different than the
    // vertex's patch grand parent)

    for (uint16_t i = 0; i < iter.size(); ++i) {
        uint32_t n_pid = iter[i].patch_id();

        int n_gp = (d_patch_proj == nullptr) ? n_pid : d_patch_proj[n_pid];

        if (v_gp != n_gp) {
            return true;
        }
    }
    return false;
}

template <uint32_t blockThreads>
__global__ static void extract_separators(const Context        context,
                                          const int*           d_patch_proj_l,
                                          const int*           d_patch_proj_l1,
                                          VertexAttribute<int> v_index,
                                          // VertexAttribute<int> v_render,
                                          int*       d_permute,
                                          int        current_level,
                                          int        depth,
                                          const int* d_dfs_index,
                                          int*       d_count,
                                          int*       d_cut_size)
{
    // d_patch_proj_l is the patch projection on this level
    // d_patch_proj_l1 is the patch projection on the next level (i.e.,
    // current_level -1)

    const int S = depth - current_level - 1;

    const int shift = (1 << S) - 1;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    Query<blockThreads> query(context);

    const uint16_t num_v = query.get_patch_info().num_vertices[0];

    int* s_v_local_id = shrd_alloc.alloc<int>(num_v);

    fill_n<blockThreads>(s_v_local_id, num_v, int(-1));

    __shared__ int s_patch_total;
    if (threadIdx.x == 0) {
        s_patch_total = 0;
    }


    auto extract = [&](VertexHandle v_id, VertexIterator& iter) {
        // this is important to check if v is on the separator before going
        // in and check if it is on the current/grant separator because we
        // have consistent criterion for if a vertex is on a separator
        // (using less than for the vertex patch id)

        int v_proj = d_patch_proj_l[v_id.patch_id()];

        int v_proj_1 = (d_patch_proj_l1 == nullptr) ?
                           v_id.patch_id() :
                           d_patch_proj_l1[v_id.patch_id()];

        int index = shift + v_proj;


        // make sure that the vertex is not counted towards a separator from
        //  previous levels in the tree
        if (v_index(v_id) < 0) {

            // if the vertex is on the separator of the current level
            if (is_v_on_separator(v_id, iter) &&
                is_v_on_grand_separator(
                    v_id, v_proj_1, d_patch_proj_l1, iter)) {

                s_v_local_id[v_id.local_id()] =
                    ::atomicAdd(&s_patch_total, int(1));

                // ::atomicAdd(d_cut_size, int(1));

                // d_permute[context.linear_id(v_id)] =
                //     ::atomicAdd(&d_count[d_dfs_index[index]], int(1));

                assert(v_index(v_id) < 0);
                v_index(v_id) = d_dfs_index[index];

                // v_render(v_id) = current_level;

            } else if (current_level == 0) {
                // get the patch index within the max match tree

                const int SS = depth - (current_level - 1) - 1;

                const int sh = (1 << SS) - 1;

                int index = sh + v_id.patch_id();

                d_permute[context.linear_id(v_id)] =
                    ::atomicAdd(&d_count[d_dfs_index[index]], int(1));

                assert(v_index(v_id) < 0);
                v_index(v_id) = d_dfs_index[index];

                // v_render(v_id) = 10;
            }
        }
    };


    query.dispatch<Op::VV>(block, shrd_alloc, extract);
    block.sync();

    if (s_patch_total > 0) {

        if (threadIdx.x == 0) {

            int p_proj = d_patch_proj_l[query.get_patch_id()];

            int index = shift + p_proj;

            int sum = ::atomicAdd(&d_count[d_dfs_index[index]], s_patch_total);

            s_patch_total = sum;
        }
        block.sync();

        for_each_vertex(query.get_patch_info(), [&](const VertexHandle v_id) {
            int local_id = s_v_local_id[v_id.local_id()];
            if (local_id != -1) {
                d_permute[context.linear_id(v_id)] = s_patch_total + local_id;
            }
        });
    }
}


}  // namespace detail

inline void create_dfs_indexing(const int                level,
                                const int                node,
                                int&                     current_id,
                                const MaxMatchTree<int>& max_match_tree,
                                std::vector<bool>&       visited,
                                std::vector<int>&        dfs_index)
{
    const int S     = max_match_tree.levels.size() - level - 1;
    const int shift = (1 << S) - 1;
    const int id    = shift + node;

    visited[id]   = true;
    dfs_index[id] = current_id++;

    if (level >= 0) {

        int lch = max_match_tree.levels[level].nodes[node].lch;
        int rch = max_match_tree.levels[level].nodes[node].rch;

        int next_level = level - 1;

        int ss = max_match_tree.levels.size() - next_level - 1;
        int sh = (1 << ss) - 1;

        int lch_id = sh + lch;
        int rch_id = sh + rch;

        if (!visited[lch_id]) {
            create_dfs_indexing(next_level,
                                lch,
                                current_id,
                                max_match_tree,
                                visited,
                                dfs_index);
        }

        if (!visited[rch_id]) {
            create_dfs_indexing(next_level,
                                rch,
                                current_id,
                                max_match_tree,
                                visited,
                                dfs_index);
        }
    }
}

inline void single_patch_nd_permute(RXMeshStatic&              rx,
                                    VertexAttribute<uint16_t>& v_local_permute)
{
    CPUTimer timer;
    GPUTimer gtimer;

    timer.start();
    gtimer.start();

    constexpr uint32_t blockThreads = 256;

    // auto attr_v  = rx.add_vertex_attribute<int>("attr_v", 1);
    // auto attr_v1 = rx.add_vertex_attribute<int>("attr_v1", 1);
    // auto attr_e  = rx.add_edge_attribute<int>("attr_e", 1);
    //
    // attr_v->reset(-1, DEVICE);
    // attr_v1->reset(-1, DEVICE);
    // attr_e->reset(-1, DEVICE);


    v_local_permute.reset(INVALID16, DEVICE);

    LaunchBox<blockThreads> lb;

#if 0
    const int maxCoarsenLevels = 5;
    rx.prepare_launch_box(
        {Op::V},
        lb,
        (void*)patch_permute_nd<blockThreads, maxCoarsenLevels>,
        false,
        false,
        false,
        [&](uint32_t v, uint32_t e, uint32_t f) {
            return
                // active_v_mis, v_mis, candidate_v_mis
                5 * detail::mask_num_bytes(v) +

                // 4*EV for vv_cur and vv_nxt
                (3 * 2 * e + std::max(v + 1, 2 * e)) * sizeof(uint16_t) +
                // matching array
                v * maxCoarsenLevels * sizeof(uint16_t) +

                // padding
                7 * ShmemAllocator::default_alignment;
        });
#else
    rx.prepare_launch_box({Op::V},
                          lb,
                          (void*)patch_permute_kmeans<blockThreads>,
                          false,
                          false,
                          false,
                          [&](uint32_t v, uint32_t e, uint32_t f) {
                              return
                                  // active_v_mis, v_mis, candidate_v_mis
                                  7 * detail::mask_num_bytes(v) +

                                  // EV for vv_cur
                                  (2 * e + std::max(v + 1, 2 * e)) *
                                      sizeof(uint16_t) +

                                  // index
                                  v * sizeof(uint16_t) +

                                  // memory used in v_v and v_e
                                  (2 * v + 1) * sizeof(uint16_t) +

                                  // padding
                                  11 * ShmemAllocator::default_alignment;
                          });
#endif

    RXMESH_TRACE("single_patch_nd_permute shared memory= {} (bytes)",
                 lb.smem_bytes_dyn);

#if 0
    patch_permute_nd<blockThreads, maxCoarsenLevels>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), v_local_permute, *attr_v, *attr_e, *attr_v1);
#else
    patch_permute_kmeans<blockThreads>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(rx.get_context(),
                                                           v_local_permute);
#endif

    // CUDA_ERROR(cudaDeviceSynchronize());

#if USE_POLYSCOPE
    // attr_v->move(DEVICE, HOST);
    //  attr_v1->move(DEVICE, HOST);
    //  attr_e->move(DEVICE, HOST);

    v_local_permute.move(DEVICE, HOST);
    //  for (int p = 0; p < rx.get_num_patches(); ++p) {
    //      rx.render_patch(p)->setEnabled(false);
    //  }

    // auto ps_mesh = rx.get_polyscope_mesh();

    // ps_mesh->addVertexScalarQuantity("attr_v", *attr_v);
    // ps_mesh->addVertexScalarQuantity("attr_v1", *attr_v1);
    // ps_mesh->addEdgeScalarQuantity("attr_e", *attr_e);
    // ps_mesh->addVertexScalarQuantity("in_patch_prem", v_local_permute);

    // render
    // polyscope::show();
#endif

    timer.stop();
    gtimer.stop();

    RXMESH_INFO("single_patch_nd_permute took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());
}

inline void permute_separators(RXMeshStatic&              rx,
                               VertexAttribute<int>&      v_index,
                               VertexAttribute<uint16_t>& v_local_permute,
                               MaxMatchTree<int>&         max_match_tree,
                               int*                       d_permute,
                               int*                       d_patch_proj_l,
                               int*                       d_patch_proj_l1)
{
    v_index.reset(-1, DEVICE);

    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> lbe;
    rx.prepare_launch_box(
        {Op::VV},
        lbe,
        (void*)detail::extract_separators<blockThreads>,
        false,
        false,
        false,
        [&](uint32_t v, uint32_t e, uint32_t f) { return v * sizeof(int); });


    // the total number of nodes in the tree is upper-bounded by 2^d where d
    // is the depth of the tree.
    const int      depth     = max_match_tree.levels.size();
    const uint32_t num_nodes = 1 << depth;

    // Every node in the max match tree will contains three pieces of
    // information:
    // 1) the size of its separator
    // 2) the number of nodes on the right
    // 3) the number of nodes on the left
    // the count here refers to the number of mesh vertices on separator,
    // left, or right. We identify the left and right nodes of a (parent)
    // node using less (<) between the left and right node ID.


    // count the number of mesh vertices at different parts in the max match
    // tree some vertices are on the different separators along the tree.
    // The remaining vertices are the one inside the interior of the patch
    // that are number randomly.
    int count_size = 1 << (depth + 1);

    int *d_dfs_index(nullptr), *d_count(nullptr);
    CUDA_ERROR(cudaMalloc((void**)&d_dfs_index, sizeof(int) * count_size));

    int* d_cut_size(nullptr);
    // CUDA_ERROR(cudaMalloc((void**)&d_cut_size, sizeof(int)));
    // CUDA_ERROR(cudaMemset(d_cut_size, 0, sizeof(int)));


    CUDA_ERROR(cudaMalloc((void**)&d_count, sizeof(int) * (count_size + 1)));
    CUDA_ERROR(cudaMemset(d_count, 0, sizeof(int) * count_size));


    int               current_id = 0;
    std::vector<bool> visited(count_size, false);
    std::vector<int>  dfs_index(count_size, -1);
    create_dfs_indexing(
        depth - 1, 0, current_id, max_match_tree, visited, dfs_index);

    CUDA_ERROR(cudaMemcpy(d_dfs_index,
                          dfs_index.data(),
                          count_size * sizeof(int),
                          cudaMemcpyHostToDevice));

    // auto v_render = *rx.add_vertex_attribute<int>("Render", 1);
    // v_render.reset(-1, LOCATION_ALL);

#ifdef USE_POLYSCOPE
    // for (int l = depth - 1; l >= 0; --l) {
    //     rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
    //         int proj     =
    //         max_match_tree.levels[l].patch_proj[vh.patch_id()]; v_render(vh)
    //         = proj;
    //     });
    //     rx.get_polyscope_mesh()->addVertexScalarQuantity(
    //         "Match " + std::to_string(l), v_render);
    // }
#endif

    int sum_edge_cut = 0;

    CUDA_ERROR(cudaMemcpy(d_patch_proj_l,
                          max_match_tree.levels.back().patch_proj.data(),
                          sizeof(int) * rx.get_num_patches(),
                          cudaMemcpyHostToDevice));

    for (int l = depth - 1; l >= 0; --l) {

        if (l != 0) {
            CUDA_ERROR(
                cudaMemcpy(d_patch_proj_l1,
                           max_match_tree.levels[l - 1].patch_proj.data(),
                           sizeof(int) * rx.get_num_patches(),
                           cudaMemcpyHostToDevice));
        }


        detail::extract_separators<blockThreads>
            <<<lbe.blocks, lbe.num_threads, lbe.smem_bytes_dyn>>>(
                rx.get_context(),
                d_patch_proj_l,
                (l == 0) ? nullptr : d_patch_proj_l1,
                v_index,
                // v_render,
                d_permute,
                l,
                depth,
                d_dfs_index,
                d_count,
                d_cut_size);
        // int h_cut_size = 0;
        // CUDA_ERROR(cudaMemcpy(
        //     &h_cut_size, d_cut_size, sizeof(int), cudaMemcpyDeviceToHost));
        // CUDA_ERROR(cudaMemset(d_cut_size, 0, sizeof(int)));
        // sum_edge_cut += h_cut_size;
        // RXMESH_INFO("Level = {},Cut Size = {}", l, h_cut_size);
        //
        // v_render.move(DEVICE, HOST);
        // rx.get_polyscope_mesh()->addVertexScalarQuantity(
        //     "Render " + std::to_string(l), v_render);

        // if (l >= depth - 3) {
        //     v_render.move(DEVICE, HOST);
        //     rx.get_polyscope_mesh()->addVertexScalarQuantity(
        //         "Render " + std::to_string(l), v_render);
        //
        //     // v_index.move(DEVICE, HOST);
        //     // rx.get_polyscope_mesh()->addVertexScalarQuantity(
        //     //     "Index " + std::to_string(l), v_index);
        // }

        std::swap(d_patch_proj_l, d_patch_proj_l1);
    }

    // RXMESH_INFO("Sum Edge Cut Size = {}", sum_edge_cut);

    thrust::exclusive_scan(
        thrust::device, d_count, d_count + count_size, d_count);

    // std::vector<int> h_count(count_size);
    // CUDA_ERROR(cudaMemcpy(h_count.data(),
    //                       d_count,
    //                       sizeof(int) * count_size,
    //                       cudaMemcpyDeviceToHost));
    //
    // max_match_tree.print([&](int l, int n) {
    //     const int S = depth - l - 1;
    //
    //     const int shift = (1 << S) - 1;
    //
    //     int index = shift + n;
    //
    //     return std::to_string(dfs_index[index]) + "_" +
    //            std::to_string(h_count[dfs_index[index]]);
    // });

    auto context = rx.get_context();
    int  num_v   = rx.get_num_vertices();
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
        int l = d_count[v_index(vh)];


        if (v_local_permute(vh) != INVALID16) {
            // if it is interior and not on any separator

            l += v_local_permute(vh);
            // l += d_permute[context.linear_id(vh)];

        } else {
            // if it is a separator
            l += d_permute[context.linear_id(vh)];
        }

        d_permute[context.linear_id(vh)] = num_v - l - 1;
    });

    GPU_FREE(d_dfs_index);
    GPU_FREE(d_count);
}

inline void nd_permute(RXMeshStatic& rx, int* h_permute)
{

    auto v_index = *rx.add_vertex_attribute<int>("index", 1);

    auto v_local_permute = *rx.add_vertex_attribute<uint16_t>("local_index", 1);

    // for a level L in the max_match_tree, d_patch_proj stores the node at
    // level L that branch off to a given patch, i.e., the projection of the
    // the patch on to level L in the tree
    int *d_patch_proj_l(nullptr), *d_patch_proj_l1(nullptr);
    CUDA_ERROR(cudaMalloc((void**)&d_patch_proj_l,
                          sizeof(int) * rx.get_num_patches()));
    CUDA_ERROR(cudaMalloc((void**)&d_patch_proj_l1,
                          sizeof(int) * rx.get_num_patches()));


    // stores the edge weight of the patch graph
    int*     d_patch_graph_edge_weight = nullptr;
    uint32_t edge_weight_size  = PatchStash::stash_size * rx.get_num_patches();
    uint32_t edge_weight_bytes = sizeof(int) * edge_weight_size;
    CUDA_ERROR(
        cudaMalloc((void**)&d_patch_graph_edge_weight, edge_weight_bytes));
    CUDA_ERROR(cudaMemset(d_patch_graph_edge_weight, 0, edge_weight_bytes));

    // stores the vertex weight of the patch graph
    int* d_patch_graph_vertex_weight = nullptr;
#ifdef USE_V_WEIGHTS
    uint32_t vertex_weight_size  = rx.get_num_patches();
    uint32_t vertex_weight_bytes = sizeof(int) * vertex_weight_size;
    CUDA_ERROR(
        cudaMalloc((void**)&d_patch_graph_vertex_weight, vertex_weight_bytes));
    CUDA_ERROR(cudaMemset(d_patch_graph_vertex_weight, 0, vertex_weight_bytes));
#endif


    std::vector<int> h_patch_graph_edge_weight(edge_weight_size, 0);

    // the new index
    int* d_permute = nullptr;
    CUDA_ERROR(
        cudaMalloc((void**)&d_permute, rx.get_num_vertices() * sizeof(int)));

    CPUTimer timer;
    GPUTimer gtimer;

    timer.start();
    gtimer.start();

    // compute edge weight
    rx.run_kernel<512>({Op::EV},
                       detail::compute_patch_graph_edge_weight<512>,
                       d_patch_graph_edge_weight,
                       d_patch_graph_vertex_weight);

    CUDA_ERROR(cudaMemcpy(h_patch_graph_edge_weight.data(),
                          d_patch_graph_edge_weight,
                          edge_weight_bytes,
                          cudaMemcpyDeviceToHost));

    // a graph representing the patch connectivity
    Graph<int> p_graph;
    construct_patches_neighbor_graph(
        rx, p_graph, h_patch_graph_edge_weight, d_patch_graph_vertex_weight);


    // create max match tree
    MaxMatchTree<int> max_match_tree;


    // std::vector<int> sizes(p_graph.n, 0);
    // for (int p = 0; p < rx.get_num_patches(); ++p) {
    //     sizes[p] = rx.get_num_vertices(p);
    // }
    // heavy_max_matching(rx, p_graph, max_match_tree, sizes);

    hierarchical_patch_graph_partitioning(rx, p_graph, max_match_tree);


    {
        // works only with meshes from create_plane with number of patches
        // that is 2^n (i.e., -n 64 or -n 128)

        // int num_levels = int(std::log2(double(rx.get_num_patches())));
        //
        // for (int l = 0; l < num_levels; ++l) {
        //     Level<int> level;
        //
        //     int num_nodes = std::pow(2.0, double(num_levels - l - 1));
        //
        //     level.nodes.resize(num_nodes);
        //     level.patch_proj.resize(rx.get_num_patches());
        //
        //     for (int n = 0; n < num_nodes; ++n) {
        //         level.nodes[n] = Node(n / 2, n * 2, n * 2 + 1);
        //     }
        //
        //     max_match_tree.levels.push_back(level);
        // }


        // Level<int> l0;
        // l0.nodes.resize(8);
        // l0.patch_proj.resize(rx.get_num_patches());
        //
        // l0.nodes[0] = Node(0, 0, 1);
        // l0.nodes[1] = Node(1, 2, 3);
        // l0.nodes[2] = Node(0, 4, 5);
        // l0.nodes[3] = Node(1, 6, 7);
        // l0.nodes[4] = Node(2, 8, 9);
        // l0.nodes[5] = Node(3, 10, 11);
        // l0.nodes[6] = Node(2, 12, 13);
        // l0.nodes[7] = Node(3, 14, 15);
        //
        // max_match_tree.levels.push_back(l0);
        //
        //
        // Level<int> l1;
        // l1.nodes.resize(4);
        // l1.patch_proj.resize(rx.get_num_patches());
        //
        // l1.nodes[0] = Node(0, 0, 2);
        // l1.nodes[1] = Node(0, 1, 3);
        // l1.nodes[2] = Node(1, 4, 6);
        // l1.nodes[3] = Node(1, 5, 7);
        //
        // max_match_tree.levels.push_back(l1);
        //
        //
        // Level<int> l2;
        // l2.nodes.resize(2);
        // l2.patch_proj.resize(rx.get_num_patches());
        //
        // l2.nodes[0] = Node(0, 0, 1);
        // l2.nodes[1] = Node(0, 2, 3);
        //
        // max_match_tree.levels.push_back(l2);
        //
        //
        // Level<int> l3;
        // l3.nodes.resize(1);
        // l3.patch_proj.resize(rx.get_num_patches());
        // l3.nodes[0] = Node(-1, 0, 1);
        //
        // max_match_tree.levels.push_back(l3);
    }

    max_match_tree.print();

    compute_projection(max_match_tree);

    single_patch_nd_permute(rx, v_local_permute);

    permute_separators(rx,
                       v_index,
                       v_local_permute,
                       max_match_tree,
                       d_permute,
                       d_patch_proj_l,
                       d_patch_proj_l1);

    timer.stop();
    gtimer.stop();

    RXMESH_INFO("nd_permute took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());

    CUDA_ERROR(cudaMemcpy(h_permute,
                          d_permute,
                          rx.get_num_vertices() * sizeof(int),
                          cudaMemcpyDeviceToHost));


    std::vector<int> helper(rx.get_num_vertices());
    inverse_permutation(rx.get_num_vertices(), h_permute, helper.data());

    GPU_FREE(d_permute);
    GPU_FREE(d_patch_proj_l);
    GPU_FREE(d_patch_proj_l1);
    GPU_FREE(d_patch_graph_edge_weight);
    GPU_FREE(d_patch_graph_vertex_weight);
}


}  // namespace rxmesh