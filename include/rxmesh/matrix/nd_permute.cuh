#pragma once

#include <functional>
#include <queue>
#include <set>
#include <vector>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/mgnd_permute.cuh"
#include "rxmesh/matrix/patch_permute.cuh"
#include "rxmesh/matrix/permute_util.h"

// if we should calc and use vertex weight in max match
// #define USE_V_WEIGHTS

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
void heavy_max_matching(const RXMeshStatic&      rx,
                        const Graph<integer_t>&  graph,
                        MaxMatchTree<integer_t>& max_match_tree)
{

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


    for (int k = 0; k < graph.n; ++k) {
        int v = rands[k];
        if (!matched[v]) {

            // the neighbor with which we will match v
            integer_t matched_neighbour = integer_t(-1);
            integer_t max_w             = 0;

            for (int i = graph.xadj[v]; i < graph.xadj[v + 1]; ++i) {
                integer_t n = graph.adjncy[i];
                integer_t w = graph.e_weights[i];

                if (!matched[n]) {
                    if (w > max_w) {
                        matched_neighbour = n;
                        max_w             = w;
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

                // store the grand parent of level -1
                if (max_match_tree.levels.empty()) {
                    l.patch_proj[v]                 = node_id;
                    l.patch_proj[matched_neighbour] = node_id;
                }
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

            // store the parent of level -1
            if (max_match_tree.levels.empty()) {
                l.patch_proj[v] = node_id;
            }
        }
    }

    // update the projection if we are not on level 0
    if (!max_match_tree.levels.empty() && l.nodes.size() > 1) {
        for (uint32_t p = 0; p < l.patch_proj.size(); ++p) {
            l.patch_proj[p] =
                parents[max_match_tree.levels.back().patch_proj[p]];
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

    // recurse to the next level
    heavy_max_matching(rx, c_graph, max_match_tree);
}

template <typename T>
void fm_refinement_boundary(const Graph<T>&   graph,
                            std::vector<int>& partition_id_vec,
                            int               max_passes = 10)
{
    T n = graph.n;

    // Helper function to update partition sizes
    auto update_partition_sizes = [&](int& size0, int& size1) {
        size0 = 0;
        size1 = 0;
        for (T v = 0; v < n; v++) {
            if (partition_id_vec[v] == 0)
                size0++;
            else
                size1++;
        }
    };

    int size0, size1;
    update_partition_sizes(size0, size1);

    // We say a vertex v is a boundary vertex if it has at least one neighbor in
    // the opposite partition
    auto is_boundary_vertex = [&](T v) {
        T   start   = graph.xadj[v];
        T   end     = graph.xadj[v + 1];
        int my_part = partition_id_vec[v];
        for (T idx = start; idx < end; idx++) {
            T nei = graph.adjncy[idx];
            if (partition_id_vec[nei] != my_part) {
                return true;
            }
        }
        return false;
    };

    // Gain array: gain[v] = external_weight - internal_weight
    std::vector<T> gain(n, 0);

    auto compute_gain_for_vertex = [&](T v) {
        T start = graph.xadj[v];
        T end   = graph.xadj[v + 1];

        T internal_cost = 0;
        T external_cost = 0;

        int current_part = partition_id_vec[v];
        for (T idx = start; idx < end; idx++) {
            T nei = graph.adjncy[idx];
            T w   = graph.e_weights[idx];
            if (partition_id_vec[nei] == current_part)
                internal_cost += w;
            else
                external_cost += w;
        }
        gain[v] = external_cost - internal_cost;
    };

    // Compute initial gains
    for (T v = 0; v < n; v++) {
        compute_gain_for_vertex(v);
    }

    // Compute the current cut
    auto compute_cut = [&]() {
        T cut_cost = 0;
        for (T v = 0; v < n; v++) {
            T start = graph.xadj[v];
            T end   = graph.xadj[v + 1];
            for (T idx = start; idx < end; idx++) {
                T nei = graph.adjncy[idx];
                if (nei > v && partition_id_vec[v] != partition_id_vec[nei]) {
                    cut_cost += graph.e_weights[idx];
                }
            }
        }
        return cut_cost;
    };

    T                best_cut       = compute_cut();
    std::vector<int> best_partition = partition_id_vec;

    // Update neighbor gains when a vertex is moved
    auto update_neighbor_gains = [&](T moved_v, int old_part) {
        T start = graph.xadj[moved_v];
        T end   = graph.xadj[moved_v + 1];
        for (T idx = start; idx < end; idx++) {
            T nei = graph.adjncy[idx];
            T w   = graph.e_weights[idx];

            // If neighbor is still in old_part, its gain increases by w.
            // If neighbor is now in the same part as moved_v, its gain
            // decreases by w.
            if (partition_id_vec[nei] == old_part)
                gain[nei] += w;
            else
                gain[nei] -= w;
        }
    };

    // Maximum allowed difference in partition sizes
    int max_diff = static_cast<int>(0.5 * n);

    for (int pass = 0; pass < max_passes; pass++) {
        bool              improvement_found = false;
        std::vector<bool> locked(n, false);

        // Build a priority queue of boundary vertices only
        // (gain, vertex).
        std::priority_queue<std::pair<T, T>> pq;
        for (T v = 0; v < n; v++) {
            if (is_boundary_vertex(v)) {
                pq.push(std::make_pair(gain[v], v));
            }
        }

        T                current_cut       = best_cut;
        std::vector<int> current_partition = partition_id_vec;

        std::vector<std::pair<T, int>> move_sequence;
        move_sequence.reserve(n);

        T   best_partial_cut = best_cut;
        int best_move_index  = -1;

        for (T move_index = 0; move_index < n; move_index++) {
            if (pq.empty()) {
                break;
            }

            // Pick the highest-gain unlocked boundary vertex
            T v      = -1;
            T v_gain = std::numeric_limits<T>::lowest();
            while (!pq.empty()) {
                auto top = pq.top();
                pq.pop();
                T cand_gain   = top.first;
                T cand_vertex = top.second;
                if (!locked[cand_vertex]) {
                    v      = cand_vertex;
                    v_gain = cand_gain;
                    break;
                }
            }

            if (v == -1) {
                break;  // No unlocked boundary vertices left
            }

            locked[v] = true;

            int old_part = partition_id_vec[v];
            int new_part = 1 - old_part;

            // Check partition size difference constraints
            // The difference must not exceed max_diff, and no partition can be
            // empty
            int new_size0 = size0;
            int new_size1 = size1;
            if (old_part == 0) {
                new_size0--;
                new_size1++;
            } else {
                new_size1--;
                new_size0++;
            }

            bool can_move = true;
            if (new_size0 < 1 || new_size1 < 1) {
                // Do not allow any partition to be empty
                can_move = false;
            }
            if (std::abs(new_size0 - new_size1) > max_diff) {
                // Do not allow size difference to exceed 0.2 * n
                can_move = false;
            }

            if (can_move) {
                partition_id_vec[v] = new_part;
                size0               = new_size0;
                size1               = new_size1;

                // Update the cut. If v_gain is positive, cut decreases. If
                // negative, cut increases.
                current_cut -= v_gain;

                // Record the move
                move_sequence.push_back({v, old_part});

                // Update neighbor gains
                update_neighbor_gains(v, old_part);

                // Rebuild the priority queue with the updated gains for
                // boundary vertices
                std::priority_queue<std::pair<T, T>> new_pq;
                for (T x = 0; x < n; x++) {
                    if (!locked[x] && is_boundary_vertex(x)) {
                        new_pq.push(std::make_pair(gain[x], x));
                    }
                }
                pq = std::move(new_pq);

                // Track the best partial state
                if (current_cut < best_partial_cut) {
                    best_partial_cut = current_cut;
                    best_move_index  = move_index;
                }
            } else {
                // Revert the locked state but no partition change
                // This vertex remains locked and in old_part
            }
        }

        // Revert any moves beyond best_move_index
        if (best_move_index >= 0 && best_partial_cut < best_cut) {
            // improvement found
            for (int i = best_move_index + 1; i < (int)move_sequence.size();
                 i++) {
                auto& mv                    = move_sequence[i];
                T     moved_vtx             = mv.first;
                int   original_part         = mv.second;
                partition_id_vec[moved_vtx] = original_part;
            }
            // Recompute partition sizes after reversion
            update_partition_sizes(size0, size1);

            best_cut          = best_partial_cut;
            best_partition    = partition_id_vec;
            improvement_found = true;
        } else {
            // revert everything in this pass
            partition_id_vec = best_partition;
            update_partition_sizes(size0, size1);
        }

        if (!improvement_found) {
            break;
        }

        // Recompute gains for next pass
        for (T v = 0; v < n; v++) {
            compute_gain_for_vertex(v);
        }
    }
}

#ifdef USE_V_WEIGHTS

// This function refines the sub-partition "id" using a
// Fiduccia-Mattheyses–like approach. local_partition_id_vec[v] is 0 or 1
// only for vertices whose global_partition_id_vec[v] == id.
template <typename T>
void fm_refinement_vweight(const Graph<T>&   graph,
                           std::vector<int>& global_partition_id_vec,
                           int               id,
                           std::vector<int>& local_partition_id_vec,
                           int               max_passes = 10)
{
    // We only refine the subset of vertices where global_partition_id_vec[v] ==
    // id. The local_partition_id_vec has the same size as
    // global_partition_id_vec, but it is only meaningful (0 or 1) for the
    // vertices in the sub-partition "id". This code will ignore vertices that
    // are not in "id".

    T n = graph.n;

    // Gather indices of vertices in the global partition "id".
    std::vector<T> subset;
    subset.reserve(n);
    for (T v = 0; v < n; v++) {
        if (global_partition_id_vec[v] == id)
            subset.push_back(v);
    }
    // If subset.size() <= 1, there is nothing to refine.
    if (subset.size() <= 1) {
        return;
    }

    // Compute total vertex weight for the subset
    T total_weight = 0;
    for (T v : subset) {
        total_weight += graph.v_weights[v];
    }

    // Gains are stored for each vertex in the subset.
    // We'll store them in a vector of length 'n', but only the subset entries
    // are used.
    std::vector<T> gain(n, 0);

    // A helper to compute gain for a single vertex, ignoring edges to vertices
    // that are not in the same global partition "id".
    auto compute_gain_for_vertex = [&](T v) {
        T start = graph.xadj[v];
        T end   = graph.xadj[v + 1];

        T   internal_cost = 0;
        T   external_cost = 0;
        int current_part  = local_partition_id_vec[v];  // 0 or 1

        for (T idx = start; idx < end; idx++) {
            T nei = graph.adjncy[idx];
            // Only consider edges within the same global partition 'id'
            if (global_partition_id_vec[nei] != id) {
                continue;
            }

            T w = graph.e_weights[idx];
            if (local_partition_id_vec[nei] == current_part)
                internal_cost += w;
            else
                external_cost += w;
        }
        gain[v] = external_cost - internal_cost;
    };

    // Initialize gains for vertices in the subset
    for (T v : subset) {
        compute_gain_for_vertex(v);
    }

    // Compute cut within this sub-partition only
    auto compute_sub_cut = [&]() {
        T cut_cost = 0;
        for (T v : subset) {
            T start = graph.xadj[v];
            T end   = graph.xadj[v + 1];
            for (T idx = start; idx < end; idx++) {
                T nei = graph.adjncy[idx];
                // Only consider edges inside the sub-partition
                if (nei <= v)
                    continue;  // avoid double counting and parallel edges
                if (global_partition_id_vec[nei] != id)
                    continue;

                if (local_partition_id_vec[v] != local_partition_id_vec[nei]) {
                    cut_cost += graph.e_weights[idx];
                }
            }
        }
        return cut_cost;
    };

    T                best_cut             = compute_sub_cut();
    std::vector<int> best_local_partition = local_partition_id_vec;  // snapshot

    // Partition weight of the sub-partition for the two local parts (0 and 1).
    auto update_partition_weights = [&](T& weight0, T& weight1) {
        weight0 = 0;
        weight1 = 0;
        for (T v : subset) {
            if (local_partition_id_vec[v] == 0)
                weight0 += graph.v_weights[v];
            else
                weight1 += graph.v_weights[v];
        }
    };

    T weight0, weight1;
    update_partition_weights(weight0, weight1);

    // After moving a vertex from old_part to new_part, update neighbor gains
    auto update_neighbor_gains = [&](T moved_v, int old_part) {
        T start = graph.xadj[moved_v];
        T end   = graph.xadj[moved_v + 1];

        for (T idx = start; idx < end; idx++) {
            T nei = graph.adjncy[idx];
            if (global_partition_id_vec[nei] != id)
                continue;

            T w = graph.e_weights[idx];
            // If neighbor is still in old_part, gain[nei] increases by w
            // If neighbor is in the new part, gain[nei] decreases by w
            if (local_partition_id_vec[nei] == old_part)
                gain[nei] += w;
            else
                gain[nei] -= w;
        }
    };

    // Check if we can move a vertex under the chosen balance criteria:
    // The difference in partition weights must be <= 0.1 * total_weight,
    // and neither partition becomes empty in terms of total vertex weight.
    auto can_move_vertex = [&](T v, int old_part, int new_part, T w0, T w1) {
        T v_w    = graph.v_weights[v];
        T new_w0 = w0;
        T new_w1 = w1;
        if (old_part == 0) {
            new_w0 -= v_w;
            new_w1 += v_w;
        } else {
            new_w1 -= v_w;
            new_w0 += v_w;
        }

        T diff = std::abs(new_w0 - new_w1);
        if (diff > (0.1 * total_weight))
            return false;
        // Do not allow an empty partition by weight
        if (new_w0 == 0 || new_w1 == 0)
            return false;

        return true;
    };

    for (int pass = 0; pass < max_passes; pass++) {
        bool              improvement_found = false;
        std::vector<bool> locked(n, false);

        // Build a priority queue of (gain, vertex) for the subset
        std::priority_queue<std::pair<T, T>> pq;
        for (T v : subset) {
            pq.push(std::make_pair(gain[v], v));
        }

        T                current_cut       = best_cut;
        std::vector<int> current_partition = local_partition_id_vec;

        std::vector<std::pair<T, int>> move_sequence;
        move_sequence.reserve(subset.size());

        T   best_partial_cut = best_cut;
        int best_move_index  = -1;

        for (size_t move_index = 0; move_index < subset.size(); move_index++) {
            // pick the highest gain unlocked vertex
            T v      = -1;
            T v_gain = std::numeric_limits<T>::lowest();
            while (!pq.empty()) {
                auto top = pq.top();
                pq.pop();
                if (!locked[top.second]) {
                    v_gain = top.first;
                    v      = top.second;
                    break;
                }
            }
            if (v == -1)
                break;

            int old_part = local_partition_id_vec[v];
            int new_part = 1 - old_part;
            locked[v]    = true;

            if (can_move_vertex(v, old_part, new_part, weight0, weight1)) {
                local_partition_id_vec[v] = new_part;
                if (old_part == 0) {
                    weight0 -= graph.v_weights[v];
                    weight1 += graph.v_weights[v];
                } else {
                    weight1 -= graph.v_weights[v];
                    weight0 += graph.v_weights[v];
                }

                current_cut -= v_gain;
                move_sequence.push_back({v, old_part});

                update_neighbor_gains(v, old_part);

                // Rebuild the priority queue with updated gains
                std::priority_queue<std::pair<T, T>> new_pq;
                for (T x : subset) {
                    if (!locked[x])
                        new_pq.push(std::make_pair(gain[x], x));
                }
                pq = std::move(new_pq);

                // record the best partial state
                if (current_cut < best_partial_cut) {
                    best_partial_cut = current_cut;
                    best_move_index  = (int)move_index;
                }
            } else {
                // If the move fails the balance constraint, keep the vertex
                // locked, but do not apply the move.
            }
        }

        if (best_move_index >= 0 && best_partial_cut < best_cut) {
            // Revert moves after best_move_index
            for (int i = best_move_index + 1; i < (int)move_sequence.size();
                 i++) {
                auto& mv                        = move_sequence[i];
                T     moved_v                   = mv.first;
                int   original_part             = mv.second;
                local_partition_id_vec[moved_v] = original_part;
            }
            // Recompute weight0, weight1
            T new_w0 = 0, new_w1 = 0;
            for (T v : subset) {
                if (local_partition_id_vec[v] == 0)
                    new_w0 += graph.v_weights[v];
                else
                    new_w1 += graph.v_weights[v];
            }
            weight0 = new_w0;
            weight1 = new_w1;

            best_cut             = best_partial_cut;
            best_local_partition = local_partition_id_vec;
            improvement_found    = true;
        } else {
            // No improvement found, revert everything
            local_partition_id_vec = best_local_partition;
            update_partition_weights(weight0, weight1);
        }

        if (!improvement_found) {
            break;
        }

        // Recompute gains for the next pass
        for (T v : subset) {
            compute_gain_for_vertex(v);
        }
    }
}

#endif

template <typename integer_t>
void GGGP(const RXMeshStatic&      rx,
          const Graph<integer_t>&  graph,
          MaxMatchTree<integer_t>& max_match_tree)
{
    std::vector<integer_t> node_partition_mapping(graph.n, 0);
    std::vector<integer_t> local_node_partition_mapping(graph.n, 0);

    // the partitioning process lambda function
    auto process_partition = [&](integer_t current_partition) {
        integer_t local_n = std::count(node_partition_mapping.begin(),
                                       node_partition_mapping.end(),
                                       current_partition);

        if (local_n <= 1) {
            // if there is only one node in the partition, then we are done
            return;
        }

        // select the node with the min edge weight sum as the starting node
        integer_t min_edge_sum = std::numeric_limits<integer_t>::max();
        integer_t start_node   = integer_t(-1);
        integer_t new_partition_node_count = 0;
        integer_t edge_cut_weight          = 0;
        for (integer_t i = 0; i < graph.n; ++i) {

            if (node_partition_mapping[i] != current_partition) {
                // skip the node if it is not in the partition
                continue;
            }

            integer_t edge_sum = 0;
            for (integer_t j = graph.xadj[i]; j < graph.xadj[i + 1]; ++j) {

                if (node_partition_mapping[graph.adjncy[j]] !=
                    current_partition) {
                    // skip the node if it is not in the partition
                    continue;
                }

                edge_sum += graph.e_weights[j];
            }
            if (edge_sum < min_edge_sum) {
                min_edge_sum = edge_sum;
                start_node   = i;
            }
        }

        assert(start_node != integer_t(-1));

        // set the initial seed start node for the new partition
        // old local partition is 0,
        // new local partition is 1
        local_node_partition_mapping[start_node] = 1;
        new_partition_node_count                 = 1;
        edge_cut_weight                          = min_edge_sum;

        while (new_partition_node_count < (local_n / 2)) {
            // add the node what would introduce the lowest edge cut weight
            // increase
            integer_t min_edge_cut = std::numeric_limits<integer_t>::max();
            integer_t next_node    = integer_t(-1);
            for (integer_t i = 0; i < graph.n; ++i) {
                if (node_partition_mapping[i] != current_partition) {
                    // skip the node if it is not in the partition
                    continue;
                }

                if (local_node_partition_mapping[i] == 1) {
                    // skip the node if it is already in the new partition
                    continue;
                }

                integer_t tmp_edge_cut            = edge_cut_weight;
                bool      is_adj_to_new_partition = false;
                for (integer_t j = graph.xadj[i]; j < graph.xadj[i + 1]; ++j) {
                    if (node_partition_mapping[graph.adjncy[j]] !=
                        current_partition) {
                        // skip the node if it is not in the partition
                        continue;
                    }

                    if (local_node_partition_mapping[graph.adjncy[j]] == 1) {
                        // the node is adjacent to the new partition, the edge
                        // cut weight will decrease
                        is_adj_to_new_partition = true;
                        tmp_edge_cut -= graph.e_weights[j];
                    } else {
                        // the node is adjacent to the old partition, the edge
                        // cut weight will increase
                        tmp_edge_cut += graph.e_weights[j];
                    }
                }

                if (is_adj_to_new_partition && tmp_edge_cut < min_edge_cut) {
                    min_edge_cut = tmp_edge_cut;
                    next_node    = i;
                }
            }

            assert(next_node != integer_t(-1));

            new_partition_node_count++;
            edge_cut_weight                         = min_edge_cut;
            local_node_partition_mapping[next_node] = 1;
        }


        // DEBUG to skip the refinement step
        // TODO do the minimum degree algorithm
        return;
        if (local_n < 5) {
            return;
        }


        // RXMESH_INFO("Partition {} has {} nodes", current_partition, local_n);
        // RXMESH_INFO("Partition {} has {} nodes in the new partition",
        //             current_partition,
        //             new_partition_node_count);
        // // print node_partition_mapping and local_node_partition_mapping
        // RXMESH_INFO("node_partition_mapping:");
        // for (int i = 0; i < graph.n; ++i) {
        //     std::cout << node_partition_mapping[i] << " ";
        // }
        // std::cout << std::endl;

        // RXMESH_INFO("local_node_partition_mapping:");
        // for (int i = 0; i < graph.n; ++i) {
        //     std::cout << local_node_partition_mapping[i] << " ";
        // }
        // std::cout << std::endl;
    };

    // start the partitioning process loop
    integer_t current_partition_process_level = 0;
    bool      is_done                         = false;
    while (!is_done) {
        // process the partition
        for (integer_t i = 0; i < (1 << current_partition_process_level); ++i) {
            process_partition(i);
        }

        if (current_partition_process_level == 0) {
            RXMESH_INFO(" ---------- REFINEMENT ---------- ");
            // before
            RXMESH_INFO("local_node_partition_mapping:");
            for (int i = 0; i < graph.n; ++i) {
                std::cout << local_node_partition_mapping[i] << " ";
            }
            std::cout << std::endl;


            fm_refinement_vweight(graph, local_node_partition_mapping, 10);

            // after
            RXMESH_INFO("local_node_partition_mapping after refinement:");
            for (int i = 0; i < graph.n; ++i) {
                std::cout << local_node_partition_mapping[i] << " ";
            }
            std::cout << std::endl;
        }

        // check the mapping before we move to the next level
        RXMESH_INFO("MAIN - node_partition_mapping:");
        for (int i = 0; i < graph.n; ++i) {
            std::cout << node_partition_mapping[i] << " ";
        }
        std::cout << std::endl;


        for (integer_t i = 0; i < graph.n; ++i) {
            node_partition_mapping[i] = (node_partition_mapping[i] << 1) +
                                        local_node_partition_mapping[i];
        }

        std::fill(local_node_partition_mapping.begin(),
                  local_node_partition_mapping.end(),
                  0);

        RXMESH_INFO(" ---------- MAIN ---------- ");

        RXMESH_INFO("MAIN - node_partition_mapping after shift:");
        for (int i = 0; i < graph.n; ++i) {
            std::cout << node_partition_mapping[i] << " ";
        }
        std::cout << std::endl;

        std::vector<integer_t> sorted_mapping;
        sorted_mapping = node_partition_mapping;
        std::sort(sorted_mapping.begin(), sorted_mapping.end());

        RXMESH_INFO("MAIN - sorted_mapping:");
        for (int i = 0; i < graph.n; ++i) {
            std::cout << sorted_mapping[i] << " ";
        }
        std::cout << std::endl;

        // fill the max match tree for current level
        max_match_tree.levels.insert(max_match_tree.levels.begin(),
                                     Level<integer_t>());
        auto& level      = max_match_tree.levels.front();
        level.patch_proj = node_partition_mapping;

        // fill current level nodes
        for (int i = 0; i < (1 << (current_partition_process_level)); ++i) {
            level.nodes.push_back(
                Node<integer_t>(integer_t(-1), integer_t(-1)));

            Node<integer_t>& node = level.nodes.back();

            assert(std::find(node_partition_mapping.begin(),
                             node_partition_mapping.end(),
                             i) !=
                   node_partition_mapping.end());  // must be found

            node.lch = (i << 1);

            assert(std::find(node_partition_mapping.begin(),
                             node_partition_mapping.end(),
                             i + 1) != node_partition_mapping.end());
            // partition found
            node.rch = (i << 1) + 1;

            node.pa = i;
        }

        RXMESH_INFO(" ---------- LEVEL DONE ---------- ");

        // check if we are done
        // if every node is in a unique partition, then we are done
        is_done = true;

        // is_done = *std::max_element(node_partition_mapping.begin(),
        //                            node_partition_mapping.end()) >
        //                            node_partition_mapping.size();

        std::unordered_set<int> elementSet;
        for (const auto& element : node_partition_mapping) {
            elementSet.insert(element);
        }

        is_done = elementSet.size() == node_partition_mapping.size();

        if (is_done) {
            break;
        }

        current_partition_process_level++;
    }

    // fill the last level of the max match tree
    Level<integer_t>& level = max_match_tree.levels.front();
    for (int i = 0; i < (1 << (current_partition_process_level)); ++i) {
        Node<integer_t>& node = level.nodes[i];

        // processing rch
        auto find_iter = std::find(node_partition_mapping.begin(),
                                   node_partition_mapping.end(),
                                   i << 1);
        assert(find_iter != node_partition_mapping.end());  // must be found
        integer_t index =
            std::distance(node_partition_mapping.begin(), find_iter);
        node.rch = index;

        // processing lch
        find_iter = std::find(node_partition_mapping.begin(),
                              node_partition_mapping.end(),
                              (i << 1) + 1);
        if (find_iter != node_partition_mapping.end()) {
            // partition found
            index    = std::distance(node_partition_mapping.begin(), find_iter);
            node.lch = index;
        } else {
            // partition not found
            node.lch = node.rch;
        }
    }
}

#ifdef USE_V_WEIGHTS
template <typename integer_t>
void GGGP_vweight(const RXMeshStatic&      rx,
                  const Graph<integer_t>&  graph,
                  MaxMatchTree<integer_t>& max_match_tree)
{
    std::vector<integer_t> node_partition_mapping(graph.n, 0);
    std::vector<integer_t> local_node_partition_mapping(graph.n, 0);

    // the partitioning process lambda function
    auto process_partition = [&](integer_t current_partition) {
        integer_t local_n = std::count(node_partition_mapping.begin(),
                                       node_partition_mapping.end(),
                                       current_partition);

        integer_t local_v_weight = 0;
        for (integer_t i = 0; i < graph.n; ++i) {
            if (node_partition_mapping[i] != current_partition) {
                // skip the node if it is not in the partition
                continue;
            }
            local_v_weight += graph.v_weights[i];
        }

        if (local_n <= 1) {
            // if there is only one node in the partition, then we are done
            return;
        }

        // select the node with the min edge weight sum as the starting node
        integer_t min_edge_sum = std::numeric_limits<integer_t>::max();
        integer_t start_node   = integer_t(-1);
        integer_t new_partition_node_count = 0;
        integer_t new_partition_v_weight   = 0;
        integer_t edge_cut_weight          = 0;

        for (integer_t i = 0; i < graph.n; ++i) {
            if (node_partition_mapping[i] != current_partition) {
                continue;
            }
            integer_t edge_sum = 0;
            for (integer_t j = graph.xadj[i]; j < graph.xadj[i + 1]; ++j) {
                if (node_partition_mapping[graph.adjncy[j]] !=
                    current_partition) {
                    continue;
                }
                edge_sum += graph.e_weights[j];
            }
            if (edge_sum < min_edge_sum) {
                min_edge_sum = edge_sum;
                start_node   = i;
            }
        }
        assert(start_node != integer_t(-1));

        // set the initial node for the new local partition
        local_node_partition_mapping[start_node] = 1;
        edge_cut_weight                          = min_edge_sum;

        new_partition_node_count = 1;
        new_partition_v_weight   = graph.v_weights[start_node];

        RXMESH_INFO(
            "new_partition_v_weight: {}, local_v_weight: {} | "
            "new_partition_node_count: {}, local_n: {}",
            new_partition_v_weight,
            local_v_weight,
            new_partition_node_count,
            local_n);

        // Enforce balance: do not exceed half of local_v_weight,
        // do not move all but one node into the new partition
        while (new_partition_v_weight < (local_v_weight / 2) &&
               new_partition_node_count < (local_n - 1)) {
            integer_t min_edge_cut = std::numeric_limits<integer_t>::max();
            integer_t next_node    = integer_t(-1);

            for (integer_t i = 0; i < graph.n; ++i) {
                if (node_partition_mapping[i] != current_partition) {
                    // skip the node if it is not in the partition
                    continue;
                }
                if (local_node_partition_mapping[i] == 1) {
                    continue;
                }

                integer_t tmp_edge_cut            = edge_cut_weight;
                bool      is_adj_to_new_partition = false;
                for (integer_t j = graph.xadj[i]; j < graph.xadj[i + 1]; ++j) {
                    if (node_partition_mapping[graph.adjncy[j]] !=
                        current_partition) {
                        continue;
                    }
                    if (local_node_partition_mapping[graph.adjncy[j]] == 1) {
                        is_adj_to_new_partition = true;
                        tmp_edge_cut -= graph.e_weights[j];
                    } else {
                        tmp_edge_cut += graph.e_weights[j];
                    }
                }

                if (is_adj_to_new_partition && tmp_edge_cut < min_edge_cut) {
                    min_edge_cut = tmp_edge_cut;
                    next_node    = i;
                }
            }

            // If no suitable node was found, or adding the next node does not
            // help, break early
            if (next_node == integer_t(-1)) {
                break;
            }

            RXMESH_INFO("Loop new_partition_v_weight: {}, local_v_weight: {}",
                        new_partition_v_weight,
                        local_v_weight);

            new_partition_node_count++;
            new_partition_v_weight += graph.v_weights[next_node];
            edge_cut_weight                         = min_edge_cut;
            local_node_partition_mapping[next_node] = 1;
        }

        return;

        // RXMESH_INFO("Partition {} has {} nodes", current_partition, local_n);
        // RXMESH_INFO("Partition {} has {} nodes in the new partition",
        //             current_partition,
        //             new_partition_node_count);
        // // print node_partition_mapping and local_node_partition_mapping
        // RXMESH_INFO("node_partition_mapping:");
        // for (int i = 0; i < graph.n; ++i) {
        //     std::cout << node_partition_mapping[i] << " ";
        // }
        // std::cout << std::endl;

        // RXMESH_INFO("local_node_partition_mapping:");
        // for (int i = 0; i < graph.n; ++i) {
        //     std::cout << local_node_partition_mapping[i] << " ";
        // }
        // std::cout << std::endl;
    };

    // start the partitioning process
    integer_t current_partition_process_level = 0;
    bool      is_done                         = false;
    while (!is_done) {
        // process the partition
        for (integer_t i = 0; i < (1 << current_partition_process_level); ++i) {
            process_partition(i);

            fm_refinement_vweight(
                graph,
                /* global_partition_id_vec */ node_partition_mapping,
                /* id */ i,
                /* local_partition_id_vec */ local_node_partition_mapping,
                /* max_passes */ 10);
        }

        RXMESH_INFO("MAIN - node_partition_mapping:");
        for (int i = 0; i < graph.n; ++i) {
            std::cout << node_partition_mapping[i] << " ";
        }
        std::cout << std::endl;

        for (integer_t i = 0; i < graph.n; ++i) {
            node_partition_mapping[i] = (node_partition_mapping[i] << 1) +
                                        local_node_partition_mapping[i];
        }

        std::fill(local_node_partition_mapping.begin(),
                  local_node_partition_mapping.end(),
                  0);

        RXMESH_INFO(" ---------- MAIN ---------- ");
        RXMESH_INFO("MAIN - node_partition_mapping after shift:");
        for (int i = 0; i < graph.n; ++i) {
            std::cout << node_partition_mapping[i] << " ";
        }
        std::cout << std::endl;

        std::vector<integer_t> sorted_mapping = node_partition_mapping;
        std::sort(sorted_mapping.begin(), sorted_mapping.end());

        RXMESH_INFO("MAIN - sorted_mapping:");
        for (int i = 0; i < graph.n; ++i) {
            std::cout << sorted_mapping[i] << " ";
        }
        std::cout << std::endl;

        max_match_tree.levels.insert(max_match_tree.levels.begin(),
                                     Level<integer_t>());
        auto& level      = max_match_tree.levels.front();
        level.patch_proj = node_partition_mapping;

        for (int i = 0; i < (1 << (current_partition_process_level)); ++i) {
            level.nodes.push_back(
                Node<integer_t>(integer_t(-1), integer_t(-1)));
            Node<integer_t>& node = level.nodes.back();

            assert(std::find(node_partition_mapping.begin(),
                             node_partition_mapping.end(),
                             i) != node_partition_mapping.end());

            node.lch = (i << 1);

            assert(std::find(node_partition_mapping.begin(),
                             node_partition_mapping.end(),
                             i + 1) != node_partition_mapping.end());
            node.rch = (i << 1) + 1;

            node.pa = i;
        }

        RXMESH_INFO(" ---------- LEVEL DONE ---------- ");

        std::unordered_set<int> elementSet;
        for (const auto& element : node_partition_mapping) {
            elementSet.insert(element);
        }

        // done if every node is in a unique partition
        is_done = (elementSet.size() ==
                   static_cast<size_t>(node_partition_mapping.size()));
        if (is_done) {
            break;
        }

        current_partition_process_level++;
    }

    // fill the last level of the max match tree
    Level<integer_t>& level = max_match_tree.levels.front();
    for (int i = 0; i < (1 << (current_partition_process_level)); ++i) {
        Node<integer_t>& node = level.nodes[i];

        auto find_iter = std::find(node_partition_mapping.begin(),
                                   node_partition_mapping.end(),
                                   i << 1);
        assert(find_iter != node_partition_mapping.end());
        integer_t index =
            std::distance(node_partition_mapping.begin(), find_iter);
        node.rch = index;

        find_iter = std::find(node_partition_mapping.begin(),
                              node_partition_mapping.end(),
                              (i << 1) + 1);
        if (find_iter != node_partition_mapping.end()) {
            index    = std::distance(node_partition_mapping.begin(), find_iter);
            node.lch = index;
        } else {
            node.lch = node.rch;
        }
    }
}
#endif

template <typename integer_t>
void heavy_max_matching_with_partition(const RXMeshStatic&      rx,
                                       const Graph<integer_t>&  graph,
                                       MaxMatchTree<integer_t>& max_match_tree)
{
    std::vector<integer_t> node_partition_mapping(graph.n, 0);
    std::vector<integer_t> local_node_partition_mapping(graph.n, 0);

    // the partitioning process lambda function
    auto process_partition = [&](integer_t current_partition) {
        integer_t local_n = std::count(node_partition_mapping.begin(),
                                       node_partition_mapping.end(),
                                       current_partition);

        if (local_n <= 1) {
            // if there is only one node in the partition, then we are done
            return;
        }

        // select the node with the min edge weight sum as the starting node
        integer_t min_edge_sum = std::numeric_limits<integer_t>::max();
        integer_t start_node   = integer_t(-1);
        integer_t new_partition_node_count = 0;
        integer_t edge_cut_weight          = 0;
        for (integer_t i = 0; i < graph.n; ++i) {

            if (node_partition_mapping[i] != current_partition) {
                // skip the node if it is not in the partition
                continue;
            }

            integer_t edge_sum = 0;
            for (integer_t j = graph.xadj[i]; j < graph.xadj[i + 1]; ++j) {

                if (node_partition_mapping[graph.adjncy[j]] !=
                    current_partition) {
                    // skip the node if it is not in the partition
                    continue;
                }

                edge_sum += graph.e_weights[j];
            }
            if (edge_sum < min_edge_sum) {
                min_edge_sum = edge_sum;
                start_node   = i;
            }
        }

        assert(start_node != integer_t(-1));

        // set the initial seed start node for the new partition
        // old local partition is 0,
        // new local partition is 1
        local_node_partition_mapping[start_node] = 1;
        new_partition_node_count                 = 1;
        edge_cut_weight                          = min_edge_sum;

        while (new_partition_node_count < (local_n / 2)) {
            // add the node what would introduce the lowest edge cut weight
            // increase
            integer_t min_edge_cut = std::numeric_limits<integer_t>::max();
            integer_t next_node    = integer_t(-1);
            for (integer_t i = 0; i < graph.n; ++i) {
                if (node_partition_mapping[i] != current_partition) {
                    // skip the node if it is not in the partition
                    continue;
                }

                if (local_node_partition_mapping[i] == 1) {
                    // skip the node if it is already in the new partition
                    continue;
                }

                integer_t tmp_edge_cut            = edge_cut_weight;
                bool      is_adj_to_new_partition = false;
                for (integer_t j = graph.xadj[i]; j < graph.xadj[i + 1]; ++j) {
                    if (node_partition_mapping[graph.adjncy[j]] !=
                        current_partition) {
                        // skip the node if it is not in the partition
                        continue;
                    }

                    if (local_node_partition_mapping[graph.adjncy[j]] == 1) {
                        // the node is adjacent to the new partition, the edge
                        // cut weight will decrease
                        is_adj_to_new_partition = true;
                        tmp_edge_cut -= graph.e_weights[j];
                    } else {
                        // the node is adjacent to the old partition, the edge
                        // cut weight will increase
                        tmp_edge_cut += graph.e_weights[j];
                    }
                }

                if (is_adj_to_new_partition && tmp_edge_cut < min_edge_cut) {
                    min_edge_cut = tmp_edge_cut;
                    next_node    = i;
                }
            }

            assert(next_node != integer_t(-1));

            new_partition_node_count++;
            edge_cut_weight                         = min_edge_cut;
            local_node_partition_mapping[next_node] = 1;
        }

        return;

        // RXMESH_INFO("Partition {} has {} nodes", current_partition, local_n);
        // RXMESH_INFO("Partition {} has {} nodes in the new partition",
        //             current_partition,
        //             new_partition_node_count);
        // // print node_partition_mapping and local_node_partition_mapping
        // RXMESH_INFO("node_partition_mapping:");
        // for (int i = 0; i < graph.n; ++i) {
        //     std::cout << node_partition_mapping[i] << " ";
        // }
        // std::cout << std::endl;

        // RXMESH_INFO("local_node_partition_mapping:");
        // for (int i = 0; i < graph.n; ++i) {
        //     std::cout << local_node_partition_mapping[i] << " ";
        // }
        // std::cout << std::endl;
    };

    // partition for one time
    process_partition(0);

    fm_refinement_boundary(graph, local_node_partition_mapping, 100);


    integer_t n = graph.n;

    // prepare the two grpahs
    Graph<integer_t> graph0;
    Graph<integer_t> graph1;


    // Count the number of vertices in each partition
    integer_t n0 = 0;
    integer_t n1 = 0;
    for (integer_t i = 0; i < n; ++i) {
        if (local_node_partition_mapping[i] == 0) {
            ++n0;
        } else if (local_node_partition_mapping[i] == 1) {
            ++n1;
        } else {
            RXMESH_ERROR("Partition id must be 0 or 1");
        }
    }

    // Create mappings from old indices to new indices in each partition
    std::vector<integer_t> old_to_new_index0(n, -1);
    std::vector<integer_t> old_to_new_index1(n, -1);
    integer_t              idx0 = 0;
    integer_t              idx1 = 0;
    for (integer_t i = 0; i < n; ++i) {
        if (local_node_partition_mapping[i] == 0) {
            old_to_new_index0[i] = idx0++;
        } else {
            old_to_new_index1[i] = idx1++;
        }
    }

    // Initialize subgraphs
    graph0.n = n0;
    graph0.xadj.resize(n0 + 1, 0);
    graph0.adjncy.clear();
    graph0.e_weights.clear();
    graph0.e_partition.clear();

    graph1.n = n1;
    graph1.xadj.resize(n1 + 1, 0);
    graph1.adjncy.clear();
    graph1.e_weights.clear();
    graph1.e_partition.clear();

    // Build subgraph for partition 0
    for (integer_t i = 0; i < n; ++i) {
        if (local_node_partition_mapping[i] != 0) {
            continue;
        }
        integer_t new_i    = old_to_new_index0[i];
        graph0.xadj[new_i] = graph0.adjncy.size();

        for (integer_t k = graph.xadj[i]; k < graph.xadj[i + 1]; ++k) {
            integer_t j = graph.adjncy[k];
            if (local_node_partition_mapping[j] == 0) {
                integer_t new_j = old_to_new_index0[j];
                graph0.adjncy.push_back(new_j);
                graph0.e_weights.push_back(graph.e_weights[k]);
                graph0.e_partition.push_back(graph.e_partition[k]);
            }
        }

        graph0.xadj[new_i + 1] = graph0.adjncy.size();
    }

    // Build subgraph for partition 1
    for (integer_t i = 0; i < n; ++i) {
        if (local_node_partition_mapping[i] != 1) {
            continue;
        }
        integer_t new_i    = old_to_new_index1[i];
        graph1.xadj[new_i] = graph1.adjncy.size();

        for (integer_t k = graph.xadj[i]; k < graph.xadj[i + 1]; ++k) {
            integer_t j = graph.adjncy[k];
            if (local_node_partition_mapping[j] == 1) {
                integer_t new_j = old_to_new_index1[j];
                graph1.adjncy.push_back(new_j);
                graph1.e_weights.push_back(graph.e_weights[k]);
                graph1.e_partition.push_back(graph.e_partition[k]);
            }
        }

        graph1.xadj[new_i + 1] = graph1.adjncy.size();
    }

    MaxMatchTree<int> max_match_tree0;
    MaxMatchTree<int> max_match_tree1;

    heavy_max_matching(rx, graph0, max_match_tree0);
    heavy_max_matching(rx, graph1, max_match_tree1);

    max_match_tree0.print();
    max_match_tree1.print();

    RXMESH_INFO("---------- Merging the two trees ----------");


    if (max_match_tree0.levels.size() != max_match_tree1.levels.size()) {
        RXMESH_INFO("The levels of the two trees must be the same");
        while (max_match_tree0.levels.size() < max_match_tree1.levels.size()) {
            Level<int> level;
            level.nodes      = max_match_tree0.levels.back().nodes;
            level.patch_proj = max_match_tree0.levels.back().patch_proj;

            assert(level.nodes.size() == 1);

            level.nodes[0].lch = 0;
            level.nodes[0].rch = 0;

            max_match_tree0.levels.push_back(level);
        }

        while (max_match_tree0.levels.size() > max_match_tree1.levels.size()) {
            Level<int> level;
            level.nodes      = max_match_tree1.levels.back().nodes;
            level.patch_proj = max_match_tree1.levels.back().patch_proj;

            assert(level.nodes.size() == 1);

            level.nodes[0].lch = 0;
            level.nodes[0].rch = 0;

            max_match_tree1.levels.push_back(level);
        }

        max_match_tree0.print();
        max_match_tree1.print();
    }

    assert(max_match_tree0.levels.size() == max_match_tree1.levels.size() &&
           "The levels of the two trees must be the same");

    // get the new to old index mapping
    std::vector<integer_t> new_to_old_index0(n0, -1);
    std::vector<integer_t> new_to_old_index1(n1, -1);
    for (integer_t i = 0; i < n; ++i) {
        if (local_node_partition_mapping[i] == 0) {
            new_to_old_index0[old_to_new_index0[i]] = i;
        } else {
            new_to_old_index1[old_to_new_index1[i]] = i;
        }
    }


    // merge the two trees
    max_match_tree.levels.clear();
    Level<integer_t> root_level;

    for (size_t i = 0; i < max_match_tree0.levels.size(); ++i) {
        Level<integer_t>       level;
        std::vector<integer_t> curr_patch_proj(n, -1);

        if (i == 0) {
            integer_t n0_offset = 0;
            integer_t n1_offset = max_match_tree0.levels[i].nodes.size();

            for (integer_t j = 0; j < max_match_tree0.levels[0].nodes.size();
                 ++j) {
                Node<integer_t> node = max_match_tree0.levels[0].nodes[j];
                node.lch             = new_to_old_index0[node.lch];
                node.rch             = new_to_old_index0[node.rch];
                level.nodes.push_back(node);

                curr_patch_proj[node.lch] = j;
                curr_patch_proj[node.rch] = j;
            }

            for (integer_t j = 0; j < max_match_tree1.levels[0].nodes.size();
                 ++j) {
                Node<integer_t> node = max_match_tree1.levels[0].nodes[j];
                node.lch             = new_to_old_index1[node.lch];
                node.rch             = new_to_old_index1[node.rch];
                level.nodes.push_back(node);

                curr_patch_proj[node.lch] = j + n1_offset;
                curr_patch_proj[node.rch] = j + n1_offset;
            }
        } else {
            assert(i > 0);

            integer_t n0_offset = 0;

            integer_t prev_n1_offset =
                max_match_tree0.levels[i - 1].nodes.size();
            integer_t curr_n1_offset = max_match_tree0.levels[i].nodes.size();

            curr_patch_proj = max_match_tree.levels[i - 1].patch_proj;

            for (integer_t j = 0; j < max_match_tree0.levels[i].nodes.size();
                 ++j) {
                Node<integer_t> node = max_match_tree0.levels[i].nodes[j];
                level.nodes.push_back(node);

                // check
                // print node.lch and node.rch and j
                RXMESH_INFO("0 - node.lch: {}, node.rch: {}, j: {}",
                            node.lch,
                            node.rch,
                            j);

                std::replace(curr_patch_proj.begin(),
                             curr_patch_proj.end(),
                             node.lch,
                             j);
                std::replace(curr_patch_proj.begin(),
                             curr_patch_proj.end(),
                             node.rch,
                             j);
            }

            for (integer_t j = 0; j < max_match_tree1.levels[i].nodes.size();
                 ++j) {
                Node<integer_t> node = max_match_tree1.levels[i].nodes[j];
                node.lch += prev_n1_offset;
                node.rch += prev_n1_offset;
                level.nodes.push_back(node);

                // check
                // print node.lch and node.rch and j
                RXMESH_INFO("1 - node.lch: {}, node.rch: {}, j: {}",
                            node.lch,
                            node.rch,
                            j + curr_n1_offset);

                std::replace(curr_patch_proj.begin(),
                             curr_patch_proj.end(),
                             node.lch,
                             j + curr_n1_offset);
                std::replace(curr_patch_proj.begin(),
                             curr_patch_proj.end(),
                             node.rch,
                             j + curr_n1_offset);
            }
        }

        // print the patch projection
        std::cout << "Patch projection for level " << i << std::endl;
        for (integer_t j = 0; j < n; ++j) {
            std::cout << curr_patch_proj[j] << " ";
        }
        std::cout << std::endl;


        level.patch_proj = curr_patch_proj;
        max_match_tree.levels.push_back(level);
    }

    max_match_tree.print();

    // merge the last level
    Level<integer_t> last_level;
    last_level.nodes.push_back(Node<integer_t>(0, 1));
    last_level.patch_proj = std::vector<integer_t>(n, 0);
    max_match_tree.levels.push_back(last_level);
}


// Helper data structure for priority queue
struct MinDegreeNode
{
    int vertex;
    int degree; // "degree" here = sum of adjacent edge weights

    bool operator>(const MinDegreeNode &rhs) const {
        return degree > rhs.degree;
    }
};

std::vector<int> minimum_degree_ordering_avg_weights(const Graph<int> &graph)
{
    const int n = graph.n;

    // Step 1: Compute initial weighted degree
    std::vector<int> weighted_degree(n, 0);
    for (int v = 0; v < n; ++v) {
        int start = graph.xadj[v];
        int end   = graph.xadj[v+1];
        int deg_sum = 0;
        for (int idx = start; idx < end; ++idx) {
            deg_sum += graph.e_weights[idx];
        }
        weighted_degree[v] = deg_sum;
    }

    // Step 2: Create a single min-priority queue for all vertices
    std::priority_queue<MinDegreeNode, std::vector<MinDegreeNode>, std::greater<MinDegreeNode>> pq;
    pq = {}; // ensure empty
    for (int v = 0; v < n; ++v) {
        pq.push({v, weighted_degree[v]});
    }

    // Step 3: Keep track of eliminated vertices
    std::vector<bool> eliminated(n, false);

    // Result ordering
    std::vector<int> ordering;
    ordering.reserve(n);

    // We also might want a quick way to lookup the weight of (u,v).
    // For naive code, we just re-scan adjacency. For bigger graphs,
    // you might want a hash map or a separate structure.

    // Helper to find weight of edge (u, v). 
    // Returns 0 if no edge found. (We assume undirected or stored as needed.)
    auto get_edge_weight = [&](int u, int v){
        int start = graph.xadj[u];
        int end   = graph.xadj[u+1];
        for(int idx = start; idx < end; ++idx){
            if(graph.adjncy[idx] == v){
                return graph.e_weights[idx];
            }
        }
        return 0; // or -1 if you want to detect "no edge"
    };

    // Step 4: Main loop
    int eliminated_count = 0;
    while (eliminated_count < n)
    {
        // Pop from PQ any vertex already eliminated
        while (!pq.empty() && eliminated[pq.top().vertex]) {
            pq.pop();
        }
        if (pq.empty()) break; // no vertices left

        // Extract the minimum-degree vertex
        MinDegreeNode mn = pq.top();
        pq.pop();
        int chosen_vertex = mn.vertex;
        if (eliminated[chosen_vertex]) {
            continue;
        }

        // Eliminate chosen_vertex
        eliminated[chosen_vertex] = true;
        ordering.push_back(chosen_vertex);
        eliminated_count++;

        // Gather neighbors that are still active
        std::vector<int> neighbors;
        int vstart = graph.xadj[chosen_vertex];
        int vend   = graph.xadj[chosen_vertex+1];
        neighbors.reserve(vend - vstart);
        for(int idx = vstart; idx < vend; ++idx){
            int nbr = graph.adjncy[idx];
            if(!eliminated[nbr]){
                neighbors.push_back(nbr);
            }
        }

        // Step 5: Update degrees of neighbors in a quotient-graph sense
        // Each neighbor loses the edge to chosen_vertex,
        // and gains edges to the other neighbors. 
        // Instead of adding a fixed 1, let's compute the *average* of the edge weights to chosen_vertex.

        // First, collect the weights from chosen_vertex to each neighbor.
        // We'll store them in a small map or array for quick reference:
        std::vector<int> neighbor_weights(neighbors.size(), 0);
        for (size_t i = 0; i < neighbors.size(); ++i){
            int nbr = neighbors[i];
            neighbor_weights[i] = get_edge_weight(chosen_vertex, nbr);
        }

        // Subtract the weight to chosen_vertex from each neighbor's weighted_degree
        for (size_t i = 0; i < neighbors.size(); ++i) {
            int nbr = neighbors[i];
            weighted_degree[nbr] -= neighbor_weights[i];
        }

        // Now form clique among neighbors. For every pair (n1, n2) of neighbors:
        //   If an edge does not exist, add a new edge whose weight is
        //   average( weight(chosen_vertex,n1), weight(chosen_vertex,n2) ).
        for (size_t i = 0; i < neighbors.size(); ++i) {
            int n1 = neighbors[i];
            int w1 = neighbor_weights[i];

            for (size_t j = i+1; j < neighbors.size(); ++j) {
                int n2 = neighbors[j];
                int w2 = neighbor_weights[j];

                // Check if n1 and n2 are already connected
                int existing_weight_n1n2 = get_edge_weight(n1, n2);
                if (existing_weight_n1n2 == 0) {
                    // no edge => create a new edge in the quotient graph
                    // Weighted MD heuristics might do something more sophisticated,
                    // but let's do the naive approach:
                    int new_edge_weight = (w1 + w2) / 2; // average

                    // We increment each neighbor’s degree by the new weight.
                    weighted_degree[n1] += new_edge_weight;
                    weighted_degree[n2] += new_edge_weight;

                    // If you want to store this new edge in the actual adjacency (to maintain consistency),
                    // you'd have to modify 'graph.adjncy' and 'graph.e_weights' to reflect the new edge (n1,n2).
                    // That is more complicated; we'd have to expand the adjacency structure dynamically.
                    // For demonstration, we're only updating the "weighted_degree" values.
                }
                else {
                    // There's already an edge (n1,n2). 
                    // Some Weighted MD variants might also update that existing weight (like merging).
                    // For simplicity, we won't re-add it, but in a rigorous approach,
                    // you might need to unify or update that weight.
                }
            }
        }

        // Step 6: Re-insert (or push updated) neighbors into PQ
        for (auto nbr : neighbors) {
            if (!eliminated[nbr]) {
                pq.push({nbr, weighted_degree[nbr]});
            }
        }
    }

    return ordering;
}


template <typename integer_t>
void min_degree_reordering(const RXMeshStatic&      rx,
                           const Graph<integer_t>&  graph,
                           MaxMatchTree<integer_t>& max_match_tree)
{
    std::vector<integer_t> min_degree_ordering =
        minimum_degree_ordering_avg_weights(graph);

    // check the ordering
    RXMESH_INFO(
        "Ordering Size {}, Graph Size {}", min_degree_ordering.size(), graph.n);
    // print the ordering
    std::cout << "Ordering: ";
    for (int i = 0; i < min_degree_ordering.size(); ++i) {
        std::cout << min_degree_ordering[i] << " ";
    }
    std::cout << std::endl;

    // create the max match tree
    max_match_tree.levels.clear();

    for (int i = 0; i < min_degree_ordering.size() - 1; ++i) {
        max_match_tree.levels.insert(max_match_tree.levels.begin(),
                                     Level<integer_t>());
        auto& level = max_match_tree.levels.front();

        if (i == 0) {
            level.nodes.push_back(Node<integer_t>(0, 1));
        } else {
            level.nodes = max_match_tree.levels[1].nodes;
            level.nodes.push_back(Node<integer_t>(i, i + 1));
        }

        level.patch_proj = std::vector<integer_t>(graph.n, i);
        for (int j = 0; j < i; ++j) {
            level.patch_proj[min_degree_ordering[j]] = j;
        }

        // check the patch projection
        std::cout << "Patch projection for level " << i << std::endl;
        for (int j = 0; j < graph.n; ++j) {
            std::cout << level.patch_proj[j] << " ";
        }
        std::cout << std::endl;
    }

    // fill the last level of the max match tree

    Level<integer_t>& level = max_match_tree.levels.front();
    for (int i = 0; i < level.nodes.size(); ++i) {
        Node<integer_t>& node = level.nodes[i];

        node.lch = min_degree_ordering[i];
        node.rch = min_degree_ordering[i];

        if (i == level.nodes.size() - 1) {
            node.rch = min_degree_ordering[i + 1];
        }
    }

    max_match_tree.print();
}


namespace detail {

template <uint32_t blockThreads>
__global__ static void compute_patch_graph_edge_weight(
    const rxmesh::Context context,
    int*                  d_edge_weight,
    int*                  d_vertex_weight)
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

                                  // padding
                                  8 * ShmemAllocator::default_alignment;
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
    // attr_v1->move(DEVICE, HOST);
    // attr_e->move(DEVICE, HOST);

    // v_local_permute.move(DEVICE, HOST);
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
}

inline void permute_separators(RXMeshStatic&              rx,
                               VertexAttribute<int>&      v_index,
                               VertexAttribute<uint16_t>& v_local_permute,
                               MaxMatchTree<int>&         max_match_tree,
                               int*                       d_permute,
                               int*                       d_patch_proj_l,
                               int*                       d_patch_proj_l1)
{


    CPUTimer timer;
    GPUTimer gtimer;

    timer.start();
    gtimer.start();

    single_patch_nd_permute(rx, v_local_permute);


    timer.stop();
    gtimer.stop();

    RXMESH_INFO("single_patch_nd_permute took {} (ms), {} (ms)",
                timer.elapsed_millis(),
                gtimer.elapsed_millis());

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

    // std::vector<int> h_sibling(rx.get_num_patches());

    //@brief every node in the max match tree will contains three pieces of
    // information:
    // 1) the size of its separator
    // 2) the number of nodes on the right
    // 3) the number of nodes on the left
    // the count here refers to the number of mesh vertices on separator,
    // left, or right We identify the left and right nodes of a (parent)
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

    auto v_render = *rx.add_vertex_attribute<int>("Render", 1);
    v_render.reset(-1, LOCATION_ALL);


    for (int l = depth - 1; l >= 0; --l) {
        rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
            int proj     = max_match_tree.levels[l].patch_proj[vh.patch_id()];
            v_render(vh) = proj;
        });
        rx.get_polyscope_mesh()->addVertexScalarQuantity(
            "Match " + std::to_string(l), v_render);
    }

    int sum_edge_cut = 0;

    for (int l = depth - 1; l >= 0; --l) {

        // TODO use swap at the end of the loop rather copying twice with
        // every iteration
        CUDA_ERROR(cudaMemcpy(d_patch_proj_l,
                              max_match_tree.levels[l].patch_proj.data(),
                              sizeof(int) * rx.get_num_patches(),
                              cudaMemcpyHostToDevice));

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
        //     &h_cut_size, d_cut_size, sizeof(int),
        //     cudaMemcpyDeviceToHost));
        // CUDA_ERROR(cudaMemset(d_cut_size, 0, sizeof(int)));
        // sum_edge_cut += h_cut_size;
        // RXMESH_INFO("Level = {},Cut Size = {}", l, h_cut_size);
        //  if (l >= depth - 3) {
        //      v_render.move(DEVICE, HOST);
        //      rx.get_polyscope_mesh()->addVertexScalarQuantity(
        //          "Render " + std::to_string(l), v_render);
        //
        //      // v_index.move(DEVICE, HOST);
        //      // rx.get_polyscope_mesh()->addVertexScalarQuantity(
        //      //     "Index " + std::to_string(l), v_index);
        //  }
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
    constexpr uint32_t      blockThreads = 512;
    LaunchBox<blockThreads> lb;
    rx.prepare_launch_box(
        {Op::EV},
        lb,
        (void*)detail::compute_patch_graph_edge_weight<blockThreads>);

    detail::compute_patch_graph_edge_weight<blockThreads>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(),
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
    // heavy_max_matching(rx, p_graph, max_match_tree);

    // test the GGGP function
    // GGGP(rx, p_graph, max_match_tree);

    // test the GGGP with vweight but not vertex count
    // GGGP_vweight(rx, p_graph, max_match_tree);

    // RXMESH_INFO("Max Match Tree");
    // max_match_tree.print();

    // RXMESH_INFO("GGGP Max Match Tree");
    // ggp_max_match_tree.print();

    // RXMESH_INFO("Max Match with Partition");
    // heavy_max_matching_with_partition(rx, p_graph, max_match_tree);

    RXMESH_INFO("min_degree_reordering");
    min_degree_reordering(rx, p_graph, max_match_tree);

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