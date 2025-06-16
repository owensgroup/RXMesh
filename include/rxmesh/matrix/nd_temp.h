#if 0

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
    int degree;  // "degree" here = sum of adjacent edge weights

    bool operator>(const MinDegreeNode& rhs) const
    {
        return degree > rhs.degree;
    }
};

inline std::vector<int> minimum_degree_ordering_avg_weights(
    const Graph<int>& graph)
{
    const int n = graph.n;

    // Step 1: Compute initial weighted degree
    std::vector<int> weighted_degree(n, 0);
    for (int v = 0; v < n; ++v) {
        int start   = graph.xadj[v];
        int end     = graph.xadj[v + 1];
        int deg_sum = 0;
        for (int idx = start; idx < end; ++idx) {
            deg_sum += graph.e_weights[idx];
        }
        weighted_degree[v] = deg_sum;
    }

    // Step 2: Create a single min-priority queue for all vertices
    std::priority_queue<MinDegreeNode,
                        std::vector<MinDegreeNode>,
                        std::greater<MinDegreeNode>>
        pq;
    pq = {};  // ensure empty
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
    auto get_edge_weight = [&](int u, int v) {
        int start = graph.xadj[u];
        int end   = graph.xadj[u + 1];
        for (int idx = start; idx < end; ++idx) {
            if (graph.adjncy[idx] == v) {
                return graph.e_weights[idx];
            }
        }
        return 0;  // or -1 if you want to detect "no edge"
    };

    // Step 4: Main loop
    int eliminated_count = 0;
    while (eliminated_count < n) {
        // Pop from PQ any vertex already eliminated
        while (!pq.empty() && eliminated[pq.top().vertex]) {
            pq.pop();
        }
        if (pq.empty())
            break;  // no vertices left

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
        int              vstart = graph.xadj[chosen_vertex];
        int              vend   = graph.xadj[chosen_vertex + 1];
        neighbors.reserve(vend - vstart);
        for (int idx = vstart; idx < vend; ++idx) {
            int nbr = graph.adjncy[idx];
            if (!eliminated[nbr]) {
                neighbors.push_back(nbr);
            }
        }

        // Step 5: Update degrees of neighbors in a quotient-graph sense
        // Each neighbor loses the edge to chosen_vertex,
        // and gains edges to the other neighbors.
        // Instead of adding a fixed 1, let's compute the *average* of the edge
        // weights to chosen_vertex.

        // First, collect the weights from chosen_vertex to each neighbor.
        // We'll store them in a small map or array for quick reference:
        std::vector<int> neighbor_weights(neighbors.size(), 0);
        for (size_t i = 0; i < neighbors.size(); ++i) {
            int nbr             = neighbors[i];
            neighbor_weights[i] = get_edge_weight(chosen_vertex, nbr);
        }

        // Subtract the weight to chosen_vertex from each neighbor's
        // weighted_degree
        for (size_t i = 0; i < neighbors.size(); ++i) {
            int nbr = neighbors[i];
            weighted_degree[nbr] -= neighbor_weights[i];
        }

        // Now form clique among neighbors. For every pair (n1, n2) of
        // neighbors:
        //   If an edge does not exist, add a new edge whose weight is
        //   average( weight(chosen_vertex,n1), weight(chosen_vertex,n2) ).
        for (size_t i = 0; i < neighbors.size(); ++i) {
            int n1 = neighbors[i];
            int w1 = neighbor_weights[i];

            for (size_t j = i + 1; j < neighbors.size(); ++j) {
                int n2 = neighbors[j];
                int w2 = neighbor_weights[j];

                // Check if n1 and n2 are already connected
                int existing_weight_n1n2 = get_edge_weight(n1, n2);
                if (existing_weight_n1n2 == 0) {
                    // no edge => create a new edge in the quotient graph
                    // Weighted MD heuristics might do something more
                    // sophisticated, but let's do the naive approach:
                    int new_edge_weight = (w1 + w2) / 2;  // average

                    // We increment each neighbor’s degree by the new weight.
                    weighted_degree[n1] += new_edge_weight;
                    weighted_degree[n2] += new_edge_weight;

                    // If you want to store this new edge in the actual
                    // adjacency (to maintain consistency), you'd have to modify
                    // 'graph.adjncy' and 'graph.e_weights' to reflect the new
                    // edge (n1,n2). That is more complicated; we'd have to
                    // expand the adjacency structure dynamically. For
                    // demonstration, we're only updating the "weighted_degree"
                    // values.
                } else {
                    // There's already an edge (n1,n2).
                    // Some Weighted MD variants might also update that existing
                    // weight (like merging). For simplicity, we won't re-add
                    // it, but in a rigorous approach, you might need to unify
                    // or update that weight.
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
#endif