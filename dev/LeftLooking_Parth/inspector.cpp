//
// Created by behrooz on 28/09/22.
//

#include <iostream>
#include "compute_etree_inv.h"
#include "parth_solver.h"
#include "spdlog/spdlog.h"
#include <queue>


namespace PARTH_SOLVER {

void ParthSolverAPI::analyze(std::vector<int>& user_defined_perm)
{
    if (!user_defined_perm.empty()) {
        A_cm.nmethods           = 1;
        A_cm.method[0].ordering = CHOLMOD_GIVEN;
        chol_L =
            cholmod_analyze_p(chol_A, user_defined_perm.data(), NULL, 0, &A_cm);
        schedule_type = ScheduleType::SERIAL;
    } else {
        parth.setReorderingType(PARTH::ReorderingType::AMD);
        parth.computePermutation(perm_vector);
        A_cm.nmethods           = 1;
        A_cm.method[0].ordering = CHOLMOD_GIVEN;
        chol_L                  = cholmod_analyze_p_custom(
            chol_A, perm_vector.data(), NULL, 0, &A_cm);
    }

    if (schedule_type == ScheduleType::SUPERNODAL_BASED) {

    } else if (schedule_type == ScheduleType::PATCH_TREE_BASED) {
        createSupernodalSchedule();
    } else if (schedule_type == ScheduleType::SERIAL) {
        // Do nothing
    } else {
        spdlog::error("Unknown schedule type.");
    }
}

// Comparison function to sort the vector elements
// by second element of tuples
bool sortbysec(const std::tuple<int, int> &a, const std::tuple<int, int> &b) {
    return (std::get<1>(a) > std::get<1>(b));
}

void ParthSolverAPI::createSupernodalSchedule()
{
    // Compute Etree
    // Create the coarsend partitions
    int  nsuper = chol_L->nsuper;
    auto super  = (int*)chol_L->super;
    int  n_elem = chol_L->n / 3;
    assert(chol_L->n % 3 == 0);

    //  Creating the inverse elemination tree
    compute_etree_inv(nsuper, super_parents.data(), tree_ptr, tree_set);

    std::vector<int> serial_nodes(nsuper, 0);
    bool             serial_node_exist = false;
    // Define serial nodes - If a serial node exist,
    // move all its parents to serial schedule
    // Note: The actual workload depends also on the dependent supernodes
    // not just size of this supernode
    for (int s = 0; s < nsuper; s++) {
        if (super[s + 1] - super[s] > serial_supernode_th) {
            serial_nodes[s]   = 1;
            serial_node_exist = true;
            // Mark all the parents as serial
            int parent = super_parents[s];
            while (parent != -1) {
                serial_nodes[parent] = 1;
                parent               = super_parents[parent];
            }
        }
    }

    // Find the first cut based on children of serial nodes that are not serial
    std::vector<std::tuple<int, int>> first_cut;
    if (serial_node_exist) {
        for (int s = 0; s < nsuper; s++) {
            if (serial_nodes[s] == 1) {
                for (int child_ptr = tree_ptr[s]; child_ptr < tree_ptr[s + 1];
                     child_ptr++) {
                    int child = tree_set[child_ptr];
                    if (serial_nodes[child] == 0) {
                        first_cut.emplace_back(std::tuple<int, int>(
                            child, super[child + 1] - super[child]));
                    }
                }
            }
        }
    }

    // If there was no serial nodes, find the roots
    for (int s = 0; s < nsuper; s++) {
        if (super_parents[s] == -1 && serial_nodes[s] == 0) {
            first_cut.emplace_back(
                std::tuple<int, int>(s, super[s + 1] - super[s]));
        }
    }
    assert(first_cut.size() != 0);

    //Push the cut down the stream to have enough parallel workload
    //Note: a reasonable implementation should use SpMP like schedule (P2P synchronization)
    while (first_cut.size() < num_cores) {
        std::sort(first_cut.begin(), first_cut.end(), sortbysec);
        int node = std::get<0>(first_cut[0]);
        assert(serial_nodes[node] == 0);
        serial_nodes[node] = 1;
        first_cut[0]       = first_cut.back();
        first_cut.pop_back();

        for (int child_ptr = tree_ptr[node]; child_ptr < tree_ptr[node + 1];
             child_ptr++) {
            int child = tree_set[child_ptr];
            if (serial_nodes[child] != 1) {
                first_cut.emplace_back(std::tuple<int, int>(
                    child, super[child + 1] - super[child]));
            }
        }
    }

#ifndef NDEBUG
    std::vector<int> first_cut_mark(nsuper, 0);
    for (auto& iter : first_cut) {
        first_cut_mark[std::get<0>(iter)] = 1;
    }
    for (auto& f : first_cut) {
        int s = std::get<0>(f);
        assert(serial_nodes[s] == 0);
        int parent = super_parents[s];
        while (parent != -1) {
            assert(serial_nodes[parent] == 1);
            assert(first_cut_mark[parent] == 0);
            parent = super_parents[parent];
        }
    }
#endif

    //-------------------------------------------------
    // The structure is as follow:
    // parallel_levels: The number of levels that we used from the region
    // tree for region parallelization
    //
    // level_ptr: is the ptr to region_ptr array that shows the supernodes
    // belong to each region region_ptr: is a ptr to supernode_idx array that
    // shows the supernodes in each reigon
    //-------------------------------------------------

    level_ptr.clear();
    part_ptr.clear();
    supernode_idx.clear();
    // Exp:|0|#regions in lvl1|#regions in lvl2 + lvl1|
    int parallel_levels = 2;
    level_ptr.reserve(parallel_levels + 1);
    // total #regions -> we may have empty regions for separated meshes so we
    // reserve
    part_ptr.reserve(first_cut.size());
    supernode_idx.reserve(nsuper);

    level_ptr.emplace_back(0);
    part_ptr.emplace_back(0);

    // Sort based on node idx
    std::sort(first_cut.begin(), first_cut.end());
    int num_parts = 0;
    //  std::vector<int> s_to_p(nsuper, -1);
    // Add parallel part
    for (auto& f : first_cut) {
        // Mark all the descendent
        std::queue<int> r_queue;
        r_queue.push(std::get<0>(f));
        while (!r_queue.empty()) {
            int node = r_queue.front();
            r_queue.pop();
            assert(serial_nodes[node] == 0);
            supernode_idx.emplace_back(node);
            //      s_to_p[node] = part_ptr.size() - 1;
            //      if (serial_nodes[super_parents[node]] == 0) {
            //        assert(s_to_p[super_parents[node]] == s_to_p[node]);
            //      }
            for (int child_ptr = tree_ptr[node]; child_ptr < tree_ptr[node + 1];
                 child_ptr++) {
                int child = tree_set[child_ptr];
                assert(serial_nodes[child] == 0);
                r_queue.push(child);
            }
        }
        //Remember that supernodes are post-order -> so dependency is based on id
        std::sort(supernode_idx.data() + part_ptr.back(),
                  supernode_idx.data() + supernode_idx.size());
        part_ptr.emplace_back(supernode_idx.size());
    }
    level_ptr.emplace_back(part_ptr.size() - 1);

#ifndef NDEBUG
    for (int s = 0; s < supernode_idx.size(); s++) {
        assert(serial_nodes[supernode_idx[s]] == 0);
    }
#endif
    // Add serial part
    for (int s = 0; s < nsuper; s++) {
        if (serial_nodes[s] == 1) {
            supernode_idx.emplace_back(s);
        }
    }
    std::sort(supernode_idx.data() + part_ptr.back(),
              supernode_idx.data() + supernode_idx.size());
    part_ptr.emplace_back(supernode_idx.size());
    level_ptr.emplace_back(part_ptr.size() - 1);
    assert(supernode_idx.size() == nsuper);
#ifndef NDEBUG
    // check data-dependency
    std::vector<int> super_to_lvl(nsuper, -1);
    std::vector<int> super_to_part(nsuper, -1);
    // Applying Checks for correct dependencies
    for (int l = 0; l < level_ptr.size() - 1; l++) {
        for (int p = level_ptr[l]; p < level_ptr[l + 1]; p++) {
            for (int s_ptr = part_ptr[p]; s_ptr < part_ptr[p + 1]; s_ptr++) {
                int s            = supernode_idx[s_ptr];
                super_to_lvl[s]  = l;
                super_to_part[s] = p;
            }
        }
    }

    // Applying Checks for correct dependencies
    for (int l = 0; l < level_ptr.size() - 1; l++) {
        for (int p = level_ptr[l]; p < level_ptr[l + 1]; p++) {
            int prev_s = supernode_idx[part_ptr[p]];
            for (int s_ptr = part_ptr[p]; s_ptr < part_ptr[p + 1]; s_ptr++) {
                int s      = supernode_idx[s_ptr];
                int parent = super_parents[s];
                if (parent == -1) {
                    continue;
                }
                if (super_to_lvl[parent] == l) {
                    assert(super_to_part[parent] == p);
                } else if (super_to_lvl[parent] > l) {
                    assert(super_to_part[parent] != p);
                } else {
                    assert(false);
                }
            }
        }
    }

    // Check existance
    std::vector<int> marked(nsuper, 0);
    for (auto& iter : supernode_idx) {
        marked[iter] = 1;
    }
    for (auto& iter : marked) {
        if (iter == 0) {
            assert(false);
        }
    }
#endif
}
//
// Comparison function to sort the vector elements
// by second element of tuples
// bool sortbysec(const std::tuple<int, int>& a, const std::tuple<int, int>& b)
// {
//     return (std::get<1>(a) > std::get<1>(b));
// }

// void ParthSolverAPI::createPatchTreeSchedule() {
//   // Compute Etree
//   // Create the coarsend partitions
//   int nsuper = chol_L->nsuper;
//   auto super = (int *)chol_L->super;
//   int n_elem = chol_L->n / 3;
//   assert(chol_L->n % 3 == 0);
//
//   //  Creating the inverse elemination tree
//   compute_(nsuper, super_parents.data(), tree_ptr,
//                                 tree_set);
//
//   int switch_th = 256;
//   std::vector<int> serial_nodes(nsuper, 0);
//   bool serial_node_exist = false;
//   // Define serial nodes //TODO: MAKE IT MORE EFFICIENT
//   for (int s = 0; s < nsuper; s++) {
//     if (super[s + 1] - super[s] > switch_th) {
//       serial_nodes[s] = 1;
//       serial_node_exist = true;
//       int parent = super_parents[s];
//       while (parent != -1) {
//         serial_nodes[parent] = 1;
//         parent = super_parents[parent];
//       }
//     }
//   }
//
//   // Find the first cut based on children of serial nodes that are not serial
//   std::vector<std::tuple<int, int>> first_cut;
//   if (serial_node_exist) {
//     for (int s = 0; s < nsuper; s++) {
//       if (serial_nodes[s] == 1) {
//         for (int child_ptr = tree_ptr[s]; child_ptr < tree_ptr[s + 1];
//              child_ptr++) {
//           int child = tree_set[child_ptr];
//           if (serial_nodes[child] == 0) {
//             first_cut.emplace_back(
//                 std::tuple<int, int>(child, super[child + 1] -
//                 super[child]));
//           }
//         }
//       }
//     }
//   }
//
//   // If there was no serial nodes, find the roots
//   for (int s = 0; s < nsuper; s++) {
//     if (super_parents[s] == -1 && serial_nodes[s] == 0) {
//       first_cut.emplace_back(std::tuple<int, int>(s, super[s + 1] -
//       super[s]));
//     }
//   }
//   assert(first_cut.size() != 0);
//
//   while (first_cut.size() < Options().getNumberOfCores()) {
//     std::sort(first_cut.begin(), first_cut.end(), sortbysec);
//     int node = std::get<0>(first_cut[0]);
//     assert(serial_nodes[node] == 0);
//     serial_nodes[node] = 1;
//     first_cut[0] = first_cut.back();
//     first_cut.pop_back();
//
//     for (int child_ptr = tree_ptr[node]; child_ptr < tree_ptr[node + 1];
//          child_ptr++) {
//       int child = tree_set[child_ptr];
//       if (serial_nodes[child] != 1) {
//         first_cut.emplace_back(
//             std::tuple<int, int>(child, super[child + 1] - super[child]));
//       }
//     }
//   }
//
// #ifndef NDEBUG
//   std::vector<int> first_cut_mark(nsuper, 0);
//   for (auto &iter : first_cut) {
//     first_cut_mark[std::get<0>(iter)] = 1;
//   }
//   for (auto &f : first_cut) {
//     int s = std::get<0>(f);
//     assert(serial_nodes[s] == 0);
//     int parent = super_parents[s];
//     while (parent != -1) {
//       assert(serial_nodes[parent] == 1);
//       assert(first_cut_mark[parent] == 0);
//       parent = super_parents[parent];
//     }
//   }
// #endif
//
//   //-------------------------------------------------
//   // The structure is as follow:
//   // parallel_levels: The number of levels that we used from the region
//   // tree for region parallelization
//   //
//   // level_ptr: is the ptr to region_ptr array that shows the supernodes
//   // belong to each region region_ptr: is a ptr to supernode_idx array that
//   // shows the supernodes in each reigon
//   //-------------------------------------------------
//
//   level_ptr.clear();
//   part_ptr.clear();
//   supernode_idx.clear();
//   // Exp:|0|#regions in lvl1|#regions in lvl2 + lvl1|
//   parallel_levels = 2;
//   level_ptr.reserve(parallel_levels + 1);
//   // total #regions -> we may have empty regions for separated meshes so we
//   // reserve
//   part_ptr.reserve(first_cut.size());
//   supernode_idx.reserve(nsuper);
//
//   level_ptr.emplace_back(0);
//   part_ptr.emplace_back(0);
//
//   // Sort based on node idx
//   std::sort(first_cut.begin(), first_cut.end());
//   int num_parts = 0;
//   //  std::vector<int> s_to_p(nsuper, -1);
//   // Add parallel part
//   for (auto &f : first_cut) {
//     // Mark all the descendent
//     std::queue<int> r_queue;
//     r_queue.push(std::get<0>(f));
//     while (!r_queue.empty()) {
//       int node = r_queue.front();
//       r_queue.pop();
//       assert(serial_nodes[node] == 0);
//       supernode_idx.emplace_back(node);
//       //      s_to_p[node] = part_ptr.size() - 1;
//       //      if (serial_nodes[super_parents[node]] == 0) {
//       //        assert(s_to_p[super_parents[node]] == s_to_p[node]);
//       //      }
//       for (int child_ptr = tree_ptr[node]; child_ptr < tree_ptr[node + 1];
//            child_ptr++) {
//         int child = tree_set[child_ptr];
//         assert(serial_nodes[child] == 0);
//         r_queue.push(child);
//       }
//     }
//     std::sort(supernode_idx.data() + part_ptr.back(),
//               supernode_idx.data() + supernode_idx.size());
//     part_ptr.emplace_back(supernode_idx.size());
//   }
//   level_ptr.emplace_back(part_ptr.size() - 1);
//
// #ifndef NDEBUG
//   for (int s = 0; s < supernode_idx.size(); s++) {
//     assert(serial_nodes[supernode_idx[s]] == 0);
//   }
// #endif
//   // Add serial part
//   for (int s = 0; s < nsuper; s++) {
//     if (serial_nodes[s] == 1) {
//       supernode_idx.emplace_back(s);
//     }
//   }
//   std::sort(supernode_idx.data() + part_ptr.back(),
//             supernode_idx.data() + supernode_idx.size());
//   part_ptr.emplace_back(supernode_idx.size());
//   level_ptr.emplace_back(part_ptr.size() - 1);
//   assert(supernode_idx.size() == nsuper);
// #ifndef NDEBUG
//   // check data-dependency
//   std::vector<int> super_to_lvl(nsuper, -1);
//   std::vector<int> super_to_part(nsuper, -1);
//   // Applying Checks for correct dependencies
//   for (int l = 0; l < level_ptr.size() - 1; l++) {
//     for (int p = level_ptr[l]; p < level_ptr[l + 1]; p++) {
//       for (int s_ptr = part_ptr[p]; s_ptr < part_ptr[p + 1]; s_ptr++) {
//         int s = supernode_idx[s_ptr];
//         super_to_lvl[s] = l;
//         super_to_part[s] = p;
//       }
//     }
//   }
//
//   // Applying Checks for correct dependencies
//   for (int l = 0; l < level_ptr.size() - 1; l++) {
//     for (int p = level_ptr[l]; p < level_ptr[l + 1]; p++) {
//       int prev_s = supernode_idx[part_ptr[p]];
//       for (int s_ptr = part_ptr[p]; s_ptr < part_ptr[p + 1]; s_ptr++) {
//         int s = supernode_idx[s_ptr];
//         int parent = super_parents[s];
//         if (parent == -1) {
//           continue;
//         }
//         if (super_to_lvl[parent] == l) {
//           assert(super_to_part[parent] == p);
//         } else if (super_to_lvl[parent] > l) {
//           assert(super_to_part[parent] != p);
//         } else {
//           assert(false);
//         }
//       }
//     }
//   }
//
//   // Check existance
//   std::vector<int> marked(nsuper, 0);
//   for (auto &iter : supernode_idx) {
//     marked[iter] = 1;
//   }
//   for (auto &iter : marked) {
//     if (iter == 0) {
//       assert(false);
//     }
//   }
// #endif
// }
// //
// int ParthSolverAPI::computeETreeCost(int *super, int *tree_ptr, int
// *tree_set,
//                                   int current_node) {
//   assert(tree_ptr != nullptr);
//   assert(tree_set != nullptr);
//   assert(super != nullptr);
//   assert(current_node > -1);
//   int child_cost = 0;
//   for (int child_ptr = tree_ptr[current_node];
//        child_ptr < tree_ptr[current_node + 1]; child_ptr++) {
//     int child = tree_set[child_ptr];
//     child_cost += computeETreeCost(super, tree_ptr, tree_set, child);
//   }
//   node_cost[current_node] =
//       child_cost + super[current_node + 1] - super[current_node];
// }
//
// void ParthSolverAPI::createReuseParallelSchedule() {
//   // Create the coarsend partitions
//   int nsuper = chol_L->nsuper;
//   auto super = (int *)chol_L->super;
//   int n_elem = chol_L->n / 3;
//   assert(chol_L->n % 3 == 0);
//
//   // A cost function that works on Bud hava
//   // Mark the serial supernodes and all of their parents
//   std::vector<int> serial_Marked(nsuper, -1);
//   int switch_th = 256;
//   std::vector<int> serial_supernodes_id;
//   for (int s = 0; s < nsuper; s++) {
//     if (serial_Marked[s] == 1) {
//       continue;
//     }
//     int max_distance = 0;
//     for (int k = super[s]; k < super[s + 1]; k++) {
//       if (k - first_nonzero_per_row[k] > max_distance) {
//         max_distance = k - first_nonzero_per_row[k];
//       }
//     }
//     if (max_distance > switch_th) {
//       serial_Marked[s] = 1;
//       serial_supernodes_id.emplace_back(s);
//       int parent = super_parents[s];
//       while (parent != -1) {
//         serial_Marked[parent] = 1;
//         serial_supernodes_id.emplace_back(parent);
//         parent = super_parents[parent];
//       }
//     }
//   }
//
//   if (Options().getVerbose()) {
//     std::cout << "+++PARTH: The percentage of serial supernodes are: "
//               << serial_supernodes_id.size() * 1.0 / nsuper << std::endl;
//   }
//
//   // Compute the cost of computation based on the size of the supernodes
//   computeInverseEliminationTree(nsuper, super_parents.data(), tree_ptr,
//                                 tree_set);
//
//   std::vector<int> roots;
//   for (int s = 0; s < nsuper; s++) {
//     if (super_parents[s] == -1) {
//       roots.emplace_back(s);
//     }
//   }
//
//   node_cost.clear();
//   node_cost.resize(nsuper, 0);
//
//   for (auto &r : roots) {
//     computeETreeCost(super, tree_ptr.data(), tree_set.data(), r);
//   }
//
//   // Start from a level with enough regions -> Note that levels are start
//   from
//   // 0
//   // So parallel_levels equal to 1 means we start with level 0 with one
//   region
//   // and level 1 with 2 regions
//   parallel_levels =
//   std::min(std::ceil(std::log2(Options().getNumberOfCores())),
//                              (double)regions.num_levels);
//
//   //-------------------------------------------------
//   // The structure is as follow:
//   // parallel_levels: The number of levels that we used from the region
//   // tree for region parallelization
//   //
//   // level_ptr: is the ptr to region_ptr array that shows the supernodes
//   // belong to each region region_ptr: is a ptr to supernode_idx array that
//   // shows the supernodes in each reigon
//   //-------------------------------------------------
//
//   level_ptr.clear();
//   part_ptr.clear();
//   supernode_idx.clear();
//   // Exp:|0|#regions in lvl1|#regions in lvl2 + lvl1|
//   level_ptr.reserve(parallel_levels + 1);
//   // total #regions -> we may have empty regions for separated meshes so we
//   // reserve
//   part_ptr.reserve(std::pow(2, parallel_levels + 1));
//   supernode_idx.reserve(nsuper);
//
//   level_ptr.emplace_back(0);
//   part_ptr.emplace_back(0);
//
//   // Find a cut with good PG
//
//   // Create the schedule
// }
//
// void ParthSolverAPI::computeFirstchild(int n, int *Ap, int *Ai) {
//   first_nonzero_per_row.clear();
//   first_nonzero_per_row.resize(n, n);
//   for (int c = 0; c < n; c++) {
//     first_nonzero_per_row[c] = Ai[Ap[c]];
//   }
// }

}  // namespace PARTH_SOLVER
