#pragma once

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
    }
};

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

void nd_permute(RXMeshStatic& rx, std::vector<int>& h_permute)
{
    // a graph representing the patch connectivity
    Graph<idx_t> p_graph;

    construct_patches_neighbor_graph(rx, p_graph);

    p_graph.print();

    std::vector<idx_t> p_graph_permute(p_graph.n);
    std::vector<idx_t> p_graph_inv_permute(p_graph.n);

    fill_with_sequential_numbers(p_graph_permute.data(),
                                 p_graph_permute.size());

    fill_with_sequential_numbers(p_graph_inv_permute.data(),
                                 p_graph_inv_permute.size());
    // run_metis(p_graph, p_graph_permute, p_graph_inv_permute);

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

}  // namespace rxmesh