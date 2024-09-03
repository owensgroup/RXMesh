#pragma once

#include <set>
#include <vector>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/permute_util.h"
#include "rxmesh/matrix/separator_tree.h"

#include "metis.h"

#include "rxmesh/matrix/mgnd_permute.cuh"

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
    std::vector<T> weights;


    /**
     * @brief return the weight between x and y. If there is no edge between
     * them return the default weight
     */
    T get_weight(T x, T y, T default = -1) const
    {
        for (T i = xadj[x]; i < xadj[x + 1]; ++i) {
            if (adjncy[i] == y) {
                return weights[i];
            }
        }
        return default;
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
template <typename T>
void construct_patches_neighbor_graph(
    RXMeshStatic&           rx,
    Graph<T>&               patches_graph,
    const std::vector<int>& h_patch_graph_edge_weight)
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
                patches_graph.weights.push_back(
                    h_patch_graph_edge_weight[PatchStash::stash_size * p + i]);
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
    std::vector<int> patch_grand_parent;
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
};

template <typename integer_t>
void random_max_matching(const RXMeshStatic&      rx,
                         const Graph<integer_t>&  graph,
                         MaxMatchTree<integer_t>& max_match_tree)
{
    // TODO workaround isolated island (comes from disconnected component input
    // mesh)
    if (graph.n <= 1) {
        return;
    }

    // graph.print();

    std::vector<bool> matched(graph.n, false);

    Level<integer_t> l;
    l.nodes.reserve(DIVIDE_UP(graph.n, 2));
    l.patch_grand_parent.resize(rx.get_num_patches());

    // we store the parent of each node which we use to create the coarse graph
    // i.e., next level graph
    std::vector<integer_t> parents;
    parents.resize(graph.n, integer_t(-1));

    std::vector<integer_t> rands(graph.n);
    fill_with_random_numbers(rands.data(), rands.size());


    for (int k = 0; k < graph.n; ++k) {
        int v = rands[k];
        if (!matched[v]) {

            // the neighbor with which we will match v
            integer_t matched_neighbour = integer_t(-1);
            integer_t max_w             = 0;

            for (int i = graph.xadj[v]; i < graph.xadj[v + 1]; ++i) {
                integer_t n = graph.adjncy[i];
                integer_t w = graph.weights[i];

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

                // store the grant parent of level -1
                if (max_match_tree.levels.empty()) {
                    l.patch_grand_parent[v]                 = node_id;
                    l.patch_grand_parent[matched_neighbour] = node_id;
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
                l.patch_grand_parent[v] = node_id;
            }
        }
    }

    // update the grand parents if we are not on level 0
    if (!max_match_tree.levels.empty() && l.nodes.size() > 1) {
        for (uint32_t p = 0; p < l.patch_grand_parent.size(); ++p) {
            l.patch_grand_parent[p] =
                parents[max_match_tree.levels.back().patch_grand_parent[p]];
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
    c_graph.weights.reserve(c_graph.n * 3);


    for (int i = 0; i < l.nodes.size(); ++i) {
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
                    integer_t new_weight = graph.weights[i];

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

        c_graph.xadj[i + 1] = c_graph.xadj[i];
        for (const auto& u : u_neighbour) {
            // actually what we insert in the coarse graph is the parents
            c_graph.adjncy.push_back(u.first);
            c_graph.weights.push_back(u.second);
            c_graph.xadj[i + 1]++;
        }
    }

    // Now that we have this level finished, we can insert in the tree
    max_match_tree.levels.push_back(l);

    // recurse to the next level
    random_max_matching(rx, c_graph, max_match_tree);
}

namespace detail {

template <uint32_t blockThreads>
__global__ static void compute_patch_graph_edge_weight(
    const rxmesh::Context context,
    int*                  d_edge_weight)
{
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
}

__inline__ __device__ bool is_v_on_grand_separator(const VertexHandle v_id,
                                                   uint32_t           v_gp,
                                                   int* d_grand_parent,
                                                   const VertexIterator& iter)
{
    for (uint16_t i = 0; i < iter.size(); ++i) {
        uint32_t n_pid = iter[i].patch_id();

        int n_gp = d_grand_parent[n_pid];

        if (v_gp != n_gp) {
            return true;
        }
    }
    return false;
}
template <uint32_t blockThreads>
__global__ static void extract_separators(const Context        context,
                                          int*                 d_grand_parent,
                                          int*                 d_permute,
                                          int*                 d_accumulate,
                                          VertexAttribute<int> assigned)
{

    auto extract = [&](VertexHandle v_id, VertexIterator& iter) {
        // this is important to check if v is on the separator before going in
        // and check if it is on the current/grant separator because we have
        // consistent criterion for if a vertex is on a separator (using less
        // than for the vertex patch id)

        if (is_v_on_separator(v_id, iter)) {
            uint32_t v_gp = d_grand_parent[v_id.patch_id()];

            if (is_v_on_grand_separator(v_id, v_gp, d_grand_parent, iter)) {
                assigned(v_id) = 1;

                int v_new_local = ::atomicAdd(&d_accumulate[v_gp], int(1));
                d_permute[context.linear_id(v_id)] = v_new_local;
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, extract);
}

template <uint32_t blockThreads>
__global__ static void separators_permutation(const Context context,
                                              int*          d_grand_parent,
                                              int*          d_permute,
                                              int*          d_accumulate,
                                              int           num_vertices,
                                              VertexAttribute<int> assigned)
{
    auto extract = [&](VertexHandle v_id, VertexIterator& iter) {
        if (is_v_on_separator(v_id, iter)) {
            uint32_t v_gp = d_grand_parent[v_id.patch_id()];

            if (is_v_on_grand_separator(v_id, v_gp, d_grand_parent, iter)) {

                int v_new_local = d_permute[context.linear_id(v_id)];

                int v_new_id = v_new_local + d_accumulate[v_gp];

                v_new_id = num_vertices - v_new_id;

                d_permute[context.linear_id(v_id)] = v_new_id;
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, extract);
}
}  // namespace detail

void permute_separators(RXMeshStatic&      rx,
                        MaxMatchTree<int>& max_match_tree,
                        int*               d_permute,
                        int*               d_grand_parent,
                        int*               d_accumulate)
{


    auto assigned = *rx.add_vertex_attribute<int>("sep", 1);

    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> lbe;
    rx.prepare_launch_box(
        {Op::VV}, lbe, (void*)detail::extract_separators<blockThreads>);

    LaunchBox<blockThreads> lba;
    rx.prepare_launch_box(
        {Op::VV}, lba, (void*)detail::separators_permutation<blockThreads>);

    rx.render_face_patch();
    rx.render_vertex_patch();

    for (int l = max_match_tree.levels.size() - 2; l >= 0; --l) {
        assigned.reset(0, DEVICE);

        CUDA_ERROR(cudaMemset(
            d_accumulate, 0, sizeof(int) * (rx.get_num_patches() + 1)));

        // move grand parents to the device
        CUDA_ERROR(
            cudaMemcpy(d_grand_parent,
                       max_match_tree.levels[l].patch_grand_parent.data(),
                       sizeof(int) * rx.get_num_patches(),
                       cudaMemcpyHostToDevice));

        detail::extract_separators<blockThreads>
            <<<lbe.blocks, lbe.num_threads, lbe.smem_bytes_dyn>>>(
                rx.get_context(),
                d_grand_parent,
                d_permute,
                d_accumulate,
                assigned);


        assigned.move(DEVICE, HOST);
        rx.get_polyscope_mesh()->addVertexScalarQuantity(
            "Sep_" + std::to_string(l), assigned);

        // assigned.reset(0, DEVICE);
        // thrust::exclusive_scan(thrust::device,
        //                        d_accumulate,
        //                        d_accumulate + rx.get_num_patches(),
        //                        d_accumulate);
        //
        // detail::separators_permutation<blockThreads>
        //     <<<lba.blocks, lba.num_threads, lba.smem_bytes_dyn>>>(
        //         rx.get_context(),
        //         d_grand_parent,
        //         d_permute,
        //         d_accumulate,
        //         rx.get_num_vertices(),
        //         assigned);
        //
        // assigned.move(DEVICE, HOST);
        // rx.get_polyscope_mesh()->addVertexScalarQuantity(
        //     "ID_" + std::to_string(l), assigned);
        //
        polyscope::show();
    }
}

void nd_permute(RXMeshStatic& rx, std::vector<int>& h_permute)
{
    h_permute.resize(rx.get_num_vertices());

    // for a level L in the max_match_tree, d_grand_parent stores the node at
    // level L that branch off to a given patch
    int* d_grand_parent = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_grand_parent,
                          sizeof(int) * rx.get_num_patches()));

    // stores the edge weight of the patch graph
    int*     d_patch_graph_edge_weight = nullptr;
    uint32_t edge_weight_size  = PatchStash::stash_size * rx.get_num_patches();
    uint32_t edge_weight_bytes = sizeof(int) * edge_weight_size;
    CUDA_ERROR(
        cudaMalloc((void**)&d_patch_graph_edge_weight, edge_weight_bytes));
    CUDA_ERROR(cudaMemset(d_patch_graph_edge_weight, 0, edge_weight_bytes));

    std::vector<int> h_patch_graph_edge_weight(edge_weight_size, 0);

    // the new index
    int* d_permute = nullptr;
    CUDA_ERROR(
        cudaMalloc((void**)&d_permute, rx.get_num_vertices() * sizeof(int)));

    int* d_accumulate = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_accumulate,
                          sizeof(int) * (rx.get_num_patches() + 1)));


    // compute edge weight
    constexpr uint32_t      blockThreads = 512;
    LaunchBox<blockThreads> lb;
    rx.prepare_launch_box(
        {rxmesh::Op::EV},
        lb,
        (void*)detail::compute_patch_graph_edge_weight<blockThreads>);

    detail::compute_patch_graph_edge_weight<blockThreads>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), d_patch_graph_edge_weight);

    CUDA_ERROR(cudaMemcpy(h_patch_graph_edge_weight.data(),
                          d_patch_graph_edge_weight,
                          edge_weight_bytes,
                          cudaMemcpyDeviceToHost));

    // a graph representing the patch connectivity
    Graph<int> p_graph;
    construct_patches_neighbor_graph(rx, p_graph, h_patch_graph_edge_weight);
    // construct_a_simple_chordal_graph(p_graph);
    // p_graph.print();
    CUDA_ERROR(cudaDeviceSynchronize());

    // create max match tree
    MaxMatchTree<int> max_match_tree;
    random_max_matching(rx, p_graph, max_match_tree);
    // max_match_tree.print();


    permute_separators(
        rx, max_match_tree, d_permute, d_grand_parent, d_accumulate);

    GPU_FREE(d_permute);
    GPU_FREE(d_grand_parent);
    GPU_FREE(d_accumulate);
    GPU_FREE(d_patch_graph_edge_weight);
}
}  // namespace rxmesh