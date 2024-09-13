#pragma once

#include <set>
#include <vector>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/mgnd_permute.cuh"
#include "rxmesh/matrix/permute_util.h"

#include "rxmesh/matrix/nd_single_patch_ordering.cuh"

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
    T get_weight(T x, T y, T default_val = -1) const
    {
        for (T i = xadj[x]; i < xadj[x + 1]; ++i) {
            if (adjncy[i] == y) {
                return weights[i];
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
    const RXMeshStatic&     rx,
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
    heavy_max_matching(rx, c_graph, max_match_tree);
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
                                                   const int* d_patch_proj,
                                                   const VertexIterator& iter)
{
    // on a certain level of the max match, return the sibling to a certain
    // mesh vertex on the tree (only if this sibling is different than the
    // vertx's patch grand parent)

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
                                          int*                 d_permute,
                                          int                  current_level,
                                          int                  depth,
                                          const int*           d_dfs_index,
                                          int*                 d_count)
{
    // d_patch_proj_l is the patch projection on this level
    // d_patch_proj_l1 is the patch projection on the next level (i.e.,
    // current_level -1)

    const int S = depth - current_level - 1;

    const int shift = (1 << S) - 1;

    auto extract = [&](VertexHandle v_id, VertexIterator& iter) {
        // this is important to check if v is on the separator before going in
        // and check if it is on the current/grant separator because we have
        // consistent criterion for if a vertex is on a separator (using less
        // than for the vertex patch id)

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

                d_permute[context.linear_id(v_id)] =
                    ::atomicAdd(&d_count[d_dfs_index[index]], int(1));

                assert(v_index(v_id) < 0);
                v_index(v_id) = d_dfs_index[index];

            } else if (current_level == 0) {
                // get the patch index within the max match tree

                const int SS = depth - (current_level - 1) - 1;

                const int sh = (1 << SS) - 1;

                int index = sh + v_id.patch_id();

                d_permute[context.linear_id(v_id)] =
                    ::atomicAdd(&d_count[d_dfs_index[index]], int(1));

                assert(v_index(v_id) < 0);
                v_index(v_id) = d_dfs_index[index];
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, extract);
}


}  // namespace detail

void create_dfs_indexing(const int                level,
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

void single_patch_nd_permute(RXMeshStatic&              rx,
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
        (void*)nd_single_patch<blockThreads, maxCoarsenLevels>,
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
                          (void*)nd_single_patch_kmeans<blockThreads>,
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
    nd_single_patch<blockThreads, maxCoarsenLevels>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), v_local_permute, *attr_v, *attr_e, *attr_v1);
#else
    nd_single_patch_kmeans<blockThreads>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(rx.get_context(),
                                                           v_local_permute);
#endif

    CUDA_ERROR(cudaDeviceSynchronize());

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

void permute_separators(RXMeshStatic&              rx,
                        VertexAttribute<int>&      v_index,
                        VertexAttribute<uint16_t>& v_local_permute,
                        MaxMatchTree<int>&         max_match_tree,
                        int*                       d_permute,
                        int*                       d_patch_proj_l,
                        int*                       d_patch_proj_l1)
{
    single_patch_nd_permute(rx, v_local_permute);

    v_index.reset(-1, DEVICE);

    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> lbe;
    rx.prepare_launch_box(
        {Op::VV}, lbe, (void*)detail::extract_separators<blockThreads>);


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
    // the count here refers to the number of mesh vertices on separator, left,
    // or right
    // We identify the left and right nodes of a (parent) node using less (<)
    // between the left and right node ID.


    // count the number of mesh vertices at different parts in the max match
    // tree some vertices are on the different separators along the tree. The
    // remaining vertices are the one inside the interior of the patch that
    // are number randomly.
    int count_size = 1 << (depth + 1);

    int *d_dfs_index(nullptr), *d_count(nullptr);
    CUDA_ERROR(cudaMalloc((void**)&d_dfs_index, sizeof(int) * count_size));


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


    for (int l = depth - 1; l >= 0; --l) {

        // TODO use swap at the end of the loop rather copying twice with every
        // iteration
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
                d_permute,
                l,
                depth,
                d_dfs_index,
                d_count);
    }


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

            // l += v_local_permute(vh);
            l += d_permute[context.linear_id(vh)];

        } else {
            // if it is a separator
            l += d_permute[context.linear_id(vh)];
        }

        d_permute[context.linear_id(vh)] = num_v - l - 1;
    });

    GPU_FREE(d_dfs_index);
    GPU_FREE(d_count);
}

void nd_permute(RXMeshStatic& rx, int* h_permute)
{

    auto v_index = *rx.add_vertex_attribute<int>("index", 1);

    auto v_local_permute = *rx.add_vertex_attribute<uint16_t>("index", 1);

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


    // create max match tree
    MaxMatchTree<int> max_match_tree;
    heavy_max_matching(rx, p_graph, max_match_tree);


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

    GPU_FREE(d_permute);
    GPU_FREE(d_patch_proj_l);
    GPU_FREE(d_patch_proj_l1);
    GPU_FREE(d_patch_graph_edge_weight);
}


}  // namespace rxmesh