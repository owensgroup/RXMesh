#pragma once

#include <cub/device/device_scan.cuh>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/dense_matrix.cuh"
#include "rxmesh/matrix/sparse_matrix.cuh"

#include "rxmesh/matrix/sparse_matrix_constant_nnz_row.h"

#include "FPSSampler.h"
#include "GMG_kernels.h"
#include "NeighborHandling.h"
#include "cluster.h"
#include "hashtable.h"

namespace rxmesh {

enum class Sampling
{
    Rand   = 0,
    FPS    = 1,
    KMeans = 2,
};

template <typename T>
struct GMG
{
    // In indexing, we always using L=0 to refer to the fine mesh.
    // The index of the first coarse level is L=1.

    float m_ratio;
    int   m_num_levels;  // numberOfLevels
    int   m_num_rows;
    int*  m_d_flag;


    std::vector<int>                m_num_samples;  // fine+levels
    std::vector<DenseMatrix<float>> m_samples_pos;  // levels

    std::vector<DenseMatrix<int>> m_sample_neighbor_size;            // levels
    std::vector<DenseMatrix<int>> m_sample_neighbor_size_prefix;     // levels
    std::vector<DenseMatrix<int>> m_sample_neighbor;                 // levels
    std::vector<SparseMatrixConstantNNZRow<float, 3>> m_prolong_op;  // levels

    std::vector<DenseMatrix<int>>   m_sample_id;       // fine+levels
    std::vector<DenseMatrix<float>> m_distance_mat;    // fine+levels
    std::vector<DenseMatrix<int>>   m_vertex_cluster;  // fine+levels

    DenseMatrix<float>     m_vertex_pos;            // fine
    VertexAttribute<float> m_distance;              // fine
    DenseMatrix<uint16_t>  m_sample_level_bitmask;  // fine

    GPUHashTable<Edge> m_edge_hash_table;


    // we reuse cub temp storage for all level (since level has the largest
    // storage) and only re-allocate if we have to
    void*  m_d_cub_temp_storage;
    size_t m_cub_temp_bytes;

    

    GMG(RXMeshStatic&    rx,
        SparseMatrix<T>& A,
        DenseMatrix<T>&  B,
        Sampling         sam                   = Sampling::FPS,
        int              reduction_ratio       = 7,
        int              num_samples_threshold = 6)
        : m_ratio(reduction_ratio),
          m_edge_hash_table(GPUHashTable(DIVIDE_UP(rx.get_num_edges(), 2)))
    {
        m_num_rows = A.rows();

        m_num_samples.push_back(m_num_rows);
        for (int i = 0; i < 16; i++) {
            int s = DIVIDE_UP(m_num_rows, std::pow(m_ratio, i));
            if (s > num_samples_threshold) {
                m_num_samples.push_back(s);
            }
        }
        m_num_levels = m_num_samples.size();

        //============
        // 1) init variables and alloc memory
        //============

        // sample_id, m_distance_mat, and vertex_cluster are stored for the fine
        // mesh and all levels.
        // m_samples_pos, m_prolong_op,
        // m_sample_neighbor_size, and m_sample_neighbor_size_prefix are
        // allocated only for coarse levels
        for (int l = 0; l < m_num_samples.size(); ++l) {
            int level_num_samples = m_num_samples[l];

            if (l > 0) {
                m_samples_pos.emplace_back(rx, level_num_samples, 3);
                m_prolong_op.emplace_back(
                    rx, level_num_samples, level_num_samples);

                // we allocate +1 for cub prefix sum
                m_sample_neighbor_size.emplace_back(
                    rx, level_num_samples + 1, 1);
                m_sample_neighbor_size.back().reset(0, DEVICE);

                m_sample_neighbor_size_prefix.emplace_back(
                    rx, level_num_samples + 1, 1);
                m_sample_neighbor_size_prefix.back().reset(0, DEVICE);
            }

            m_sample_id.emplace_back(rx, level_num_samples, 1);
            m_vertex_cluster.emplace_back(rx, level_num_samples, 1);
            m_vertex_cluster.back().reset(std::numeric_limits<int>::max(),
                                          LOCATION_ALL);
            m_distance_mat.emplace_back(rx, level_num_samples, 1);
        }


        m_vertex_pos = *rx.get_input_vertex_coordinates()->to_matrix();
        m_distance   = *rx.add_vertex_attribute<float>("d", 1);
        m_sample_level_bitmask = DenseMatrix<uint16_t>(rx, m_num_rows, 1);


        CUDA_ERROR(cudaMalloc((void**)&m_d_flag, sizeof(int)));
                

        // allocate CUB stuff here
        m_cub_temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            m_d_cub_temp_storage,
            m_cub_temp_bytes,
            m_sample_neighbor_size[0].data(DEVICE),
            m_sample_neighbor_size_prefix[0].data(DEVICE),
            m_num_samples[1] + 1);
        CUDA_ERROR(cudaMalloc((void**)&m_d_cub_temp_storage, m_cub_temp_bytes));


        //============
        // 2) Sampling
        //============
        switch (sam) {
            case Sampling::Rand: {
                random_sampling(rx);
                break;
            }
            case Sampling::FPS: {
                fps_sampling(rx);
                break;
            }
            case Sampling::KMeans: {
                kmeans_sampling(rx);
                break;
            }
            default: {
                RXMESH_ERROR("GMG::GMG() Invalid input Sampling");
                break;
            }
        }


        for (int l = 1; l < m_num_levels; ++l) {

            //============
            // 3) Clustering
            //============
            clustering(rx, l);


            //============
            // 4) Create coarse mesh compressed representation of
            //============
            create_compressed_representation(rx, l);
        }

        //============
        // 5) Create prolongation operator
        //============
        create_all_prolongation();

        // move prolongation operator from device to host (for no obvious
        // reason)
        for (auto& prolong_op : m_prolong_op) {
            prolong_op.move_col_idx(DEVICE, HOST);
            prolong_op.move(DEVICE, HOST);
        }

        // release temp memory
        GPU_FREE(m_d_flag);
        GPU_FREE(m_d_cub_temp_storage);
    }

    /**
     * @brief Samples for all levels using FPS sampling
     */
    void fps_sampling(RXMeshStatic& rx)
    {
        FPSSampler(rx,
                   m_distance,
                   m_vertex_pos,
                   m_sample_id[0],
                   m_sample_level_bitmask,
                   m_samples_pos[0],
                   m_ratio,
                   m_num_rows,
                   m_num_levels,
                   m_num_samples[1],
                   m_d_flag);
    }

    /**
     * @brief Samples for all levels using random sampling
     * TODO implement this
     */
    void random_sampling(RXMeshStatic& rx)
    {
    }

    /**
     * @brief Samples for all levels using k-means
     * TODO implement this
     */
    void kmeans_sampling(RXMeshStatic& rx)
    {
    }


    /**
     * @brief compute clustering at all levels in the hierarchy
     */
    void clustering(RXMeshStatic& rx, int l)
    {
        if (l == 1) {
            clustering_1st_level(rx,
                                 1,  // first coarse level
                                 m_vertex_pos,
                                 m_sample_level_bitmask,
                                 m_distance,
                                 m_sample_id[0],
                                 m_vertex_cluster[0],
                                 m_d_flag);
        } else {
            clustering_nth_level(m_num_samples[0],
                                 l,
                                 m_sample_neighbor_size_prefix[l - 1],
                                 m_sample_neighbor[l - 1],
                                 m_vertex_cluster[l],
                                 m_distance_mat[l],
                                 m_sample_id[l],
                                 m_sample_level_bitmask,
                                 m_samples_pos[l - 1],
                                 m_d_flag);
        }
    }

    /**
     * @brief create compressed represent for specific level in the hierarchy
     */
    void create_compressed_representation(RXMeshStatic& rx, uint32_t level)
    {
        constexpr uint32_t blockThreads = 256;

        m_edge_hash_table.clear();

        // a) for each sample, count the number of neighbor samples
        if (level == 1) {
            // if we are building the compressed format for level 1, then we use
            // the mesh itself to tell us who is neighbor to whom on the at
            // level (1)
            rx.run_kernel<blockThreads>(
                {Op::VV},
                detail::count_neighbors_1st_level<blockThreads>,
                m_vertex_cluster[0],
                m_edge_hash_table);
        } else {
            // if we are building the compressed format for any other level,
            // then we use level-1 to tell us how to get the neighbor of that
            // level

            auto& vertex_cluster = m_vertex_cluster[level - 2];
            auto& prv_sample_neighbor_size_prefix =
                m_sample_neighbor_size_prefix[level - 2];
            auto& prv_sample_neighbor = m_sample_neighbor[level - 2];

            // the number of parallel stuff we wanna do is equal to the number
            // of the samples in the previous level

            uint32_t blocks = DIVIDE_UP(
                prv_sample_neighbor_size_prefix.rows() - 1, blockThreads);

            assert(prv_sample_neighbor_size_prefix.rows() - 1 ==
                   m_num_samples[level - 1]);

            for_each_item<<<blocks, blockThreads>>>(
                prv_sample_neighbor_size_prefix.rows() - 1,
                [=] __device__(int i) mutable {
                    int start = prv_sample_neighbor_size_prefix(i);
                    int end   = prv_sample_neighbor_size_prefix(i + 1);
                    for (int j = start; j < end; ++j) {
                        int n = prv_sample_neighbor(j);

                        int a = vertex_cluster(i);
                        int b = vertex_cluster(n);

                        if (a != b) {
                            Edge e(a, b);
                            m_edge_hash_table.insert(e);
                        }
                    }
                });
        }


        // b) iterate over all entries in the hashtable, for non sentinel
        // entries, find
        uint32_t blocks =
            DIVIDE_UP(m_edge_hash_table.get_capacity(), blockThreads);

        auto& sample_neighbor_size = m_sample_neighbor_size[level - 1];

        auto& sample_neighbor_size_prefix =
            m_sample_neighbor_size_prefix[level - 1];

        auto& sample_neighbor = m_sample_neighbor[level - 1];


        for_each_item<<<blocks, blockThreads>>>(
            m_edge_hash_table.get_capacity(),
            [=] __device__(uint32_t i) mutable {
                const Edge e = m_edge_hash_table.m_table[i];

                if (e != Edge::sentinel()) {
                    std::pair<int, int> p = e.unpack();

                    ::atomicAdd(&sample_neighbor_size(p.first), 1);
                    ::atomicAdd(&sample_neighbor_size(p.second), 1);
                }
            });

        // c) compute the exclusive sum of the number of neighbor samples
        cub::DeviceScan::ExclusiveSum(m_d_cub_temp_storage,
                                      m_cub_temp_bytes,
                                      sample_neighbor_size.data(DEVICE),
                                      sample_neighbor_size_prefix.data(DEVICE),
                                      m_num_samples[1] + 1);

        // d) allocate memory to store the neighbor samples
        int s = 0;
        CUDA_ERROR(cudaMemcpy(&s,
                              sample_neighbor_size.data() + m_num_samples[1],
                              sizeof(int),
                              cudaMemcpyDeviceToHost));

        // e) allocate memory for sample_neighbour
        m_sample_neighbor.emplace_back(DenseMatrix<int>(rx, s, 1));

        sample_neighbor_size.reset(0, DEVICE);

        // f) store the neighbor samples in the compressed format
        for_each_item<<<blocks, blockThreads>>>(
            m_edge_hash_table.get_capacity(),
            [=] __device__(uint32_t i) mutable {
                const Edge e = m_edge_hash_table.m_table[i];

                if (e != Edge::sentinel()) {
                    std::pair<int, int> p = e.unpack();

                    // add a to b
                    // and add b to a
                    int a = p.first;
                    int b = p.second;

                    int a_id = ::atomicAdd(&sample_neighbor_size(a), 1);
                    int b_id = ::atomicAdd(&sample_neighbor_size(b), 1);

                    int a_pre = sample_neighbor_size_prefix(a);
                    int b_pre = sample_neighbor_size_prefix(b);

                    sample_neighbor(a_pre + a_id) = b;
                    sample_neighbor(b_pre + b_id) = a;
                }
            });
    }


    /**
     * @brief Create prolongation operator for all levels beyond the 1st level
     */
    void create_all_prolongation()
    {
        for (int level = 1; level < m_num_levels - 1; level++) {

            int current_num_vertices = m_num_samples[level];

            // TODO i think the indexing here is not right
            create_prolongation(current_num_vertices,
                                m_sample_neighbor_size_prefix[level - 1],
                                m_sample_neighbor[level - 1],
                                m_prolong_op[level - 1],
                                m_samples_pos[level],
                                m_vertex_cluster[level - 1]);
        }
    }

    /**
     * \brief Constructs single prolongation operator
     * \param num_samples number of samples in the next coarse level
     * \param row_ptr row pointer for next level csr mesh
     * \param value_ptr column index pointer for next level csr mesh
     * \param operator_value_ptr prolongation operator column index pointer
     * \param operator_data_ptr prolongation operator value pointer
     */
    void create_prolongation(int               num_samples,
                             DenseMatrix<int>& sample_neighbor_size_prefix,
                             DenseMatrix<int>& sample_neighbor,
                             SparseMatrixConstantNNZRow<float, 3>& prolong_op,
                             DenseMatrix<float>&                   samples_pos,
                             DenseMatrix<int>& vertex_cluster)
    {

        uint32_t threads = 256;
        uint32_t blocks  = DIVIDE_UP(num_samples, threads);

        for_each_item<<<blocks, threads>>>(
            num_samples, [=] __device__(int sample_id) mutable {
                // go through every triangle of the cluster
                const int cluster_point = vertex_cluster(sample_id);
                const int start = sample_neighbor_size_prefix(cluster_point);
                const int end = sample_neighbor_size_prefix(cluster_point + 1);

                float min_distance = std::numeric_limits<float>::max();

                Eigen::Vector3<float> selectedv1{0, 0, 0}, selectedv2{0, 0, 0},
                    selectedv3{0, 0, 0};

                const Eigen::Vector3<float> q{samples_pos(sample_id, 0),
                                              samples_pos(sample_id, 1),
                                              samples_pos(sample_id, 2)};

                int selected_neighbor             = 0;
                int selected_neighbor_of_neighbor = 0;

                // We need at least 2 neighbors to form a triangle
                int neighbors_count = end - start;
                if (neighbors_count >= 2) {
                    // Iterate through all possible triangle combinations
                    for (int j = start; j < end; ++j) {
                        for (int k = j + 1; k < end; ++k) {
                            int v1_idx = cluster_point;
                            int v2_idx = sample_neighbor(j);
                            int v3_idx = sample_neighbor(k);

                            // Verify v2 and v3 are connected (are neighbors)
                            bool are_neighbors = false;

                            const int n1_start =
                                sample_neighbor_size_prefix(v2_idx);

                            const int n1_end =
                                sample_neighbor_size_prefix(v2_idx + 1);

                            for (int m = n1_start; m < n1_end; m++) {
                                if (v3_idx == sample_neighbor(m)) {
                                    are_neighbors = true;
                                    break;
                                }
                            }

                            if (are_neighbors) {

                                Eigen::Vector3<float> v1{
                                    samples_pos(v1_idx, 0),
                                    samples_pos(v1_idx, 1),
                                    samples_pos(v1_idx, 2)};

                                Eigen::Vector3<float> v2{
                                    samples_pos(v2_idx, 0),
                                    samples_pos(v2_idx, 1),
                                    samples_pos(v2_idx, 2)};

                                Eigen::Vector3<float> v3{
                                    samples_pos(v3_idx, 0),
                                    samples_pos(v3_idx, 1),
                                    samples_pos(v3_idx, 2)};

                                float distance =
                                    detail::projected_distance(v1, v2, v3, q);

                                if (distance < min_distance) {
                                    min_distance                  = distance;
                                    selectedv1                    = v1;
                                    selectedv2                    = v2;
                                    selectedv3                    = v3;
                                    selected_neighbor             = v2_idx;
                                    selected_neighbor_of_neighbor = v3_idx;
                                }
                            }
                        }
                    }
                }
                assert(selectedv1 != selectedv2 && selectedv2 != selectedv3 &&
                       selectedv3 != selectedv1);

                // Compute barycentric coordinates for the closest triangle
                float b1 = 0, b2 = 0, b3 = 0;
                detail::compute_barycentric(
                    selectedv1, selectedv2, selectedv3, q, b1, b2, b3);

                assert(!isnan(b1));
                assert(!isnan(b2));
                assert(!isnan(b3));

                prolong_op.col_idx()[sample_id * 3 + 0] = cluster_point;
                prolong_op.col_idx()[sample_id * 3 + 1] = selected_neighbor;
                prolong_op.col_idx()[sample_id * 3 + 2] =
                    selected_neighbor_of_neighbor;

                prolong_op.get_val_at(sample_id * 3 + 0) = b1;
                prolong_op.get_val_at(sample_id * 3 + 1) = b2;
                prolong_op.get_val_at(sample_id * 3 + 2) = b3;
            });
    }
};
}  // namespace rxmesh