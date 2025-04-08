#pragma once

#include <cub/device/device_scan.cuh>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix.cuh"

#include "rxmesh/matrix/sparse_matrix_constant_nnz_row.h"

#include "GMG_kernels.h"
#include "NeighborHandling.h"
#include "cluster.h"
#include "fps_sampler.h"
#include "hashtable.h"

#include "polyscope/point_cloud.h"

namespace rxmesh {

enum class Sampling
{
    Rand   = 0,
    FPS    = 1,
    KMeans = 2,
};

void rearrangeDenseMatrix(std::vector<DenseMatrix<int>>& m_sample_neighbor_size,
                          std::vector<DenseMatrix<int>>& m_sample_neighbor)
{
    int       num_rows      = m_sample_neighbor_size[0].rows();
    const int MAX_LOOKAHEAD = 4;

    for (int row = 0; row < num_rows; ++row) {
        int num_neighbors = m_sample_neighbor_size[0](row, 0);
        if (num_neighbors < 2)
            continue;

        for (int i = 0; i < num_neighbors - 1; i++) {
            int current = m_sample_neighbor[0](row, i);
            int next    = m_sample_neighbor[0](row, i + 1);

            bool already_connected = false;
            for (int k = 0; k < m_sample_neighbor_size[0](current, 0); k++) {
                if (m_sample_neighbor[0](current, k) == next) {
                    already_connected = true;
                    break;
                }
            }

            if (!already_connected) {
                int max_look = std::min(i + 1 + MAX_LOOKAHEAD, num_neighbors);
                for (int j = i + 2; j < max_look; j++) {
                    int candidate = m_sample_neighbor[0](row, j);
                    for (int k = 0; k < m_sample_neighbor_size[0](current, 0);
                         k++) {
                        if (m_sample_neighbor[0](current, k) == candidate) {
                            std::swap(m_sample_neighbor[0](row, i + 1),
                                      m_sample_neighbor[0](row, j));
                            goto next_vertex;
                        }
                    }
                }
            }
        next_vertex:
            continue;
        }
    }
}

void GetRenderDataFromDenseMatrices(
    std::vector<std::array<double, 3>>& vertexPositions,
    std::vector<std::vector<size_t>>&   faceIndices,
    std::vector<DenseMatrix<int>>&      m_sample_neighbor_size,
    std::vector<DenseMatrix<int>>&      m_sample_neighbor,
    DenseMatrix<float>&                 vertex_pos)
{
    rearrangeDenseMatrix(m_sample_neighbor_size, m_sample_neighbor);

    int num_rows = m_sample_neighbor_size[0].rows();

    vertexPositions.resize(num_rows);
    for (int i = 0; i < num_rows; ++i) {
        vertexPositions[i] = {
            vertex_pos(i, 0), vertex_pos(i, 1), vertex_pos(i, 2)};
    }

    faceIndices.clear();
    for (int i = 0; i < num_rows; ++i) {
        int num_neighbors = m_sample_neighbor_size[0](i, 0);
        if (num_neighbors >= 2) {
            for (int j = 0; j < num_neighbors; ++j) {
                int a = i;
                int b = m_sample_neighbor[0](i, j);
                int c = (j == num_neighbors - 1) ?
                            m_sample_neighbor[0](i, 0) :
                            m_sample_neighbor[0](i, j + 1);

                for (int k = 0; k < m_sample_neighbor_size[0](b, 0); ++k) {
                    if (m_sample_neighbor[0](b, k) == c) {
                        faceIndices.push_back({static_cast<size_t>(a),
                                               static_cast<size_t>(b),
                                               static_cast<size_t>(c)});
                        break;
                    }
                }
            }
        }
    }
}

void renderFromDenseMatrices(
    std::vector<DenseMatrix<int>>& m_sample_neighbor_size,
    std::vector<DenseMatrix<int>>& m_sample_neighbor,
    DenseMatrix<float>&            vertex_pos)
{
    std::vector<std::array<double, 3>> vertexPositions;
    std::vector<std::vector<size_t>>   faceIndices;

    GetRenderDataFromDenseMatrices(vertexPositions,
                                   faceIndices,
                                   m_sample_neighbor_size,
                                   m_sample_neighbor,
                                   vertex_pos);

    polyscope::registerSurfaceMesh(
        "dense matrix mesh", vertexPositions, faceIndices);
}


template <typename T>
struct GMG
{
    // In indexing, we always using L=0 to refer to the fine mesh.
    // The index of the first coarse level is L=1.

    GMG(const GMG&) = delete;

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

    std::vector<DenseMatrix<float>> m_distance_mat;    // fine+levels
    std::vector<DenseMatrix<int>>   m_vertex_cluster;  // levels

    DenseMatrix<float>     m_vertex_pos;            // fine
    VertexAttribute<float> m_distance;              // fine
    DenseMatrix<uint16_t>  m_sample_level_bitmask;  // fine

    GPUHashTable<Edge> m_edge_hash_table;


    // we reuse cub temp storage for all level (since level has the largest
    // storage) and only re-allocate if we have to
    void*  m_d_cub_temp_storage;
    size_t m_cub_temp_bytes;


    GMG(RXMeshStatic& rx,
        Sampling      sam                   = Sampling::FPS,
        int           reduction_ratio       = 20,
        int           num_samples_threshold = 7)
        : m_ratio(reduction_ratio),
          m_edge_hash_table(
              GPUHashTable<Edge>(DIVIDE_UP(rx.get_num_edges(), 2)))
    {
        m_num_rows = rx.get_num_vertices();
        // m_num_samples.push_back(m_num_rows);
        for (int i = 0; i < 16; i++) {
            int s = DIVIDE_UP(m_num_rows, std::pow(m_ratio, i));
            if (s > num_samples_threshold) {
                m_num_samples.push_back(s);
                std::cout << "\nNumber of samples for level " << i << ": " << s;
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

                // we allocate +1 for cub prefix sum
                m_sample_neighbor_size.emplace_back(
                    rx, level_num_samples + 1, 1);
                m_sample_neighbor_size.back().reset(0, DEVICE);

                m_sample_neighbor_size_prefix.emplace_back(
                    rx, level_num_samples + 1, 1);
                m_sample_neighbor_size_prefix.back().reset(0, DEVICE);
            }
            if (l < m_num_samples.size() - 1) {
                m_vertex_cluster.emplace_back(rx, level_num_samples, 1);
                m_vertex_cluster.back().reset(-1, LOCATION_ALL);

                m_prolong_op.emplace_back(
                    rx, level_num_samples, m_num_samples[l + 1]);
            }


            m_distance_mat.emplace_back(rx, level_num_samples, 1);
        }


        m_vertex_pos = *rx.get_input_vertex_coordinates()->to_matrix();
        m_distance   = *rx.add_vertex_attribute<float>("d", 1);


        m_sample_level_bitmask = DenseMatrix<uint16_t>(rx, m_num_rows, 1);

        CUDA_ERROR(cudaMalloc((void**)&m_d_flag, sizeof(int)));

        // allocate CUB stuff here
        m_cub_temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            nullptr,
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

        // renderFromDenseMatrices(
        //     m_sample_neighbor_size, m_sample_neighbor, m_vertex_pos);

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
                   m_vertex_cluster[0],
                   m_sample_level_bitmask,
                   m_samples_pos[0],
                   m_ratio,
                   m_num_rows,
                   m_num_levels,
                   m_num_samples[1],
                   m_d_flag);

        constexpr uint32_t blockThreads = 256;

        for (int level = 2; level < m_num_levels; ++level) {
            uint32_t blocks = DIVIDE_UP(m_num_samples[level], blockThreads);

            auto& current_samples_pos = m_samples_pos[level - 1];

            const auto& prv_samples_pos = m_samples_pos[level - 2];

            auto& current_v_cluster = m_vertex_cluster[level - 1];

            const auto& prv_v_cluster = m_vertex_cluster[level - 2];

            // if (level == 2) {
            //     // when we are at level 2, we are reading from level 1. Level
            //     // 1 samples are scatter across the mesh vertices. Thus, only
            //     // for level 2, we read from the mesh and try to populate the
            //     // samples position and vertex cluster
            //     int num_samples = m_num_samples[level];
            //     rx.for_each_vertex(
            //         DEVICE,
            //         [num_samples, current_v_cluster, prv_v_cluster]
            //         __device__(
            //             const VertexHandle vh) mutable {
            //             int tid = blockIdx.x * blockDim.x + threadIdx.x;
            //             if (tid < num_samples && prv_v_cluster(vh) != -1) {
            //                 current_v_cluster(tid, 0) = prv_v_cluster(vh, 0);
            //             }
            //         });
            // } else {
            //     // for other levels, we always take the first N samples from
            //     the
            //     // previous level (where N is the m_num_samples[level])
            //     because
            //     // in the previous level, the first M samples are the one
            //     from
            //     // FPS (where M is m_num_samples[level-1])
            //
            //     for_each_item<<<blocks, blockThreads>>>(
            //         m_num_samples[level],
            //         [current_v_cluster,
            //          prv_v_cluster] __device__(int i) mutable {
            //             current_v_cluster(i, 0) = prv_v_cluster(i, 0);
            //         });
            // }

            for_each_item<<<blocks, blockThreads>>>(
                m_num_samples[level],
                [current_v_cluster,
                 prv_v_cluster,
                 current_samples_pos,
                 prv_samples_pos] __device__(int i) mutable {
                    current_v_cluster(i, 0) = prv_v_cluster(i, 0);

                    current_samples_pos(i, 0) = prv_samples_pos(i, 0);
                    current_samples_pos(i, 1) = prv_samples_pos(i, 1);
                    current_samples_pos(i, 2) = prv_samples_pos(i, 2);
                });
        }

        // m_sample_level_bitmask.move(DEVICE, HOST);
        // std::vector<int> num_samples(m_num_samples.size());
        // num_samples.resize(m_num_samples.size(), 0);
        // rx.for_each_vertex(
        //     HOST,
        //     [&](VertexHandle vh) {
        //         if (m_sample_id[0](vh) == -1) {
        //             return;
        //         }
        //
        //         for (int i = 1; i < num_samples.size(); ++i) {
        //             if ((m_sample_level_bitmask(vh) & (1 << i - 1)) != 0) {
        //                 num_samples[i]++;
        //             }
        //         }
        //     },
        //     nullptr,
        //     false);

        // DEBUG Code
        for (int l = 1; l < m_num_levels; ++l) {

            m_vertex_cluster[l - 1].move(DEVICE, HOST);
            m_distance.move(DEVICE, HOST);

            if (l == 1) {
                auto at =
                    *rx.add_vertex_attribute<int>("C" + std::to_string(l), 1);
                at.from_matrix(&m_vertex_cluster[l - 1]);
                rx.get_polyscope_mesh()->addVertexScalarQuantity(
                    "C" + std::to_string(l), at);

                rx.get_polyscope_mesh()->addVertexScalarQuantity(
                    "dist" + std::to_string(l), m_distance);
            } else {
                std::vector<glm::vec3> points(m_num_samples[l]);

                m_samples_pos[l - 1].move(DEVICE, HOST);

                for (int i = 0; i < m_samples_pos[l - 1].rows(); i++) {
                    points[i][0] = m_samples_pos[l - 1](i, 0);
                    points[i][1] = m_samples_pos[l - 1](i, 1);
                    points[i][2] = m_samples_pos[l - 1](i, 2);
                }

                std::vector<int> xC(points.size());
                for (int i = 0; i < points.size(); i++) {
                    xC[i] = m_vertex_cluster[l - 1](i);
                }

                polyscope::PointCloud* psCloud = polyscope::registerPointCloud(
                    "L" + std::to_string(l), points);
                psCloud->addScalarQuantity("C" + std::to_string(l), xC);
            }

            polyscope::show();
        }
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
                                 m_vertex_cluster[l - 1],  // 0
                                 m_d_flag);
        } else {
            clustering_nth_level(m_num_samples[l - 1],
                                 l,
                                 m_sample_neighbor_size_prefix[l - 2],
                                 m_sample_neighbor[l - 2],
                                 m_vertex_cluster[l - 1],
                                 m_distance_mat[l - 1],
                                 m_sample_level_bitmask,
                                 m_samples_pos[0],
                                 m_samples_pos[0],
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

        auto edge_hash_table = m_edge_hash_table;

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

            auto& vertex_cluster = m_vertex_cluster[level - 1];
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
                [edge_hash_table,
                 prv_sample_neighbor_size_prefix,
                 prv_sample_neighbor,
                 vertex_cluster] __device__(int i) mutable {
                    int start = prv_sample_neighbor_size_prefix(i);
                    int end   = prv_sample_neighbor_size_prefix(i + 1);
                    for (int j = start; j < end; ++j) {
                        int n = prv_sample_neighbor(j);

                        int a = vertex_cluster(i);
                        int b = vertex_cluster(n);

                        if (a != b) {
                            Edge e(a, b);
                            edge_hash_table.insert(e);
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

        for_each_item<<<blocks, blockThreads>>>(
            m_edge_hash_table.get_capacity(),
            [edge_hash_table,
             sample_neighbor_size] __device__(uint32_t i) mutable {
                const Edge e = edge_hash_table.m_table[i];
                if (!e.is_sentinel()) {
                    std::pair<int, int> p = e.unpack();
                    assert(p.first < sample_neighbor_size.rows());

                    ::atomicAdd(&sample_neighbor_size(p.first), 1);
                    ::atomicAdd(&sample_neighbor_size(p.second), 1);
                }
            });

        int* h_sample_neighbor_size = new int[m_num_samples[level] + 1];
        cudaMemcpy(h_sample_neighbor_size,
                   sample_neighbor_size.data(DEVICE),
                   (m_num_samples[level] + 1) * sizeof(int),
                   cudaMemcpyDeviceToHost);

        delete[] h_sample_neighbor_size;
        // c) compute the exclusive sum of the number of neighbor samples
        cudaError_t err = cub::DeviceScan::ExclusiveSum(
            m_d_cub_temp_storage,
            m_cub_temp_bytes,
            sample_neighbor_size.data(DEVICE),
            sample_neighbor_size_prefix.data(DEVICE),
            m_num_samples[level] + 1);
        cudaDeviceSynchronize();  // Ensure execution completes

        if (err != cudaSuccess) {
            std::cerr << "ExclusiveSum failed: " << cudaGetErrorString(err)
                      << std::endl;
        }


        // d) allocate memory to store the neighbor samples
        int s = 0;
        CUDA_ERROR(cudaMemcpy(
            &s,
            sample_neighbor_size_prefix.data() + m_num_samples[level],
            sizeof(int),
            cudaMemcpyDeviceToHost));

        // e) allocate memory for sample_neighbour
        m_sample_neighbor.emplace_back(DenseMatrix<int>(rx, s, 1));
        auto& sample_neighbor = m_sample_neighbor[level - 1];

        sample_neighbor_size.reset(0, DEVICE);

        // f) store the neighbor samples in the compressed format
        for_each_item<<<blocks, blockThreads>>>(
            m_edge_hash_table.get_capacity(),
            [edge_hash_table,
             sample_neighbor_size,
             sample_neighbor_size_prefix,
             sample_neighbor] __device__(uint32_t i) mutable {
                const Edge e = edge_hash_table.m_table[i];

                if (!e.is_sentinel()) {
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
        for (int level = 1; level < m_num_levels; level++) {

            int current_num_vertices = m_num_samples[level - 1];

            create_prolongation(current_num_vertices,
                                m_sample_neighbor_size_prefix[level - 1],
                                m_sample_neighbor[level - 1],
                                m_prolong_op[level - 1],
                                m_samples_pos[0],
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
                // assert(sample_id < num_samples);

                // go through every triangle of the cluster
                const int cluster_point = vertex_cluster(sample_id);
                const int start = sample_neighbor_size_prefix(cluster_point);
                const int end = sample_neighbor_size_prefix(cluster_point + 1);

                float min_distance = std::numeric_limits<float>::max();

                Eigen::Vector3<float> selectedv1{0, 0, 0}, selectedv2{0, 0, 0},
                    selectedv3{0, 0, 0};

                const Eigen::Vector3<float> q{samples_pos(cluster_point, 0),
                                              samples_pos(cluster_point, 1),
                                              samples_pos(cluster_point, 2)};

                /*printf("\nQ %d %d %f %f %f",
                       sample_id,
                       cluster_point,
                       q[0],
                       q[1],
                       q[2]);*/

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
                if (selected_neighbor_of_neighbor != 0 &&
                    selected_neighbor != 0) {
                    assert(selectedv1 != selectedv2 &&
                           selectedv2 != selectedv3 &&
                           selectedv3 != selectedv1);
                }
                // Compute barycentric coordinates for the closest triangle
                float b1 = 0, b2 = 0, b3 = 0;
                /*if (selected_neighbor == selected_neighbor_of_neighbor &&
                    selected_neighbor == 0) {
                    b1 = 1.0f;
                } else*/
                detail::compute_barycentric(
                    selectedv1, selectedv2, selectedv3, q, b1, b2, b3);

                printf("\n%d: %d %d %d %f %f %f",
                       sample_id,
                       cluster_point,
                       selected_neighbor,
                       selected_neighbor_of_neighbor,
                       b1,
                       b2,
                       b3);

                assert(!isnan(b2));

                prolong_op.col_idx()[sample_id * 3 + 2] =
                    selected_neighbor_of_neighbor;
                prolong_op.get_val_at(sample_id * 3 + 2) = b3;

                prolong_op.col_idx()[sample_id * 3 + 1]  = selected_neighbor;
                prolong_op.get_val_at(sample_id * 3 + 1) = b2;

                prolong_op.col_idx()[sample_id * 3 + 0]  = cluster_point;
                prolong_op.get_val_at(sample_id * 3 + 0) = b1;
            });
    }
};
}  // namespace rxmesh