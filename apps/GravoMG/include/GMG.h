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

namespace rxmesh {

template <typename T>
struct GMG
{
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


    // we reuse cub temp storage for all level (since level has the largest
    // storage) and only re-allocate if we have to
    void*  m_d_cub_temp_storage;
    size_t m_cub_temp_bytes;

    // mutex used in populating neighbor samples. allocated based on samples
    // on first level and then used in other levels
    DenseMatrix<int> m_mutex;


    GMG(RXMeshStatic&    rx,
        SparseMatrix<T>& A,
        DenseMatrix<T>&  B,
        int              reduction_ratio       = 7,
        int              num_samples_threshold = 6)
        : m_ratio(reduction_ratio)
    {
        constexpr uint32_t blockThreads = 256;

        m_num_rows = A.rows();

        m_num_samples.push_back(m_num_rows);
        for (int i = 0; i < 16; i++) {
            int s = DIVIDE_UP(m_num_rows, std::pow(m_ratio, i));
            if (s > num_samples_threshold) {
                m_num_samples.push_back(s);
            }
        }
        m_num_levels = m_num_samples.size();


        // init variables and alloc memory

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

        m_mutex = DenseMatrix<int>(rx, m_num_samples[0], 1);


        // allocate CUB stuff here
        m_cub_temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            m_d_cub_temp_storage,
            m_cub_temp_bytes,
            m_sample_neighbor_size[0].data(DEVICE),
            m_sample_neighbor_size_prefix[0].data(DEVICE),
            m_num_samples[0] + 1);
        CUDA_ERROR(cudaMalloc((void**)&m_d_cub_temp_storage, m_cub_temp_bytes));


        // =====================
        // ====== Level 1 ======
        // =====================

        // 1) FPS sampling
        FPSSampler(rx,
                   m_distance,
                   m_vertex_pos,
                   m_sample_id[0],
                   m_sample_level_bitmask,
                   m_samples_pos[0],
                   m_ratio,
                   m_num_rows,
                   m_num_levels,
                   m_num_samples[0],
                   m_d_flag);

        // 2) clustering
        clustering_1st_level(rx,
                             1,  // first coarse level
                             m_vertex_pos,
                             m_sample_level_bitmask,
                             m_distance,
                             m_sample_id[0],
                             m_vertex_cluster[0],
                             m_d_flag);

        // 3) create compressed representation of
        // 3.a) for each sample, count the number of neighbor samples
        // TODO this need to be fixed
        rx.run_kernel<blockThreads>(
            {Op::VV},
            detail::count_num_neighbor_samples<blockThreads>,
            m_vertex_cluster[0],
            m_sample_neighbor_size[0],
            m_mutex);


        // 3.b) compute the exclusive sum of the number of neighbor samples
        cub::DeviceScan::ExclusiveSum(
            m_d_cub_temp_storage,
            m_cub_temp_bytes,
            m_sample_neighbor_size[0].data(DEVICE),
            m_sample_neighbor_size_prefix[0].data(DEVICE),
            m_num_samples[0] + 1);

        // 3.c) allocate memory to store the neighbor samples
        int s = 0;
        CUDA_ERROR(
            cudaMemcpy(&s,
                       m_sample_neighbor_size[0].data() + m_num_samples[0],
                       sizeof(int),
                       cudaMemcpyDeviceToHost));

        m_sample_neighbor.emplace_back(DenseMatrix<int>(rx, s, 1));

        // 3.d) store the neighbor samples in the compressed format
        // TODO this need to be fixed
        rx.run_kernel<blockThreads>(
            {Op::VV},
            detail::populate_neighbor_samples<blockThreads>,
            m_vertex_cluster[0],
            m_sample_neighbor_size[0],
            m_sample_neighbor_size_prefix[0],
            m_sample_neighbor[0],
            m_mutex);


        // move prolongation operator from device to host
        for (auto& prolong_op : m_prolong_op) {
            prolong_op.move_col_idx(DEVICE, HOST);
            prolong_op.move(DEVICE, HOST);
        }

        create_all_prolongation();

        // release memory
        GPU_FREE(m_d_flag);
        GPU_FREE(m_d_cub_temp_storage);
    }

    /**
     * @brief Create prolongation operator for all levels beyond the 1st level
     */
    void create_all_prolongation()
    {
        int current_num_vertices = m_num_rows;
        int current_num_samples  = m_num_samples[0];

        for (int level = 1; level < m_num_levels - 1; level++) {

            int current_num_vertices = m_num_samples[level];
            int current_num_samples  = m_num_samples[level + 1];

            // TODO i think the indexing here is not right
            clustering_nth_level(
                current_num_vertices,
                level + 1,
                m_sample_neighbor_size[level],
                m_sample_neighbor[level],
                m_vertex_cluster[level],
                m_distance_mat[level],  // TODO figure out this distance
                m_sample_id[level],
                m_sample_level_bitmask,
                m_samples_pos[level],
                m_d_flag);

            // TODO i think the indexing here is not right
            create_prolongation(current_num_vertices,
                                m_sample_neighbor_size_prefix[level],
                                m_sample_neighbor[level],
                                m_prolong_op[level],
                                m_samples_pos[level],
                                m_vertex_cluster[level]);
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