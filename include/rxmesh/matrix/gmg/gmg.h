#pragma once

#include <cub/device/device_scan.cuh>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/sparse_matrix_constant_nnz_row.h"


#include "rxmesh/matrix/gmg/cluster.h"
#include "rxmesh/matrix/gmg/fps_sampler.h"
#include "rxmesh/matrix/gmg/gmg_kernels.h"
#include "rxmesh/matrix/gmg/hashtable.h"

#include "polyscope/point_cloud.h"

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

    GMG(const GMG&) = delete;
    GMG()           = default;
    GMG(GMG&&)      = default;
    GMG& operator=(const GMG&) = default;
    GMG& operator=(GMG&&) = default;
    ~GMG()                = default;

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

    GPUStorage<Edge> m_edge_storage;


    // we reuse cub temp storage for all level (since level has the largest
    // storage) and only re-allocate if we have to
    void*  m_d_cub_temp_storage;
    size_t m_cub_temp_bytes;

    GMG(RXMeshStatic& rx,
        int           numberOfLevels,
        int threshold = 1000,
        Sampling      sam       = Sampling::FPS
        )
        : GMG(rx,
              sam,
              compute_ratio(rx.get_num_vertices(), numberOfLevels, threshold),
              threshold)
    {
    }

    static int compute_ratio(int n, int numberOfLevels, int threshold)
    {
        for (int j = 2; j < 1000; j++) {
            int a = DIVIDE_UP(n, std::pow(j, numberOfLevels - 1));
            if (a <= threshold)
                return j - 1;
        }
        return 999;
    }



    GMG(RXMeshStatic& rx,
        Sampling      sam                   = Sampling::Rand,
        int           reduction_ratio       = 4,
        int           num_samples_threshold = 20)
        : m_ratio(reduction_ratio),
          m_edge_storage(GPUStorage<Edge>(rx.get_num_edges()))
    {
        m_num_rows = rx.get_num_vertices();
        // m_num_samples.push_back(m_num_rows);
        for (int i = 0; i < 16; i++) {
            int s = DIVIDE_UP(m_num_rows, std::pow(m_ratio, i));
            if (s > num_samples_threshold) {
                m_num_samples.push_back(s);
                RXMESH_TRACE("GMG: #samples at level {}: {}", i, s);
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

                m_distance_mat.emplace_back(rx, level_num_samples, 1);
                m_distance_mat.back().reset(std::numeric_limits<float>::max(),
                                            LOCATION_ALL);
            }
            if (l < m_num_samples.size() - 1) {
                m_vertex_cluster.emplace_back(rx, level_num_samples, 1);
                m_vertex_cluster.back().reset(-1, LOCATION_ALL);

                m_prolong_op.emplace_back(
                    rx, level_num_samples, m_num_samples[l + 1]);
            }
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


        for (int l = 0; l < m_num_levels - 1; ++l) {

            //============
            // 3) Clustering
            //============

            clustering(rx, l);


            //============
            // 4) Create coarse mesh compressed representation of
            //============

            create_compressed_representation(rx, l + 1);
        }

        // render_hierarchy();
        // render_point_clouds(rx);

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

        // re-init m_distance because it is used in clustering
        auto& distance = m_distance;

        const auto& vc = m_vertex_cluster[0];

        rx.for_each_vertex(
            DEVICE, [distance, vc] __device__(const VertexHandle vh) mutable {
                if (vc(vh) == -1) {
                    distance(vh, 0) = std::numeric_limits<float>::max();
                }
            });

        for (int level = 2; level < m_num_levels; ++level) {
            uint32_t blocks = DIVIDE_UP(m_num_samples[level], blockThreads);

            auto& current_samples_pos = m_samples_pos[level - 1];

            const auto& prv_samples_pos = m_samples_pos[level - 2];

            auto& current_v_cluster = m_vertex_cluster[level - 1];

            const auto& prv_v_cluster = m_vertex_cluster[level - 2];

            const auto& pos = m_vertex_pos;

            // set sample position of this level
            for_each_item<<<blocks, blockThreads>>>(
                m_num_samples[level],
                [current_samples_pos,
                 prv_samples_pos] __device__(int i) mutable {
                    current_samples_pos(i, 0) = prv_samples_pos(i, 0);
                    current_samples_pos(i, 1) = prv_samples_pos(i, 1);
                    current_samples_pos(i, 2) = prv_samples_pos(i, 2);
                });

            // set vertex cluster of the previous level
            blocks = DIVIDE_UP(m_num_samples[level - 1], blockThreads);

            auto& prv_level_distance = m_distance_mat[level - 2];
            for_each_item<<<blocks, blockThreads>>>(
                m_num_samples[level - 1],
                [current_v_cluster,
                 prv_level_distance,
                 n = m_num_samples[level]] __device__(int i) mutable {
                    if (i < n) {
                        current_v_cluster(i, 0)  = i;
                        prv_level_distance(i, 0) = 0;
                    } else {
                        current_v_cluster(i, 0) = -1;
                        prv_level_distance(i, 0) =
                            std::numeric_limits<float>::max();
                    }
                });
        }
    }

    void render_point_clouds(RXMeshStatic& rx)
    {
        // DEBUG Code
        for (int l = 1; l < m_num_levels; ++l) {

            m_vertex_cluster[l - 1].move(DEVICE, HOST);
            m_distance.move(DEVICE, HOST);

            if (l == 1) {
                auto at =
                    *rx.add_vertex_attribute<int>("C" + std::to_string(l), 1);
                at.from_matrix(&m_vertex_cluster[l - 1]);
                rx.get_polyscope_mesh()->addVertexScalarQuantity(
                    "C" + std::to_string(l - 1), at);

                rx.get_polyscope_mesh()->addVertexScalarQuantity(
                    "dist" + std::to_string(l), m_distance);
                rx.remove_attribute("C" + std::to_string(l));
            } else {

                // Level L points
                std::vector<glm::vec3> points(m_num_samples[l]);
                m_samples_pos[l - 1].move(DEVICE, HOST);
                for (int i = 0; i < m_samples_pos[l - 1].rows(); i++) {
                    points[i][0] = m_samples_pos[l - 1](i, 0);
                    points[i][1] = m_samples_pos[l - 1](i, 1);
                    points[i][2] = m_samples_pos[l - 1](i, 2);
                }

                polyscope::PointCloud* psCloud = polyscope::registerPointCloud(
                    "L" + std::to_string(l), points);
                psCloud->setPointRadius(0.02);

                // std::vector<int> xC(points.size());
                // for (int i = 0; i < points.size(); i++) {
                //     xC[i] = m_vertex_cluster[l - 1](i);
                // }
                //  psCloud->setPointRadiusaddScalarQuantity("C" +
                //  std::to_string(l), xC);

                // Level L-1 clustering
                std::vector<glm::vec3> prv_points(m_num_samples[l - 1]);
                m_samples_pos[l - 2].move(DEVICE, HOST);
                for (int i = 0; i < m_samples_pos[l - 2].rows(); i++) {
                    prv_points[i][0] = m_samples_pos[l - 2](i, 0);
                    prv_points[i][1] = m_samples_pos[l - 2](i, 1);
                    prv_points[i][2] = m_samples_pos[l - 2](i, 2);
                }

                std::vector<int> prv_xC(prv_points.size());
                for (int i = 0; i < prv_points.size(); i++) {
                    prv_xC[i] = m_vertex_cluster[l - 1](i);
                }
                polyscope::PointCloud* prv_psCloud =
                    polyscope::registerPointCloud("LC" + std::to_string(l - 1),
                                                  prv_points);
                prv_psCloud->setPointRadius(0.01);
                prv_psCloud
                    ->addScalarQuantity("C" + std::to_string(l - 1), prv_xC)
                    ->setEnabled(true);
            }
        }
        polyscope::show();
    }


    void render_hierarchy()
    {
        for (int l = 1; l < m_num_levels; ++l) {

            auto& prefix      = m_sample_neighbor_size_prefix[l - 1];
            auto& neighbor    = m_sample_neighbor[l - 1];
            auto& v_pos       = m_samples_pos[l - 1];
            int   num_samples = m_num_samples[l];


            prefix.move(DEVICE, HOST);
            neighbor.move(DEVICE, HOST);
            v_pos.move(DEVICE, HOST);

            auto is_connected = [&](int u, int s) {
                int u_start = prefix(u);
                int u_end   = prefix(u + 1);
                for (int i = u_start; i < u_end; ++i) {
                    if (neighbor(i) == s) {
                        return true;
                    }
                }

                return false;
            };

            std::vector<std::array<int, 3>> fv;

            std::vector<glm::vec3> pos(num_samples);

            // Neighbor samples are not sorted. Thus we do,
            // For each samples v and its neighbor u, we iterate over v
            // neighbors and try to find the samples s such that s is also
            // connected to u. There should be two s samples (s0, s1). We only
            // render a triangle u, v, s if u is the smallest index (to avoid
            // rendering a triangle multiple times)

            for (int v = 0; v < num_samples; ++v) {
                pos[v][0] = v_pos(v, 0);
                pos[v][1] = v_pos(v, 1);
                pos[v][2] = v_pos(v, 2);

                int start = prefix(v);
                int end   = prefix(v + 1);

                for (int i = start; i < end; ++i) {
                    int u = neighbor(i);
                    assert(u != v);
                    if (v > u) {
                        continue;
                    }
                    // now we iterate over v neighbor
                    for (int j = start; j < end; ++j) {
                        if (i == j) {
                            continue;
                        }
                        int s = neighbor(j);

                        assert(v != s);
                        assert(u != s);

                        if (v > s) {
                            continue;
                        }

                        if (is_connected(s, u)) {
                            std::array<int, 3> t = {u, v, s};
                            fv.push_back(t);
                        }
                    }
                }
            }

            polyscope::registerSurfaceMesh(
                "Level" + std::to_string(l), pos, fv);
        }
        polyscope::show();
    }

    /**
     * @brief Samples for all levels using random sampling
     * TODO implement this
     */
    void random_sampling(RXMeshStatic& rx)
    {
        constexpr uint32_t blockThreads = 256;
        bool nested = true;
        int  max    = 1;
        if (!nested) {
            max = m_num_levels - 1;
        }
        m_vertex_pos.move(DEVICE, HOST);
        for (int i = 0; i < max; i++) {
            // re-init m_distance because it is used in clustering
            auto& distance = m_distance;
            const auto& vc = m_vertex_cluster[i];
            if (i==0)
            rx.for_each_vertex(
                DEVICE,
                [distance, vc] __device__(const VertexHandle vh) mutable {
                    //vc(vh,0) = -1;
                    distance(vh, 0) = std::numeric_limits<float>::max();
                });
            distance.move(DEVICE, HOST);
            std::random_device         rd;
            std::default_random_engine generator(rd());
            auto&            current_vertex_cluster = m_vertex_cluster[i];
            std::vector<int> samples(m_num_samples[i]);
            std::iota(samples.begin(), samples.end(), 1);
            std::shuffle(samples.begin(), samples.end(), generator);
            samples.resize(m_num_samples[i + 1]);
            int j = 0;
            // TODO parallelise
            for (auto& a : samples) {
                current_vertex_cluster(a - 1, 0) = j;
                if(i==0) distance(a - 1, 0)               = 0;
                else m_distance_mat[i](a - 1, 0)       = 0;
                
                m_samples_pos[i](j, 0) = m_vertex_pos(a - 1, 0);
                m_samples_pos[i](j, 1) = m_vertex_pos(a - 1, 1);
                m_samples_pos[i](j, 2) = m_vertex_pos(a - 1, 2);
                j++;
            }
            m_samples_pos[i].move(HOST, DEVICE);
            m_vertex_cluster[i].move(HOST, DEVICE);
            m_distance_mat[i].move(HOST, DEVICE);
            distance.move(HOST, DEVICE);
        }
        if (nested) {
            for (int level = 2; level < m_num_levels; ++level) {
                uint32_t blocks = DIVIDE_UP(m_num_samples[level], blockThreads);
                auto& current_samples_pos = m_samples_pos[level - 1];
                const auto& prv_samples_pos = m_samples_pos[level - 2];
                auto& current_v_cluster = m_vertex_cluster[level - 1];
                const auto& prv_v_cluster = m_vertex_cluster[level - 2];
                const auto& pos = m_vertex_pos;
                // set sample position of this level
                for_each_item<<<blocks, blockThreads>>>(
                    m_num_samples[level],
                    [current_samples_pos,
                     prv_samples_pos] __device__(int i) mutable {
                        current_samples_pos(i, 0) = prv_samples_pos(i, 0);
                        current_samples_pos(i, 1) = prv_samples_pos(i, 1);
                        current_samples_pos(i, 2) = prv_samples_pos(i, 2);
                    });
                // set vertex cluster of the previous level
                blocks = DIVIDE_UP(m_num_samples[level - 1], blockThreads);
                auto& prv_level_distance = m_distance_mat[level - 2];
                for_each_item<<<blocks, blockThreads>>>(
                    m_num_samples[level - 1],
                    [current_v_cluster,
                     prv_level_distance,
                     n = m_num_samples[level]] __device__(int i) mutable {
                        if (i < n) {
                            current_v_cluster(i, 0)  = i;
                            prv_level_distance(i, 0) = 0;
                        } else {
                            current_v_cluster(i, 0) = -1;
                            prv_level_distance(i, 0) =
                                std::numeric_limits<float>::max();
                        }
                    });
            }
        }
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
        if (l == 0) {
            clustering_1st_level(rx,
                                 1,  // first coarse level
                                 m_vertex_pos,
                                 m_sample_level_bitmask,
                                 m_distance,
                                 m_vertex_cluster[l],  // 0
                                 m_d_flag);
        } else {
            clustering_nth_level(m_num_samples[l],
                                 l + 1,
                                 m_sample_neighbor_size_prefix[l - 1],
                                 m_sample_neighbor[l - 1],
                                 m_vertex_cluster[l],
                                 m_distance_mat[l - 1],
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

        m_edge_storage.clear();

        auto edge_storage = m_edge_storage;

        // a) for each sample, count the number of neighbor samples
        if (level == 1) {
            // if we are building the compressed format for level 1, then we use
            // the mesh itself to tell us who is neighbor to whom on the at
            // level (1)
            rx.run_kernel<blockThreads>(
                {Op::EV},
                detail::populate_edge_hashtable_1st_level<blockThreads>,
                m_vertex_cluster[0],
                m_edge_storage);
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
                [edge_storage,
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
                            edge_storage.insert(e);
                        }
                    }
                });
        }


        // a.2) uniquify the storage
        m_edge_storage.uniquify();


        // b) iterate over all entries in the edge hashtable, for non sentinel
        // entries, atomic add each vertex to one another

        auto& sample_neighbor_size = m_sample_neighbor_size[level - 1];
        auto& sample_neighbor_size_prefix =
            m_sample_neighbor_size_prefix[level - 1];

        m_edge_storage.for_each(
            [sample_neighbor_size] __device__(Edge e) mutable {
                if (!e.is_sentinel()) {
                    std::pair<int, int> p = e.unpack();
                    assert(p.first < sample_neighbor_size.rows());

                    ::atomicAdd(&sample_neighbor_size(p.first), 1);
                    ::atomicAdd(&sample_neighbor_size(p.second), 1);
                }
            });


        // c) compute the exclusive sum of the number of neighbor samples
        CUDA_ERROR(cub::DeviceScan::ExclusiveSum(
            m_d_cub_temp_storage,
            m_cub_temp_bytes,
            sample_neighbor_size.data(DEVICE),
            sample_neighbor_size_prefix.data(DEVICE),
            m_num_samples[level] + 1));

        // cudaDeviceSynchronize();  // Ensure execution completes

        // d) allocate memory for sample_neighbour
        int s = 0;
        CUDA_ERROR(cudaMemcpy(
            &s,
            sample_neighbor_size_prefix.data() + m_num_samples[level],
            sizeof(int),
            cudaMemcpyDeviceToHost));


        m_sample_neighbor.emplace_back(DenseMatrix<int>(rx, s, 1));
        auto& sample_neighbor = m_sample_neighbor[level - 1];

        sample_neighbor_size.reset(0, DEVICE);

        // e) store the neighbor samples in the compressed format
        m_edge_storage.for_each([sample_neighbor_size,
                                 sample_neighbor_size_prefix,
                                 sample_neighbor] __device__(Edge e) mutable {
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

            if (level == 1)
                create_prolongation(current_num_vertices,
                                    m_sample_neighbor_size_prefix[level - 1],
                                    m_sample_neighbor[level - 1],
                                    m_prolong_op[level - 1],
                                    m_samples_pos[level - 1],
                                    m_vertex_cluster[level - 1],
                                    m_vertex_pos);

            else
                create_prolongation(current_num_vertices,
                                    m_sample_neighbor_size_prefix[level - 1],
                                    m_sample_neighbor[level - 1],
                                    m_prolong_op[level - 1],
                                    m_samples_pos[level - 1],
                                    m_vertex_cluster[level - 1],
                                    m_samples_pos[level - 2]);
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
                             DenseMatrix<int>&   vertex_cluster,
                             DenseMatrix<float>& vertex_pos)
    {

        uint32_t threads = 256;
        uint32_t blocks  = DIVIDE_UP(num_samples, threads);
        for_each_item<<<blocks, threads>>>(
            num_samples, [=] __device__(int sample_id) mutable {
                bool tri_chosen = false;

                assert(sample_id < num_samples);

                // go through every triangle of the cluster
                const int cluster_point = vertex_cluster(sample_id);
                const int start = sample_neighbor_size_prefix(cluster_point);
                const int end = sample_neighbor_size_prefix(cluster_point + 1);

                float min_distance = std::numeric_limits<float>::max();

                Eigen::Vector3<float> selectedv1{0, 0, 0}, selectedv2{0, 0, 0},
                    selectedv3{0, 0, 0};

                const Eigen::Vector3<float> q{vertex_pos(sample_id, 0),
                                              vertex_pos(sample_id, 1),
                                              vertex_pos(sample_id, 2)};


                int selected_neighbor             = 0;
                int selected_neighbor_of_neighbor = 0;

                // We need at least 2 neighbors to form a triangle
                int neighbors_count = end - start;
                if (neighbors_count == 2) {
                    int                   v1_idx = cluster_point;
                    int                   v2_idx = sample_neighbor(start);
                    int                   v3_idx = sample_neighbor(start + 1);
                    Eigen::Vector3<float> v1{samples_pos(v1_idx, 0),
                                             samples_pos(v1_idx, 1),
                                             samples_pos(v1_idx, 2)};

                    Eigen::Vector3<float> v2{samples_pos(v2_idx, 0),
                                             samples_pos(v2_idx, 1),
                                             samples_pos(v2_idx, 2)};

                    Eigen::Vector3<float> v3{samples_pos(v3_idx, 0),
                                             samples_pos(v3_idx, 1),
                                             samples_pos(v3_idx, 2)};

                    tri_chosen                    = true;
                    selectedv1                    = v1;
                    selectedv2                    = v2;
                    selectedv3                    = v3;
                    selected_neighbor             = v2_idx;
                    selected_neighbor_of_neighbor = v3_idx;

                } else if (neighbors_count >= 2) {
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
                                    tri_chosen                    = true;
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
                float           b1 = 0, b2 = 0, b3 = 0;
                Eigen::Vector3f bcoords;
                detail::compute_positive_barycentric_coords(
                    q, selectedv1, selectedv2, selectedv3, bcoords);

                b1 = bcoords[0];
                b2 = bcoords[1];
                b3 = bcoords[2];


                assert(tri_chosen);
                assert(b1 >= 0);
                assert(b2 >= 0);
                assert(b3 >= 0);
                assert(!isnan(b1));
                assert(!isnan(b2));
                assert(!isnan(b3));

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