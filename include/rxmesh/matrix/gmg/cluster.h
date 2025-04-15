#pragma once

#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/rxmesh_static.h"

namespace rxmesh {

namespace detail {
template <typename T, uint32_t blockThreads>
__global__ static void cluster_points(const Context        context,
                                      const DenseMatrix<T> vertex_pos,
                                      VertexAttribute<T>   distance,
                                      DenseMatrix<int>     vertex_cluster,
                                      int*                 flag)
{

    auto cluster = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); i++) {

            float dist =
                sqrtf(powf(vertex_pos(v_id, 0) - vertex_pos(vv[i], 0), 2) +
                      powf(vertex_pos(v_id, 1) - vertex_pos(vv[i], 1), 2) +
                      powf(vertex_pos(v_id, 2) - vertex_pos(vv[i], 2), 2)) +
                distance(vv[i], 0);

            if (dist < distance(v_id, 0) && vertex_cluster(vv[i]) != -1) {
                distance(v_id, 0)    = dist;
                *flag                = 15;
                vertex_cluster(v_id) = vertex_cluster(vv[i]);
            }
        }
    };
    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, cluster);
}

}  // namespace detail

/**
 * \brief Clustering the data after sampling for the 1st level
 */
void clustering_1st_level(RXMeshStatic&                rx,
                          int                          current_level,
                          const DenseMatrix<float>&    vertex_pos,
                          const DenseMatrix<uint16_t>& sample_level_bitmask,
                          VertexAttribute<float>&      distance,
                          DenseMatrix<int>&            vertex_cluster,
                          int*                         d_flag)
{
    constexpr uint32_t blockThreads = 256;

    int h_flag = 0;

    LaunchBox<blockThreads> lb;
    rx.prepare_launch_box(
        {Op::VV}, lb, (void*)detail::cluster_points<float, blockThreads>);

    do {
        CUDA_ERROR(cudaMemset(d_flag, 0, sizeof(int)));
        rx.run_kernel<blockThreads>(lb,
                                    detail::cluster_points<float, blockThreads>,
                                    vertex_pos,
                                    distance,
                                    vertex_cluster,
                                    d_flag);

        h_flag = 0;
        CUDA_ERROR(
            cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    } while (h_flag != 0);
}


/**
 * \brief Clustering the data for any level other than the 1st level
 */
inline void clustering_nth_level(
    int                          num_samples,
    int                          current_level,
    const DenseMatrix<int>&      sample_neighbor_size_prefix,
    const DenseMatrix<int>&      sample_neighbor,
    DenseMatrix<int>&            vertex_cluster,
    DenseMatrix<float>&          distance,
    const DenseMatrix<uint16_t>& sample_level_bitmask,
    const DenseMatrix<float>&    prev_samples_pos,
    int*                         d_flag)
{
    int h_flag = 0;

    uint32_t threads = 256;
    uint32_t blocks  = DIVIDE_UP(num_samples, threads);

    do {
        CUDA_ERROR(cudaMemset(d_flag, 0, sizeof(int)));

        for_each_item<<<blocks, threads>>>(
            num_samples, [=] __device__(int id) mutable {
                const float sample_x = prev_samples_pos(id, 0);
                const float sample_y = prev_samples_pos(id, 1);
                const float sample_z = prev_samples_pos(id, 2);

                const int start = sample_neighbor_size_prefix(id);
                const int end   = sample_neighbor_size_prefix(id + 1);
                for (int i = start; i < end; i++) {
                    int         current_v = sample_neighbor(i);
                    const float v_x       = prev_samples_pos(current_v, 0);
                    const float v_y       = prev_samples_pos(current_v, 1);
                    const float v_z       = prev_samples_pos(current_v, 2);

                    float dist = sqrtf(powf(sample_x - v_x, 2) +
                                       powf(sample_y - v_y, 2) +
                                       powf(sample_z - v_z, 2)) +
                                 distance(current_v);


                    if (dist < distance(id) &&
                        vertex_cluster(current_v) != -1) {
                        distance(id)       = dist;
                        *d_flag            = 15;
                        vertex_cluster(id) = vertex_cluster(current_v);
                    }
                }
            });
        h_flag = 0;
        CUDA_ERROR(
            cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    } while (h_flag != 0);
}
}  // namespace rxmesh