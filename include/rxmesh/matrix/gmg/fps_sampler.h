#pragma once

#include <random>

#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"

namespace rxmesh {

namespace detail {
template <uint32_t blockThreads>
__global__ static void sample_points(const Context          context,
                                     DenseMatrix<float>     vertex_pos,
                                     VertexAttribute<float> distance,
                                     int*                   flag)
{

    auto sampler = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); i++) {

            float dist =
                sqrtf(powf(vertex_pos(v_id, 0) - vertex_pos(vv[i], 0), 2) +
                      powf(vertex_pos(v_id, 1) - vertex_pos(vv[i], 1), 2) +
                      powf(vertex_pos(v_id, 2) - vertex_pos(vv[i], 2), 2)) +
                distance(vv[i], 0);

            if (dist < distance(v_id, 0)) {
                distance(v_id, 0) = dist;
                *flag             = 15;
            }
        }
    };
    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, sampler);
}
}  // namespace detail

/**
 * \brief FPS Sampling in parallel
 */

void FPSSampler(RXMeshStatic&           rx,
                VertexAttribute<float>& distance,
                DenseMatrix<float>&     vertex_pos,
                DenseMatrix<int>&       vertex_cluster,
                DenseMatrix<uint16_t>&  sample_level_bitmask,
                DenseMatrix<float>&     samples_pos,
                float                   ratio,
                int                     N,
                int                     numberOfLevels,
                int                     numberOfSamplesForFirstLevel,
                int*                    d_flag)
{

    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dist(0, N - 1);

    int seed = 0;  // dist(gen);

    VertexReduceHandle<float> reducer(distance);

    const Context context = rx.get_context();

    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> lb;
    rx.prepare_launch_box(
        {Op::VV}, lb, (void*)detail::sample_points<blockThreads>);


    int currentSampleLevel = numberOfLevels;

    cub::KeyValuePair<VertexHandle, float> farthestPoint;

    int h_flag;

    for (int i = 0; i < numberOfSamplesForFirstLevel; i++) {

        if (i > N / (int)powf(ratio, currentSampleLevel)) {
            currentSampleLevel--;
        }

        rx.for_each_vertex(
            DEVICE, [=] __device__(const VertexHandle vh) mutable {
                if (seed == context.linear_id(vh)) {

                    vertex_cluster(vh) = i;
                    distance(vh)       = 0;

                    samples_pos(i, 0) = vertex_pos(vh, 0);
                    samples_pos(i, 1) = vertex_pos(vh, 1);
                    samples_pos(i, 2) = vertex_pos(vh, 2);

                    for (int k = 0; k < currentSampleLevel; k++) {
                        sample_level_bitmask(vh, 0) |= (1 << k);
                    }
                } else {
                    if (i == 0) {
                        distance(vh, 0) = std::numeric_limits<float>::max();
                    }
                }
            });

        do {
            CUDA_ERROR(cudaMemset(d_flag, 0, sizeof(int)));

            rx.run_kernel(lb,
                          detail::sample_points<blockThreads>,
                          vertex_pos,
                          distance,
                          d_flag);

            h_flag = 0;
            CUDA_ERROR(cudaMemcpy(
                &h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));

        } while (h_flag != 0);


        // reduction step
        farthestPoint = reducer.arg_max(distance, 0);

        seed = rx.linear_id(farthestPoint.key);
    }
}

}  // namespace rxmesh