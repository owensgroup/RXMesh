#include <assert.h>
#include "gtest/gtest.h"

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"

using namespace rxmesh;

template <uint32_t blockThreads>
__global__ static void tri_refine(Context                context,
                                  VertexAttribute<float> coords,
                                  DenseMatrix<float>     avg_area)
{
    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::FE> cavity(
        block, context, shrd_alloc, true);


    if (cavity.patch_id() == INVALID32) {
        return;
    }

    auto should_refine = [&](const FaceHandle& fh, const VertexIterator& iter) {
        Eigen::Vector3f x0 = coords.to_eigen<3>(iter[0]);
        Eigen::Vector3f x1 = coords.to_eigen<3>(iter[1]);
        Eigen::Vector3f x2 = coords.to_eigen<3>(iter[2]);

        float a = 0.5f * ((x1 - x0).cross(x2 - x0)).norm();
        if (a > avg_area(0, 0)) {
            cavity.create(fh);
        }
    };

    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::FV>(block, shrd_alloc, should_refine);
    block.sync();

    block.sync();

    if (cavity.prologue(block, shrd_alloc, coords)) {


        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            const VertexHandle new_v = cavity.add_vertex();

            float new_v_coord[3] = {0, 0, 0};

            for (int i = 0; i < size; ++i) {
                const VertexHandle vh = cavity.get_cavity_vertex(c, i);

                for (int j = 0; j < 3; ++j) {
                    new_v_coord[j] += coords(vh, j);
                }
            }

            coords(new_v, 0) = new_v_coord[0] / float(size);
            coords(new_v, 1) = new_v_coord[1] / float(size);
            coords(new_v, 2) = new_v_coord[2] / float(size);

            DEdgeHandle e0 =
                cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));
            const DEdgeHandle e_init = e0;

            if (e0.is_valid()) {
                for (uint16_t i = 0; i < size; ++i) {
                    const DEdgeHandle e = cavity.get_cavity_edge(c, i);
                    const DEdgeHandle e1 =
                        (i == size - 1) ?
                            e_init.get_flip_dedge() :
                            cavity.add_edge(cavity.get_cavity_vertex(c, i + 1),
                                            new_v);
                    if (!e1.is_valid()) {
                        break;
                    }
                    const FaceHandle f = cavity.add_face(e0, e, e1);
                    if (!f.is_valid()) {
                        break;
                    }

                    e0 = e1.get_flip_dedge();
                }
            }
        });
    }

    cavity.epilogue(block);
}

void compute_avg_area(RXMeshDynamic&          rx,
                      VertexAttribute<float>& coords,
                      DenseMatrix<float>&     avg_area)
{
    rx.run_query_kernel<Op::FV, 256>(
        [coords, avg_area] __device__(const FaceHandle     fh,
                                      const VertexIterator iter) mutable {
            Eigen::Vector3f x0 = coords.to_eigen<3>(iter[0]);
            Eigen::Vector3f x1 = coords.to_eigen<3>(iter[1]);
            Eigen::Vector3f x2 = coords.to_eigen<3>(iter[2]);

            float a = 0.5f * ((x1 - x0).cross(x2 - x0)).norm();

            ::atomicAdd(&avg_area(0, 0), a);
        });
    CUDA_ERROR(cudaDeviceSynchronize());

    avg_area.move(DEVICE, HOST);
    avg_area(0, 0) /= float(rx.get_num_faces());
    avg_area.move(HOST, DEVICE);

    RXMESH_INFO("avg_area = {}", avg_area(0, 0));
}

void add_to_polyscope(RXMeshDynamic&          rx,
                      VertexAttribute<float>& coords,
                      DenseMatrix<float>&     avg_area)
{
#if USE_POLYSCOPE

    auto to_refine = *rx.add_face_attribute<int>("to_refine", 1);
    rx.run_query_kernel<Op::FV, 256>(
        [coords, to_refine, avg_area] __device__(const FaceHandle     fh,
                                                 const VertexIterator iter) {
            Eigen::Vector3f x0 = coords.to_eigen<3>(iter[0]);
            Eigen::Vector3f x1 = coords.to_eigen<3>(iter[1]);
            Eigen::Vector3f x2 = coords.to_eigen<3>(iter[2]);

            float a = 0.5f * ((x1 - x0).cross(x2 - x0)).norm();

            if (a > avg_area(0, 0)) {
                to_refine(fh) = 1;
            } else {
                to_refine(fh) = 0;
            }
        });
    to_refine.move(DEVICE, HOST);
    rx.get_polyscope_mesh()->addFaceScalarQuantity("toRefine", to_refine);
#endif
}

TEST(RXMeshDynamic, TriangleRefinement)
{

    RXMeshDynamic rx(STRINGIFY(INPUT_DIR) "rocker-arm.obj");

    EXPECT_TRUE(rx.validate());

    auto coords = *rx.get_input_vertex_coordinates();

    DenseMatrix<float> avg_area(rx, 1, 1, LOCATION_ALL);
    avg_area.reset(0, LOCATION_ALL);

    compute_avg_area(rx, coords, avg_area);

    add_to_polyscope(rx, coords, avg_area);


    constexpr uint32_t blockThreads = 256;

    int iter = 0;
    while (!rx.is_queue_empty()) {
        RXMESH_INFO("iter = {}", ++iter);
        LaunchBox<blockThreads> launch_box;
        rx.prepare_launch_box({}, launch_box, (void*)tri_refine<blockThreads>);

        tri_refine<blockThreads>
            <<<launch_box.blocks,
               launch_box.num_threads,
               launch_box.smem_bytes_dyn>>>(rx.get_context(), coords, avg_area);

        rx.slice_patches(coords);
        rx.cleanup();
    }

    CUDA_ERROR(cudaDeviceSynchronize());

    rx.update_host();

    coords.move(DEVICE, HOST);

    EXPECT_TRUE(rx.validate());

#if USE_POLYSCOPE
    rx.update_polyscope();
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(coords);

    polyscope::show();
#endif
}