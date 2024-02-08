#include <cuda_profiler_api.h>
#include <glm/glm.hpp>
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_dynamic.h"


template <typename T>
using Vec3 = glm::vec<3, T, glm::defaultp>;

template <typename T>
using Vec4 = glm::vec<4, T, glm::defaultp>;

template <typename T>
using Mat4 = glm::mat<4, 4, T, glm::defaultp>;

template <typename T>
__device__ __inline__ __host__ T
compute_cost(const Mat4<T>& quadric, const T x, const T y, const T z)
{
    // clang-format off
    return     quadric[0][0] * x * x + 
           2 * quadric[0][1] * x * y +
           2 * quadric[0][2] * x * z + 
           2 * quadric[0][3] * x +

               quadric[1][1] * y * y + 
           2 * quadric[1][2] * y * z +
           2 * quadric[1][3] * y +

               quadric[2][2] * z * z + 
           2 * quadric[2][3] * z +

               quadric[3][3];
    // clang-format on
}

template <typename T>
__device__ __inline__ void calc_edge_cost(
    const rxmesh::EdgeHandle&         e,
    const rxmesh::VertexHandle&       v0,
    const rxmesh::VertexHandle&       v1,
    const rxmesh::VertexAttribute<T>& coords,
    const rxmesh::VertexAttribute<T>& vertex_quadrics,
    rxmesh::EdgeAttribute<T>&         edge_cost,
    rxmesh::EdgeAttribute<T>&         edge_col_coord)
{
    Mat4<T> edge_quadric;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            edge_quadric[i][j] =
                vertex_quadrics(v0, i * 4 + j) + vertex_quadrics(v1, i * 4 + j);
        }
    }

    // the edge_quadric but with 4th row is identity
    Mat4<T> edge_quadric_3x3 = edge_quadric;
    edge_quadric_3x3[3][0]   = 0;
    edge_quadric_3x3[3][1]   = 0;
    edge_quadric_3x3[3][2]   = 0;
    edge_quadric_3x3[3][3]   = 1;

    if (std::abs(glm::determinant(edge_quadric_3x3)) >
        std::numeric_limits<T>::epsilon()) {

        Vec4<T> b(0, 0, 0, 1);

        auto invvv = glm::inverse(edge_quadric_3x3);

        Vec4<T> x = glm::transpose(glm::inverse(edge_quadric_3x3)) * b;

        edge_col_coord(e, 0) = x[0];
        edge_col_coord(e, 1) = x[1];
        edge_col_coord(e, 2) = x[2];

        edge_cost(e) =
            std::max(T(0.0), compute_cost(edge_quadric, x[0], x[1], x[2]));
    } else {
        const Vec3<T> p0(coords(v0, 0), coords(v0, 1), coords(v0, 2));
        const Vec3<T> p1(coords(v1, 0), coords(v1, 1), coords(v1, 2));
        const Vec3<T> p2 = T(0.5) * (p0 + p1);

        const T e0 = compute_cost(edge_quadric, p0[0], p0[1], p0[2]);
        const T e1 = compute_cost(edge_quadric, p1[0], p1[1], p1[2]);
        const T e2 = compute_cost(edge_quadric, p2[0], p2[1], p2[2]);

        if (e0 < e1 && e1 < e2) {
            edge_cost(e)         = std::max(T(0.0), e0);
            edge_col_coord(e, 0) = p0[0];
            edge_col_coord(e, 1) = p0[1];
            edge_col_coord(e, 2) = p0[2];

        } else if (e1 < e2) {
            edge_cost(e)         = std::max(T(0.0), e1);
            edge_col_coord(e, 0) = p1[0];
            edge_col_coord(e, 1) = p1[1];
            edge_col_coord(e, 2) = p1[2];

        } else {
            edge_cost(e)         = std::max(T(0.0), e2);
            edge_col_coord(e, 0) = p2[0];
            edge_col_coord(e, 1) = p2[1];
            edge_col_coord(e, 2) = p2[2];
        }
    }
}

template <typename T>
__device__ __inline__ Vec4<T> compute_face_plane(const Vec3<T>& v0,
                                                 const Vec3<T>& v1,
                                                 const Vec3<T>& v2)
{
    Vec3<T> n = glm::cross(v1 - v0, v2 - v0);

    n = glm::normalize(n);

    Vec4<T> ret(n[0], n[1], n[2], -glm::dot(n, v0));

    return ret;
}

template <typename T, uint32_t blockThreads>
__global__ static void compute_edge_cost(
    const rxmesh::Context            context,
    const rxmesh::VertexAttribute<T> coords,
    const rxmesh::VertexAttribute<T> vertex_quadrics,
    rxmesh::EdgeAttribute<T>         edge_cost,
    rxmesh::EdgeAttribute<T>         edge_col_coord,
    float*                           average_cost)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    auto cost = [&](const EdgeHandle edge_id, const VertexIterator& ev) {
        const VertexHandle v0 = ev[0];
        const VertexHandle v1 = ev[1];

        calc_edge_cost(edge_id,
                       v0,
                       v1,
                       coords,
                       vertex_quadrics,
                       edge_cost,
                       edge_col_coord);
        ::atomicAdd(average_cost, edge_cost(edge_id));
    };

    Query<blockThreads> query(context);
    query.dispatch<Op::EV>(block, shrd_alloc, cost);
}

template <typename T, uint32_t blockThreads>
__global__ static void compute_vertex_quadric_fv(
    const rxmesh::Context            context,
    const rxmesh::VertexAttribute<T> coords,
    rxmesh::VertexAttribute<T>       vertex_quadrics)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    auto calc_quadrics = [&](const FaceHandle      face_id,
                             const VertexIterator& fv) {
        // TODO avoid boundary faces

        const Vec3<T> v0(coords(fv[0], 0), coords(fv[0], 1), coords(fv[0], 2));
        const Vec3<T> v1(coords(fv[1], 0), coords(fv[1], 1), coords(fv[1], 2));
        const Vec3<T> v2(coords(fv[2], 0), coords(fv[2], 1), coords(fv[2], 2));

        const Vec4<T> face_plane = compute_face_plane(v0, v1, v2);

        const T a = face_plane[0];
        const T b = face_plane[1];
        const T c = face_plane[2];
        const T d = face_plane[3];

        Mat4<T> quadric;

        quadric[0][0] = a * a;
        quadric[0][1] = a * b;
        quadric[0][2] = a * c;
        quadric[0][3] = a * d;
        quadric[1][0] = b * a;
        quadric[1][1] = b * b;
        quadric[1][2] = b * c;
        quadric[1][3] = b * d;
        quadric[2][0] = c * a;
        quadric[2][1] = c * b;
        quadric[2][2] = c * c;
        quadric[2][3] = c * d;
        quadric[3][0] = d * a;
        quadric[3][1] = d * b;
        quadric[3][2] = d * c;
        quadric[3][3] = d * d;

        for (uint16_t v = 0; v < 3; ++v) {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    ::atomicAdd(&vertex_quadrics(fv[v], i * 4 + j),
                                quadric[i][j]);
                }
            }
        }
    };

    Query<blockThreads> query(context);
    query.dispatch<Op::FV>(block, shrd_alloc, calc_quadrics);
}

template <typename T, uint32_t blockThreads>
__global__ static void simplify(rxmesh::Context            context,
                                rxmesh::VertexAttribute<T> coords,
                                rxmesh::VertexAttribute<T> vertex_quadrics,
                                rxmesh::EdgeAttribute<T>   edge_cost,
                                rxmesh::EdgeAttribute<T>   edge_col_coord)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    CavityManager<blockThreads, CavityOp::EV> cavity(
        block, context, shrd_alloc, true);

    const uint32_t pid = cavity.patch_id();

    if (pid == INVALID32) {
        return;
    }

    uint16_t num_edges = cavity.patch_info().num_edges[0];

    uint8_t* edge_predicate = shrd_alloc.alloc<uint8_t>(num_edges);

    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        edge_predicate[e] = 0;
    }
    block.sync();

    // following the same edge selection as "Instant Level-of-Detail" paper
    auto collapse = [&](const VertexHandle& vh, const EdgeIterator& iter) {
        // TODO handle boundary edges

        T   min_cost      = std::numeric_limits<T>::max();
        int min_cost_edge = -1;
        for (int e = 0; e < iter.size(); ++e) {
            if (iter[e].patch_id() != pid) {
                continue;
            }
            T cost = edge_cost(iter[e]);
            if (cost < min_cost) {
                cost          = min_cost;
                min_cost_edge = iter[e].local_id();
            }
        }

        if (min_cost_edge != -1) {
            atomicAdd(edge_predicate + min_cost_edge, 1);
        }
    };

    Query<blockThreads> query(context, pid);
    query.dispatch<Op::VE>(block, shrd_alloc, collapse);
    block.sync();

    for_each_edge(cavity.patch_info(), [&](const EdgeHandle eh) {
        // if two vertices says that this edge is the edge with min cost
        // connected to them, then this edge is eligible for collapse
        if (edge_predicate[eh.local_id()] == 2) {
            cavity.create(eh);
        }
    });
    block.sync();


    // create the cavity
    if (cavity.prologue(block,
                        shrd_alloc,
                        coords,
                        vertex_quadrics,
                        edge_cost,
                        edge_col_coord)) {

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);

            VertexHandle v0, v1;

            cavity.get_vertices(src, v0, v1);

            const VertexHandle new_v = cavity.add_vertex();

            if (new_v.is_valid()) {

                coords(new_v, 0) = edge_col_coord(src, 0);
                coords(new_v, 1) = edge_col_coord(src, 1);
                coords(new_v, 2) = edge_col_coord(src, 2);

                for (int i = 0; i < 16; ++i) {
                    vertex_quadrics(new_v, i) =
                        vertex_quadrics(v0, i) + vertex_quadrics(v1, i);
                }

                DEdgeHandle e0 =
                    cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));

                if (e0.is_valid()) {
                    const DEdgeHandle e_init = e0;

                    for (uint16_t i = 0; i < size; ++i) {
                        const DEdgeHandle e = cavity.get_cavity_edge(c, i);

                        const VertexHandle v_end =
                            cavity.get_cavity_vertex(c, (i + 1) % size);

                        const DEdgeHandle e1 =
                            (i == size - 1) ?
                                e_init.get_flip_dedge() :
                                cavity.add_edge(
                                    cavity.get_cavity_vertex(c, i + 1), new_v);

                        if (!e1.is_valid()) {
                            break;
                        }

                        calc_edge_cost(e1.get_edge_handle(),
                                       v_end,
                                       new_v,
                                       coords,
                                       vertex_quadrics,
                                       edge_cost,
                                       edge_col_coord);

                        const FaceHandle new_f = cavity.add_face(e0, e, e1);

                        if (!new_f.is_valid()) {
                            break;
                        }
                        e0 = e1.get_flip_dedge();
                    }
                }
            }
        });
    }

    cavity.epilogue(block);
}


inline void simplification_rxmesh(rxmesh::RXMeshDynamic& rx,
                                  const uint32_t         final_num_faces)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    const uint32_t num_vertices = rx.get_num_vertices();
    const uint32_t num_edges    = rx.get_num_edges();
    const uint32_t num_faces    = rx.get_num_faces();


    auto coords = rx.get_input_vertex_coordinates();

    EXPECT_TRUE(rx.validate());

    auto vertex_quadrics = rx.add_vertex_attribute<float>("quadrics", 16);
    vertex_quadrics->reset(0, DEVICE);

    auto edge_cost = rx.add_edge_attribute<float>("cost", 1);
    edge_cost->reset(0, DEVICE);

    auto edge_col_coord = rx.add_edge_attribute<float>("eCoord", 3);
    edge_col_coord->reset(0, DEVICE);

    float total_time = 0;

    float app_time     = 0;
    float slice_time   = 0;
    float cleanup_time = 0;

    float* average_cost;
    CUDA_ERROR(cudaMallocManaged((void**)&average_cost, sizeof(float)));

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({Op::FV},
                          launch_box,
                          (void*)compute_vertex_quadric_fv<float, blockThreads>,
                          false);
    compute_vertex_quadric_fv<float, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *vertex_quadrics);


    rx.prepare_launch_box({Op::EV},
                          launch_box,
                          (void*)compute_edge_cost<float, blockThreads>,
                          false);
    compute_edge_cost<float, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                        *coords,
                                        *vertex_quadrics,
                                        *edge_cost,
                                        *edge_col_coord,
                                        average_cost);

    CUDA_ERROR(cudaDeviceSynchronize());

    average_cost[0] /= rx.get_num_edges();

#if USE_POLYSCOPE
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    edge_cost->move(DEVICE, HOST);
    // rx.get_polyscope_mesh()->addEdgeScalarQuantity("ECost", *edge_cost);
    // for (uint32_t p = 0; p < rx.get_num_patches(); ++p) {
    //     rx.render_patch(p)->setEnabled(false);
    // }
    polyscope::show();
#endif


    bool validate = true;

    CUDA_ERROR(cudaProfilerStart());
    while (rx.get_num_faces() > final_num_faces) {

        rx.reset_scheduler();
        int inner_iter = 0;
        while (!rx.is_queue_empty() && rx.get_num_faces() > final_num_faces) {

            rx.prepare_launch_box({Op::VE},
                                  launch_box,
                                  (void*)simplify<float, blockThreads>,
                                  true,
                                  false,
                                  false,
                                  false,
                                  [](uint32_t v, uint32_t e, uint32_t f) {
                                      return e * sizeof(uint8_t);
                                  });

            GPUTimer timer;
            timer.start();

            GPUTimer app_timer;
            app_timer.start();
            simplify<float, blockThreads>
                <<<launch_box.blocks,
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *coords,
                                                *vertex_quadrics,
                                                *edge_cost,
                                                *edge_col_coord);
            app_timer.stop();

            GPUTimer slice_timer;
            slice_timer.start();
            rx.slice_patches(*coords, *vertex_quadrics);
            slice_timer.stop();

            GPUTimer cleanup_timer;
            cleanup_timer.start();
            rx.cleanup();
            cleanup_timer.stop();


            timer.stop();
            CUDA_ERROR(cudaDeviceSynchronize());
            CUDA_ERROR(cudaGetLastError());

            total_time += timer.elapsed_millis();
            app_time += app_timer.elapsed_millis();
            slice_time += slice_timer.elapsed_millis();
            cleanup_time += cleanup_timer.elapsed_millis();

            if (validate) {
                rx.update_host();
                EXPECT_TRUE(rx.validate());
            }
        }
    }

    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO("simplification_rxmesh() RXMesh simplification took {} (ms)",
                total_time);
    RXMESH_INFO("simplification_rxmesh() App time {} (ms)", app_time);
    RXMESH_INFO("simplification_rxmesh() Slice timer {} (ms)", slice_time);
    RXMESH_INFO("simplification_rxmesh() Cleanup timer {} (ms)", cleanup_time);

    if (!validate) {
        rx.update_host();
    }
    coords->move(DEVICE, HOST);

#if USE_POLYSCOPE
    rx.update_polyscope();

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(*coords);
    ps_mesh->setEnabled(false);

    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    polyscope::show();
#endif
}