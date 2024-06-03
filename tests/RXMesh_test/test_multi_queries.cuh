#include "gtest/gtest.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "rxmesh/rxmesh_static.h"


template <uint32_t blockThreads, typename T>
__global__ static void sum_edges_ev(const rxmesh::Context            context,
                                    const rxmesh::VertexAttribute<T> coords,
                                    rxmesh::VertexAttribute<T>       vertex_sum)
{
    // Ever edge atomicAdd its length to its two end vertices
    using namespace rxmesh;


    auto sum_edges = [&](const EdgeHandle& id, const VertexIterator& iter) {
        const vec3<T> p0(
            coords(iter[0], 0), coords(iter[0], 1), coords(iter[0], 2));
        const vec3<T> p1(
            coords(iter[1], 0), coords(iter[1], 1), coords(iter[1], 2));

        const T edge_len = glm::distance2(p0, p1);

        ::atomicAdd(&vertex_sum(iter[0]), edge_len);
        ::atomicAdd(&vertex_sum(iter[1]), edge_len);
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, sum_edges);
}


template <uint32_t blockThreads, typename T>
__global__ static void sum_edges_multi_queries(
    const rxmesh::Context            context,
    const rxmesh::VertexAttribute<T> coords,
    rxmesh::VertexAttribute<T>       vertex_sum)
{
    // For each vertex, loop over its edges. For each incident, compute its
    // length and add it to the vertex
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    // initiate the secondary query (EV)
    Query<blockThreads> ev_query(context);
    ev_query.prologue<Op::EV>(block, shrd_alloc);


    auto sum_edges = [&](const VertexHandle& vertex,
                         const EdgeIterator& eiter) {
        const vec3<T> p0(
            coords(vertex, 0), coords(vertex, 1), coords(vertex, 2));

        // for each incident to the vertex
        for (uint16_t i = 0; i < eiter.size(); ++i) {
            //  access the edge's (eiter[i]) two end vertices
            VertexIterator viter =
                ev_query.template get_iterator<VertexIterator>(eiter.local(i));
            // note that  we use local index from the iterator ^^^^^^^

            assert(viter.size() == 2);

            const VertexHandle vh0(viter[0]), vh1(viter[1]);

            // sanity check: one of the two end vertices should be the same as
            // vertex
            assert(vh0 == vertex || vh1 == vertex);

            // get the other end vertex coordinates
            vec3<T> p1;
            if (vertex != vh0) {
                p1 = vec3<T>(coords(vh0, 0), coords(vh0, 1), coords(vh0, 2));
            } else {
                p1 = vec3<T>(coords(vh1, 0), coords(vh1, 1), coords(vh1, 2));
            }
            const T edge_len = glm::distance2(p0, p1);

            vertex_sum(vertex) += edge_len;
        }
    };

    // the primary query
    Query<blockThreads> ve_query(context);
    ve_query.dispatch<Op::VE>(block, shrd_alloc, sum_edges);

    ev_query.epilogue(block, shrd_alloc);
}

TEST(RXMeshStatic, MultiQueries)
{
    // Compute vertex edge-sum (i.e., for each vertex, accumulate the edge
    // length of all edges incident to the vertex) using
    // 1. EV queries such that each edge computes its length and then atomically
    // adds its length to its two end vertices. This is the ground truth
    // 2. VE and EV where we launch the "primary" queries as VE i.e., a thread
    // is assigned to each vertex. Each vertex than iterates over its edges
    // (VE), and for each incident edge, we access the other end vertex (i.e.,
    // EV which we call "secondary query"). Then, compute the edge length and
    // then add it to the vertex.

    using namespace rxmesh;
    constexpr uint32_t blockThreads = 320;

    // Select device
    cuda_query(rxmesh_args.device_id);

    RXMeshStatic rx(rxmesh_args.obj_file_name);

    const auto coords = rx.get_input_vertex_coordinates();

    auto vertex_sum = rx.add_vertex_attribute<float>("vertex_sum", 1);
    vertex_sum->reset(0, LOCATION_ALL);

    // ground truth
    auto vertex_sum_gt = rx.add_vertex_attribute<float>("vertex_sum_gt", 1);
    vertex_sum_gt->reset(0, LOCATION_ALL);

    LaunchBox<blockThreads> launch_box;

    // compute vertex sum ground truth using EV queries
    rx.prepare_launch_box(
        {Op::EV}, launch_box, (void*)sum_edges_ev<blockThreads, float>);
    sum_edges_ev<blockThreads>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *vertex_sum_gt);

    // compute vertex sum using multi-queries
    rx.prepare_launch_box({Op::VE, Op::EV},
                          launch_box,
                          (void*)sum_edges_multi_queries<blockThreads, float>,
                          false,
                          false,
                          true);
    sum_edges_multi_queries<blockThreads>
        <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *vertex_sum);

    CUDA_ERROR(cudaDeviceSynchronize());

    // move data to the host
    vertex_sum_gt->move(DEVICE, HOST);
    vertex_sum->move(DEVICE, HOST);


    // verify
    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        ASSERT_NEAR((*vertex_sum_gt)(vh), (*vertex_sum)(vh), 0.0001);
    });
}