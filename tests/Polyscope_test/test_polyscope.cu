#include <vector>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/cuda_query.h"
#include "rxmesh/util/log.h"

#include "rxmesh/query.cuh"

template <typename T, uint32_t blockThreads>
__global__ static void compute_vertex_normal(const rxmesh::Context      context,
                                             rxmesh::VertexAttribute<T> coords,
                                             rxmesh::VertexAttribute<T> normals)
{
    using namespace rxmesh;

    auto vn_lambda = [&](FaceHandle face_id, VertexIterator& fv) {
        // get the face's three vertices coordinates
        glm::fvec3 c0(coords(fv[0], 0), coords(fv[0], 1), coords(fv[0], 2));
        glm::fvec3 c1(coords(fv[1], 0), coords(fv[1], 1), coords(fv[1], 2));
        glm::fvec3 c2(coords(fv[2], 0), coords(fv[2], 1), coords(fv[2], 2));

        // compute the face normal
        glm::fvec3 n = cross(c1 - c0, c2 - c0);

        // the three edges length
        glm::fvec3 l(glm::distance2(c0, c1),
                     glm::distance2(c1, c2),
                     glm::distance2(c2, c0));

        // add the face's normal to its vertices
        for (uint32_t v = 0; v < 3; ++v) {      // for every vertex in this face
            for (uint32_t i = 0; i < 3; ++i) {  // for the vertex 3 coordinates
                atomicAdd(&normals(fv[v], i), n[i] / (l[v] + l[(v + 2) % 3]));
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, vn_lambda);
}

int main(int argc, char** argv)
{
    rxmesh::Log::init();
    rxmesh::cuda_query(0);

    polyscope::view::upDir = polyscope::UpDir::ZUp;


    rxmesh::RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

    auto polyscope_mesh = rx.get_polyscope_mesh();

    polyscope_mesh->setEdgeWidth(1.0);

    // Vertex Color
    auto vertex_pos   = *rx.get_input_vertex_coordinates();
    auto vertex_color = *rx.add_vertex_attribute<float>("vColor", 3);
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [vertex_color, vertex_pos] __device__(const rxmesh::VertexHandle vh) {
            vertex_color(vh, 0) = 0.9;
            vertex_color(vh, 1) = vertex_pos(vh, 1);
            vertex_color(vh, 2) = 0.9;
        });

    vertex_color.move(rxmesh::DEVICE, rxmesh::HOST);

    polyscope_mesh->addVertexColorQuantity("vColor", vertex_color);

    // rx.render_face_patch();
    // rx.render_vertex_patch();
    // rx.render_edge_patch();

    // Vertex Normal
    auto vertex_normals = rx.add_vertex_attribute<float>("vNormals", 3);
    vertex_normals->reset(0, rxmesh::LOCATION_ALL);

    constexpr uint32_t               CUDABlockSize = 256;
    rxmesh::LaunchBox<CUDABlockSize> launch_box;
    rx.prepare_launch_box({rxmesh::Op::FV},
                          launch_box,
                          (void*)compute_vertex_normal<float, CUDABlockSize>);
    compute_vertex_normal<float, CUDABlockSize><<<launch_box.blocks,
                                                  launch_box.num_threads,
                                                  launch_box.smem_bytes_dyn>>>(
        rx.get_context(), vertex_pos, *vertex_normals);

    vertex_normals->move(rxmesh::DEVICE, rxmesh::HOST);

    polyscope_mesh->addVertexVectorQuantity("vNormal", *vertex_normals);

    polyscope::show();

    return 0;
}
