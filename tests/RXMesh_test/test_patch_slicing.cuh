#include <assert.h>
#include "gtest/gtest.h"

#include "rxmesh/rxmesh_dynamic.h"

__global__ static void set_patch_should_slice(rxmesh::Context context, int p)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // if (tid < context.m_num_patches[0]) {
    if (tid == p) {
        context.m_patches_info[tid].should_slice = true;
    }
}

TEST(RXMeshDynamic, PatchSlicing)
{
    using namespace rxmesh;
 
    RXMeshDynamic rx(rxmesh_args.obj_file_name,
                     STRINGIFY(OUTPUT_DIR) +
                         extract_file_name(rxmesh_args.obj_file_name) +
                         "_patches");

    auto coords = rx.get_input_vertex_coordinates();

    const uint32_t num_vertices = rx.get_num_vertices();
    const uint32_t num_edges    = rx.get_num_edges();
    const uint32_t num_faces    = rx.get_num_faces();

    EXPECT_TRUE(rx.validate());

#if USE_POLYSCOPE
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    // rx.render_patch(0)->setEnabled(false);
    // polyscope::show();
#endif

    RXMESH_INFO("Input mesh #Patches {}", rx.get_num_patches());

    // set_patch_should_slice<<<rx.get_num_patches(), 1>>>(rx.get_context());
    //  rx.copy_patch_debug(0, *coords);
    uint32_t num_pp = rx.get_num_patches();
    for (uint32_t p = 0; p < num_pp; ++p) {
        set_patch_should_slice<<<rx.get_num_patches(), 1>>>(rx.get_context(),
                                                            p);
        rx.slice_patches(*coords);
        rx.cleanup();
    }

    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());

    rx.update_host();
    EXPECT_TRUE(rx.validate());

    coords->move(DEVICE, HOST);


    EXPECT_EQ(num_vertices, rx.get_num_vertices());
    EXPECT_EQ(num_edges, rx.get_num_edges());
    EXPECT_EQ(num_faces, rx.get_num_faces());


#if USE_POLYSCOPE
    rx.update_polyscope();
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();

    for (uint32_t p = 0; p < rx.get_num_patches(); ++p) {
        rx.render_patch(p)->setEnabled(false);
    }

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(*coords);
    ps_mesh->setEnabled(false);
    // polyscope::show();
#endif
}