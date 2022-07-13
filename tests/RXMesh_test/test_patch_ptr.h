#include "gtest/gtest.h"

#include "rxmesh/patch_ptr.h"

TEST(RXMeshStatic, PatchPtr)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj", rxmesh_args.quite);

    PatchPtr ptch_ptr(rx);

    // move the patch ptr to the host so we can test it
    std::vector<uint32_t> h_ptchptr(rx.get_num_patches() + 1);

    // vertices
    CUDA_ERROR(cudaMemcpy(h_ptchptr.data(),
                          ptch_ptr.get_pointer(ELEMENT::VERTEX),
                          h_ptchptr.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_ptchptr.back(), rx.get_num_vertices());

    for (uint32_t i = 0; i < rx.get_num_patches(); ++i) {
        EXPECT_EQ(h_ptchptr[i + 1] - h_ptchptr[i],
                  rx.get_patches_info()[i].num_owned_vertices);
    }


    // edges
    CUDA_ERROR(cudaMemcpy(h_ptchptr.data(),
                          ptch_ptr.get_pointer(ELEMENT::EDGE),
                          h_ptchptr.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_ptchptr.back(), rx.get_num_edges());

    for (uint32_t i = 0; i < rx.get_num_patches(); ++i) {
        EXPECT_EQ(h_ptchptr[i + 1] - h_ptchptr[i],
                  rx.get_patches_info()[i].num_owned_edges);
    }


    // faces
    CUDA_ERROR(cudaMemcpy(h_ptchptr.data(),
                          ptch_ptr.get_pointer(ELEMENT::FACE),
                          h_ptchptr.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_ptchptr.back(), rx.get_num_faces());

    for (uint32_t i = 0; i < rx.get_num_patches(); ++i) {
        EXPECT_EQ(h_ptchptr[i + 1] - h_ptchptr[i],
                  rx.get_patches_info()[i].num_owned_faces);
    }


    ptch_ptr.free();
}