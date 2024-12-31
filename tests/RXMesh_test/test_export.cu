#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"


TEST(RXMeshStatic, Export)
{
    using namespace rxmesh;

    CUDA_ERROR(cudaDeviceReset());

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto v_attr_scalar = *rx.add_vertex_attribute<float>("vScalar", 1);
    auto v_attr_vec2   = *rx.add_vertex_attribute<float>("vVector2", 2);
    auto v_attr_vec3   = *rx.add_vertex_attribute<float>("vVector3", 3);

    auto f_attr_scalar = *rx.add_face_attribute<float>("fScalar", 1);
    auto f_attr_vec2   = *rx.add_face_attribute<float>("fVector2", 2);
    auto f_attr_vec3   = *rx.add_face_attribute<float>("fVector3", 3);

    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        v_attr_scalar(vh, 0) = rand() % 100;

        for (uint32_t i = 0; i < v_attr_vec2.get_num_attributes(); ++i) {
            v_attr_vec2(vh, i) = rand() % 100;
        }

        for (uint32_t i = 0; i < v_attr_vec3.get_num_attributes(); ++i) {
            v_attr_vec3(vh, i) = rand() % 100;
        }
    });


    rx.for_each_face(HOST, [&](const FaceHandle& fh) {
        f_attr_scalar(fh, 0) = rand() % 100;

        for (uint32_t i = 0; i < f_attr_vec2.get_num_attributes(); ++i) {
            f_attr_vec2(fh, i) = rand() % 100;
        }

        for (uint32_t i = 0; i < f_attr_vec3.get_num_attributes(); ++i) {
            f_attr_vec3(fh, i) = rand() % 100;
        }
    });


    rx.export_obj("sphere3.obj", *rx.get_input_vertex_coordinates());

    rx.export_vtk("sphere3.vtk",
                  *rx.get_input_vertex_coordinates(),
                  v_attr_scalar,
                  v_attr_vec2,
                  v_attr_vec3,
                  f_attr_scalar,
                  f_attr_vec2,
                  f_attr_vec3);


    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}