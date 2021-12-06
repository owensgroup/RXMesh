#include "gtest/gtest.h"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/vector.h"

/*__global__ static void test_vector(
    rxmesh::RXMeshAttribute<rxmesh::Vector3f> mesh_attr,
    uint32_t*                                 suceess)
{

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *suceess = 1;

        assert((mesh_attr.get_allocated() & rxmesh::DEVICE) == rxmesh::DEVICE);
        uint32_t num_mesh_elements = mesh_attr.get_num_mesh_elements();
        for (uint32_t i = 0; i < num_mesh_elements; ++i) {
            const auto& vec = mesh_attr(i);
            if (vec[0] != i + 0 || vec[1] != i + 1 || vec[2] != i + 2) {
                *suceess = 0;
                return;
            }
        }
    }
}*/


/*TEST(RXMeshAttribute, Vector)
{
    using namespace rxmesh;
    // mesh attr on host
    uint32_t                         num_mesh_elements = 2048;
    rxmesh::RXMeshAttribute<Vector3f> rxmesh_attr;

    rxmesh_attr.set_name("vector3f_attr");
    rxmesh_attr.init(num_mesh_elements, 1, rxmesh::HOST, rxmesh::AoS);

    // generate some numbers as AoS
    for (uint32_t i = 0; i < num_mesh_elements; ++i) {
        auto& vec = rxmesh_attr(i);
        vec[0] = i + 0;
        vec[1] = i + 1;
        vec[2] = i + 2;
    }

    // move memory to device
    rxmesh_attr.move(rxmesh::HOST, rxmesh::DEVICE);


    // device success variable
    uint32_t* d_success = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_success, sizeof(uint32_t)));


    // actual testing
    test_vector<<<1, 1>>>(rxmesh_attr, d_success);

    CUDA_ERROR(cudaPeekAtLastError());
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaDeviceSynchronize());

    // host success variable
    uint32_t h_success(0);
    CUDA_ERROR(cudaMemcpy(&h_success, d_success, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // free device
    GPU_FREE(d_success);

    // release rxmesh_attribute memory on host and device
    rxmesh_attr.release();

    EXPECT_EQ(h_success, 1);
}*/
template <typename T>
void norm2_populate(rxmesh::RXMeshStatic&             rxmesh,
                    rxmesh::RXMeshVertexAttribute<T>& v,
                    T                                 val)
{
    rxmesh.for_each_vertex(
        rxmesh::DEVICE,
        [v, val] __device__(const rxmesh::VertexHandle vh) { v(vh) = val; });
}

TEST(RXMeshAttribute, Norm2)
{

    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    ASSERT_TRUE(
        import_obj(STRINGIFY(INPUT_DIR) "sphere3.obj", Verts, Faces, true));

    RXMeshStatic rxmesh(Faces, rxmesh_args.quite);

    auto attr = rxmesh.add_vertex_attribute<float>("v", 3, rxmesh::DEVICE);

    const float val(2.0);

    norm2_populate<float>(rxmesh, *attr, val);

    ReduceHandle reduce(*attr);

    float output = reduce.norm2(*attr);

    CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_FLOAT_EQ(output, std::sqrt(val * val * rxmesh.get_num_vertices()));
}

template <typename T>
void dot_populate(rxmesh::RXMeshStatic&             rxmesh,
                  rxmesh::RXMeshVertexAttribute<T>& v1,
                  rxmesh::RXMeshVertexAttribute<T>& v2,
                  T                                 v1_val,
                  T                                 v2_val)
{
    rxmesh.for_each_vertex(
        rxmesh::DEVICE,
        [v1, v2, v1_val, v2_val] __device__(const rxmesh::VertexHandle vh) {
            v1(vh) = v1_val;
            v2(vh) = v2_val;
        });
}
TEST(RXMeshAttribute, Dot)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    ASSERT_TRUE(
        import_obj(STRINGIFY(INPUT_DIR) "sphere3.obj", Verts, Faces, true));

    RXMeshStatic rxmesh(Faces, rxmesh_args.quite);

    auto v1_attr = rxmesh.add_vertex_attribute<float>("v1", 3, rxmesh::DEVICE);
    auto v2_attr = rxmesh.add_vertex_attribute<float>("v2", 3, rxmesh::DEVICE);

    const float v1_val(2.0);
    const float v2_val(3.0);

    dot_populate<float>(rxmesh, *v1_attr, *v2_attr, v1_val, v2_val);

    ReduceHandle reduce(*v1_attr);

    float output = reduce.dot(*v1_attr, *v2_attr);

    CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_FLOAT_EQ(output, v1_val * v2_val * rxmesh.get_num_vertices());
}


TEST(RXMeshAttribute, Copy)
{
    using namespace rxmesh;
    uint32_t                       num_mesh_elements = 2048;
    rxmesh::RXMeshAttribute<float> rxmesh_attr;
    rxmesh_attr.set_name("float_attr");
    rxmesh_attr.init(num_mesh_elements, 1, rxmesh::LOCATION_ALL, rxmesh::AoS);

    for (uint32_t i = 0; i < num_mesh_elements; ++i) {
        rxmesh_attr(i) = i;
    }

    auto copy = rxmesh_attr;

    EXPECT_EQ(num_mesh_elements, copy.get_num_mesh_elements());
    EXPECT_EQ(1, copy.get_num_attributes());
    std::string name(copy.get_name());

    EXPECT_TRUE(name == "float_attr");

    for (uint32_t i = 0; i < num_mesh_elements; ++i) {
        EXPECT_EQ(rxmesh_attr(i), i) << " TestAttributes::test_copy failed";
    }

    rxmesh_attr.release();
}

TEST(RXMeshAttribute, AddingAndRemoving)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    ASSERT_TRUE(
        import_obj(STRINGIFY(INPUT_DIR) "sphere3.obj", Verts, Faces, true));


    RXMeshStatic rxmesh(Faces, rxmesh_args.quite);

    std::string attr_name = "v_attr";

    auto vertex_attr =
        rxmesh.add_vertex_attribute<float>(attr_name, 3, rxmesh::LOCATION_ALL);

    EXPECT_TRUE(rxmesh.does_attribute_exist(attr_name));


    vertex_attr->move_v1(rxmesh::HOST, rxmesh::DEVICE);

    CUDA_ERROR(cudaDeviceSynchronize());

    // this is not neccessary in general but we are just testing the
    // functionality here
    rxmesh.remove_attribute(attr_name);
}