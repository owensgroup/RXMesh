#include "gtest/gtest.h"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/util/macros.h"

template <typename T>
void populate(rxmesh::RXMeshStatic&             rxmesh,
              rxmesh::RXMeshVertexAttribute<T>& v,
              T                                 val)
{
    rxmesh.for_each_vertex(
        rxmesh::DEVICE,
        [v, val] __device__(const rxmesh::VertexHandle vh) { v(vh) = val; });

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}


template <typename T>
void populate(rxmesh::RXMeshStatic&           rxmesh,
              rxmesh::RXMeshFaceAttribute<T>& f,
              T                               val)
{
    rxmesh.for_each_face(
        rxmesh::DEVICE,
        [f, val] __device__(const rxmesh::FaceHandle fh) { f(fh) = val; });

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

template <typename T>
void populate(rxmesh::RXMeshStatic&             rxmesh,
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
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(RXMeshAttribute, Norm2)
{
    using namespace rxmesh;

    CUDA_ERROR(cudaDeviceReset());

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    ASSERT_TRUE(
        import_obj(STRINGIFY(INPUT_DIR) "sphere3.obj", Verts, Faces, true));

    RXMeshStatic rxmesh(Faces, rxmesh_args.quite);

    auto attr = rxmesh.add_vertex_attribute<float>("v", 3, rxmesh::DEVICE);

    const float val(2.0);

    populate<float>(rxmesh, *attr, val);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ReduceHandle reduce(*attr);

    float output = reduce.norm2(*attr);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_FLOAT_EQ(output, std::sqrt(val * val * rxmesh.get_num_vertices()));
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

    populate<float>(rxmesh, *v1_attr, *v2_attr, v1_val, v2_val);

    ReduceHandle reduce(*v1_attr);

    float output = reduce.dot(*v1_attr, *v2_attr);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_FLOAT_EQ(output, v1_val * v2_val * rxmesh.get_num_vertices());
}


TEST(RXMeshAttribute, CopyFrom)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    ASSERT_TRUE(
        import_obj(STRINGIFY(INPUT_DIR) "sphere3.obj", Verts, Faces, true));


    RXMeshStatic rxmesh(Faces, rxmesh_args.quite);

    auto f_device = rxmesh.add_face_attribute<uint32_t>("d", 3, DEVICE);

    auto f_host = rxmesh.add_face_attribute<uint32_t>("h", 3, HOST);

    uint32_t val = 99;

    populate<uint32_t>(rxmesh, *f_device, val);

    f_host->copy_from(*f_device, DEVICE, HOST);

    rxmesh.for_each_face(
        HOST, [&](const FaceHandle fh) { EXPECT_EQ((*f_host)(fh), val); });
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


    vertex_attr->move(rxmesh::HOST, rxmesh::DEVICE);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // this is not neccessary in general but we are just testing the
    // functionality here
    rxmesh.remove_attribute(attr_name);
}