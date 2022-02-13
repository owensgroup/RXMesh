#include "gtest/gtest.h"
#include "rxmesh/attribute.h"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/util/macros.h"

template <typename T>
void populate(rxmesh::RXMeshStatic&       rxmesh,
              rxmesh::VertexAttribute<T>& v,
              T                           val)
{
    rxmesh.for_each_vertex(
        rxmesh::DEVICE,
        [v, val] __device__(const rxmesh::VertexHandle vh) { v(vh) = val; });

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}


template <typename T>
void populate(rxmesh::RXMeshStatic& rxmesh, rxmesh::FaceAttribute<T>& f, T val)
{
    rxmesh.for_each_face(
        rxmesh::DEVICE,
        [f, val] __device__(const rxmesh::FaceHandle fh) { f(fh) = val; });

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

template <typename T>
void populate(rxmesh::RXMeshStatic& rxmesh, rxmesh::EdgeAttribute<T>& e, T val)
{
    rxmesh.for_each_edge(rxmesh::DEVICE,
                         [e, val] __device__(const rxmesh::EdgeHandle eh) {
                             auto pl = eh.unpack();
                             e(eh)   = pl.first * pl.second;
                         });

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

template <typename T>
void populate(rxmesh::RXMeshStatic&       rxmesh,
              rxmesh::VertexAttribute<T>& v1,
              rxmesh::VertexAttribute<T>& v2,
              T                           v1_val,
              T                           v2_val)
{
    rxmesh.for_each_vertex(
        rxmesh::DEVICE,
        [v1, v2, v1_val, v2_val] __device__(const rxmesh::VertexHandle vh) {
            v1(vh) = v1_val;
            v2(vh) = v2_val;
        });
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(Attribute, Norm2)
{
    using namespace rxmesh;

    CUDA_ERROR(cudaDeviceReset());

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshStatic rxmesh(STRINGIFY(INPUT_DIR) "sphere3.obj", rxmesh_args.quite);

    auto attr = rxmesh.add_vertex_attribute<float>("v", 3, rxmesh::DEVICE);

    const float val(2.0);

    populate<float>(rxmesh, *attr, val);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ReduceHandle reduce_handle(*attr);

    float output = reduce_handle.norm2(*attr);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_FLOAT_EQ(output, std::sqrt(val * val * rxmesh.get_num_vertices()));
}


TEST(Attribute, Dot)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshStatic rxmesh(STRINGIFY(INPUT_DIR) "sphere3.obj", rxmesh_args.quite);

    auto v1_attr = rxmesh.add_vertex_attribute<float>("v1", 3, rxmesh::DEVICE);
    auto v2_attr = rxmesh.add_vertex_attribute<float>("v2", 3, rxmesh::DEVICE);

    const float v1_val(2.0);
    const float v2_val(3.0);

    populate<float>(rxmesh, *v1_attr, *v2_attr, v1_val, v2_val);

    ReduceHandle reduce_handle(*v1_attr);

    float output = reduce_handle.dot(*v1_attr, *v2_attr);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_FLOAT_EQ(output, v1_val * v2_val * rxmesh.get_num_vertices());
}

TEST(Attribute, Reduce)
{
    using namespace rxmesh;

    CUDA_ERROR(cudaDeviceReset());

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshStatic rxmesh(STRINGIFY(INPUT_DIR) "sphere3.obj", rxmesh_args.quite);

    auto attr = rxmesh.add_edge_attribute<uint32_t>("e", 3, rxmesh::DEVICE);

    const uint32_t val(2.0);

    populate<uint32_t>(rxmesh, *attr, val);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ReduceHandle reduce_handle(*attr);

    uint32_t output = reduce_handle.reduce(*attr, cub::Max(), 0);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t result = 0;
    rxmesh.for_each_edge(rxmesh::HOST, [&](const rxmesh::EdgeHandle eh) {
        auto pl = eh.unpack();
        result  = std::max(result, pl.first * pl.second);
    });

    EXPECT_EQ(output, result);
}


TEST(Attribute, CopyFrom)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshStatic rxmesh(STRINGIFY(INPUT_DIR) "sphere3.obj", rxmesh_args.quite);

    auto f_device = rxmesh.add_face_attribute<uint32_t>("d", 3, DEVICE);

    auto f_host = rxmesh.add_face_attribute<uint32_t>("h", 3, HOST);

    uint32_t val = 99;

    populate<uint32_t>(rxmesh, *f_device, val);

    f_host->copy_from(*f_device, DEVICE, HOST);

    rxmesh.for_each_face(
        HOST, [&](const FaceHandle fh) { EXPECT_EQ((*f_host)(fh), val); });
}

TEST(Attribute, AddingAndRemoving)
{
    using namespace rxmesh;

    cuda_query(rxmesh_args.device_id, rxmesh_args.quite);

    RXMeshStatic rxmesh(STRINGIFY(INPUT_DIR) "sphere3.obj", rxmesh_args.quite);

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