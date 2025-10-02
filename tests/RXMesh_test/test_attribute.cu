#include "gtest/gtest.h"
#include "rxmesh/attribute.h"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/macros.h"

template <typename T>
void populate(rxmesh::RXMeshStatic& rx, rxmesh::VertexAttribute<T>& v, T val)
{
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [v, val] __device__(const rxmesh::VertexHandle vh) { v(vh) = val; });

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}


template <typename T>
void populate(rxmesh::RXMeshStatic& rx, rxmesh::FaceAttribute<T>& f, T val)
{
    rx.for_each_face(
        rxmesh::DEVICE,
        [f, val] __device__(const rxmesh::FaceHandle fh) { f(fh) = val; });

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

template <typename T>
void populate(rxmesh::RXMeshStatic& rx, rxmesh::EdgeAttribute<T>& e, T val)
{
    rx.for_each_edge(rxmesh::DEVICE,
                     [e, val] __device__(const rxmesh::EdgeHandle eh) {
                         auto pl = eh.unpack();
                         e(eh)   = pl.first * pl.second;
                     });

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

template <typename T>
void populate(rxmesh::RXMeshStatic&       rx,
              rxmesh::VertexAttribute<T>& v1,
              rxmesh::VertexAttribute<T>& v2,
              T                           v1_val,
              T                           v2_val)
{
    rx.for_each_vertex(
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

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto attr = rx.add_vertex_attribute<float>("v", 3, rxmesh::DEVICE);

    const float val(2.0);

    populate<float>(rx, *attr, val);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ReduceHandle reduce_handle(*attr);

    float output = reduce_handle.norm2(*attr);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_FLOAT_EQ(output, std::sqrt(val * val * rx.get_num_vertices()));
}


TEST(Attribute, Dot)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto v1_attr = rx.add_vertex_attribute<float>("v1", 3, rxmesh::DEVICE);
    auto v2_attr = rx.add_vertex_attribute<float>("v2", 3, rxmesh::DEVICE);

    const float v1_val(2.0);
    const float v2_val(3.0);

    populate<float>(rx, *v1_attr, *v2_attr, v1_val, v2_val);

    ReduceHandle reduce_handle(*v1_attr);

    float output = reduce_handle.dot(*v1_attr, *v2_attr);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_FLOAT_EQ(output, v1_val * v2_val * rx.get_num_vertices());
}

TEST(Attribute, Reduce)
{
    using namespace rxmesh;

    CUDA_ERROR(cudaDeviceReset());

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto attr = rx.add_edge_attribute<uint32_t>("e", 3, rxmesh::DEVICE);

    const uint32_t val(2.0);

    populate<uint32_t>(rx, *attr, val);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ReduceHandle reduce_handle(*attr);

    uint32_t output = reduce_handle.reduce(*attr, cub::Max(), 0);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t result = 0;
    rx.for_each_edge(
        rxmesh::HOST,
        [&](const rxmesh::EdgeHandle eh) {
            auto pl = eh.unpack();
            result  = std::max(result, pl.first * pl.second);
        },
        NULL,
        false);

    EXPECT_EQ(output, result);
}


TEST(Attribute, ArgMax)
{
    using namespace rxmesh;

    CUDA_ERROR(cudaDeviceReset());

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bumpy-cube.obj");

    auto attr = *rx.add_vertex_attribute<float>("v", 1);

    const float val(2.0);

    populate<float>(rx, attr, val);

    attr.move(DEVICE, HOST);

    uint32_t chosenVertex = rx.get_num_vertices() - 1;

    VertexHandle chosenHandle;

    float chosenValue = 10;

    rx.for_each_vertex(rxmesh::HOST, [&](const rxmesh::VertexHandle vh) {
        if (rx.linear_id(vh) == chosenVertex) {
            attr(vh)     = chosenValue;
            chosenHandle = vh;
        }
    });

    EXPECT_TRUE(chosenHandle.is_valid());

    attr.move(HOST, DEVICE);

    ReduceHandle reduce_handle(attr);

    auto output = reduce_handle.arg_max(attr);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(output.key, chosenHandle);
    EXPECT_EQ(output.value, chosenValue);
}


TEST(Attribute, CopyFrom)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto f_device = rx.add_face_attribute<uint32_t>("d", 3, DEVICE);

    auto f_host = rx.add_face_attribute<uint32_t>("h", 3, HOST);

    uint32_t val = 99;

    populate<uint32_t>(rx, *f_device, val);

    f_host->copy_from(*f_device, DEVICE, HOST);

    rx.for_each_face(
        HOST, [&](const FaceHandle fh) { EXPECT_EQ((*f_host)(fh), val); });
}

TEST(Attribute, AddingAndRemoving)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    std::string attr_name = "v_attr";

    auto vertex_attr =
        rx.add_vertex_attribute<float>(attr_name, 3, rxmesh::LOCATION_ALL);

    EXPECT_TRUE(rx.does_attribute_exist(attr_name));


    vertex_attr->move(rxmesh::HOST, rxmesh::DEVICE);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // this is not neccessary in general but we are just testing the
    // functionality here
    rx.remove_attribute(attr_name);
}