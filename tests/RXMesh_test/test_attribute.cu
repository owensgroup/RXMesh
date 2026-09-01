#include "gtest/gtest.h"
#include "rxmesh/attribute.h"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/macros.h"

#include <array>
#include <limits>

using namespace rxmesh;

template <typename T>
void populate(RXMeshStatic& rx, VertexAttribute<T>& v, T val)
{
    rx.for_each_vertex(
        DEVICE, [v, val] __device__(const VertexHandle vh) { v(vh) = val; });

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}


template <typename T>
void populate(RXMeshStatic& rx, FaceAttribute<T>& f, T val)
{
    rx.for_each_face(DEVICE,
                     [f, val] __device__(const FaceHandle fh) { f(fh) = val; });

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

template <typename T>
void populate(RXMeshStatic& rx, EdgeAttribute<T>& e, T val)
{
    rx.for_each_edge(DEVICE, [e, val] __device__(const EdgeHandle eh) {
        auto pl = eh.unpack();
        e(eh)   = pl.first * pl.second;
    });

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

template <typename T>
void populate(RXMeshStatic&       rx,
              VertexAttribute<T>& v1,
              VertexAttribute<T>& v2,
              T                   v1_val,
              T                   v2_val)
{
    rx.for_each_vertex(
        DEVICE, [v1, v2, v1_val, v2_val] __device__(const VertexHandle vh) {
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

    auto attr = rx.add_vertex_attribute<float>("v", 3, DEVICE);

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

    auto v1_attr = rx.add_vertex_attribute<float>("v1", 3, DEVICE);
    auto v2_attr = rx.add_vertex_attribute<float>("v2", 3, DEVICE);

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

    auto attr = rx.add_edge_attribute<uint32_t>("e", 3, DEVICE);

    const uint32_t val(2.0);

    populate<uint32_t>(rx, *attr, val);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ReduceHandle reduce_handle(*attr);

    uint32_t output = reduce_handle.reduce(*attr, cub::Max(), 0);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint32_t result = 0;
    rx.for_each_edge(
        HOST,
        [&](const EdgeHandle eh) {
            auto pl = eh.unpack();
            result  = std::max(result, pl.first * pl.second);
        },
        NULL,
        false);

    EXPECT_EQ(output, result);
}

TEST(Attribute, ReduceDevice)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");
    auto attr = rx.add_vertex_attribute<float>("reduce_device", 1, DEVICE);
    constexpr float value = 2.5f;
    populate<float>(rx, *attr, value);

    ReduceHandle<float, VertexHandle> reducer(*attr);
    float*                            device_output = nullptr;
    CUDA_ERROR(cudaMalloc(&device_output, sizeof(float)));
    cudaStream_t stream = nullptr;
    CUDA_ERROR(cudaStreamCreate(&stream));

    reducer.reduce_device(*attr, cub::Sum(), 0.0f, device_output, 0, stream);
    float output = 0.0f;
    CUDA_ERROR(cudaMemcpyAsync(
        &output, device_output, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_ERROR(cudaStreamSynchronize(stream));
    EXPECT_FLOAT_EQ(output, value * rx.get_num_vertices());

    reducer.reduce_device(*attr,
                          cub::Max(),
                          std::numeric_limits<float>::lowest(),
                          device_output,
                          0,
                          stream);
    CUDA_ERROR(cudaMemcpyAsync(
        &output, device_output, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_ERROR(cudaStreamSynchronize(stream));
    EXPECT_FLOAT_EQ(output, value);

    CUDA_ERROR(cudaStreamDestroy(stream));
    GPU_FREE(device_output);
}

TEST(Attribute, ReduceDeviceZeroPatchWritesInit)
{
    Attribute<float, VertexHandle>    empty_attr;
    ReduceHandle<float, VertexHandle> reducer(0);
    float*                            device_output = nullptr;
    CUDA_ERROR(cudaMalloc(&device_output, sizeof(float)));

    reducer.reduce_device(
        empty_attr, cub::Sum(), 7.25f, device_output, INVALID32, nullptr);
    float output = 0.0f;
    CUDA_ERROR(cudaMemcpy(
        &output, device_output, sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_FLOAT_EQ(output, 7.25f);

    GPU_FREE(device_output);
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

    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
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
        rx.add_vertex_attribute<float>(attr_name, 3, LOCATION_ALL);

    EXPECT_TRUE(rx.does_attribute_exist(attr_name));


    vertex_attr->move(HOST, DEVICE);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // this is not neccessary in general but we are just testing the
    // functionality here
    rx.remove_attribute(attr_name);
}

TEST(Attribute, DefaultLayoutIsAoSoA)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto attr = rx.add_vertex_attribute<float>("default_layout", 3, HOST);

    EXPECT_EQ(attr->get_layout(), AoSoA);
    EXPECT_STREQ(layout_to_string(attr->get_layout()).c_str(), "AoSoA");
}

TEST(Attribute, TrueSoAHostStorageIsColumnMajor)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto attr = rx.add_vertex_attribute<float>("true_soa", 3, HOST, SoA);

    const uint32_t n = rx.get_num_vertices();
    ASSERT_TRUE(attr->is_tensor_layout());
    ASSERT_EQ(attr->storage_size(), size_t(n) * attr->get_num_attributes());

    float* data = attr->data(HOST);
    for (uint32_t c = 0; c < attr->get_num_attributes(); ++c) {
        for (uint32_t i = 0; i < n; ++i) {
            data[c * n + i] = float(c * 1000 + i);
        }
    }

    rx.for_each_vertex(
        HOST,
        [&](const VertexHandle vh) {
            const uint32_t row = rx.linear_id(vh);
            for (uint32_t c = 0; c < attr->get_num_attributes(); ++c) {
                EXPECT_FLOAT_EQ((*attr)(vh, c), float(c * 1000 + row));
                EXPECT_FLOAT_EQ((*attr)(row, c), float(c * 1000 + row));
            }
        },
        NULL,
        false);
}

template <typename AttrT>
void device_write(RXMeshStatic& rx, AttrT& attr)
{
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle vh) mutable {
        attr(vh, 0) = 11.0f;
        attr(vh, 1) = 22.0f;
        attr(vh, 2) = 33.0f;
    });
}
TEST(Attribute, TrueSoADeviceWritesColumnMajor)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto attr =
        rx.add_vertex_attribute<float>("true_soa_device", 3, LOCATION_ALL, SoA);

    device_write(rx, *attr);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    attr->move(DEVICE, HOST);

    const uint32_t n    = rx.get_num_vertices();
    const float*   data = attr->data(HOST);
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(data[i], 11.0f);
        EXPECT_FLOAT_EQ(data[n + i], 22.0f);
        EXPECT_FLOAT_EQ(data[2 * n + i], 33.0f);
    }
}

TEST(Attribute, ResetSetsAllComponents)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    const std::array<layoutT, 3> layouts = {AoS, AoSoA, SoA};

    for (layoutT layout : layouts) {
        auto attr = rx.add_vertex_attribute<float>(
            "reset_" + layout_to_string(layout), 3, HOST, layout);

        attr->reset(5.0f, HOST);

        rx.for_each_vertex(
            HOST,
            [&](const VertexHandle vh) {
                for (uint32_t c = 0; c < attr->get_num_attributes(); ++c) {
                    EXPECT_FLOAT_EQ((*attr)(vh, c), 5.0f);
                }
            },
            NULL,
            false);
    }
}

TEST(Attribute, ToAndFromMatrixPreserveLayouts)
{
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    const std::array<layoutT, 3> layouts = {AoS, AoSoA, SoA};

    for (layoutT layout : layouts) {
        auto src = rx.add_vertex_attribute<float>(
            "matrix_src_" + layout_to_string(layout), 3, HOST, layout);

        rx.for_each_vertex(
            HOST,
            [&](const VertexHandle vh) {
                const uint32_t row = rx.linear_id(vh);
                for (uint32_t c = 0; c < src->get_num_attributes(); ++c) {
                    (*src)(vh, c) = float(c * 1000 + row);
                }
            },
            NULL,
            false);

        auto mat = src->to_matrix<>();

        auto dst = rx.add_vertex_attribute<float>(
            "matrix_dst_" + layout_to_string(layout), 3, HOST, layout);
        dst->from_matrix(mat.get());

        rx.for_each_vertex(
            HOST,
            [&](const VertexHandle vh) {
                for (uint32_t c = 0; c < src->get_num_attributes(); ++c) {
                    EXPECT_FLOAT_EQ((*dst)(vh, c), (*src)(vh, c));
                }
            },
            NULL,
            false);
    }
}

TEST(Attribute, ExternalDeviceBuffer)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto attr =
        rx.add_vertex_attribute<float>("external_soa", 3, LOCATION_NONE, SoA);

    const size_t bytes  = size_t(rx.get_num_vertices()) * 3 * sizeof(float);
    float*       first  = nullptr;
    float*       second = nullptr;
    CUDA_ERROR(cudaMalloc(&first, bytes));
    CUDA_ERROR(cudaMalloc(&second, bytes));

    ASSERT_TRUE(attr->attach_device_buffer(first, bytes));
    EXPECT_TRUE(attr->is_external_device_buffer());
    EXPECT_TRUE(attr->is_device_allocated());
    EXPECT_EQ(attr->data(DEVICE), first);

    device_write(rx, *attr);
    std::vector<float> values(rx.get_num_vertices() * 3);
    ASSERT_EQ(cudaMemcpy(values.data(), first, bytes, cudaMemcpyDeviceToHost),
              cudaSuccess);
    const size_t n = rx.get_num_vertices();
    for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(values[i], 11.0f);
        EXPECT_FLOAT_EQ(values[n + i], 22.0f);
        EXPECT_FLOAT_EQ(values[2 * n + i], 33.0f);
    }

    ASSERT_TRUE(attr->attach_device_buffer(second, bytes));
    EXPECT_EQ(attr->data(DEVICE), second);
    EXPECT_EQ(cudaMemset(first, 0, bytes), cudaSuccess);

    attr->release(DEVICE);
    EXPECT_FALSE(attr->is_external_device_buffer());
    EXPECT_FALSE(attr->is_device_allocated());
    EXPECT_EQ(attr->data(DEVICE), nullptr);
    EXPECT_EQ(cudaMemset(second, 0, bytes), cudaSuccess);

    attr->detach_device_buffer();
    GPU_FREE(first);
    GPU_FREE(second);
}

TEST(Attribute, ExternalDeviceBufferValidation)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto attr = rx.add_vertex_attribute<float>("ext", 3, LOCATION_NONE, SoA);

    const size_t bytes     = size_t(rx.get_num_vertices()) * 3 * sizeof(float);
    float*       external  = nullptr;
    float*       too_small = nullptr;
    CUDA_ERROR(cudaMalloc(&external, bytes));
    CUDA_ERROR(cudaMalloc(&too_small, sizeof(float)));

    ASSERT_TRUE(attr->attach_device_buffer(external, bytes));
    EXPECT_FALSE(attr->attach_device_buffer(too_small, sizeof(float)));
    EXPECT_EQ(attr->data(DEVICE), external);
    EXPECT_TRUE(attr->is_external_device_buffer());

    std::vector<float> host(rx.get_num_vertices() * 3);
    EXPECT_FALSE(attr->attach_device_buffer(host.data(), bytes));
    EXPECT_EQ(attr->data(DEVICE), external);

    auto non_soa =
        rx.add_vertex_attribute<float>("ext_non_soa", 3, LOCATION_NONE, AoSoA);
    EXPECT_FALSE(non_soa->attach_device_buffer(external, bytes));
    EXPECT_FALSE(non_soa->is_device_allocated());

    rx.remove_attribute("ext");
    EXPECT_FALSE(attr->is_external_device_buffer());
    EXPECT_EQ(cudaMemset(external, 0, bytes), cudaSuccess);

    GPU_FREE(external);
    GPU_FREE(too_small);
}

TEST(Attribute, DeviceBufferReleasedBeforeExternalAttach)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto attr =
        rx.add_vertex_attribute<float>("owned_to_external", 1, DEVICE, SoA);

    float* owned = attr->data(DEVICE);
    ASSERT_NE(owned, nullptr);
    cudaPointerAttributes pointer_attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&pointer_attributes, owned),
              cudaSuccess);

    const size_t bytes    = size_t(rx.get_num_vertices()) * sizeof(float);
    float*       external = nullptr;
    CUDA_ERROR(cudaMalloc(&external, bytes));
    ASSERT_NE(external, owned);

    ASSERT_TRUE(attr->attach_device_buffer(external, bytes));
    EXPECT_EQ(attr->data(DEVICE), external);
    EXPECT_TRUE(attr->is_external_device_buffer());

    // The previous owned allocation was released, while the borrowed one is
    // still owned by caller and usable after Attribute release.
    const cudaError_t owned_status =
        cudaPointerGetAttributes(&pointer_attributes, owned);
    if (owned_status == cudaSuccess) {
        // CUDA 11+ can report a successfully queried, freed pointer as
        // unregistered host memory instead of returning an error.
        EXPECT_EQ(pointer_attributes.type, cudaMemoryTypeUnregistered);
    } else {
        cudaGetLastError();
    }
    attr->release(DEVICE);
    EXPECT_EQ(cudaMemset(external, 0, bytes), cudaSuccess);
    GPU_FREE(external);
}

TEST(Attribute, ExternalDeviceBufferRequiresDetachBeforeMove)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");
    auto         attr =
        rx.add_vertex_attribute<float>("external_detach_move", 1, HOST, SoA);
    attr->reset(4.0f, HOST);

    const size_t bytes    = size_t(rx.get_num_vertices()) * sizeof(float);
    float*       external = nullptr;
    CUDA_ERROR(cudaMalloc(&external, bytes));
    CUDA_ERROR(cudaMemset(external, 0, bytes));
    ASSERT_TRUE(attr->attach_device_buffer(external, bytes));

    // Moving to DEVICE cannot silently overwrite borrowed storage
    attr->move(HOST, DEVICE);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> values(rx.get_num_vertices(), -1.0f);
    ASSERT_EQ(
        cudaMemcpy(values.data(), external, bytes, cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }

    attr->detach_device_buffer();
    attr->move(HOST, DEVICE);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_TRUE(attr->is_device_allocated());
    EXPECT_FALSE(attr->is_external_device_buffer());
    EXPECT_NE(attr->data(DEVICE), external);

    attr->release(DEVICE);
    EXPECT_EQ(cudaMemset(external, 0, bytes), cudaSuccess);
    GPU_FREE(external);
}

TEST(Attribute, MeshDestructionDoesNotFreeExternalDeviceBuffer)
{
    float* external = nullptr;
    size_t bytes    = 0;
    {
        RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");
        bytes = size_t(rx.get_num_vertices()) * sizeof(float);
        CUDA_ERROR(cudaMalloc(&external, bytes));
        auto attr = rx.add_vertex_attribute<float>(
            "external_mesh_lifetime", 1, LOCATION_NONE, SoA);
        ASSERT_TRUE(attr->attach_device_buffer(external, bytes));
    }

    EXPECT_EQ(cudaMemset(external, 0, bytes), cudaSuccess);
    GPU_FREE(external);
}

struct TetCustomMin
{
    template <typename T>
    __device__ __forceinline__ T operator()(const T& a, const T& b) const
    {
        return b < a ? b : a;
    }
};

static std::vector<std::vector<uint32_t>> make_tet_chain(uint32_t num_tets)
{
    std::vector<std::vector<uint32_t>> tets;
    for (uint32_t t = 0; t < num_tets; ++t) {
        tets.push_back({t, t + 1, t + 2, t + 3});
    }
    return tets;
}

TEST(Attribute, TetLayoutsAndAPI)
{
    auto         tets = make_tet_chain(4);
    RXMeshStatic rx(tets, "", 1);
    ASSERT_EQ(rx.get_num_tets(), tets.size());
    ASSERT_GT(rx.get_num_patches(), 1);

    std::vector<std::vector<float>> values(rx.get_num_tets(),
                                           std::vector<float>(3));
    std::vector<float>              scalar_values(rx.get_num_tets());
    for (uint32_t t = 0; t < rx.get_num_tets(); ++t) {
        scalar_values[t] = float(10 + t);
        for (uint32_t c = 0; c < 3; ++c) {
            values[t][c] = float(100 * c + t);
        }
    }

    const std::array<layoutT, 3> layouts = {AoS, AoSoA, SoA};

    TetHandle owner_handle;


    for (layoutT layout : layouts) {
        const std::string suffix = layout_to_string(layout);

        std::shared_ptr<TetAttribute<float>> attr =
            rx.add_tet_attribute<float>(values, "tet_" + suffix, layout);

        EXPECT_EQ(attr->get_layout(), layout);
        EXPECT_TRUE(attr->is_host_allocated());
        EXPECT_TRUE(attr->is_device_allocated());
        EXPECT_EQ(attr->rows(), rx.get_num_tets());
        EXPECT_EQ(attr->cols(), 3);

        rx.for_each_tet(
            HOST,
            [&](const TetHandle th) {
                const uint32_t global = rx.map_to_global(th);
                const uint32_t row    = rx.linear_id(th);
                for (uint32_t c = 0; c < 3; ++c) {
                    EXPECT_FLOAT_EQ((*attr)(th, c), values[global][c]);
                    EXPECT_FLOAT_EQ((*attr)(row, c), values[global][c]);
                }
            },
            nullptr,
            false);

        CUDA_ERROR(cudaGetLastError());

        auto copy =
            rx.add_tet_attribute_like<float>("tet_copy_" + suffix, *attr);

        copy->copy_from(*attr, DEVICE, DEVICE);
        copy->reset(-1.0f, HOST);
        copy->move(DEVICE, HOST);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        auto matrix      = attr->to_matrix<>();
        auto from_matrix = rx.add_tet_attribute<float>(
            "tet_matrix_" + suffix, 3, HOST, layout);
        from_matrix->from_matrix(matrix.get());

        rx.for_each_tet(
            HOST,
            [&](const TetHandle th) {
                const DenseMatrix<float>::IndexT row =
                    static_cast<DenseMatrix<float>::IndexT>(rx.linear_id(th));
                for (uint32_t c = 0; c < 3; ++c) {
                    EXPECT_FLOAT_EQ((*matrix)(row, c), (*attr)(th, c));
                    EXPECT_FLOAT_EQ((*copy)(th, c), (*attr)(th, c));
                    EXPECT_FLOAT_EQ((*from_matrix)(th, c), (*attr)(th, c));
                }
            },
            nullptr,
            false);

        const std::string copy_name = "tet_copy_" + suffix;
        EXPECT_TRUE(rx.does_attribute_exist(copy_name));
        rx.remove_attribute(copy_name);
        EXPECT_FALSE(rx.does_attribute_exist(copy_name));
    }


    auto flat = rx.add_tet_attribute<float>(scalar_values, "tet_flat", SoA);
    rx.for_each_tet(
        HOST,
        [&](const TetHandle th) {
            EXPECT_FLOAT_EQ((*flat)(th), scalar_values[rx.map_to_global(th)]);
        },
        nullptr,
        false);

    auto host =
        rx.add_attribute<float, TetHandle>("tet_generic_host", 2, HOST, AoS);
    host->reset(-4.0f, HOST);
    auto host_like =
        rx.add_attribute_like<float, TetHandle>("tet_generic_host_like", *host);
    host_like->copy_from(*host, HOST, HOST);

    auto device = rx.add_tet_attribute<float>("tet_device", 2, DEVICE, AoSoA);
    device->reset(6.0f, DEVICE);
    device->move(DEVICE, HOST);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    rx.for_each_tet(
        HOST,
        [&](const TetHandle th) {
            for (uint32_t c = 0; c < 2; ++c) {
                EXPECT_FLOAT_EQ((*host_like)(th, c), -4.0f);
                EXPECT_FLOAT_EQ((*device)(th, c), 6.0f);
            }
        },
        nullptr,
        false);
}

TEST(Attribute, TetReductions)
{
    auto         tets = make_tet_chain(4);
    RXMeshStatic rx(tets, "", 1);
    ASSERT_GT(rx.get_num_patches(), 1);

    std::vector<std::array<float, 3>> values(rx.get_num_tets());
    std::vector<std::array<float, 3>> rhs(rx.get_num_tets());
    for (uint32_t t = 0; t < rx.get_num_tets(); ++t) {
        values[t] = {float(3 * t) - 5.0f, float(t + 1), float(3 * t) - 2.0f};
        rhs[t]    = {float(t + 2), -float(t + 1), 0.5f * float(t + 1)};
    }

    float expected_dot   = 0.0f;
    float expected_dot_1 = 0.0f;
    float expected_norm  = 0.0f;
    float expected_sum_0 = 0.0f;
    float expected_sum_2 = 0.0f;
    for (uint32_t t = 0; t < rx.get_num_tets(); ++t) {
        expected_sum_0 += values[t][0];
        expected_sum_2 += values[t][2];
        expected_dot_1 += values[t][1] * rhs[t][1];
        for (uint32_t c = 0; c < 3; ++c) {
            expected_dot += values[t][c] * rhs[t][c];
            expected_norm += values[t][c] * values[t][c];
        }
    }
    expected_norm = std::sqrt(expected_norm);

    float* device_output = nullptr;
    CUDA_ERROR(cudaMalloc(&device_output, sizeof(float)));
    cudaStream_t stream = nullptr;
    CUDA_ERROR(cudaStreamCreate(&stream));

    const std::array<layoutT, 3> layouts = {AoS, AoSoA, SoA};
    for (layoutT layout : layouts) {
        const std::string suffix = layout_to_string(layout);

        auto attr = rx.add_tet_attribute<float>(
            "tet_reduce_" + suffix, 3, LOCATION_ALL, layout);

        auto other = rx.add_tet_attribute<float>(
            "tet_reduce_rhs_" + suffix, 3, LOCATION_ALL, layout);

        attr->reset(1000.0f, HOST);
        other->reset(-1000.0f, HOST);
        rx.for_each_tet(
            HOST,
            [&](const TetHandle th) {
                const uint32_t global = rx.map_to_global(th);
                for (uint32_t c = 0; c < 3; ++c) {
                    (*attr)(th, c)  = values[global][c];
                    (*other)(th, c) = rhs[global][c];
                }
            },
            nullptr,
            false);
        attr->move(HOST, DEVICE, stream);
        other->move(HOST, DEVICE, stream);

        TetReduceHandle<float> reducer(*attr);
        EXPECT_NEAR(
            reducer.dot(*attr, *other, INVALID32, stream), expected_dot, 1e-5f);
        EXPECT_NEAR(
            reducer.dot(*attr, *other, 1, stream), expected_dot_1, 1e-5f);
        EXPECT_NEAR(
            reducer.norm2(*attr, INVALID32, stream), expected_norm, 1e-5f);
        EXPECT_FLOAT_EQ(reducer.reduce(*attr, cub::Sum(), 0.0f, 0, stream),
                        expected_sum_0);
        EXPECT_FLOAT_EQ(reducer.reduce(*attr,
                                       TetCustomMin(),
                                       std::numeric_limits<float>::max(),
                                       0,
                                       stream),
                        values.front()[0]);
        EXPECT_FLOAT_EQ(reducer.reduce(*attr,
                                       cub::Max(),
                                       std::numeric_limits<float>::lowest(),
                                       0,
                                       stream),
                        values.back()[0]);

        const auto arg_min = reducer.arg_min(*attr, 0, stream);
        const auto arg_max = reducer.arg_max(*attr, 0, stream);
        EXPECT_EQ(rx.map_to_global(arg_min.key), 0);
        EXPECT_EQ(rx.map_to_global(arg_max.key), rx.get_num_tets() - 1);
        EXPECT_FLOAT_EQ(arg_min.value, values.front()[0]);
        EXPECT_FLOAT_EQ(arg_max.value, values.back()[0]);

        reducer.reduce_device(
            *attr, cub::Sum(), 0.0f, device_output, 2, stream);
        float output = 0.0f;
        CUDA_ERROR(cudaMemcpyAsync(&output,
                                   device_output,
                                   sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_ERROR(cudaStreamSynchronize(stream));
        EXPECT_FLOAT_EQ(output, expected_sum_2);
    }

    CUDA_ERROR(cudaStreamDestroy(stream));
    GPU_FREE(device_output);
}
