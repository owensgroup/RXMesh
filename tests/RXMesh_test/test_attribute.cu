#include "gtest/gtest.h"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/vector.h"

/**
 * test_vector()
 */
__global__ static void test_vector(
    RXMESH::RXMeshAttribute<RXMESH::Vector3f> mesh_attr,
    uint32_t*                                 suceess)
{

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *suceess = 1;

        assert((mesh_attr.get_allocated() & RXMESH::DEVICE) == RXMESH::DEVICE);
        uint32_t num_mesh_elements = mesh_attr.get_num_mesh_elements();
        for (uint32_t i = 0; i < num_mesh_elements; ++i) {
            const auto& vec = mesh_attr(i);
            if (vec[0] != i + 0 || vec[1] != i + 1 || vec[2] != i + 2) {
                *suceess = 0;
                return;
            }
        }
    }
}

/**
 * test_values()
 */
template <class T>
__global__ static void test_values(RXMESH::RXMeshAttribute<T> mesh_attr,
                                   uint32_t*                  suceess)
{

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *suceess = 1;

        assert((mesh_attr.get_allocated() & RXMESH::DEVICE) == RXMESH::DEVICE);
        uint32_t num_mesh_elements = mesh_attr.get_num_mesh_elements();
        for (uint32_t i = 0; i < num_mesh_elements; ++i) {
            for (uint32_t j = 0; j < mesh_attr.get_num_attribute_per_element();
                 ++j) {
                if (mesh_attr(i, j) != i + j) {

                    *suceess = 0;
                    return;
                }
            }
        }
    }
}

/**
 * generate_values()
 */
template <class T>
__global__ static void generate_values(RXMESH::RXMeshAttribute<T> mesh_attr)
{

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        assert((mesh_attr.get_allocated() & RXMESH::DEVICE) == RXMESH::DEVICE);

        uint32_t num_mesh_elements = mesh_attr.get_num_mesh_elements();
        for (uint32_t i = 0; i < num_mesh_elements; ++i) {
            for (uint32_t j = 0; j < mesh_attr.get_num_attribute_per_element();
                 ++j) {
                mesh_attr(i, j) = i + j;
            }
        }
    }
}


bool test_host(uint32_t attributes_per_element)
{
    using namespace RXMESH;
    // mesh attr on host
    uint32_t                       num_mesh_elements = 2048;
    RXMESH::RXMeshAttribute<float> rxmesh_attr;

    rxmesh_attr.set_name("float_attr");
    rxmesh_attr.init(num_mesh_elements, attributes_per_element, RXMESH::HOST,
                     RXMESH::AoS);

    // generate some numbers as AoS
    for (uint32_t i = 0; i < num_mesh_elements; ++i) {
        for (uint32_t j = 0; j < attributes_per_element; ++j) {
            rxmesh_attr(i, j) = i + j;
        }
    }

    // change the layout to SoA (good for gpu)
    rxmesh_attr.change_layout(RXMESH::HOST);

    // move memory to device
    rxmesh_attr.move(RXMESH::HOST, RXMESH::DEVICE);


    // device success variable
    uint32_t* d_success = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_success, sizeof(uint32_t)));


    // actual testing
    test_values<float><<<1, 1>>>(rxmesh_attr, d_success);

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

    // reporting
    return h_success == 1;
}


bool test_device(uint32_t attributes_per_element)
{
    using namespace RXMESH;
    // Test generating values on device and processing it on host

    // mesh attr on host (but allocated on device)
    uint32_t                          num_mesh_elements = 2048;
    RXMESH::RXMeshAttribute<uint32_t> rxmesh_attr;
    rxmesh_attr.set_name("int_attr");
    rxmesh_attr.init(num_mesh_elements, attributes_per_element, RXMESH::DEVICE);


    // generate some numbers on device
    generate_values<<<1, 1>>>(rxmesh_attr);

    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());


    // move the generate values to host
    rxmesh_attr.move(RXMESH::DEVICE, RXMESH::HOST);

    // change the layout to SoA
    rxmesh_attr.change_layout(RXMESH::HOST);

    // testing
    bool suceess = true;
    assert((rxmesh_attr.get_allocated() & RXMESH::HOST) == RXMESH::HOST);
    num_mesh_elements = rxmesh_attr.get_num_mesh_elements();

    for (uint32_t i = 0; i < num_mesh_elements; ++i) {
        for (uint32_t j = 0; j < attributes_per_element; ++j) {
            if (rxmesh_attr(i, j) != i + j) {
                suceess = false;
                break;
            }
        }
        if (!suceess) {
            break;
        }
    }

    // release rxmesh_attribute memory on host and device
    rxmesh_attr.release();

    return suceess;
}

/*bool test_vector()
{
    using namespace RXMESH;
    // mesh attr on host
    uint32_t                         num_mesh_elements = 2048;
    RXMESH::RXMeshAttribute<Vector3f> rxmesh_attr;

    rxmesh_attr.set_name("vector3f_attr");
    rxmesh_attr.init(num_mesh_elements, 1, RXMESH::HOST, RXMESH::AoS);

    // generate some numbers as AoS
    for (uint32_t i = 0; i < num_mesh_elements; ++i) {
        auto& vec = rxmesh_attr(i);
        vec[0] = i + 0;
        vec[1] = i + 1;
        vec[2] = i + 2;
    }

    // move memory to device
    rxmesh_attr.move(RXMESH::HOST, RXMESH::DEVICE);


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

    // reporting
    return h_success == 1;
}*/

bool test_axpy(uint32_t attributes_per_element)
{
    using namespace RXMESH;

    float x_val(1.0), y_val(3.0), alpha_val(5.0), beta_val(7.0);

    uint32_t                       num_mesh_elements = 2048;
    RXMESH::RXMeshAttribute<float> X;
    RXMESH::RXMeshAttribute<float> Y;

    X.set_name("X");
    Y.set_name("Y");
    X.init(num_mesh_elements, attributes_per_element, RXMESH::HOST,
           RXMESH::AoS);
    Y.init(num_mesh_elements, attributes_per_element, RXMESH::HOST,
           RXMESH::AoS);

    // generate some numbers as AoS
    for (uint32_t i = 0; i < num_mesh_elements; ++i) {
        for (uint32_t j = 0; j < attributes_per_element; ++j) {
            X(i, j) = x_val;
            Y(i, j) = y_val;
        }
    }

    X.change_layout(RXMESH::HOST);
    Y.change_layout(RXMESH::HOST);
    X.move(RXMESH::HOST, RXMESH::DEVICE);
    Y.move(RXMESH::HOST, RXMESH::DEVICE);

    // call axpy
    Vector<3, float> alpha(alpha_val);
    Vector<3, float> beta(beta_val);
    Y.axpy(X, alpha, beta);

    // sync
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaPeekAtLastError());
    CUDA_ERROR(cudaGetLastError());


    // move to host (don't need to move X
    Y.move(RXMESH::DEVICE, RXMESH::HOST);

    // check results
    bool is_passed = true;
    for (uint32_t i = 0; i < num_mesh_elements; ++i) {
        for (uint32_t j = 0; j < attributes_per_element; ++j) {
            if (std::abs(Y(i, j) - (alpha_val * x_val + beta_val * y_val)) >
                0.0001) {
                is_passed = false;
                break;
            }
        }
        if (!is_passed) {
            break;
        }
    }

    // release rxmesh_attribute memory on host and device
    X.release();
    Y.release();


    return is_passed;
}


bool test_reduce()
{
    using namespace RXMESH;
    constexpr uint32_t             attributes_per_element = 3;
    uint32_t                       num_mesh_elements = 2048;
    RXMESH::RXMeshAttribute<float> X;

    X.set_name("X");
    X.init(num_mesh_elements, attributes_per_element, RXMESH::HOST,
           RXMESH::AoS);

    // generate some numbers as AoS
    for (uint32_t i = 0; i < num_mesh_elements; ++i) {
        for (uint32_t j = 0; j < attributes_per_element; ++j) {
            X(i, j) = j + 1;
        }
    }

    X.change_layout(RXMESH::HOST);
    X.move(RXMESH::HOST, RXMESH::DEVICE);
    Vector<attributes_per_element, float> output;

    // call reduce
    X.reduce(output, RXMESH::SUM);


    // sync
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaPeekAtLastError());
    CUDA_ERROR(cudaGetLastError());

    bool is_passed = true;

    for (uint32_t j = 0; j < attributes_per_element; ++j) {
        if (output[j] != num_mesh_elements * (j + 1)) {
            is_passed = false;
            break;
        }
    }

    // release rxmesh_attribute memory on host and device
    X.release();


    return is_passed;
}


bool test_norm2()
{
    using namespace RXMESH;
    constexpr uint32_t             attributes_per_element = 3;
    uint32_t                       num_mesh_elements = 2048;
    RXMESH::RXMeshAttribute<float> X;

    X.set_name("X");
    X.init(num_mesh_elements, attributes_per_element, RXMESH::HOST,
           RXMESH::AoS);

    // generate some numbers as AoS
    for (uint32_t i = 0; i < num_mesh_elements; ++i) {
        for (uint32_t j = 0; j < attributes_per_element; ++j) {
            X(i, j) = 2;
        }
    }

    X.change_layout(RXMESH::HOST);
    X.move(RXMESH::HOST, RXMESH::DEVICE);
    Vector<attributes_per_element, float> output;

    // call reduce
    X.reduce(output, RXMESH::NORM2);


    // sync
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaPeekAtLastError());
    CUDA_ERROR(cudaGetLastError());

    bool is_passed = true;

    for (uint32_t j = 0; j < attributes_per_element; ++j) {
        if (output[j] != 4 * num_mesh_elements) {
            is_passed = false;
            break;
        }
    }

    // release rxmesh_attribute memory on host and device
    X.release();


    return is_passed;
}


bool test_dot()
{
    using namespace RXMESH;
    constexpr uint32_t             attributes_per_element = 3;
    uint32_t                       num_mesh_elements = 2048;
    RXMESH::RXMeshAttribute<float> X;
    RXMESH::RXMeshAttribute<float> Y;

    X.set_name("X");
    Y.set_name("Y");
    X.init(num_mesh_elements, attributes_per_element, RXMESH::HOST,
           RXMESH::AoS);
    Y.init(num_mesh_elements, attributes_per_element, RXMESH::HOST,
           RXMESH::AoS);

    // generate some numbers as AoS
    for (uint32_t i = 0; i < num_mesh_elements; ++i) {
        for (uint32_t j = 0; j < attributes_per_element; ++j) {
            X(i, j) = 2;
            Y(i, j) = 3;
        }
    }

    X.change_layout(RXMESH::HOST);
    X.move(RXMESH::HOST, RXMESH::DEVICE);
    Y.change_layout(RXMESH::HOST);
    Y.move(RXMESH::HOST, RXMESH::DEVICE);
    Vector<attributes_per_element, float> output;

    // call reduce
    X.reduce(output, RXMESH::DOT, &Y);


    // sync
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaPeekAtLastError());
    CUDA_ERROR(cudaGetLastError());

    bool is_passed = true;

    for (uint32_t j = 0; j < attributes_per_element; ++j) {
        if (output[j] != 6 * num_mesh_elements) {
            is_passed = false;
            break;
        }
    }

    // release rxmesh_attribute memory on host and device
    X.release();
    Y.release();


    return is_passed;
}


TEST(RXMesh, Attributes)
{
    using namespace RXMESH;
    EXPECT_TRUE(test_host(3u)) << " TestAttributes::tes_host failed";
    EXPECT_TRUE(test_device(3u)) << " TestAttributes::tes_device failed";
    // EXPECT_TRUE(test_vector()) << " TestAttributes::test_vector failed";
    EXPECT_TRUE(test_axpy(3u)) << " TestAttributes::test_axpy failed";
    EXPECT_TRUE(test_reduce()) << " TestAttributes::test_reduce failed";
    EXPECT_TRUE(test_norm2()) << " TestAttributes::test_norm2 failed";
    EXPECT_TRUE(test_dot()) << " TestAttributes::test_dot failed";

    CUDA_ERROR(cudaDeviceSynchronize());
}