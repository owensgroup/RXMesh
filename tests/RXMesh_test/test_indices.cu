#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/dense_matrix.h"

#include "rxmesh/kernels/util.cuh"

using namespace rxmesh;

template <typename HandleT>
void from_linear_to_handle(RXMeshStatic& rx)
{

    uint32_t size = rx.get_num_elements<HandleT>();

    HandleT* handles = nullptr;

    CUDA_ERROR(cudaMalloc((void**)&handles, sizeof(HandleT) * size));

    DenseMatrix<int> ret(rx, 1, 1, LOCATION_ALL);
    ret.reset(0, LOCATION_ALL);

    const int threads = 256;
    const int blocks  = DIVIDE_UP(size, threads);

    rxmesh::memsett<<<blocks, threads>>>(handles, HandleT(), size);

    auto ctx = rx.get_context();
        
    rx.for_each<HandleT>(DEVICE, [=] __device__(const HandleT h) {
        uint32_t id = ctx.template linear_id<HandleT>(h);
        assert(id < size);
        handles[id] = h;
    });

    for_each_item<<<blocks, threads>>>(size, [=] __device__(int i) mutable {
        HandleT h = ctx.template get_handle<HandleT>(i);
        if (h != handles[i]) {            
            ret(0, 0) = 1;
        }
    });


    ret.move(DEVICE, HOST);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(ret(0, 0), 0);

    GPU_FREE(handles);
    ret.release();
}

TEST(RXMeshStatic, Indices)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    from_linear_to_handle<VertexHandle>(rx);
    from_linear_to_handle<EdgeHandle>(rx);
    from_linear_to_handle<FaceHandle>(rx);
}