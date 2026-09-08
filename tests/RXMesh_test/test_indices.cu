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

__global__ static void check_tet_owner(Context context, int* error)
{
    const uint32_t p = blockIdx.x;
    if (p >= context.get_num_patches()) {
        return;
    }

    const PatchInfo& patch = context.m_patches_info[p];
    for (uint16_t t = threadIdx.x; t < patch.num_tets[0]; t += blockDim.x) {
        const LocalTetT local_tet(t);
        if (patch.is_deleted(local_tet)) {
            continue;
        }

        const TetHandle tet(p, local_tet);
        const TetHandle owner = context.get_owner_handle(tet);
        if (!owner.is_valid() ||
            !context.m_patches_info[owner.patch_id()].is_owned(
                LocalTetT(owner.local_id()))) {
            atomicExch(error, 1);
            continue;
        }

        const uint32_t id = context.linear_id(tet);
        if (id >= context.get_num<TetHandle>() ||
            context.get_handle<TetHandle>(id) != owner) {
            atomicExch(error, 1);
        }
    }
}

TEST(RXMeshStatic, IndicesTriangle)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    from_linear_to_handle<VertexHandle>(rx);
    from_linear_to_handle<EdgeHandle>(rx);
    from_linear_to_handle<FaceHandle>(rx);

    RXMeshStatic tet_rx(STRINGIFY(INPUT_DIR) "car.msh");

    ASSERT_EQ(tet_rx.get_num_elements<TetHandle>(), tet_rx.get_num_tets());
    from_linear_to_handle<TetHandle>(tet_rx);

    int* error = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&error, sizeof(int)));
    CUDA_ERROR(cudaMemset(error, 0, sizeof(int)));

    check_tet_owner<<<tet_rx.get_num_patches(), 256>>>(tet_rx.get_context(),
                                                       error);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int host_error = 0;
    CUDA_ERROR(
        cudaMemcpy(&host_error, error, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(host_error, 0);

    CUDA_ERROR(cudaFree(error));
}

TEST(RXMeshStatic, IndicesTet)
{
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "car.msh");

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    from_linear_to_handle<VertexHandle>(rx);
    from_linear_to_handle<EdgeHandle>(rx);
    from_linear_to_handle<FaceHandle>(rx);
    from_linear_to_handle<TetHandle>(rx);

    int* error = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&error, sizeof(int)));
    CUDA_ERROR(cudaMemset(error, 0, sizeof(int)));

    check_tet_owner<<<rx.get_num_patches(), 256>>>(rx.get_context(), error);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int host_error = 0;
    CUDA_ERROR(
        cudaMemcpy(&host_error, error, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(host_error, 0);

    CUDA_ERROR(cudaFree(error));
}