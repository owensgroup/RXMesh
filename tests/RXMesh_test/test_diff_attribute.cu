#include "gtest/gtest.h"

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/scalar.h"


template <typename T, int Size, bool WithHessian>
void populate(rxmesh::RXMeshStatic&                              rx,
              rxmesh::DiffVertexAttribute<T, Size, WithHessian>& v,
              rxmesh::Scalar<T, Size, WithHessian>               val)
{
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [v, val] __device__(const rxmesh::VertexHandle vh) { v(vh) = val; });

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}


TEST(DiffAttribute, Simple)
{
    // write diff vertex attribute on the device and verify it on the host
    using namespace rxmesh;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto v_attr = *rx.add_diff_vertex_attribute<float, 1, true>("v");

    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) { v_attr(vh) = 1; });

    auto val = Scalar<float, 1, true>::known_derivatives(2.0, 3.0, 4.0);

    populate<float>(rx, v_attr, val);

    v_attr.move(DEVICE, HOST);

    bool is_okay = true;

    rx.for_each_vertex(
        HOST,
        [&](const VertexHandle& vh) {
            if (v_attr(vh).val != val.val || v_attr(vh).grad != val.grad ||
                v_attr(vh).Hess != val.Hess) {
                is_okay = false;
            }
        },
        NULL,
        false);

    EXPECT_TRUE(is_okay);

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // #if USE_POLYSCOPE
    //  rx.get_polyscope_mesh()->addVertexScalarQuantity("vAttr", v_attr);
    //  polyscope::show();
    // #endif
}
