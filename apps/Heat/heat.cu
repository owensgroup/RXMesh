#include "rxmesh/query.cuh"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"

using namespace rxmesh;


int main(int argc, char** argv)
{
    Log::init();



    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto constraints = *rx.add_vertex_attribute<float>("v", 1);

    auto context = rx.get_context();

    rx.for_each_vertex(DEVICE, [context, constraints] __device__(const VertexHandle& vh) {
            constraints(vh, 0) = context.linear_id(vh);
    });
    constraints.move(DEVICE, HOST);

    //ReduceHandle reduce_handle(constraints);

    ///auto output= reduce_handle.arg_maxOrmin(constraints, false);

    VertexReduceHandle<float> reduce_handle(constraints);

    reduce_handle.arg_max(constraints,0);

    rx.get_polyscope_mesh()->addVertexScalarQuantity("value", constraints);

    //std::cout << "OUTPUT:" << output;
#if USE_POLYSCOPE
    polyscope::show();
#endif
}