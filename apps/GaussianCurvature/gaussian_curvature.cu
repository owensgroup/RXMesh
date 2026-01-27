// Compute the gaussian curvature according to
// Mark Meyer, Mathieu Desbrun, Peter Schroder, and Alan H. Barr. "Discrete
// Differential-Geometry Operators for Triangulated 2-Manifolds"
// International Workshop on Visualization and Mathematics

#include <CLI/CLI.hpp>
#include <glm/gtc/constants.hpp>
#include "rxmesh/attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"

#include "gaussian_curvature_kernel.cuh"
#include "gaussian_curvature_ref.h"

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "bumpy-cube.obj";
    uint32_t    device_id     = 0;
} Arg;

template <typename T>
void gaussian_curvature_rxmesh()
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    RXMeshStatic rx(Arg.obj_file_name);

    // input coordinates
    auto coords = *rx.get_input_vertex_coordinates();

    // gaussian curvatures
    auto v_gc = *rx.add_vertex_attribute<T>("v_gc", 1, rxmesh::LOCATION_ALL);

    // mixed area for integration
    auto v_amix =
        *rx.add_vertex_attribute<T>("v_amix", 1, rxmesh::LOCATION_ALL);

    // launch box
    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({rxmesh::Op::FV},
                          launch_box,
                          (void*)compute_gaussian_curvature<T, blockThreads>);

    // initialization
    v_gc.reset(2.0 * glm::pi<T>(), rxmesh::DEVICE);
    v_amix.reset(0, rxmesh::DEVICE);

    compute_gaussian_curvature<T, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), coords, v_gc, v_amix);

    rx.for_each_vertex(DEVICE,
                       [v_gc, v_amix] __device__(const VertexHandle vh) {
                           v_gc(vh, 0) = v_gc(vh, 0) / v_amix(vh, 0);
                       });

    CUDA_ERROR(cudaDeviceSynchronize());


    // Move to host for visualization
    v_gc.move(rxmesh::DEVICE, rxmesh::HOST);

#if USE_POLYSCOPE
    // visualize
    polyscope::view::upDir = polyscope::UpDir::ZUp;
    auto polyscope_mesh    = rx.get_polyscope_mesh();
    polyscope_mesh->setEdgeWidth(1.0);
    polyscope_mesh->addVertexScalarQuantity("vGaussianCurv", v_gc);
    polyscope::show();
#endif
}

int main(int argc, char** argv)
{
    using namespace rxmesh;
    
    CLI::App app{"GaussianCurvature - Compute Gaussian curvature of a mesh"};
    
    app.add_option("-i,--input", Arg.obj_file_name, "Input OBJ mesh file")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "bumpy-cube.obj"));
    
    app.add_option("-d,--device_id", Arg.device_id, "GPU device ID")
        ->default_val(0u);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    rx_init(Arg.device_id);
    
    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    // RXMesh Impl
    gaussian_curvature_rxmesh<float>();

    return 0;
}