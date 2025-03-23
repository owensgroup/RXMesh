#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/geometry_util.cuh"


using namespace rxmesh;

#include "include/General/GenericCG.h"
#include "include/MCF/mcfCG.h"



int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");
    constexpr uint32_t blockThreads = 256;

    auto input_coord = rx.get_input_vertex_coordinates();
     
    // B in CG
    auto B = rx.add_vertex_attribute<float>("B", 3, rxmesh::DEVICE, rxmesh::SoA);
    B->reset(0.0, rxmesh::DEVICE);

    // X in CG (the output)
    auto X = rx.add_vertex_attribute<float>("X", 3, rxmesh::LOCATION_ALL);
    X->copy_from(*input_coord, rxmesh::DEVICE, rxmesh::DEVICE);


    // RXMesh launch box
    LaunchBox<blockThreads> launch_box_init_B;
    LaunchBox<blockThreads> launch_box_matvec;
    
    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_matvec,
                          (void*)rxmesh_matvec<float, blockThreads>,
                          false);



    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_init_B,
                          (void*)init_B<float, blockThreads>,
                          true);

    // init kernel to initialize RHS (B)
    init_B<float, blockThreads><<<launch_box_init_B.blocks,
                              launch_box_init_B.num_threads,
                              launch_box_init_B.smem_bytes_dyn>>>(
        rx.get_context(), *X, *B, true);


    auto mcf_matvec = [launch_box_matvec, input_coord](
                          rxmesh::RXMeshStatic&                 rx,
                          const rxmesh::VertexAttribute<float>& input,
                          rxmesh::VertexAttribute<float>&       output) {
        rxmesh_matvec<float, blockThreads>
            <<<launch_box_matvec.blocks,
               launch_box_matvec.num_threads,
               launch_box_matvec.smem_bytes_dyn>>>(
                rx.get_context(), *input_coord, input, output, true, 10);
    };

    CG<float> cg(*X, *B, mcf_matvec, 1000, 10, 0.000001);

    cg.solve(rx);

    X->move(rxmesh::DEVICE, rxmesh::HOST);

    rx.get_polyscope_mesh()->updateVertexPositions(*X);


#if USE_POLYSCOPE
    polyscope::show();
#endif
}