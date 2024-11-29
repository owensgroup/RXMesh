#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"

using namespace rxmesh;


//farthest point sampling

/*
 *
 * pick a random point on the mesh
 * sample the furthest point, add it to a list
 * do a foreach vertex operation to determine furthest
 * make sure the list u get finally has some ordering
 *
 */

int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto vertex_pos = *rx.get_input_vertex_coordinates();

    //attribute to sample,store and order samples
    auto samples = *rx.add_vertex_attribute<float>("samples", 1);



    int N = rx.get_num_vertices();

    std::vector<int> sampleID(N);

    Eigen::VectorXd D(N);
    D.setConstant(std::numeric_limits<double>::infinity());

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dist(0, N - 1);  // From 0 to (number of points - 1)
    const int seed = dist(gen);

    sampleID[0] = seed;
    std::cout << seed<<std::endl;

//      vertex_pos.to_eigen<N>();
    auto context = rx.get_context();
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [vertex_pos, seed,context,samples] __device__(const rxmesh::VertexHandle vh) {

        if (seed == context.linear_id(vh)) {
                               samples(vh, 0) = 10;
            printf("\nFOUND");
        }
                           else
                               samples(vh, 0) = 0;
        printf("\nLocal: %d Patch: %d Unique: %u",
                     vh.local_id(),
                     vh.patch_id(),
                     context.linear_id(vh));
            
        });

    samples.move(DEVICE, HOST);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("samples", samples);



#if USE_POLYSCOPE
    polyscope::show();
#endif

}