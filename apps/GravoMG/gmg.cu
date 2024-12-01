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

template <typename T, uint32_t blockThreads>
__global__ static void sample_points(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<T> vertex_pos,
    rxmesh::VertexAttribute<T> distance,
    rxmesh::SparseMatrix<T>    weight_matrix)
{
    auto sampler = [&](VertexHandle v_id, VertexIterator& vv)
    {
        for (int i = 0; i < vv.size(); i++) 
        {
            float dist = sqrt(pow(vertex_pos(v_id, 0) - vertex_pos(v_id, 0), 2) +
                              pow(vertex_pos(v_id, 1) - vertex_pos(v_id, 1), 2) +
                              pow(vertex_pos(v_id, 2) - vertex_pos(v_id, 2), 2))+ distance(vv[i], 0);
            if (dist < distance(v_id,0)) 
            {
                distance(v_id, 0) = dist;
            }
        }
    };
    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    bool                done = false;
    query.dispatch<Op::VV>(block, shrd_alloc,sampler);
}


int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto vertex_pos = *rx.get_input_vertex_coordinates();

    //attribute to sample,store and order samples
    auto samples = *rx.add_vertex_attribute<float>("samples", 1);
    auto distance = *rx.add_vertex_attribute<float>("distance", 1);



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
        [vertex_pos, seed,context,samples,distance] __device__(const rxmesh::VertexHandle vh) {

        if (seed == context.linear_id(vh)) {
            samples(vh, 0) = 10;
            distance(vh, 0) = 0;

            printf("\nFOUND");
        }
        else {
            samples(vh, 0) = 0;
            distance(vh,0)=INFINITY;
        }
        printf("\nLocal: %d Patch: %d Unique: %u",
            vh.local_id(),
            vh.patch_id(),
            context.linear_id(vh));
        });

    samples.move(DEVICE, HOST);
    distance.move(DEVICE, HOST);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("samples", samples);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("distance", distance);



#if USE_POLYSCOPE
    polyscope::show();
#endif

}