#include "rxmesh/query.cuh"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"

using namespace rxmesh;


template <typename T, uint32_t blockThreads>
__global__ static void sample_points(const rxmesh::Context      context,
                                     rxmesh::VertexAttribute<T> vertex_pos,
                                     rxmesh::VertexAttribute<T> distance,
                                     int*                       flag)
{

    auto sampler = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); i++) {

            float dist =
                sqrtf(powf(vertex_pos(v_id, 0) - vertex_pos(vv[i], 0), 2) +
                      powf(vertex_pos(v_id, 1) - vertex_pos(vv[i], 1), 2) +
                      powf(vertex_pos(v_id, 2) - vertex_pos(vv[i], 2), 2))+
                distance(vv[i], 0);

            //printf("\nVertex: %u Distance : %f", context.linear_id(v_id), dist);


            if (dist < distance(v_id, 0)) {
                distance(v_id, 0) = dist;
                *flag             = 15;
            }
        }
         //printf("\nFLAG : %d", *flag);
    };
    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, sampler);
}



int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");

    auto vertex_pos = *rx.get_input_vertex_coordinates();

    //attribute to sample,store and order samples
    auto sample_number = *rx.add_vertex_attribute<int>("sample_number", 1);
    auto distance = *rx.add_vertex_attribute<float>("distance", 1);

    int* flagger;
    cudaMallocManaged(&flagger, sizeof(int));
    *flagger = 0;

    auto context = rx.get_context();

    constexpr uint32_t CUDABlockSize = 256;
    rxmesh::LaunchBox<CUDABlockSize> lb;
    rx.prepare_launch_box({rxmesh::Op::EVDiamond},
                          lb,
                          (void*)sample_points<float, CUDABlockSize>);



    float ratio = 8;
    int   numberOfLevels = 1;
    int   numberOfSamples = 300;

    
    int N = rx.get_num_vertices();

    std::random_device
        rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(
        rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dist(
        0, N - 1);  // From 0 to (number of points - 1)
    int seed = dist(gen);

    std::cout << "\nSeed: " << seed;

    VertexReduceHandle<float> reducer(distance);
    cub::KeyValuePair<VertexHandle, float> farthestPoint;

    int j = 0;
    for (int i = 0; i < numberOfSamples; i++) {

        rx.for_each_vertex(rxmesh::DEVICE,
                           [seed, context, sample_number, distance, i] __device__(
                               const rxmesh::VertexHandle vh) {
                               if (seed == context.linear_id(vh)) {
                                   sample_number(vh, 0) = i;
                                   distance(vh, 0)      = 0;
                               }
                               else {
                                   if (i==0)
                                   {
                                       distance(vh, 0) = INFINITY;
                                       sample_number(vh, 0) = -1;

                                   }
                               }

                           });

        j = 0;
        // we may need to call this repeatedly
        do {
            cudaDeviceSynchronize();
            *flagger = 0;
            sample_points<float, CUDABlockSize>
                <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                    rx.get_context(), vertex_pos, distance, flagger);
            cudaDeviceSynchronize();
            //std::cout << "\nflag: "<<*flagger
              //        << "\n\niteration: " << j << std::endl;

            j++;

        } while (*flagger != 0);

        // reduction step
        farthestPoint = reducer.arg_max(distance, 0);
        seed          = rx.linear_id(farthestPoint.key);

    }


    sample_number.move(DEVICE, HOST);
    distance.move(DEVICE, HOST);



   
    std::cout << std::endl << farthestPoint.value;

    std::cout << std::endl << rx.linear_id(farthestPoint.key);

    
    rx.get_polyscope_mesh()->addVertexScalarQuantity("sample_number",sample_number);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("distance", distance);



#if USE_POLYSCOPE
    polyscope::show();
#endif

}