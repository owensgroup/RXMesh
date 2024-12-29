#include "rxmesh/query.cuh"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/util/bitmask_util.h"

#include "cuda_runtime.h"

using namespace rxmesh;
class BitMatrix
{
   public:
    int       N;         // Size of the matrix
    int       row_size;  // Number of 32-bit integers per row
    uint32_t* d_data;    // Device pointer for GPU

    // Constructor
    __host__ BitMatrix(int size) : N(size)
    {
        row_size = (N + 31) / 32;  // Number of integers per row
        cudaMalloc(&d_data, row_size * N * sizeof(uint32_t));
        cudaMemset(
            d_data, 0, row_size * N * sizeof(uint32_t));  // Initialize to 0
    }

    // Destructor
    __host__ ~BitMatrix()
    {
        cudaFree(d_data);
    }

    // Helper function to calculate index and bit position
    __device__ inline std::pair<int, int> getIndex(int row, int col) const
    {
        int bit_index    = row * N + col;
        int int_index    = bit_index / 32;  // Which 32-bit integer
        int bit_position = bit_index % 32;  // Position within the integer
        return {int_index, bit_position};
    }

    __device__ inline void set(int row, int col, bool value)
    {
        auto [int_index, bit_position] = getIndex(row, col);
        uint32_t* address              = &d_data[int_index];
        uint32_t  mask                 = 1 << bit_position;

        uint32_t old_val, new_val;
        do {
            old_val = *address;  // Read the current value
            if (value) {
                new_val = old_val | mask;  // Set the bit
            } else {
                new_val = old_val & ~mask;  // Clear the bit
            }
        } while (atomicCAS(address, old_val, new_val) != old_val);
    }
    __device__ inline bool trySet(int row, int col)
    {
        auto [int_index, bit_position] = getIndex(row, col);
        uint32_t* address              = &d_data[int_index];
        uint32_t  mask                 = 1 << bit_position;

        uint32_t old_val, new_val;
        do {
            old_val = *address;  // Read the current value
            if (old_val & mask)
                return false;          // Already set, another thread got it
            new_val = old_val | mask;  // Set the bit
        } while (atomicCAS(address, old_val, new_val) != old_val);

        return true;  // Successfully set
    }

   __device__ inline bool get(int row, int col) const
    {
        auto [int_index, bit_position] = getIndex(row, col);
        const uint32_t value = atomicCAS(&d_data[int_index], 0, 0);  // Atomic read
        return (value >> bit_position) & 1;
    }
};


template <typename T, uint32_t blockThreads>
__global__ static void findNumberOfCoarseNeighbors(
    const rxmesh::Context        context,
    rxmesh::VertexAttribute<int> clustered_vertices,
    BitMatrix bitMatrix,
    int* number_of_neighbors)
{

    auto cluster = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); i++) {
            
            if (clustered_vertices(v_id, 0) != clustered_vertices(vv[i], 0)) 
            {
                if (bitMatrix.trySet(clustered_vertices(v_id, 0),
                                     clustered_vertices(vv[i], 0))) {
                    /* printf("\nFound %d with %d since the matrix value is 0",
                           clustered_vertices(v_id, 0),
                           clustered_vertices(vv[i], 0));
                    */
                    atomicAdd(&number_of_neighbors[clustered_vertices(v_id, 0)],
                              1);
                }


            }
        }
    };
    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, cluster);
}



template <typename T, uint32_t blockThreads>
__global__ static void cluster_points(const rxmesh::Context      context,
                                     rxmesh::VertexAttribute<T> vertex_pos,
                                     rxmesh::VertexAttribute<T> distance,
                                     rxmesh::VertexAttribute<int> clustered_vertices,
                                     int*                       flag)
{

    auto cluster = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); i++) {

            float dist =
                sqrtf(powf(vertex_pos(v_id, 0) - vertex_pos(vv[i], 0), 2) +
                      powf(vertex_pos(v_id, 1) - vertex_pos(vv[i], 1), 2) +
                      powf(vertex_pos(v_id, 2) - vertex_pos(vv[i], 2), 2)) +
                distance(vv[i], 0);


            if (dist < distance(v_id, 0) && clustered_vertices(vv[i], 0)!=-1) {
                distance(v_id, 0) = dist;
                *flag             = 15;
                clustered_vertices(v_id, 0) = clustered_vertices(vv[i], 0);

            } 
        }
    };
    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, cluster);
}


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


    auto sample_level_bitmask = *rx.add_vertex_attribute<uint16_t>("bitmask", 1);
    auto clustered_vertex = *rx.add_vertex_attribute<int>("clustering", 1);

    auto number_of_neighbors_coarse =*rx.add_vertex_attribute<int>("number of neighbors", 1);


    int* flagger;
    cudaMallocManaged(&flagger, sizeof(int));
    *flagger = 0;

    auto context = rx.get_context();

    constexpr uint32_t CUDABlockSize = 256;
    rxmesh::LaunchBox<CUDABlockSize> lb;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          lb,
                          (void*)sample_points<float, CUDABlockSize>);



    float ratio = 8;
    int   N     = rx.get_num_vertices();
    int   numberOfLevels = 1;
    int   currentLevel    = 1; //first coarse mesh
    int   numberOfSamples = N / powf(ratio , currentLevel);
    

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dist(0, N - 1);  // From 0 to (number of points - 1)
    int seed = dist(gen);

    std::cout << "\nSeed: " << seed;

    VertexReduceHandle<float> reducer(distance);
    cub::KeyValuePair<VertexHandle, float> farthestPoint;


    // pre processing step
    //gathers samples for every level
    int j = 0;
    for (int i = 0; i < numberOfSamples; i++) {

        rx.for_each_vertex(rxmesh::DEVICE,
                           [seed,
                            context,
                            sample_number,
                            sample_level_bitmask,
                            distance,
                            i, 
                            currentLevel] __device__(
                               const rxmesh::VertexHandle vh) {
                               if (seed == context.linear_id(vh)) {
                                   sample_number(vh, 0) = i;
                                   distance(vh, 0)      = 0;

                                   /*
                                    *something like
                                    *
                                    *for each level if samples_current_level [currentLevel] > i 
                                    * then
                                    * sample level bitmask (vh,0) |= (1<<currentevel)
                                    */
                                   sample_level_bitmask(vh,0) |= (1 << currentLevel);
                               }
                               else {
                                   if (i==0)
                                   {
                                       distance(vh, 0) = INFINITY;
                                       sample_number(vh, 0) = -1;
                                   }
                               }
                           });

        do {
            cudaDeviceSynchronize();
            *flagger = 0;
            sample_points<float, CUDABlockSize>
                <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                    rx.get_context(), vertex_pos, distance, flagger);
            cudaDeviceSynchronize();
            //std::cout << "\nflag: "<<*flagger
            //          << "\n\niteration: " << j << std::endl;

            j++;

        } while (*flagger != 0);


        // reduction step
        farthestPoint = reducer.arg_max(distance, 0);
        seed          = rx.linear_id(farthestPoint.key);
    }

    std::cout << "\nSampling iterations: " << j;


    sample_number.move(DEVICE, HOST);
    distance.move(DEVICE, HOST);
    sample_level_bitmask.move(DEVICE, HOST);


    rxmesh::LaunchBox<CUDABlockSize> cb;
    rx.prepare_launch_box(
        {rxmesh::Op::VV}, cb, (void*)cluster_points<float, CUDABlockSize>);


    //clustering step
     j = 0;
     rx.for_each_vertex(
            rxmesh::DEVICE,
            [
             sample_number,
             sample_level_bitmask,
             distance,
             currentLevel,
            clustered_vertex, context] __device__(const rxmesh::VertexHandle vh) {
                if (sample_number(vh,0) > -1) //we could replace this with the bitmask check instead
                {
                 clustered_vertex(vh, 0) =  sample_number(vh,0);
                    distance(vh, 0)         = 0;
                } else {
                        distance(vh, 0)      = INFINITY;
                        clustered_vertex(vh, 0) = -1;
                    
                }
            });
         do {
            cudaDeviceSynchronize();
            *flagger = 0;
            cluster_points<float, CUDABlockSize>
                <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                    rx.get_context(), vertex_pos, distance,clustered_vertex, flagger);
            cudaDeviceSynchronize();
             j++;
        } while (*flagger != 0);

    clustered_vertex.move(DEVICE, HOST);
    std::cout << "\Clustering iterations: " << j;

    //find number of neighbors x
    int* number_of_neighbors;
    cudaMallocManaged(&number_of_neighbors, numberOfSamples * sizeof(int));
    for (int i = 0; i < numberOfSamples; i++) {
        number_of_neighbors[i] = 0;
    }
    BitMatrix bitMatrix(numberOfSamples * numberOfSamples);
    cudaDeviceSynchronize();

    rxmesh::LaunchBox<CUDABlockSize> nn;
    rx.prepare_launch_box(
        {rxmesh::Op::VV}, nn, (void*)findNumberOfCoarseNeighbors<float, CUDABlockSize>);


    findNumberOfCoarseNeighbors<float, CUDABlockSize>
        <<<nn.blocks, nn.num_threads, nn.smem_bytes_dyn>>>(
            rx.get_context(), clustered_vertex, bitMatrix,number_of_neighbors);
    cudaDeviceSynchronize();

    // construct row pointers -> prefix sum
    // Number of rows in your matrix
    int num_rows = numberOfSamples;  // Set this appropriately

    // Allocate unified memory for row counts and row pointers
    int* row_ptr;

    cudaMallocManaged(&row_ptr, (num_rows + 1) * sizeof(int));
    

    // Temporary storage for CUB
    void*  d_cub_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Compute the required temporary storage size
    cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                  temp_storage_bytes,
                                  number_of_neighbors,
                                  row_ptr,
                                  num_rows + 1);

    // Allocate temporary storage
    cudaMallocManaged(&d_cub_temp_storage, temp_storage_bytes);

    // Perform the prefix sum
    cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                  temp_storage_bytes,
                                  number_of_neighbors,
                                  row_ptr,
                                  num_rows + 1);

    cudaDeviceSynchronize();  // Synchronize to ensure prefix sum completion

    // Free the temporary storage
    cudaFree(d_cub_temp_storage);
    /*
    printf("\nFINAL ROW POINTERS\n");
    for (int i = 0; i <= num_rows; ++i) {
        printf("row_ptr[%d] = %d\n", i, row_ptr[i]);
        printf("add %d values\n", number_of_neighbors[i]);

    }
    */






    //for debug purposes
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [sample_number,
         clustered_vertex,
         number_of_neighbors,
        number_of_neighbors_coarse,
         context] __device__(const rxmesh::VertexHandle vh) {
                           number_of_neighbors_coarse(vh, 0) = number_of_neighbors[sample_number(vh, 0)];
        });

    number_of_neighbors_coarse.move(DEVICE, HOST);

    rx.get_polyscope_mesh()->addVertexScalarQuantity("sample_number",sample_number);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("distance", distance);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("sample_level_bitmask",sample_level_bitmask);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("clusterPoint", clustered_vertex);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("number of neighbors", number_of_neighbors_coarse);
    


#if USE_POLYSCOPE
    polyscope::show();
#endif

}