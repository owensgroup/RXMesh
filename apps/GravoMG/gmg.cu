#include "rxmesh/query.cuh"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/util/bitmask_util.h"

#include "cuda_runtime.h"

// Function to compute the projected distance from a point to a triangle
__device__ float projectedDistance(const Eigen::Vector3f& v0,
                                   const Eigen::Vector3f& v1,
                                   const Eigen::Vector3f& v2,
                                   const Eigen::Vector3f& p)
{
    // Compute edges of the triangle
    Eigen::Vector3f edge1 = v1 - v0;
    Eigen::Vector3f edge2 = v2 - v0;

    // Compute the triangle normal
    Eigen::Vector3f normal        = edge1.cross(edge2);
    float           normal_length = normal.norm();

    if (normal_length < 1e-6f) {
        return -1.0f;  // Return -1 to indicate an error
    }

    // Normalize the normal
    normal.normalize();

    // Compute vector from point to the triangle vertex
    Eigen::Vector3f point_to_vertex = p - v0;

    // Project the vector onto the normal
    float distance = point_to_vertex.dot(normal);

    // Return the absolute distance
    return std::fabs(distance);
}



__device__ std::tuple<float, float, float> computeBarycentricCoordinates(
    const Eigen::Vector3f& v0,
    const Eigen::Vector3f& v1,
    const Eigen::Vector3f& v2,
    const Eigen::Vector3f& p)
{
    // Compute edges of the triangle
    Eigen::Vector3f edge1    = v1 - v0;
    Eigen::Vector3f edge2    = v2 - v0;
    Eigen::Vector3f pointVec = p - v0;

    // Compute normal of the triangle
    Eigen::Vector3f normal = edge1.cross(edge2);
    float area2 = normal.squaredNorm();  // Area of the triangle multiplied by 2

    // Compute barycentric coordinates
    float lambda0, lambda1, lambda2;

    // Sub-area with respect to v0
    Eigen::Vector3f normal1 = (v1 - p).cross(v2 - p);
    lambda0                 = normal1.squaredNorm() / area2;

    // Sub-area with respect to v1
    Eigen::Vector3f normal2 = (v2 - p).cross(v0 - p);
    lambda1                 = normal2.squaredNorm() / area2;

    // Sub-area with respect to v2
    Eigen::Vector3f normal3 = (v0 - p).cross(v1 - p);
    lambda2                 = normal3.squaredNorm() / area2;

    // Return the barycentric coordinates
    return std::make_tuple(lambda0, lambda1, lambda2);
}


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
        cudaError_t err;

        // Allocate memory
        err = cudaMalloc(&d_data, row_size * N * sizeof(uint32_t));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        // Initialize to 0
        err = cudaMemset(d_data, 0, row_size * N * sizeof(uint32_t));
        if (err != cudaSuccess) {
            cudaFree(d_data);  // Free allocated memory in case of failure
            throw std::runtime_error("cudaMemset failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    // Destructor
    __host__ ~BitMatrix()
    {
        if (d_data) {
            cudaFree(d_data);
            d_data = nullptr;
        }
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
        }
        while (atomicCAS(address, old_val, new_val) != old_val);
        
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
                int a = clustered_vertices(vv[i], 0);
                int b = clustered_vertices(v_id, 0);

                //something wrong here
                if (bitMatrix.trySet(b,a)) {

                    //printf("\nFound %d with %d since the matrix value is 0",
                    //       clustered_vertices(v_id, 0),
                    //       clustered_vertices(vv[i], 0));

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



struct Vec3
{
    float x, y, z;
};

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

    constexpr uint32_t CUDABlockSize = 512;
    rxmesh::LaunchBox<CUDABlockSize> lb;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          lb,
                          (void*)sample_points<float, CUDABlockSize>);



    float ratio = 8;
    int   N     = rx.get_num_vertices();
    int   numberOfLevels = 1;
    int   currentLevel    = 2; //first coarse mesh
    int   numberOfSamples = N / powf(ratio , currentLevel);
    

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dist(0, N - 1);  // From 0 to (number of points - 1)
    int seed = dist(gen);

    std::cout << "\nSeed: " << seed;

    VertexReduceHandle<float> reducer(distance);
    cub::KeyValuePair<VertexHandle, float> farthestPoint;

    Vec3* vertices;

    // Allocate unified memory
    cudaMallocManaged(&vertices, numberOfSamples * sizeof(Vec3));


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
                            currentLevel,
                           vertex_pos,
                           vertices] __device__(
                               const rxmesh::VertexHandle vh) {
                               if (seed == context.linear_id(vh)) {
                                   sample_number(vh, 0) = i;
                                   distance(vh, 0)      = 0;
                                   vertices[i].x        = vertex_pos(vh, 0);
                                   vertices[i].y        = vertex_pos(vh, 1);
                                   vertices[i].z        = vertex_pos(vh, 2);
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


    ////
    ///
    //find number of neighbors x
    int* number_of_neighbors;
    cudaMallocManaged(&number_of_neighbors, numberOfSamples * sizeof(int));
    for (int i = 0; i < numberOfSamples; i++) {
        number_of_neighbors[i] = 0;
    }
    BitMatrix bitMatrix(numberOfSamples);
    cudaDeviceSynchronize();

    //ASSERT here that every vertex has a neighbor

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

    cudaMallocManaged(&d_cub_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                  temp_storage_bytes,
                                  number_of_neighbors,
                                  row_ptr,
                                  num_rows + 1);

    cudaDeviceSynchronize();  
    cudaFree(d_cub_temp_storage);

    //ASSERT the memory may not be enough, 

    /*
    printf("\nFINAL ROW POINTERS\n");
    for (int i = 0; i <= num_rows; ++i) {
        printf("row_ptr[%d] = %d\n", i, row_ptr[i]);
        printf("add %d values\n", number_of_neighbors[i]);

    }
    */


    // then populate the row pointers
    
    int* value_ptr;

    cudaMallocManaged(&value_ptr, row_ptr[num_rows] * sizeof(int));


    // make this based on thresh
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [sample_number,
         bitMatrix,
         number_of_neighbors, row_ptr,
         value_ptr, numberOfSamples] __device__(const rxmesh::VertexHandle v_id) {
        int number = sample_number(v_id, 0); 
        if (number != -1) {
            /// find start pointer
            int start_pointer = row_ptr[number];
            /// find end pointer
            int end_pointer = row_ptr[number + 1];


            /// for each pointer, go to the bitmatrix, read the neighbors one
            /// after the other, if we find a 1, add it to the list
            int j = 0;
            for (int i = 0; i < numberOfSamples; i++) {
                if (bitMatrix.get(number, i)) {
                    value_ptr[start_pointer + j] = i;
                    j++;
                }
            }
            //#ifndef NDEBUG
            //THIS CAN BE AN ASSERT
            if (j != number_of_neighbors[number]) {
                printf(
                    "ERROR: Number of neighbors does not match number of "
                    "entries");
            }
        }
    });

    cudaDeviceSynchronize();

    printf("\nCSR Array: \n");
     for (int i = 0; i < num_rows; ++i) {
        printf("row_ptr[%d] = %d\n", i, row_ptr[i]);
        printf("add %d values\n", number_of_neighbors[i]);
        for (int q = row_ptr[i]; q < row_ptr[i + 1]; q++) {
            printf("vertex %d\n", value_ptr[q]);
        }
    }


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

        
    //for each vertex, get a triangle
    //project
    /// for a fine vertex -> get source, get closest triangle, project
    ///
    ///

    //csr
    // row pointer
    // value pointer
    //
    // projectToTriangle -> take a vertex in (row pointer index value), return the projected barycentric coordinates
    // take those bary coords, put them into prolongation matrix -> normal sparse matrix for now
    //

    float* prolongation_operator;
    cudaMallocManaged(&prolongation_operator, N * numberOfSamples*sizeof(float));

    cudaDeviceSynchronize();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < numberOfSamples; j++) {
            prolongation_operator[numberOfSamples * i + j] = 0;
        }
    }

    rx.for_each_vertex(
        rxmesh::DEVICE,
        [sample_number,
         clustered_vertex,
         number_of_neighbors,
         row_ptr,value_ptr,
         context,
        vertices,vertex_pos, prolongation_operator,numberOfSamples] __device__(const rxmesh::VertexHandle vh) {

        //go through every triangle of my cluster
        const int cluster_point = clustered_vertex(vh, 0);
        const int start_pointer = row_ptr[clustered_vertex(vh,0)];
        const int end_pointer = row_ptr[clustered_vertex(vh,0)+1];

        float min_distance = 99999;
        Eigen::Vector3<float> selectedv1{0,0,0}, selectedv2{0, 0, 0},
            selectedv3{0, 0, 0};
        const Eigen::Vector3<float> q{
            vertex_pos(vh, 0), vertex_pos(vh, 1), vertex_pos(vh, 2)};

        int neighbor=0;
        int selected_neighbor=0;
        int neighbor_of_neighbor=0;
        int selected_neighbor_of_neighbor=0;


        for (int i=start_pointer;i<end_pointer;i++) {

            float distance;
             // Get the neighbor vertex
            neighbor = value_ptr[i];  // Assuming col_idx stores column
                                        // indices of neighbors in CSR.

            // Get the range of neighbors for this neighbor
            const int neighbor_start = row_ptr[neighbor];
            const int neighbor_end   = row_ptr[neighbor + 1];

            for (int j = neighbor_start; j < neighbor_end; j++) {
                neighbor_of_neighbor = value_ptr[j];

                for (int k=i+1;k<end_pointer;k++)
                {
                    if (value_ptr[k]==neighbor_of_neighbor) 
                    {

                        /* printf("\n%d %d and %d are a triangle",
                               cluster_point,
                               neighbor,
                               neighbor_of_neighbor);
                               */



                        
                        Eigen::Vector3<float> v1{vertices[cluster_point].x,
                                                 vertices[cluster_point].y,
                                                 vertices[cluster_point].z};
                        Eigen::Vector3<float> v2{vertices[neighbor].x,
                                                 vertices[neighbor].y,
                                                 vertices[neighbor].z};
                        Eigen::Vector3<float> v3{
                            vertices[neighbor_of_neighbor].x,
                            vertices[neighbor_of_neighbor].y,
                            vertices[neighbor_of_neighbor].z};

                        //find distance , if less than min dist, find bary coords, save them
                        float distance = projectedDistance(v1, v2, v3, q);
                        if (distance<min_distance) {
                            
                            min_distance = distance;
                            selectedv1   = v1;
                            selectedv2   = v2;
                            selectedv3   = v3;
                            selected_neighbor = neighbor;
                            selected_neighbor_of_neighbor =neighbor_of_neighbor;
                        }
                    }
                }
            }
        }
        // take the best bary coords
        auto [b1, b2, b3] = computeBarycentricCoordinates(
            selectedv1, selectedv2, selectedv3, q);
        // put it inside prolongation row, it will be unique so no race
        // condition
        int l = context.linear_id(vh);

        printf("\n %d final coords: %f %f %f", l, b1, b2, b3);


        prolongation_operator[l * numberOfSamples + cluster_point]        = b1;
        prolongation_operator[l * numberOfSamples + selected_neighbor]             = b2;
        prolongation_operator[l * numberOfSamples + selected_neighbor_of_neighbor] = b3;

    });
    cudaDeviceSynchronize();
    std::cout << std::endl;
    std::cout << std::endl;

    
    for (int i=0;i<N;i++) {
        std::cout << std::endl << i << "  ";
        for (int j=0;j<numberOfSamples;j++) {
            std::cout << prolongation_operator[i * numberOfSamples + j] << " ";
        }
    }
    





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