#include "rxmesh/query.cuh"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"
#include "rxmesh/util/bitmask_util.h"

#include "cuda_runtime.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>


struct Vec3
{
    float x, y, z;
};



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

    //printf("\ncalculated coords are %f, %f, %f", lambda0, lambda1, lambda2);

    // Return the barycentric coordinates
    return std::make_tuple(lambda0, lambda1, lambda2);
}

__device__ void computeBarycentricCoordinates(
    const Eigen::Vector3f& v0,
    const Eigen::Vector3f& v1,
    const Eigen::Vector3f& v2,
    const Eigen::Vector3f& p,
    float &a,float &b, float &c)
{
    // Compute edges of the triangle
    Eigen::Vector3f edge1    = v1 - v0;
    Eigen::Vector3f edge2    = v2 - v0;
    Eigen::Vector3f pointVec = p - v0;

    // Compute normal of the triangle
    Eigen::Vector3f normal = edge1.cross(edge2);
    float area2 = normal.squaredNorm();  // Area of the triangle multiplied by 2

    // Compute barycentric coordinates
    float lambda0=0, lambda1=0, lambda2=0;

    // Sub-area with respect to v0
    Eigen::Vector3f normal1 = (v1 - p).cross(v2 - p);
    lambda0                 = normal1.squaredNorm() / area2;

    // Sub-area with respect to v1
    Eigen::Vector3f normal2 = (v2 - p).cross(v0 - p);
    lambda1                 = normal2.squaredNorm() / area2;

    // Sub-area with respect to v2
    Eigen::Vector3f normal3 = (v0 - p).cross(v1 - p);
    lambda2                 = normal3.squaredNorm() / area2;

    a = lambda0;
    b = lambda1;
    c = lambda2;
    //printf("\ncalculated coords are %f, %f, %f", lambda0, lambda1, lambda2);

    // Return the barycentric coordinates
    //return std::make_tuple(lambda0, lambda1, lambda2);
}


using namespace rxmesh;


#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

struct Mutex
{
#ifdef __CUDA_ARCH__
#if (__CUDA_ARCH__ < 700)
#error ShmemMutex requires compiling with sm70 or higher since it relies on Independent Thread Scheduling
#endif
#endif

    __device__ Mutex() : m_mutex(0)
    {

    }
    __device__ void lock()
    {
#ifdef __CUDA_ARCH__
        assert(&m_mutex);
        __threadfence();
        while (::atomicCAS(&m_mutex, 0, 1) != 0) {
            __threadfence();
        }
        __threadfence();
#endif
    }
    __device__ void unlock()
    {
#ifdef __CUDA_ARCH__
        assert(&m_mutex);
        __threadfence();
        ::atomicExch(&m_mutex, 0);
        __threadfence();
#endif
    }
    int m_mutex;
};

// Node structure for linked list of neighbors, storing a chunk of 64 neighbors
struct NeighborNode
{
    static const int CHUNK_SIZE = 64;
    int              neighbors[CHUNK_SIZE];
    int           count;  // Number of neighbors currently stored in this chunk
    NeighborNode* next;

    __device__ NeighborNode() : count(0), next(nullptr)
    {
    }

    // Add a neighbor to the chunk, returns true if successfully added, false if
    // full
    __device__ bool addNeighbor(int neighbor)
    {
        if (count < CHUNK_SIZE) {
            printf("\nneighbor added %d", neighbor);
            neighbors[count++] = neighbor;
            return true;
        }
        return false;
    }

    // Check if a neighbor exists in this chunk
    __device__ bool containsNeighbor(int neighbor)
    {
        for (int i = 0; i < count; ++i) {
            if (neighbors[i] == neighbor) {
                return true;
            }
        }
        return false;
    }
};

class VertexNeighbors
{
   public:
    __device__ VertexNeighbors() : head(nullptr), tail(nullptr), mutex()
    {
        //cudaMallocManaged(mutex, sizeof(Mutex));
        cudaMallocManaged(&neighborsList, sizeof(int) * 64);
    }

    // Add a neighbor to the list
    __device__ void addNeighbor(int neighbor)
    {
        mutex.lock();
        if (containsNeighbor(neighbor)) {
            mutex.unlock();
            //printf("\nneighbor %d exists", neighbor);
            return;  // Neighbor already exists, do nothing
        }
        if (!tail || !tail->addNeighbor(neighbor)) {
            // Create a new node if the current tail is full or doesn't exist
            NeighborNode* newNode = new NeighborNode();
            if (tail) {
                tail->next = newNode;
            } else {
                head = newNode;
            }
            tail = newNode;
            tail->addNeighbor(neighbor);  // Add the neighbor to the new node
            //printf("\nneighbor %d added",neighbor);

        }
        mutex.unlock();
    }

    // Retrieve all neighbors
    __device__ int getNumberOfNeighbors(/*int* thread_neighbors,*/ int max_neighbors=0) const
    {
        int           count   = 0;  // Track the number of neighbors added
        NeighborNode* current = head;

        // Traverse the linked list
        while (current){// && count < max_neighbors) {
            for (int i = 0; i < current->count;
                 /* && count < max_neighbors;*/ ++i) {
                //thread_neighbors[count] = current->neighbors[i];
                ++count;
            }
            current = current->next;  // Move to the next node in the list
        }

        return count;  // Return the total number of neighbors added
    }

    __device__ void getNeighbors(int* thread_neighbors) const
    {
        int           count   = 0;  // Track the number of neighbors added
        NeighborNode* current = head;

        // Traverse the linked list
        while (current) {  // && count < max_neighbors) {
            for (int i = 0; i < current->count;
                 /* && count < max_neighbors;*/ ++i) {
                 thread_neighbors[count] = current->neighbors[i];
                ++count;
            }
            current = current->next;  // Move to the next node in the list
        }

        //return count;  // Return the total number of neighbors added
    }


    // Destructor to clean up memory
    __device__ ~VertexNeighbors()
    {
        NeighborNode* current = head;
        while (current) {
            NeighborNode* temp = current;
            current            = current->next;
            delete temp;
        }
    }

   private:
    int*               neighborsList;
    NeighborNode*      head;
    NeighborNode*      tail;
    mutable Mutex      mutex;

    // Check if a neighbor already exists in the list
    __device__ bool containsNeighbor(int neighbor) const
    {
        NeighborNode* current = head;
        while (current) {
            if (current->containsNeighbor(neighbor)) {
                return true;
            }
            current = current->next;
        }
        return false;
    }
};



void createValuePointer(int       numberOfSamples,
                                     int*      row_ptr_raw,
                                     int*      value_ptr_raw,
                                     int*      number_of_neighbors_raw,
                                     VertexNeighbors* vns,
                                     int       N)
{
    thrust::device_vector<int> samples(numberOfSamples);
    thrust::sequence(samples.begin(), samples.end());

    thrust::for_each(
        thrust::device,
        samples.begin(),
        samples.end(),
        [=] __device__(int number) {
            //printf("\n The number %d is used here\n", number);
            //printf("\n The number of samples %d is used here\n",numberOfSamples);

            int* neighbors = new int[number_of_neighbors_raw[number]];
            vns[number].getNeighbors(neighbors);

            const int n = vns[number].getNumberOfNeighbors();
            
            int start_pointer = row_ptr_raw[number];
            int j = 0;

            for (int i = 0; i < n; i++) {

                value_ptr_raw[start_pointer + j] = neighbors[j];
                j++;
               
            }

            free(neighbors);

             if (j != n) {
            printf("ERROR: Number of neighbors does not match for sample %d\n",
                   number);
            }
        });
}

void createProlongationOperator(int              numberOfSamples,
                        int*             row_ptr,
                        int*             value_ptr,
                        int*             number_of_neighbors,
                        int              N,
    
    int* clustered_vertex, Vec3* vertex_pos, Vec3* sample_pos,
    float* prolongation_operator)
{
    thrust::device_vector<int> samples(N);
    thrust::sequence(samples.begin(), samples.end());

    thrust::for_each(
        thrust::device,
        samples.begin(),
        samples.end(),
        [=] __device__(int number) {
            // go through every triangle of my cluster
            const int cluster_point = clustered_vertex[number];
            const int start_pointer = row_ptr[clustered_vertex[number]];
            const int end_pointer   = row_ptr[clustered_vertex[number] + 1];

            float                 min_distance = 99999;
            Eigen::Vector3<float> selectedv1{0, 0, 0}, selectedv2{0, 0, 0},
                selectedv3{0, 0, 0};
            const Eigen::Vector3<float> q{
                vertex_pos[number].x, vertex_pos[number].y, vertex_pos[number].z};

            int neighbor                      = 0;
            int selected_neighbor             = 0;
            int neighbor_of_neighbor          = 0;
            int selected_neighbor_of_neighbor = 0;


            for (int i = start_pointer; i < end_pointer; i++) {

                float distance;
                // Get the neighbor vertex
                neighbor = value_ptr[i];  // Assuming col_idx stores column
                                          // indices of neighbors in CSR.

                // Get the range of neighbors for this neighbor
                const int neighbor_start = row_ptr[neighbor];
                const int neighbor_end   = row_ptr[neighbor + 1];

                for (int j = neighbor_start; j < neighbor_end; j++) {
                    neighbor_of_neighbor = value_ptr[j];

                    for (int k = i + 1; k < end_pointer; k++) {
                        if (value_ptr[k] == neighbor_of_neighbor) {


                            Eigen::Vector3<float> v1{sample_pos[cluster_point].x,
                                sample_pos[cluster_point].y,
                                sample_pos[cluster_point].z};
                            Eigen::Vector3<float> v2{sample_pos[neighbor].x,
                                                     sample_pos[neighbor].y,
                                                     sample_pos[neighbor].z};
                            Eigen::Vector3<float> v3{
                                sample_pos[neighbor_of_neighbor].x,
                                sample_pos[neighbor_of_neighbor].y,
                                sample_pos[neighbor_of_neighbor].z};

                            // find distance , if less than min dist, find bary
                            // coords, save them
                            float distance = projectedDistance(v1, v2, v3, q);
                            if (distance < min_distance) {

                                min_distance      = distance;
                                selectedv1        = v1;
                                selectedv2        = v2;
                                selectedv3        = v3;
                                selected_neighbor = neighbor;
                                selected_neighbor_of_neighbor =
                                    neighbor_of_neighbor;
                            }
                        }
                    }
                }
            }


        /*
        printf("\n%d %f %f %f Selected: %d %d %d",
                   number,
                   vertex_pos[number].x,
                   vertex_pos[number].y,
                   vertex_pos[number].z,
                   cluster_point,
                   selected_neighbor,
                   selected_neighbor_of_neighbor);
          */         

            float b1=0, b2=0, b3=0;
            computeBarycentricCoordinates(
                selectedv1, selectedv2, selectedv3, q, b1, b2, b3);

           

            // put it inside prolongation row, it will be unique so no race
            // condition
            int l = number;

            //printf("\n %d final coords: %f %f %f", l, b1, b2, b3);


            prolongation_operator[l * numberOfSamples + cluster_point]     = b1;
            prolongation_operator[l * numberOfSamples + selected_neighbor] = b2;
            prolongation_operator[l * numberOfSamples +
                                  selected_neighbor_of_neighbor]           = b3;


        });
}






void numberOfNeighbors(int       numberOfSamples,
    VertexNeighbors* neighbors,
    int* vertexClusters, int N

                                     )
{
    thrust::device_vector<int> samples(N);
    thrust::sequence(samples.begin(), samples.end());

    int* neighborList;
    cudaMallocManaged(&neighborList, sizeof(int) * numberOfSamples);

    for (int i = 0; i < numberOfSamples; i++)
        neighborList[i] = 0;
    thrust::for_each(
        thrust::device,
        samples.begin(),
        samples.end(),
        [=] __device__(int number) {


        int cluster = vertexClusters[number];

        //neighbors[cluster].getNeighbors(neighborList);




        //std::cout << neighbors[0].getNeighbors().size();//this is the neighbor count


        });
}


struct CSR
{
    int* row_ptr;
    int* value_ptr;
    int* number_of_neighbors;
    int  num_rows;
    CSR(int n_rows, int* num_of_neighbors, VertexNeighbors* vns, int N)
    {

        num_rows = n_rows;
        cudaMallocManaged(&row_ptr, (num_rows + 1) * sizeof(int));
        cudaDeviceSynchronize();

        number_of_neighbors = num_of_neighbors;


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

        cudaFree(d_cub_temp_storage);
        cudaDeviceSynchronize();

        cudaMallocManaged(&value_ptr, row_ptr[num_rows] * sizeof(int));
        cudaDeviceSynchronize();

        createValuePointer(num_rows, row_ptr,value_ptr,number_of_neighbors,vns, N);

        
    }

    void createValuePointer(int              numberOfSamples,
                            int*             row_ptr_raw,
                            int*             value_ptr_raw,
                            int*             number_of_neighbors_raw,
                            VertexNeighbors* vns,
                            int              N)
    {
        thrust::device_vector<int> samples(numberOfSamples);
        thrust::sequence(samples.begin(), samples.end());

        thrust::for_each(
            thrust::device,
            samples.begin(),
            samples.end(),
            [=] __device__(int number) {
                //printf("\n The number %d is used here\n", number);
                //printf("\n The number of samples %d is used here\n",numberOfSamples);

                int* neighbors = new int[number_of_neighbors_raw[number]];
                vns[number].getNeighbors(neighbors);

                const int n = vns[number].getNumberOfNeighbors();

                int start_pointer = row_ptr_raw[number];
                int j             = 0;

                for (int i = 0; i < n; i++) {

                    value_ptr_raw[start_pointer + j] = neighbors[j];
                    j++;
                }

                free(neighbors);

                if (j != n) {
                    printf(
                        "ERROR: Number of neighbors does not match for sample "
                        "%d\n",
                        number);
                }
            });
    }
    
    void printCSR()
    {
        printf("\nCSR Array: \n");
        for (int i = 0; i < num_rows; ++i) {
            printf("row_ptr[%d] = %d\n", i, row_ptr[i]);
            printf("add %d values\n", number_of_neighbors[i]);
            for (int q = row_ptr[i]; q < row_ptr[i + 1]; q++) {
                printf("vertex %d\n", value_ptr[q]);
            }
        }
    }
    
    ~CSR()
    {
    }
};




template <typename T, uint32_t blockThreads>
__global__ static void findNumberOfCoarseNeighbors(
    const rxmesh::Context        context,
    rxmesh::VertexAttribute<int> clustered_vertices,
    int* number_of_neighbors,
    VertexNeighbors* vns)
{

    auto cluster = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); i++) {
            
            if (clustered_vertices(v_id, 0) != clustered_vertices(vv[i], 0)) 
            {
                int a = clustered_vertices(vv[i], 0);
                int b = clustered_vertices(v_id, 0);

               
                //atomicAdd(&number_of_neighbors[clustered_vertices(v_id, 0)],1);

                vns[b].addNeighbor(a);
               

                //neighbor adding logic here where we say that b is a neighbor of a
                

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

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bumpy-cube.obj");

    auto vertex_pos = *rx.get_input_vertex_coordinates();

    //attribute to sample,store and order samples
    auto sample_number = *rx.add_vertex_attribute<int>("sample_number", 1);
    auto distance      = *rx.add_vertex_attribute<float>("distance", 1);


    auto sample_level_bitmask = *rx.add_vertex_attribute<
        uint16_t>("bitmask", 1);
    auto clustered_vertex = *rx.add_vertex_attribute<int>("clustering", 1);

    auto number_of_neighbors_coarse = *rx.add_vertex_attribute<int>(
        "number of neighbors",
        1);


    int* flagger;
    cudaMallocManaged(&flagger, sizeof(int));
    *flagger = 0;

    auto context = rx.get_context();

    constexpr uint32_t               CUDABlockSize = 512;
    rxmesh::LaunchBox<CUDABlockSize> lb;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          lb,
                          (void*)sample_points<float, CUDABlockSize>);


    float ratio           = 8;
    int   N               = rx.get_num_vertices();
    int   numberOfLevels  = 4;
    int   currentLevel    = 2; // first coarse mesh
    int   numberOfSamples = N / powf(ratio, 1);


    std::random_device rd;
    // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dist(0, N - 1);
    // From 0 to (number of points - 1)
    int seed = dist(gen);

    std::cout << "\nSeed: " << seed;

    VertexReduceHandle<float>              reducer(distance);
    cub::KeyValuePair<VertexHandle, float> farthestPoint;

    Vec3* vertices;
    Vec3* sample_pos;
    uint8_t* bitmask;
    float*    distanceArray;
    int*    clusterVertices;
    
    // Allocate unified memory
    cudaMallocManaged(&sample_pos, numberOfSamples * sizeof(Vec3));
    cudaMallocManaged(&vertices, N * sizeof(Vec3));
    cudaMallocManaged(&bitmask, N * sizeof(int));
    cudaMallocManaged(&distanceArray, N * sizeof(int));
    cudaMallocManaged(&clusterVertices, N * sizeof(int));

    cudaDeviceSynchronize();

    for (int i=0;i<N;i++) {
        bitmask[i] = 0;
    }


    // pre processing step
    //gathers samples for every level
    int j = 0;
    int currentSampleLevel = numberOfLevels;
    std::cout << "levels:";

    for (int q=0;q<numberOfLevels;q++) {
        std::cout << "\n  level " << q << " : " << N / powf(ratio, q);
    }
    for (int i = 0; i < numberOfSamples; i++) {
        if (i == N / (int)powf(ratio,  currentSampleLevel)) {
            currentSampleLevel--;
            std::cout << "\nNext sample level: " << currentSampleLevel;
        }

        rx.for_each_vertex(rxmesh::DEVICE,
                           [seed,
                               context,
                               sample_number,
                               sample_level_bitmask,
                               bitmask,
                               distance,
                               i,
                               currentSampleLevel,
             sample_pos,
                               vertex_pos] __device__(
                           const rxmesh::VertexHandle vh) {
                               if (seed == context.linear_id(vh)) {
                                   sample_number(vh, 0) = i;
                                   distance(vh, 0)      = 0;
                                   sample_pos[i].x        = vertex_pos(vh, 0);
                                   sample_pos[i].y      = vertex_pos(vh, 1);
                                   sample_pos[i].z        = vertex_pos(vh, 2);
                                   /*
                                    *something like
                                    *
                                    *for each level if samples_current_level [currentLevel] > i 
                                    * then
                                    * sample level bitmask (vh,0) |= (1<<currentevel)
                                    */

                                   for (int k = 0; k < currentSampleLevel;
                                        k++) {
                                       sample_level_bitmask(vh, 0) |= (1 << k);
                                       bitmask[seed] |= (1 << k);
                                   }
                               } else {
                                   if (i == 0) {
                                       distance(vh, 0)      = INFINITY;
                                       sample_number(vh, 0) = -1;
                                   }
                               }
                           });

        do {
            cudaDeviceSynchronize();
            *flagger = 0;
            sample_points<float, CUDABlockSize>
                <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                    rx.get_context(),
                    vertex_pos,
                    distance,
                    flagger);
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
        {rxmesh::Op::VV},
        cb,
        (void*)cluster_points<float, CUDABlockSize>);


    setCluster(N, distanceArray, clusterVertices, bitmask, currentLevel);

    //clustering step
    j = 0;
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [
            sample_number,
            sample_level_bitmask,
            distance,
            currentLevel,
            clustered_vertex, context, vertices, vertex_pos ]
        __device__(const rxmesh::VertexHandle vh) {

                vertices[context.linear_id(vh)].x = vertex_pos(vh, 0);
                vertices[context.linear_id(vh)].y = vertex_pos(vh, 1);
                vertices[context.linear_id(vh)].z = vertex_pos(vh, 2);

            //if (sample_number(vh, 0) > -1)
            if ((sample_level_bitmask(vh,0) & (1 << currentLevel)) !=0)
            {
                clustered_vertex(vh, 0) = sample_number(vh, 0);
                distance(vh, 0)         = 0;
            } else {
                distance(vh, 0)         = INFINITY;
                clustered_vertex(vh, 0) = -1;
            }
        });
    do {
        cudaDeviceSynchronize();
        *flagger = 0;
        cluster_points<float, CUDABlockSize>
            <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                rx.get_context(),
                vertex_pos,
                distance,
                clustered_vertex,
                flagger);
        cudaDeviceSynchronize();
        j++;
    } while (*flagger != 0);

    clustered_vertex.move(DEVICE, HOST);
    std::cout << "\Clustering iterations: " << j;

    int* vertexCluster;
    cudaMallocManaged(&vertexCluster, sizeof(int) * N);
    rx.for_each_vertex(rxmesh::DEVICE,
                       [
                           clustered_vertex,
                           context,vertexCluster] __device__(
                       const rxmesh::VertexHandle vh) {
                           vertexCluster[context.linear_id(vh)] =
                               clustered_vertex(vh, 0);


                       });
    cudaDeviceSynchronize();


    ////
    ///
    //find number of neighbors x
    int* number_of_neighbors;
    cudaMallocManaged(&number_of_neighbors, numberOfSamples * sizeof(int));
    for (int i = 0; i < numberOfSamples; i++) {
        number_of_neighbors[i] = 0;
    }
    cudaDeviceSynchronize();

    //ASSERT here that every vertex has a neighbor

    rxmesh::LaunchBox<CUDABlockSize> nn;
    rx.prepare_launch_box(
        {rxmesh::Op::VV},
        nn,
        (void*)findNumberOfCoarseNeighbors<float, CUDABlockSize>);


    //find number of neighbors without the bit matrix


    // Allocate memory for vertex neighbors
    VertexNeighbors* vertexNeighbors;
    cudaError_t      err = cudaMallocManaged(&vertexNeighbors,
                                             numberOfSamples * sizeof(
                                                 VertexNeighbors));

    findNumberOfCoarseNeighbors<float, CUDABlockSize>
        <<<nn.blocks, nn.num_threads, nn.smem_bytes_dyn>>>(
            rx.get_context(),
            clustered_vertex,
            number_of_neighbors,
            vertexNeighbors);
    cudaDeviceSynchronize();


    rx.for_each_vertex(rxmesh::DEVICE,
                       [numberOfSamples,
                           context,
                           vertexNeighbors,
                           clustered_vertex,
                           sample_number,
                           number_of_neighbors] __device__(
                       const rxmesh::VertexHandle vh) {

                           if (clustered_vertex(vh, 0) ==
                               sample_number(vh, 0)) {
                               number_of_neighbors[clustered_vertex(vh, 0)] =
                                   vertexNeighbors[sample_number(vh, 0)].
                                   getNumberOfNeighbors();


                               /* printf("\n vertex %d : %d neighbors",
                       sample_number(vh, 0),
                       vertexNeighbors[sample_number(vh, 0)].getNumberOfNeighbors());
                       */
                           }
                       });
    cudaDeviceSynchronize();


    //numberOfNeighbors(numberOfSamples, vertexNeighbors,vertexCluster,N);

    //cudaFree(vertexNeighbors);


    // construct row pointers -> prefix sum
    // Number of rows in your matrix

    int num_rows = numberOfSamples; // Set this appropriately
    CSR csr(num_rows, number_of_neighbors, vertexNeighbors, N);

    csr.printCSR();

    cudaDeviceSynchronize(); // Ensure data is synchronized before accessing


    //for debug purposes
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [sample_number,
            clustered_vertex,
            number_of_neighbors,
            number_of_neighbors_coarse,
            context] __device__(const rxmesh::VertexHandle vh) {
            number_of_neighbors_coarse(vh, 0) = number_of_neighbors[
                sample_number(vh, 0)];
        });


    float* prolongation_operator;
    cudaMallocManaged(&prolongation_operator,
                      N * numberOfSamples * sizeof(float));

    cudaDeviceSynchronize();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < numberOfSamples; j++) {
            prolongation_operator[numberOfSamples * i + j] = 0;
        }
    }

    cudaDeviceSynchronize();

    
   /* createProlongationOperator(csr.num_rows,
                                csr.row_ptr,
                                csr.value_ptr,
                                csr.number_of_neighbors,
                                N,
                                vertexCluster,
                                vertices, sample_pos,
                                prolongation_operator);
                                
    cudaDeviceSynchronize();*/


    /*
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
             // get the neighbor vertex
            neighbor = value_ptr[i];  // assuming col_idx stores column
                                        // indices of neighbors in csr.

            // get the range of neighbors for this neighbor
            const int neighbor_start = row_ptr[neighbor];
            const int neighbor_end   = row_ptr[neighbor + 1];

            for (int j = neighbor_start; j < neighbor_end; j++) {
                neighbor_of_neighbor = value_ptr[j];

                for (int k=i+1;k<end_pointer;k++)
                {
                    if (value_ptr[k]==neighbor_of_neighbor) 
                    { 


                        
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


        //prolongation_operator[l * numberOfSamples + cluster_point]        = b1;
        //prolongation_operator[l * numberOfSamples + selected_neighbor] = b2;
        //prolongation_operator[l * numberOfSamples + selected_neighbor_of_neighbor]              = b3;

        
        prolongation_operator[l * numberOfSamples + cluster_point] =
            cluster_point;
        prolongation_operator[l * numberOfSamples + selected_neighbor] =
            selected_neighbor;
        prolongation_operator[l * numberOfSamples +
                              selected_neighbor_of_neighbor] =
            selected_neighbor_of_neighbor;

        //printf("\n%d at %d", l, l * numberOfSamples);



    });
    */


    std::cout << std::endl;
    std::cout << std::endl;

    cudaDeviceSynchronize();

    
    /*for (int i = 0; i < N; i++) {
        std::cout << std::endl << i << "  ";
        for (int j = 0; j < numberOfSamples; j++) {
            std::cout << prolongation_operator[i * numberOfSamples + j] << " ";
        }
    }*/
    

    //csr clustering


    //new set of vertices
    //new set of distances
    //use the same flag

    cudaFree(vertices);
    cudaMallocManaged(&vertices, sizeof(Vec3) * csr.num_rows);

    float* distances;
    cudaMallocManaged(&distances, sizeof(float) * csr.num_rows);
    


    
    ///general prolongation operator construction
    ///
    /// cluster the next set
    /// csr construct
    /// prolongate
    ///
    ///

    //


    number_of_neighbors_coarse.move(DEVICE, HOST);

    rx.get_polyscope_mesh()->addVertexScalarQuantity(
        "sample_number",
        sample_number);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("distance", distance);
    rx.get_polyscope_mesh()->addVertexScalarQuantity(
        "sample_level_bitmask",
        sample_level_bitmask);
    rx.get_polyscope_mesh()->addVertexScalarQuantity(
        "clusterPoint",
        clustered_vertex);
    rx.get_polyscope_mesh()->addVertexScalarQuantity(
        "number of neighbors",
        number_of_neighbors_coarse);


#if USE_POLYSCOPE
    polyscope::show();
#endif
}

