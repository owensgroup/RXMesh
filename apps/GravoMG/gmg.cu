
#include "include/NeighborHandling.h"


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>


#include "include/GMGProlongatorCreation.h"
std::vector<int> intPointerArrayToVector(int* array, size_t size)
{
    return std::vector<int>(array, array + size);
}


void CreateNextLevelData(int      N,
                         int      numberOfSamples,
                         VertexData* vData_old,
                         VertexData* vData_new)
{

    thrust::device_vector<int> samples(N);
    thrust::sequence(samples.begin(), samples.end());

    thrust::for_each(thrust::device,
                     samples.begin(),
                     samples.end(),
        [=] __device__(int number) { vData_new[number].distance = 0;
                     });
}




void numberOfNeighbors(int              numberOfSamples,
                       VertexNeighbors* neighbors ,
    int              N,
                       CSR              csr,
                        VertexData* vData, int* number_of_neighbors

)
{
    thrust::device_vector<int> samples(N);
    thrust::sequence(samples.begin(), samples.end());

    int* neighborList;
    cudaMallocManaged(&neighborList, sizeof(int) * numberOfSamples);

    for (int i = 0; i < numberOfSamples; i++)
        neighborList[i] = 0;
    thrust::for_each(thrust::device,
                     samples.begin(),
                     samples.end(),
                     [=] __device__(int number) {
                         //int currentCluster = vertexClusters[number];
                         int currentCluster = vData[number].cluster;

                         //neighbors[currentCluster].getNeighbors(neighborList);
                         for (int i = csr.row_ptr[number]; i < csr.row_ptr[number+1];i++) {
                             int currentNode = csr.value_ptr[i];
                             if (vData[currentNode].cluster != currentCluster) {
                                 //neighbors
                                 neighbors[currentCluster].addNeighbor(vData[currentNode].cluster);
                             }
                         }

                         // std::cout <<
                         // neighbors[0].getNeighbors().size();//this is the
                         // neighbor count
                     });


     thrust::device_vector<int> samples2(numberOfSamples);
    thrust::sequence(samples2.begin(), samples2.end());

    thrust::for_each(thrust::device,
                     samples.begin(),
                     samples.end(),
                     [=] __device__(int number) {

                        number_of_neighbors[number] =
                             neighbors[number].getNumberOfNeighbors();

                     });


}




void setCluster(int    n,
                float* distance,
    int currentLevel,
    VertexData* vertex_data)
{
    thrust::device_vector<int> samples(n);
    thrust::sequence(samples.begin(), samples.end());

    thrust::for_each(thrust::device,
                     samples.begin(),
                     samples.end(),
                     [=] __device__(int number) {
                         // take bitmask
                         // if sample, the cluster is its own
                         // if not a sample, set cluster as -1
                         // set distance as infinity or 0 based on whether it is
                         // not or is a sample

                         if ((vertex_data[number].bitmask & (1 << currentLevel-1)) != 0) 
                         {
                            distance[number]        = 0;
                             vertex_data[number].cluster =
                                 vertex_data[number].sample_number;
                            
                             printf(
                                "\n%d which is sample %d is now a cluster vertex",
                                number,
                                vertex_data[number].sample_number);
                                
                         }
                         else 
                         {
                             vertex_data[number].cluster =-1;
                             /*
                             printf(
                                 "\n%d which is sample %d is not a cluster "
                                 "vertex",
                                 number,
                                 vertex_data[number].sample_number);
                                 */
                             distance[number]        = INFINITY;
                         }
        });
}







int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

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
    int   currentLevel    = 1; // first coarse mesh
    int   numberOfSamplesForFirstLevel = N / powf(ratio, 1);//start
    int   numberOfSamples = N / powf(ratio, currentLevel);//start


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
    float*    distanceArray;
    int*    clusterVertices;

    
    // Allocate unified memory
    cudaMallocManaged(&sample_pos, numberOfSamples * sizeof(Vec3));
    cudaMallocManaged(&vertices, N * sizeof(Vec3));
    cudaMallocManaged(&distanceArray, N * sizeof(int));
    cudaMallocManaged(&clusterVertices, N * sizeof(int));

    cudaDeviceSynchronize();


    // pre processing step
    //gathers samples for every level
    int j = 0;
    int currentSampleLevel = numberOfLevels;
    std::cout << "levels:";

    for (int q=0;q<numberOfLevels;q++) {
        std::cout << "\n  level " << q << " : " << N / powf(ratio, q);
    }
    for (int i = 0; i < numberOfSamplesForFirstLevel; i++) {
        if (i == N / (int)powf(ratio,  currentSampleLevel)) {
            currentSampleLevel--;
            std::cout << "\nNext sample level: " << currentSampleLevel;
        }

        rx.for_each_vertex(rxmesh::DEVICE,
                           [seed,
                               context,
                               sample_number,
                               sample_level_bitmask,
                               distance,
                               i,
                               currentSampleLevel,
             sample_pos,
                               vertex_pos] __device__(
                           const rxmesh::VertexHandle vh) {
                               if (seed == context.linear_id(vh)) {
                                   sample_number(vh, 0) = i;
                                   //sample_number_point
                                   distance(vh, 0)      = 0;
                                   sample_pos[i].x        = vertex_pos(vh, 0);
                                   sample_pos[i].y      = vertex_pos(vh, 1);
                                   sample_pos[i].z        = vertex_pos(vh, 2);

                                   for (int k = 0; k < currentSampleLevel;
                                        k++) {
                                       sample_level_bitmask(vh, 0) |= (1 << k);
                                   }

                                   /* printf(
                                       "\n\n Sample %d, vertex %d is level %d",
                                          i, 
                                          seed,
                                          currentSampleLevel);
                                          */
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
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///
    /// first level

    rxmesh::LaunchBox<CUDABlockSize> cb;
    rx.prepare_launch_box(
        {rxmesh::Op::VV},
        cb,
        (void*)cluster_points<float, CUDABlockSize>);



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
            if ((sample_level_bitmask(vh,0) & (1 << (currentLevel-1))) !=0)
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

    int* number_of_neighbors;
    cudaMallocManaged(&number_of_neighbors, numberOfSamples * sizeof(int));
    for (int i = 0; i < numberOfSamples; i++) {
        number_of_neighbors[i] = 0;
    }
    cudaDeviceSynchronize();

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

    int num_rows = numberOfSamples; // Set this appropriately
    CSR csr(num_rows, number_of_neighbors, vertexNeighbors, N);

    //csr.printCSR();

    cudaDeviceSynchronize(); // Ensure data is synchronized before accessing

   
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


    float* prolongation_operator;
    cudaMallocManaged(&prolongation_operator,
                      N * numberOfSamples * sizeof(float));
    cudaDeviceSynchronize();
   createProlongationOperator(csr.num_rows,
                                csr.row_ptr,
                                csr.value_ptr,
                                csr.number_of_neighbors,
                                N,
                                vertexCluster,
                                vertices, sample_pos,
                                prolongation_operator);
                                
    cudaDeviceSynchronize();
   /*
    for (int i=0;i<N;i++) {
        std::cout << "\n" << i << " ";
        for (int k=0;k<numberOfSamples;k++) {
            std::cout << prolongation_operator[i * numberOfSamples + k] << " ";
        }
    }
    */

    Eigen::MatrixXd verts;
    Eigen::MatrixXi faces;
    std::vector<std::array<double, 3>>
        vertexPositions;                            // To store vertex positions
        std::vector<std::vector<size_t>> faceIndices;  // To store face indices
 
    csr.GetRenderData(vertexPositions, faceIndices, sample_pos);

    polyscope::registerSurfaceMesh("mesh level 1", vertexPositions, faceIndices);


    //set 1st level node data
    VertexData* oldVdata;
    cudaMallocManaged(&oldVdata, sizeof(VertexData) * numberOfSamples);

    rx.for_each_vertex(rxmesh::DEVICE,
                       [sample_number,
                        oldVdata,
                        clustered_vertex,
                        vertex_pos,
         sample_level_bitmask,
                        context] __device__(const rxmesh::VertexHandle vh) {


            if (sample_number(vh, 0) != -1) {
                //printf("\nputting data for sample %d", sample_number(vh, 0));

                oldVdata[sample_number(vh, 0)].distance = 0;
                oldVdata[sample_number(vh, 0)].linear_id =
                    context.linear_id(vh);
                oldVdata[sample_number(vh, 0)].sample_number =
                    sample_number(vh, 0);
                oldVdata[sample_number(vh, 0)].bitmask =
                    sample_level_bitmask(vh, 0);
                oldVdata[sample_number(vh, 0)].position.x = vertex_pos(vh, 0);
                oldVdata[sample_number(vh, 0)].position.y = vertex_pos(vh, 1);
                oldVdata[sample_number(vh, 0)].position.z = vertex_pos(vh, 2);
                oldVdata[sample_number(vh, 0)].cluster =
                    clustered_vertex(vh, 0);
            }

                       });
    

    ////////////////////////////////////////////////////////////////////////////////////////
    ///
    ///next levels




    CSR lastCSR                 = csr;
    CSR currentCSR              = csr;
    int currentNumberOfVertices = numberOfSamples;
    int currentNumberOfSamples  = numberOfSamples / ratio;
    std::vector<float*> prolongationOperators;
    prolongationOperators.resize(numberOfLevels-1);

    prolongationOperators[0] = prolongation_operator;

    std::vector<Eigen::MatrixXd> vertsArray;
    std::vector<Eigen::MatrixXi> facesArray;
    std::vector<std::vector<std::array<double, 3>>>
        vertexPositionsArray;  // To store vertex positions
    std::vector<std::vector<std::vector<size_t>>>
        faceIndicesArray;  // To store face indices

    std::vector<std::vector<int>> clustering;



    vertsArray.resize(numberOfLevels);
    facesArray.resize(numberOfLevels);
    vertexPositionsArray.resize(numberOfLevels);
    faceIndicesArray.resize(numberOfLevels);
    clustering.resize(numberOfLevels);

    for (int level = 1; level < numberOfLevels - 1; level++) {

        std::cout << "\nlevel : " << level;
        std::cout << "\n current number of samples: " << currentNumberOfSamples;
        std::cout << "\n current number of vertices: "
                  << currentNumberOfVertices;
        setCluster(currentNumberOfVertices, distanceArray, level + 1, oldVdata);

        do {
            *flagger = 0;
            clusterCSR(currentNumberOfVertices,
                       sample_pos,
                       distanceArray,
                       vertexCluster,
                       flagger,
                       lastCSR,
                       oldVdata);
            cudaDeviceSynchronize();
        } while (*flagger != 0);

        //clustering[level - 1].resize(currentNumberOfVertices);
        //clustering[level-1]=intPointerArrayToVector(vertexCluster, currentNumberOfVertices);


        //polyscope::getSurfaceMesh("mesh level " + std::to_string(level))
        //->addVertexScalarQuantity("clustered vertices", clustering[level-1]);


        VertexNeighbors* vertexNeighbors2;
        err = cudaMallocManaged(
            &vertexNeighbors2,
            currentNumberOfVertices * sizeof(VertexNeighbors));

        int* number_of_neighbors2;
        cudaMallocManaged(&number_of_neighbors2,
                          currentNumberOfVertices * sizeof(int));


        numberOfNeighbors(currentNumberOfSamples,
                          vertexNeighbors2,
                          currentNumberOfVertices,
                          lastCSR,
                          oldVdata,
                          number_of_neighbors2);


        cudaDeviceSynchronize();
        currentCSR = CSR(currentNumberOfSamples,
                         number_of_neighbors2,
                         vertexNeighbors2,
                         currentNumberOfVertices);

        // currentCSR.printCSR();

        currentCSR.GetRenderData(vertexPositionsArray[level - 1],
                                 faceIndicesArray[level - 1],
                                 sample_pos);


        polyscope::registerSurfaceMesh(
            "mesh level " + std::to_string(level + 1),
            vertexPositionsArray[level - 1],
            faceIndicesArray[level - 1]);



        float* prolongationOperator2;
        cudaMallocManaged(&prolongationOperator2,
                          sizeof(float) * currentNumberOfSamples * currentNumberOfVertices);
        cudaDeviceSynchronize();
        prolongationOperators[level] = prolongationOperator2;
        cudaDeviceSynchronize();

        createProlongationOperator(currentNumberOfSamples,
                                   currentCSR.row_ptr,
                                   currentCSR.value_ptr,
                                   number_of_neighbors2,
                                   currentNumberOfVertices,
                                   oldVdata,
                                   prolongationOperator2);

        cudaDeviceSynchronize();  // Ensure data is synchronized before
                                  // accessing


        currentNumberOfSamples /= 8;
        currentNumberOfVertices /= 8;
        lastCSR = currentCSR;

    

    }

    cudaDeviceSynchronize();
    for (int q = 2; q < numberOfLevels - 1; q++) {
        std::cout << "\n\n Level: " << q << " to " << q + 1;
        int rows = N / static_cast<int>(std::round(powf(ratio, q)));
        int cols =
            numberOfSamples / static_cast<int>(std::round(powf(ratio, q)));
        std::cout << " Rows: " << rows;
        std::cout << " Cols: " << cols;

        for (int l = 0; l < rows; l++) {
            std::cout << "\n" << l << ": ";
            for (int m = 0; m < cols; m++) {
                int index = l * cols + m;
                auto a     = prolongationOperators[q][index];
                std::cout << prolongationOperators[q][index] << " ";
            }
        }
    }

   
    /*
    setCluster(numberOfSamples, distanceArray, currentLevel+1, oldVdata);

    do {
        *flagger = 0;
        clusterCSR(numberOfSamples,
                   sample_pos,
                   distanceArray,
                   vertexCluster,
                   flagger,
                   csr,oldVdata);
        cudaDeviceSynchronize();
    } while (*flagger != 0);

    std::vector<int> a = intPointerArrayToVector(vertexCluster,numberOfSamples);

    polyscope::getSurfaceMesh("mesh level 1")
        ->addVertexScalarQuantity("cluster Vertex", a);

    //neighbor stuff
    
    VertexNeighbors* vertexNeighbors2;
    err = cudaMallocManaged(&vertexNeighbors2,
                            numberOfSamples * sizeof(VertexNeighbors));

    int* number_of_neighbors2;
    cudaMallocManaged(&number_of_neighbors2,
                      numberOfSamples * sizeof(int));


    numberOfNeighbors(numberOfSamples / 8,
                      vertexNeighbors2,
                      numberOfSamples,
                      csr,
                      oldVdata,number_of_neighbors2);


        cudaDeviceSynchronize();
    num_rows = numberOfSamples / 8;  // Set this appropriately
    CSR csr2(num_rows,
             number_of_neighbors2,
             vertexNeighbors2,
             numberOfSamples);

    csr2.printCSR();


    float* prolongationOperator2;
    cudaMallocManaged(&prolongationOperator2,
                      sizeof(float) * numberOfSamples * num_rows);


    createProlongationOperator(num_rows,
                               csr2.row_ptr,
                               csr2.value_ptr,
                               number_of_neighbors2,
                               numberOfSamples,
                               oldVdata,
                               prolongationOperator2);

    cudaDeviceSynchronize();  // Ensure data is synchronized before accessing


     Eigen::MatrixXd verts2;
    Eigen::MatrixXi faces2;
    std::vector<std::array<double, 3>>
        vertexPositions2;                           // To store vertex positions
    std::vector<std::vector<size_t>> faceIndices2;  // To store face indices

    csr2.GetRenderData(vertexPositions2, faceIndices2, sample_pos);

    polyscope::registerSurfaceMesh(
        "mesh level 2", vertexPositions2, faceIndices2);
        
        */


    //////////////////////////////////////////////////////////////////

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




    cudaFree(vertices);
    cudaMallocManaged(&vertices, sizeof(Vec3) * csr.num_rows);

    float* distances;
    cudaMallocManaged(&distances, sizeof(float) * csr.num_rows);


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

