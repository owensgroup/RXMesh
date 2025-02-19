#pragma once



#include "GMGRXMeshKernels.h"

#include "GMGCSR.h"


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

    // printf("\ncalculated coords are %f, %f, %f", lambda0, lambda1, lambda2);

    // Return the barycentric coordinates
    return std::make_tuple(lambda0, lambda1, lambda2);
}

__device__ void computeBarycentricCoordinates(const Eigen::Vector3f& v0,
                                              const Eigen::Vector3f& v1,
                                              const Eigen::Vector3f& v2,
                                              const Eigen::Vector3f& p,
                                              float&                 a,
                                              float&                 b,
                                              float&                 c)
{
    // Compute edges of the triangle
    Eigen::Vector3f edge1    = v1 - v0;
    Eigen::Vector3f edge2    = v2 - v0;
    Eigen::Vector3f pointVec = p - v0;

    // Compute normal of the triangle
    Eigen::Vector3f normal = edge1.cross(edge2);
    float area2 = normal.squaredNorm();  // Area of the triangle multiplied by 2

    // Compute barycentric coordinates
    float lambda0 = 0, lambda1 = 0, lambda2 = 0;

    lambda0 = (v1 - p).cross(v2 - p).dot(normal) / area2;
    lambda1 = (v2 - p).cross(v0 - p).dot(normal) / area2;
    lambda2 = (v0 - p).cross(v1 - p).dot(normal) / area2;

    a = lambda0;
    b = lambda1;
    c = lambda2;
     //printf("\ncalculated coords are %f, %f, %f", lambda0, lambda1, lambda2);
}

void createProlongationOperator(int  numberOfSamples,
                                int* row_ptr,
                                int* value_ptr,
                                int* number_of_neighbors,
                                int  N,
                                int*   clustered_vertex,
                                Vec3*  vertex_pos,
                                Vec3*  sample_pos,
                                int*   operator_value_ptr,
                                float* operator_data_ptr)
{
    thrust::device_vector<int> samples(N);
    thrust::sequence(samples.begin(), samples.end());
    /*
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
            const Eigen::Vector3<float> q{vertex_pos[number].x,
                                          vertex_pos[number].y,
                                          vertex_pos[number].z};

            int neighbor                      = 0;
            int selected_neighbor             = 0;
            int neighbor_of_neighbor          = 0;
            int selected_neighbor_of_neighbor = 0;

        /*

            for (int i = start_pointer; i < end_pointer; i++) {
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


                            Eigen::Vector3<float> v1{
                                sample_pos[cluster_point].x,
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
                            if (distance < min_distance ) {

                                min_distance      = distance;
                                selected_neighbor = neighbor;
                                selected_neighbor_of_neighbor =
                                    neighbor_of_neighbor;
                                
                                selectedv1        = v1;
                                selectedv2        = v2;
                                selectedv3        = v3;

                                Eigen::Vector3f edge1 = v2 - v1;
                                Eigen::Vector3f edge2 = v3 - v1;

                                // Cross product gives us the normal vector
                                Eigen::Vector3f normal = edge1.cross(edge2);

                                // Check the z-component of the normal (assuming
                                // Z-up coordinate system) If normal.z() < 0,
                                // the vertices are in a clockwise order
                                if (normal.z() < 0) 
                                {
                                    Eigen::Vector3f temp = selectedv2;
                                    selectedv2 = selectedv3;
                                    selectedv3 = temp;
                                    selected_neighbor = neighbor_of_neighbor;
                                    selected_neighbor_of_neighbor = neighbor;
                                }
                                
                            }
                        }
                    }
                }
            }

            

                int neighbors_count = end_pointer - start_pointer;
                if (neighbors_count >= 2) {  // Process only if it can form faces
                    for (int j = start_pointer; j < end_pointer; ++j) {
                        // Create a triangular face (i, value_ptr[j],
                        // value_ptr[j+1])

                        int a, b, c;
                        if (j == end_pointer - 1) {
                            a = cluster_point;
                            b = value_ptr[j];
                            c = value_ptr[row_ptr[cluster_point]];
                        } else {
                            a = cluster_point;
                            b = value_ptr[j];
                            c = value_ptr[j + 1];
                        }
                        // check if b,c are neighbors
                        int n1_start = row_ptr[b];
                        int n1_end   = row_ptr[b + 1];
                        for (int k = n1_start; k < n1_end; k++) {
                            if (c == value_ptr[k]) 
                            {
                                //std::cout << "\n Triangle formed with " << a
                                  //        << " and " << b << " and " << c;

                                
                                 Eigen::Vector3<float> v1{
                                    sample_pos[a].x,
                                    sample_pos[a].y,
                                    sample_pos[a].z};
                                Eigen::Vector3<float> v2{
                                    sample_pos[b].x,
                                    sample_pos[b].y,
                                    sample_pos[b].z};
                                Eigen::Vector3<float> v3{
                                    sample_pos[c].x,
                                    sample_pos[c].y,
                                    sample_pos[c].z};
                                float                 distance = projectedDistance(v1, v2, v3, q);
                                if (distance < min_distance) 
                                {
                                    min_distance = distance;
                                    selectedv1   = v1;
                                    selectedv2   = v2;
                                    selectedv3   = v3;
                                    selected_neighbor = b;
                                    selected_neighbor_of_neighbor = c;
                                }
                                break;
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
              
            
            float b1 = 0, b2 = 0, b3 = 0;
            computeBarycentricCoordinates(
                selectedv1, selectedv2, selectedv3, q, b1, b2, b3);

        }

            // put it inside prolongation row, it will be unique so no race
            // condition
            int l = number;

            operator_value_ptr[l * 3]     = cluster_point;
            operator_value_ptr[l * 3 + 1] = selected_neighbor;
            operator_value_ptr[l * 3 + 2] = selected_neighbor_of_neighbor;
            operator_data_ptr[l * 3]      = b1;
            operator_data_ptr[l * 3 + 1]  = b2;
            operator_data_ptr[l * 3 + 2]  = b3;
        });
        */

    /*
    std::cout << "\n\n\n";
    for (int i = 0; i < N; i++) {
        std::cout << "\n" << i << " ";
        for (int j = 0; j < numberOfSamples; j++) {
            std::cout << prolongation_operator[i * numberOfSamples + j] << " ";
        }
    }
    */
    /*
        thrust::for_each(
        thrust::device,
        samples.begin(),
        samples.end(),
        [=] __device__(int number) {
            const int cluster_point = clustered_vertex[number];
            const int start_pointer = row_ptr[cluster_point];
            const int end_pointer   = row_ptr[cluster_point + 1];

            float                 min_distance = 99999.0f;
            Eigen::Vector3<float> selectedv1{0, 0, 0}, selectedv2{0, 0, 0},
                selectedv3{0, 0, 0};
            const Eigen::Vector3<float> q{vertex_pos[number].x,
                                          vertex_pos[number].y,
                                          vertex_pos[number].z};

            int selected_neighbor             = 0;
            int selected_neighbor_of_neighbor = 0;

            // We need at least 2 neighbors to form a triangle
            int neighbors_count = end_pointer - start_pointer;
            //if (neighbors_count >= 2) {
                // Process consecutive triples, including the wrap-around case
                for (int j = start_pointer; j < end_pointer; ++j) {
                    int v1_idx = cluster_point;
                    int v2_idx = value_ptr[j];
                    int v3_idx;

                    // Handle wrap-around for the last vertex
                    if (j == end_pointer - 1) {
                        v3_idx = value_ptr[start_pointer];
                    } else {
                        v3_idx = value_ptr[j + 1];
                    }

                    Eigen::Vector3<float> v1{sample_pos[v1_idx].x,
                                             sample_pos[v1_idx].y,
                                             sample_pos[v1_idx].z};
                    Eigen::Vector3<float> v2{sample_pos[v2_idx].x,
                                             sample_pos[v2_idx].y,
                                             sample_pos[v2_idx].z};
                    Eigen::Vector3<float> v3{sample_pos[v3_idx].x,
                                             sample_pos[v3_idx].y,
                                             sample_pos[v3_idx].z};

                    float distance = projectedDistance(v1, v2, v3, q);
                    if (distance < min_distance) {
                        min_distance                  = distance;
                        selectedv1                    = v1;
                        selectedv2                    = v2;
                        selectedv3                    = v3;
                        selected_neighbor             = v2_idx;
                        selected_neighbor_of_neighbor = v3_idx;
                    }
                }
            //}

            // Compute barycentric coordinates for the closest triangle
            float b1 = 0, b2 = 0, b3 = 0;
            computeBarycentricCoordinates(
                selectedv1, selectedv2, selectedv3, q, b1, b2, b3);

            // Store results
            int l                         = number;
            operator_value_ptr[l * 3]     = cluster_point;
            operator_value_ptr[l * 3 + 1] = selected_neighbor;
            operator_value_ptr[l * 3 + 2] = selected_neighbor_of_neighbor;
            operator_data_ptr[l * 3]      = b1;
            operator_data_ptr[l * 3 + 1]  = b2;
            operator_data_ptr[l * 3 + 2]  = b3;
        });
        */

    thrust::for_each(
        thrust::device,
        samples.begin(),
        samples.end(),
        [=] __device__(int number) {
            const int cluster_point = clustered_vertex[number];
            const int start_pointer = row_ptr[cluster_point];
            const int end_pointer   = row_ptr[cluster_point + 1];

            float                 min_distance = 99999.0f;
            Eigen::Vector3<float> selectedv1{0, 0, 0}, selectedv2{0, 0, 0},
                selectedv3{0, 0, 0};
            const Eigen::Vector3<float> q{vertex_pos[number].x,
                                          vertex_pos[number].y,
                                          vertex_pos[number].z};

            int selected_neighbor             = 0;
            int selected_neighbor_of_neighbor = 0;

            // We need at least 2 neighbors to form a triangle
            int neighbors_count = end_pointer - start_pointer;
            if (neighbors_count >= 2) {
                // Iterate through all possible triangle combinations
                for (int j = start_pointer; j < end_pointer; ++j) {
                    for (int k = j + 1; k < end_pointer; ++k) {
                        int v1_idx = cluster_point;
                        int v2_idx = value_ptr[j];
                        int v3_idx = value_ptr[k];

                        // Verify v2 and v3 are connected (are neighbors)
                        bool      are_neighbors = false;
                        const int n1_start      = row_ptr[v2_idx];
                        const int n1_end        = row_ptr[v2_idx + 1];
                        
                        for (int m = n1_start; m < n1_end; m++) {
                            if (v3_idx == value_ptr[m]) {
                                are_neighbors = true;
                                break;
                            }
                        }
                        
                        if (are_neighbors) {
                            Eigen::Vector3<float> v1{sample_pos[v1_idx].x,
                                                     sample_pos[v1_idx].y,
                                                     sample_pos[v1_idx].z};
                            Eigen::Vector3<float> v2{sample_pos[v2_idx].x,
                                                     sample_pos[v2_idx].y,
                                                     sample_pos[v2_idx].z};
                            Eigen::Vector3<float> v3{sample_pos[v3_idx].x,
                                                     sample_pos[v3_idx].y,
                                                     sample_pos[v3_idx].z};


                            float distance = projectedDistance(v1, v2, v3, q);
                            if (distance < min_distance) {
                                min_distance                  = distance;
                                selectedv1                    = v1;
                                selectedv2                    = v2;
                                selectedv3                    = v3;
                                selected_neighbor             = v2_idx;
                                selected_neighbor_of_neighbor = v3_idx;
                            }
                        }
                    }
                }
            }

            // Compute barycentric coordinates for the closest triangle
            float b1 = 0, b2 = 0, b3 = 0;
            computeBarycentricCoordinates(
                selectedv1, selectedv2, selectedv3, q, b1, b2, b3);

            // Store results
            int l                         = number;
            operator_value_ptr[l * 3]     = cluster_point;
            operator_value_ptr[l * 3 + 1] = selected_neighbor;
            operator_value_ptr[l * 3 + 2] = selected_neighbor_of_neighbor;
            operator_data_ptr[l * 3]      = b1;
            operator_data_ptr[l * 3 + 1]  = b2;
            operator_data_ptr[l * 3 + 2]  = b3;
        });

    
}


void createProlongationOperator(int         numberOfSamples,
                                int*        row_ptr,
                                int*        value_ptr,
                                int* operator_value_ptr,
                                float*        operator_data_ptr,
                                int*        number_of_neighbors,
                                int         N,
                                VertexData* vData)
{
    thrust::device_vector<int> samples(N);
    thrust::sequence(samples.begin(), samples.end());

    thrust::for_each(
        thrust::device,
        samples.begin(),
        samples.end(),
        [=] __device__(int number) {
            // go through every triangle of my cluster
            const int cluster_point = vData[number].cluster;

            //printf("\n cluster point of %d is %d", number, cluster_point);

            const int start_pointer = row_ptr[cluster_point];
            const int end_pointer   = row_ptr[cluster_point + 1];

            float                 min_distance = 99999;
            Eigen::Vector3<float> selectedv1{0, 0, 0}, selectedv2{0, 0, 0},
                selectedv3{0, 0, 0};
            const Eigen::Vector3<float> q{vData[number].position.x,
                                          vData[number].position.y,
                                          vData[number].position.z};

            int neighbor                      = 0;
            int selected_neighbor             = 0;
            int neighbor_of_neighbor          = 0;
            int selected_neighbor_of_neighbor = 0;


            for (int i = start_pointer; i < end_pointer; i++) {
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


                            Eigen::Vector3<float> v1{
                                vData[cluster_point].position.x,
                                vData[cluster_point].position.y,
                                vData[cluster_point].position.z};

                            Eigen::Vector3<float> v2{
                                vData[neighbor].position.x,
                                vData[neighbor].position.y,
                                vData[neighbor].position.z};

                            Eigen::Vector3<float> v3{
                                vData[neighbor_of_neighbor].position.x,
                                vData[neighbor_of_neighbor].position.y,
                                vData[neighbor_of_neighbor].position.z};


                            // find distance , if less than min dist, find bary
                            // coords, save them
                            float distance = projectedDistance(v1, v2, v3, q);
                            if (distance < min_distance) {

                                min_distance      = distance;
                                selected_neighbor = neighbor;
                                selected_neighbor_of_neighbor =
                                    neighbor_of_neighbor;

                                selectedv1 = v1;
                                selectedv2 = v2;
                                selectedv3 = v3;


                                
                            }
                        }
                    }
                }
            }


            /*

            printf("\n%d %f %f %f Selected: %d %d %d",
                       number,
                       vData[number].position.x,
                       vData[number].position.y,
                       vData[number].position.z,
                       cluster_point,
                       selected_neighbor,
                       selected_neighbor_of_neighbor);

              */
            float b1 = 0, b2 = 0, b3 = 0;
            computeBarycentricCoordinates(
                selectedv1, selectedv2, selectedv3, q, b1, b2, b3);

            if (b1 != b1 || b2 != b2 || b3 != b3) {
                printf(
                    "\nDEGENERATE TRIANGLE FOUND \n %d %f %f %f Selected: %d "
                    "%d %d",
                    number,
                    vData[number].position.x,
                    vData[number].position.y,
                    vData[number].position.z,
                    cluster_point,
                    selected_neighbor,
                    selected_neighbor_of_neighbor);
            }


            // printf("\n %d final coords: %f %f %f", number, b1, b2, b3);


            // put it inside prolongation row, it will be unique so no race
            // condition
            int l = number;

            // printf("\nfirst b goes into %d x %d + %d = %d",
            // l,
            // numberOfSamples,
            // cluster_point,
            // l * numberOfSamples + cluster_point);

            operator_value_ptr[l * 3]   = cluster_point;
            operator_value_ptr[l * 3+1]   = selected_neighbor;
            operator_value_ptr[l * 3+2]   = selected_neighbor_of_neighbor;
            operator_data_ptr[l * 3] = b1;
            operator_data_ptr[l * 3 + 1]  = b2;
            operator_data_ptr[l * 3 + 2]  = b3;

            //prolongation_operator[l * numberOfSamples + cluster_point]     = b1;
            //prolongation_operator[l * numberOfSamples + selected_neighbor] = b2;
            //prolongation_operator[l * numberOfSamples + selected_neighbor_of_neighbor]           = b3;
        });

    cudaDeviceSynchronize();
    /*
    std::cout << "\n\n\n";
    for (int i=0;i<N;i++) {
        std::cout << "\n" << i << " ";
        for (int j=0;j<numberOfSamples;j++) {
            std::cout << prolongation_operator[i * numberOfSamples + j] << " ";
        }
    }
    */
}

void numberOfNeighbors(int              numberOfSamples,
                       VertexNeighbors* neighbors,
                       int              N,
                       CSR              csr,
                       VertexData*      vData,
                       int*             number_of_neighbors)
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
                         // int currentCluster = vertexClusters[number];
                         int currentCluster = vData[number].cluster;

                         // neighbors[currentCluster].getNeighbors(neighborList);
                         for (int i = csr.row_ptr[number];
                              i < csr.row_ptr[number + 1];
                              i++) {
                             int currentNode = csr.value_ptr[i];
                             if (vData[currentNode].cluster != currentCluster) {
                                 // neighbors
                                 neighbors[currentCluster].addNeighbor(
                                     vData[currentNode].cluster);
                             }
                         }
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



void setCluster(int         n,
                float*      distance,
                int         currentLevel,
                VertexData* vertex_data)
{
    thrust::device_vector<int> samples(n);
    thrust::sequence(samples.begin(), samples.end());

    thrust::for_each(
        thrust::device,
        samples.begin(),
        samples.end(),
        [=] __device__(int number) {
            // take bitmask
            // if sample, the cluster is its own
            // if not a sample, set cluster as -1
            // set distance as infinity or 0 based on whether it is
            // not or is a sample
            if ((vertex_data[number].bitmask & (1 << currentLevel - 1)) != 0) {
                distance[number]            = 0;
                vertex_data[number].cluster = vertex_data[number].sample_number;

                /*
                printf("\n%d which is sample %d is now a cluster vertex",
                       number,
                       vertex_data[number].sample_number);
                       */
            } else {
                vertex_data[number].cluster = -1;
                distance[number]            = INFINITY;
            }
        });
}



std::vector<int> intPointerArrayToVector(int* array, size_t size)
{
    return std::vector<int>(array, array + size);
}

void createProlongationOperators(int               N,
                                 int               numberOfSamples,
                                 int               numberOfLevels,
                                 float             ratio,
                                 Vec3*             sample_pos,
                                 CSR               csr,
                                 std::vector<CSR>& prolongationOperatorCSR,
                                 VertexData*       oldVdata,
                                 float*            distanceArray,
                                 int*              vertexCluster)
{
    cudaDeviceSynchronize();
    int* flag;
    cudaMallocManaged(&flag, sizeof(int));
    *flag = 0;

    cudaError_t err;
    CSR         lastCSR                 = csr;
    CSR         currentCSR              = csr;
    int         currentNumberOfVertices = N;
    // numberOfSamples;
    int currentNumberOfSamples = numberOfSamples;
    /// ratio;

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


    CSR a(numberOfSamples);

    // operatorsCSR.resize(numberOfLevels-1);

    for (int level = 1; level < numberOfLevels - 1; level++) {

        currentNumberOfSamples /= ratio;
        currentNumberOfVertices /= ratio;
        a = CSR(currentNumberOfVertices);

        std::cout << "\nlevel : " << level;
        std::cout << "\n current number of samples: " << currentNumberOfSamples;
        std::cout << "\n current number of vertices: "
                  << currentNumberOfVertices;
        setCluster(currentNumberOfVertices, distanceArray, level + 1, oldVdata);

        do {

            *flag = 0;
            clusterCSR(currentNumberOfVertices,
                       sample_pos,
                       distanceArray,
                       vertexCluster,
                       flag,
                       lastCSR,
                       oldVdata);
            cudaDeviceSynchronize();
        } while (*flag != 0);

        clustering[level - 1].resize(currentNumberOfVertices);
        clustering[level - 1] =
            intPointerArrayToVector(vertexCluster, currentNumberOfVertices);


        polyscope::getSurfaceMesh("mesh level " + std::to_string(level))
            ->addVertexScalarQuantity("clustered vertices",
                                      clustering[level - 1]);


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

        //currentCSR.printCSR();

        currentCSR.GetRenderData(vertexPositionsArray[level - 1],
                                 faceIndicesArray[level - 1],
                                 sample_pos);

        polyscope::registerSurfaceMesh(
            "mesh level " + std::to_string(level + 1),
            vertexPositionsArray[level - 1],
            faceIndicesArray[level - 1]);


        createProlongationOperator(currentNumberOfSamples,
                                   currentCSR.row_ptr,
                                   currentCSR.value_ptr,
                                   a.value_ptr,
                                   a.data_ptr,
                                   number_of_neighbors2,
                                   currentNumberOfVertices,
                                   oldVdata);
        prolongationOperatorCSR.push_back(a);

        cudaDeviceSynchronize();  // Ensure data is synchronized before
                                  // accessing

        lastCSR = currentCSR;  // next mesh level
    }
    cudaFree(flag);
    cudaDeviceSynchronize();
}
