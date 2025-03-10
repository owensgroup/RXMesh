#pragma once

#include "GMGCSR.h"
#include "GMGRXMeshKernels.h"

#include "NeighborHandling.h"

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
    // printf("\ncalculated coords are %f, %f, %f", lambda0, lambda1, lambda2);
}

/**
 * \brief Function to create prolongation operator for  level 1
 * \param numberOfSamples Number of samples for the first level
 * \param row_ptr row pointer of the mesh CSR
 * \param value_ptr column index pointer for the mesh CSR
 * \param number_of_neighbors pointer containing number of neighbors for each
 * vertex in coarse mesh \param N number of vertices in fine mesh \param
 * clustered_vertex pointer which gives the cluster each vertex is associated
 * with \param vertex_pos position of each fine vertex \param sample_pos
 * position of each coarse vertex \param operator_value_ptr column index pointer
 * for first prolongation operator \param operator_data_ptr pointer for
 * associated value for a given column in a given row
 */
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
            assert(selectedv1 != selectedv2 &&
                   selectedv2 != selectedv3 & selectedv1 != selectedv3);
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

/**
 * \brief Constructs prolongation operator for a level beyond 1
 * \param row_ptr row pointer for next level csr mesh
 * \param value_ptr column index pointer for next level csr mesh
 * \param operator_value_ptr prolongation operator column index pointer
 * \param operator_data_ptr prolongation operator value pointer
 * \param N
 * \param vData
 */
void createProlongationOperator(int*        row_ptr,
                                int*        value_ptr,
                                int*        operator_value_ptr,
                                float*      operator_data_ptr,
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

            // printf("\n cluster point of %d is %d", number, cluster_point);

            const int start_pointer = row_ptr[cluster_point];
            const int end_pointer   = row_ptr[cluster_point + 1];

            float                 min_distance = 99999;
            Eigen::Vector3<float> selectedv1{0, 0, 0}, selectedv2{0, 0, 0},
                selectedv3{0, 0, 0};
            const Eigen::Vector3<float> q{vData[number].position.x,
                                          vData[number].position.y,
                                          vData[number].position.z};
            int                         selected_neighbor             = 0;
            int                         selected_neighbor_of_neighbor = 0;

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
                            Eigen::Vector3<float> v1{vData[v1_idx].position.x,
                                                     vData[v1_idx].position.y,
                                                     vData[v1_idx].position.z};
                            Eigen::Vector3<float> v2{vData[v2_idx].position.x,
                                                     vData[v2_idx].position.y,
                                                     vData[v2_idx].position.z};
                            Eigen::Vector3<float> v3{vData[v3_idx].position.x,
                                                     vData[v3_idx].position.y,
                                                     vData[v3_idx].position.z};


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
            assert(selectedv1 != selectedv2 && selectedv2 != selectedv3 &&
                   selectedv3 != selectedv1);
            // Compute barycentric coordinates for the closest triangle
            float b1 = 0, b2 = 0, b3 = 0;
            computeBarycentricCoordinates(
                selectedv1, selectedv2, selectedv3, q, b1, b2, b3);


            if (isnan(b1) || isnan(b2) || isnan(b3)) {

                printf("\nNAN found for vertex %d", number);
            }

            // Store results
            int l                         = number;
            operator_value_ptr[l * 3]     = cluster_point;
            operator_value_ptr[l * 3 + 1] = selected_neighbor;
            operator_value_ptr[l * 3 + 2] = selected_neighbor_of_neighbor;
            operator_data_ptr[l * 3]      = b1;
            operator_data_ptr[l * 3 + 1]  = b2;
            operator_data_ptr[l * 3 + 2]  = b3;
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
    thrust::for_each(
        thrust::device,
        samples.begin(),
        samples.end(),
        [=] __device__(int number) {
            int currentCluster = vData[number].cluster;

            for (int i = csr.row_ptr[number]; i < csr.row_ptr[number + 1];
                 i++) {
                int currentNode = csr.value_ptr[i];
                if (vData[currentNode].cluster != currentCluster) {
                    // add neighbors
                    neighbors[currentCluster].addNeighbor(
                        vData[currentNode].cluster);
                    neighbors[currentNode].addNeighbor(currentCluster);
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


/**
 * \brief Determine cluster for a set of vertices at a certain level
 * \param n
 * \param distance
 * \param currentLevel
 * \param vertex_data
 */
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


/**
 * \brief Create prolongation operator CSR for all levels beyond level 1
 * \param N
 * \param numberOfSamples
 * \param numberOfLevels
 * \param ratio
 * \param sample_pos
 * \param csr
 * \param prolongationOperatorCSR
 * \param prolongationOperatorCSRTransposed
 * \param oldVdata
 * \param distanceArray
 * \param vertexCluster
 */
void createProlongationOperators(
    int               N,
    int               numberOfSamples,
    int               numberOfLevels,
    float             ratio,
    Vec3*             sample_pos,
    CSR               csr,
    std::vector<CSR>& prolongationOperatorCSR,
    std::vector<CSR>& prolongationOperatorCSRTransposed,
    VertexData*       oldVdata,
    float*            distanceArray,
    int*              vertexCluster)
{
    CUDA_ERROR(cudaDeviceSynchronize());
    int* flag;
    CUDA_ERROR(cudaMallocManaged(&flag, sizeof(int)));
    *flag = 0;


    CSR lastCSR                 = csr;
    CSR currentCSR              = csr;
    int currentNumberOfVertices = N;
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

        /*std::cout << "\nlevel : " << level;
        std::cout << "\n current number of samples: " << currentNumberOfSamples;
        std::cout << "\n current number of vertices: "
                  << currentNumberOfVertices;*/
        setCluster(currentNumberOfVertices, distanceArray, level + 1, oldVdata);

        do {

            *flag = 0;
            clusterCSR(currentNumberOfVertices,
                       distanceArray,
                       vertexCluster,
                       flag,
                       lastCSR,
                       oldVdata);
            CUDA_ERROR(cudaDeviceSynchronize());
        } while (*flag != 0);


        VertexNeighbors* vertexNeighbors2;
        CUDA_ERROR(cudaMallocManaged(
            &vertexNeighbors2,
            currentNumberOfVertices * sizeof(VertexNeighbors)));

        int* number_of_neighbors2;
        CUDA_ERROR(cudaMallocManaged(&number_of_neighbors2,
                                     currentNumberOfVertices * sizeof(int)));


        numberOfNeighbors(currentNumberOfSamples,
                          vertexNeighbors2,
                          currentNumberOfVertices,
                          lastCSR,
                          oldVdata,
                          number_of_neighbors2);


        CUDA_ERROR(cudaDeviceSynchronize());

        currentCSR = CSR(currentNumberOfSamples,
                         number_of_neighbors2,
                         vertexNeighbors2,
                         currentNumberOfVertices);

        createProlongationOperator(currentCSR.row_ptr,
                                   currentCSR.value_ptr,
                                   a.value_ptr,
                                   a.data_ptr,
                                   currentNumberOfVertices,
                                   oldVdata);
        prolongationOperatorCSR.push_back(a);
        prolongationOperatorCSRTransposed.push_back(transposeCSR(a));

        CUDA_ERROR(cudaDeviceSynchronize());

        lastCSR = currentCSR;
    }


    CUDA_ERROR(cudaFree(flag));
    CUDA_ERROR(cudaDeviceSynchronize());
}
