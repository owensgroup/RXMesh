#pragma once


struct Vec3
{
    float x, y, z;
};


// used from 2nd level onwards
struct VertexData
{
    int     cluster;
    float   distance;
    uint8_t bitmask;
    Vec3    position;
    int     sample_number;
    int     linear_id;
};


struct CSR
{
    int* row_ptr;
    int* value_ptr;
    int* number_of_neighbors;
    float* data_ptr;
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
        cudaMallocManaged(&data_ptr, row_ptr[num_rows] * sizeof(float));

        cudaDeviceSynchronize();

        createValuePointer(
            num_rows, row_ptr, value_ptr, number_of_neighbors, vns, N,data_ptr);
    }


    void createValuePointer(int              numberOfSamples,
                            int*             row_ptr_raw,
                            int*             value_ptr_raw,
                            int*             number_of_neighbors_raw,
                            VertexNeighbors* vns,
                            int              N,
        float* data_pointer)
    {
        thrust::device_vector<int> samples(numberOfSamples);
        thrust::sequence(samples.begin(), samples.end());

        thrust::for_each(
            thrust::device,
            samples.begin(),
            samples.end(),
            [=] __device__(int number) {
                // printf("\n The number %d is used here\n", number);
                // printf("\n The number of samples %d is used
                // here\n",numberOfSamples);

                int* neighbors = new int[number_of_neighbors_raw[number]];
                vns[number].getNeighbors(neighbors);

                const int n = vns[number].getNumberOfNeighbors();

                int start_pointer = row_ptr_raw[number];
                int count            = 0;
                /*
                int j             = 0;
                for (int i = 0; i < n; i++) 
                {
                    value_ptr_raw[start_pointer+j] = neighbors[j];
                    data_ptr[start_pointer+j] = 1;
                    j++;
                }
                */

                 for (int i = 0; i < n; i++) {
                    int key = neighbors[i];  // Value to insert
                    int j   = i - 1;

                    // Shift elements in value_ptr_raw to make space for the key
                    while (j >= 0 && value_ptr_raw[start_pointer + j] > key) {
                        value_ptr_raw[start_pointer + j + 1] =
                            value_ptr_raw[start_pointer + j];
                        j--;
                    }

                    // Insert the key in its sorted position
                    value_ptr_raw[start_pointer + j + 1] = key;
                    data_pointer[start_pointer + count]  = 1;  // default value

                    count++;
                }

                free(neighbors);

                if (count != n) {
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
                printf("vertex %d value: %f\n", value_ptr[q], data_ptr[q]);
            }
        }
    }

    void GetRenderData(
        std::vector<std::array<double, 3>>&
            vertexPositions,  // To store vertex positions
        std::vector<std::vector<size_t>>& faceIndices,  // To store face indices
        Vec3* vertex_pos // Array of vertex positions (assumed 3D)
    )
    {
        // Initialize vertex positions
        vertexPositions.resize(num_rows);
        for (int i = 0; i < num_rows; ++i) {
            vertexPositions[i] = {
                vertex_pos[i].x, vertex_pos[i].y, vertex_pos[i].z};
        }

        // Collect faces from the CSR structure
        faceIndices.clear();
        for (int i = 0; i < num_rows; ++i) {
            int neighbors_count = row_ptr[i + 1] - row_ptr[i];
            if (neighbors_count >= 3) {  // Process only if it can form faces
                for (int j = row_ptr[i]; j < row_ptr[i + 1] - 2; ++j) {
                    // Create a triangular face (i, value_ptr[j],
                    // value_ptr[j+1])


                    int a = i, b = value_ptr[j], c = value_ptr[j + 1];

                    
                     Eigen::Vector3<float> v1{vertex_pos[a].x,
                                             vertex_pos[a].y,
                                             vertex_pos[a].z};
                    Eigen::Vector3<float> v2{
                        vertex_pos[b].x, vertex_pos[b].y, vertex_pos[b].z};
                    Eigen::Vector3<float> v3{
                        vertex_pos[c].x, vertex_pos[c].y, vertex_pos[c].z};
                    
                    
                    Eigen::Vector3f edge1 = v2 - v1;
                    Eigen::Vector3f edge2 = v3 - v1;

                    // Cross product gives us the normal vector
                    Eigen::Vector3f normal = edge1.cross(edge2);

                    // Check the z-component of the normal (assuming
                    // Z-up coordinate system) If normal.z() < 0,
                    // the vertices are in a clockwise order
                    if (normal.z() < 0) {
                        faceIndices.push_back(
                            {
                            static_cast<size_t>(a), static_cast<size_t>(b),
                                static_cast<size_t>(c)
                            });
                    } else
                        faceIndices.push_back({static_cast<size_t>(a),
                                               static_cast<size_t>(c),
                                               static_cast<size_t>(b)});
                            

                    
                }
            }
        }
    }


    ~CSR()
    {
    }
};



void clusterCSR(int    n,
                Vec3*  vertex_pos,
                float* distance,
                int*   clusterVertices,
                int*   flag,
                CSR    csr,
                VertexData* vertex_data)
{
    thrust::device_vector<int> samples(n);
    thrust::sequence(samples.begin(), samples.end());

    thrust::for_each(
        thrust::device,
        samples.begin(),
        samples.end(),
        [=] __device__(int number) {
            // V-V query
            Vec3 ourPos = vertex_data[number].position;

            for (int i = csr.row_ptr[number]; i < csr.row_ptr[number + 1];i++) 
            {
                int current_v = csr.value_ptr[i];

                Vec3 currentPos = vertex_data[current_v].position;
                float dist =
                    sqrtf(powf(ourPos.x - currentPos.x,
                               2) +
                          powf(ourPos.y - currentPos.y,
                               2) +
                                   powf(ourPos.z - currentPos.z,
                               2)) +
                    distance[current_v];

                if (dist < distance[number] && vertex_data[current_v].cluster!=-1)
                {
                    distance[number]        = dist;
                    *flag                   = 15;
                    clusterVertices[number] = clusterVertices[current_v];
                    vertex_data[number].cluster = vertex_data[current_v].cluster;
                }
            }
        });
}