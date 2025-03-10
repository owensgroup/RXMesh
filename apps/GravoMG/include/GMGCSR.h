#pragma once

#include "rxmesh/matrix/sparse_matrix.cuh"

#include "NeighborHandling.h"

#include "GMGRXMeshKernels.h"

namespace rxmesh {

struct CSR
{
    int* row_ptr;
    int* value_ptr;
    int* number_of_neighbors;
    float* data_ptr;
    int  num_rows;
    int    non_zeros;// number of non-zero values
    CSR(){}

    CSR(int a, int b, int c)
    {
        num_rows = 3;
        cudaMallocManaged(&row_ptr, sizeof(int) * (num_rows + 1));
        cudaMallocManaged(&value_ptr, sizeof(int) * num_rows * 3);
        cudaMallocManaged(&data_ptr, sizeof(float) * num_rows * 3);
        non_zeros = 9;

        // Point 1: heavy diagonal influence (6) with small neighbor influence
        // (-1)
        value_ptr[0] = 0;
        data_ptr[0]  = 6;  // diagonal
        value_ptr[1] = 1;
        data_ptr[1]  = -1;  // connection to point 2
        value_ptr[2] = 2;
        data_ptr[2]  = -1;  // connection to point 3

        // Point 2: similar pattern
        value_ptr[3] = 0;
        data_ptr[3]  = -1;  // connection to point 1
        value_ptr[4] = 1;
        data_ptr[4]  = 6;  // diagonal
        value_ptr[5] = 2;
        data_ptr[5]  = -1;  // connection to point 3

        // Point 3: similar pattern
        value_ptr[6] = 0;
        data_ptr[6]  = -1;  // connection to point 1
        value_ptr[7] = 1;
        data_ptr[7]  = -1;  // connection to point 2
        value_ptr[8] = 2;
        data_ptr[8]  = 6;  // diagonal

        row_ptr[0]        = 0;
        row_ptr[1]        = 3;
        row_ptr[2]        = 6;
        row_ptr[num_rows] = non_zeros;
    }
    CSR(int number_of_rows, int num_cols) //to be used only if we're multiplying and this is used as result
    {
        num_rows=number_of_rows;
        cudaMallocManaged(&row_ptr, sizeof(int) * (num_rows + 1));
        cudaMallocManaged(&value_ptr, sizeof(int) * num_rows);
        cudaMallocManaged(&data_ptr, sizeof(float) * num_rows);
        non_zeros = num_rows;

        cudaDeviceSynchronize();
         for (int i = 0; i < num_rows; ++i) 
         {
            row_ptr[i] = i;
         }
         row_ptr[num_rows] = non_zeros;  // Ensure the last element is properly set
         // Initialize values and data
         for (int i = 0; i < non_zeros; ++i) {
             value_ptr[i] = 2;     // Column indices (modify as per structure)
             data_ptr[i]  = i;  // Example values
         }
         

        cudaDeviceSynchronize();

    }

    CSR(int number_of_rows)
    {
        // 3 values per row
        // take a row, look through all values,
        // or just allocate 3 x number of rows x float
        // allocate values in parallel
        num_rows = number_of_rows;
        cudaMallocManaged(&row_ptr, sizeof(int) * (num_rows+1));
        cudaMallocManaged(&value_ptr, sizeof(int) * num_rows * 3);
        cudaMallocManaged(&data_ptr, sizeof(float) * num_rows * 3);
        non_zeros = num_rows * 3;

        cudaDeviceSynchronize();

        for (int i = 0; i < num_rows; ++i) {
            row_ptr[i] = 3*i;
        }
        row_ptr[num_rows] =
            non_zeros;  // Ensure the last element is properly set

        cudaDeviceSynchronize();
    }

    __device__ void setValue(int   row,
                  int   colNumber1,
                  int   colNumber2,
                  int   colNumber3,
                  float value1,
                  float value2,
                  float value3)  // implicitly, this can only be called 3 times
    {
        int q            = row_ptr[row];
        value_ptr[q]     = colNumber1;
        value_ptr[q + 1] = colNumber2;
        value_ptr[q + 2] = colNumber3;
        data_ptr[q]      = value1;
        data_ptr[q + 1]  = value2;
        data_ptr[q + 2]  = value3;
    }

    CSR(SparseMatrix<float> A,
        const int*                A_row_pointer,
        const int*                A_column_pointer,
        int                 number_of_values)
    {
        num_rows = A.rows();
        non_zeros = A.non_zeros();
        cudaMallocManaged(&row_ptr, (num_rows + 1) * sizeof(int));
        cudaMallocManaged(&data_ptr, sizeof(float) * A.non_zeros());
        cudaMallocManaged(&value_ptr, sizeof(int) * A.non_zeros());
        cudaDeviceSynchronize();
          for (int i = 0; i < num_rows; ++i) 
          {
            row_ptr[i] = A_row_pointer[i];
            for (int q = A_row_pointer[i]; q < A_row_pointer[i + 1]; q++) 
            {
                value_ptr[q] =  A_column_pointer[q];
                data_ptr[q] = A(i, A_column_pointer[q]);
            }
          }
          row_ptr[num_rows] =non_zeros;  // Ensure the last element is properly set

          cudaDeviceSynchronize();
    }

    //used for the mesh
    CSR(int n_rows, int* num_of_neighbors, VertexNeighbors* vns, int N)
    {

        num_rows = n_rows;
        cudaMallocManaged(&row_ptr, (num_rows + 1) * sizeof(int));
        cudaDeviceSynchronize();
        // Temporary storage for CUB
        void*  d_cub_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        // Compute the required temporary storage size
        cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                      temp_storage_bytes,
                                      num_of_neighbors,
                                      row_ptr,
                                      num_rows + 1);

        cudaMallocManaged(&d_cub_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_cub_temp_storage,
                                      temp_storage_bytes,
                                      num_of_neighbors,
                                      row_ptr,
                                      num_rows + 1);

        cudaFree(d_cub_temp_storage);
        cudaDeviceSynchronize();

        cudaMallocManaged(&value_ptr, row_ptr[num_rows] * sizeof(int));
        cudaMallocManaged(&data_ptr, row_ptr[num_rows] * sizeof(float));

        non_zeros = row_ptr[num_rows];
        cudaDeviceSynchronize();

        createValuePointer(
            num_rows, row_ptr, value_ptr, num_of_neighbors, vns, N,data_ptr);
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
                int count         = 0;               

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


    void printCSR(bool showOnlyNAN = false)
    {
        printf("\nCSR Array:");
        std::cout << "\nNumber of non-zeros = " << non_zeros << "\n";
        for (int i = 0; i < num_rows; ++i) {
            if (!showOnlyNAN) {
                printf("row_ptr[%d] = %d\n", i, row_ptr[i]);
                printf("add %d values\n", row_ptr[i + 1] - row_ptr[i]);
            }
            for (int q = row_ptr[i]; q < row_ptr[i + 1]; q++) {
                if (/*!data_ptr[q] ||*/ std::isnan(data_ptr[q]) &&
                    showOnlyNAN) {
                    printf("row_ptr[%d] = %d\n", i, row_ptr[i]);
                    printf("vertex %d value: %f\n", value_ptr[q], data_ptr[q]);
                } else if (!showOnlyNAN)
                    printf("vertex %d value: %f\n", value_ptr[q], data_ptr[q]);
            }
        }
    }

    void rearrange(int* row_ptr_raw,
                   int* value_ptr_raw,
                   int  num_rows,
                   int  non_zeros)
    {
        thrust::device_vector<int> temp_values(value_ptr_raw,
                                               value_ptr_raw + non_zeros);
        int* temp_ptr = thrust::raw_pointer_cast(temp_values.data());

        thrust::device_vector<int> samples(num_rows);
        thrust::sequence(samples.begin(), samples.end());

        // Limit how far we look ahead to avoid timeout
        const int MAX_LOOKAHEAD = 4;

        thrust::for_each(
            thrust::device,
            samples.begin(),
            samples.end(),
            [=] __device__(int row) {
                int start_pointer = row_ptr_raw[row];
                int end_pointer   = row_ptr_raw[row + 1];
                int num_neighbors = end_pointer - start_pointer;

                if (num_neighbors < 2)
                    return;

                // Single pass through the row
                for (int i = start_pointer; i < end_pointer - 1; i++) {
                    int current = temp_ptr[i];
                    int next    = temp_ptr[i + 1];

                    // Quick check if current and next are already connected
                    bool already_connected = false;
                    int  current_start     = row_ptr_raw[current];
                    int  current_end       = row_ptr_raw[current + 1];

                    for (int k = current_start; k < current_end; k++) {
                        if (value_ptr_raw[k] == next) {
                            already_connected = true;
                            break;
                        }
                    }

                    if (!already_connected) {
                        // Look only a few positions ahead
                        int max_look = min(i + 1 + MAX_LOOKAHEAD, end_pointer);

                        for (int j = i + 2; j < max_look; j++) {
                            int candidate = temp_ptr[j];

                            // Check if current and candidate are connected
                            for (int k = current_start; k < current_end; k++) {
                                if (value_ptr_raw[k] == candidate) {
                                    thrust::swap(temp_ptr[i + 1], temp_ptr[j]);
                                    goto next_vertex;
                                }
                            }
                        }
                    }
                next_vertex:
                    continue;
                }
            });

        // Copy back the rearranged values
        thrust::copy(temp_values.begin(), temp_values.end(), value_ptr_raw);
    }


    void GetRenderData(
        std::vector<std::array<double, 3>>&
            vertexPositions,  // To store vertex positions
        std::vector<std::vector<size_t>>& faceIndices,  // To store face indices
        Vec3* vertex_pos  // Array of vertex positions (assumed 3D)
    )
    {
        rearrange(row_ptr, value_ptr, num_rows, non_zeros);

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
            if (neighbors_count >= 2) {  // Process only if it can form faces
                for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                    // Create a triangular face
                    int a, b, c;
                    if (j == row_ptr[i + 1] - 1) {
                        a = i;
                        b = value_ptr[j];
                        c = value_ptr[row_ptr[i]];
                    } else {
                        a = i;
                        b = value_ptr[j];
                        c = value_ptr[j + 1];
                    }
                    // check if b,c are neighbors
                    int n1_start = row_ptr[b];
                    int n1_end   = row_ptr[b + 1];
                    for (int k = n1_start; k < n1_end; k++) {
                        if (c == value_ptr[k]) {
                            // std::cout << "\n Triangle formed with " << a
                            //           << " and " << b << " and " << c;

                            faceIndices.push_back({static_cast<size_t>(a),
                                                   static_cast<size_t>(b),
                                                   static_cast<size_t>(c)});
                            break;
                        }
                    }
                }
            }
        }
    }


    void render(Vec3* vertex_pos)
    {
        std::vector<std::array<double, 3>>
            vertexPositions;  // To store vertex positions
        std::vector<std::vector<size_t>> faceIndices;  // To store face indices

        GetRenderData(vertexPositions, faceIndices, vertex_pos);
        polyscope::registerSurfaceMesh(
            "mesh level 1", vertexPositions, faceIndices);
    }

    ~CSR()
    {
    }
};


void clusterCSR(int         n,
                float*      distance,
                int*        clusterVertices,
                int*        flag,
                CSR         csr,
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

            for (int i = csr.row_ptr[number]; i < csr.row_ptr[number + 1];
                 i++) {
                int current_v = csr.value_ptr[i];

                Vec3  currentPos = vertex_data[current_v].position;
                float dist       = sqrtf(powf(ourPos.x - currentPos.x, 2) +
                                   powf(ourPos.y - currentPos.y, 2) +
                                   powf(ourPos.z - currentPos.z, 2)) +
                             distance[current_v];

                if (dist < distance[number] &&
                    vertex_data[current_v].cluster != -1) {
                    distance[number]        = dist;
                    *flag                   = 15;
                    clusterVertices[number] = clusterVertices[current_v];
                    vertex_data[number].cluster =
                        vertex_data[current_v].cluster;
                }
            }
        });
}

CSR transposeCSR(const CSR& input)
{
    // Create handle for cuSPARSE operations
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Initialize result CSR
    CSR result;
    result.num_rows = input.non_zeros > 0 ?
                          *std::max_element(input.value_ptr,
                                            input.value_ptr + input.non_zeros) +
                              1 :
                          0;

    // If input is empty, return an empty result
    if (input.non_zeros == 0) {
        result.non_zeros = 0;
        cudaMallocManaged(&result.row_ptr, sizeof(int) * (result.num_rows + 1));
        cudaMallocManaged(&result.value_ptr, sizeof(int));
        cudaMallocManaged(&result.data_ptr, sizeof(float));
        cudaDeviceSynchronize();

        for (int i = 0; i <= result.num_rows; i++) {
            result.row_ptr[i] = 0;
        }

        return result;
    }

    // Allocate memory for result
    cudaMallocManaged(&result.row_ptr, sizeof(int) * (result.num_rows + 1));
    cudaMallocManaged(&result.value_ptr, sizeof(int) * input.non_zeros);
    cudaMallocManaged(&result.data_ptr, sizeof(float) * input.non_zeros);
    result.non_zeros = input.non_zeros;

    // Temporary storage for cuSPARSE
    int*   csr_row_ptr_B;
    int*   csr_col_ind_B;
    float* csr_val_B;

    cudaMalloc(&csr_row_ptr_B, sizeof(int) * (result.num_rows + 1));
    cudaMalloc(&csr_col_ind_B, sizeof(int) * input.non_zeros);
    cudaMalloc(&csr_val_B, sizeof(float) * input.non_zeros);

    // Prepare transpose operation
    size_t buffer_size = 0;
    void*  buffer      = nullptr;

    // Step 1: Get buffer size
    cusparseCsr2cscEx2_bufferSize(
        handle,
        input.num_rows,
        result.num_rows,
        input.non_zeros,
        input.data_ptr,
        input.row_ptr,
        input.value_ptr,
        csr_val_B,
        csr_row_ptr_B,
        csr_col_ind_B,
        CUDA_R_32F,                // Data type for values
        CUSPARSE_ACTION_NUMERIC,   // Copy values, not just structure
        CUSPARSE_INDEX_BASE_ZERO,  // 0-based indexing
        CUSPARSE_CSR2CSC_ALG1,     // Algorithm selection
        &buffer_size);

    // Allocate temporary buffer
    cudaMalloc(&buffer, buffer_size);

    // Step 2: Perform the transpose operation (CSR to CSC is equivalent to
    // transpose)
    cusparseCsr2cscEx2(handle,
                       input.num_rows,
                       result.num_rows,
                       input.non_zeros,
                       input.data_ptr,
                       input.row_ptr,
                       input.value_ptr,
                       csr_val_B,
                       csr_row_ptr_B,
                       csr_col_ind_B,
                       CUDA_R_32F,
                       CUSPARSE_ACTION_NUMERIC,
                       CUSPARSE_INDEX_BASE_ZERO,
                       CUSPARSE_CSR2CSC_ALG1,
                       buffer);

    // Copy results from device to managed memory
    cudaMemcpy(result.row_ptr,
               csr_row_ptr_B,
               sizeof(int) * (result.num_rows + 1),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(result.value_ptr,
               csr_col_ind_B,
               sizeof(int) * input.non_zeros,
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(result.data_ptr,
               csr_val_B,
               sizeof(float) * input.non_zeros,
               cudaMemcpyDeviceToDevice);

    // Synchronize to ensure all operations are complete
    cudaDeviceSynchronize();

    // Free temporary resources
    cudaFree(buffer);
    cudaFree(csr_row_ptr_B);
    cudaFree(csr_col_ind_B);
    cudaFree(csr_val_B);
    cusparseDestroy(handle);

    return result;
}


__global__ void csrMultiplyKernel(int*   A_row_ptr,
                                  int*   A_col_idx,
                                  float* A_vals,
                                  int    A_rows,
                                  int*   B_row_ptr,
                                  int*   B_col_idx,
                                  float* B_vals,
                                  int    B_cols,
                                  int*   C_row_ptr,
                                  int*   C_col_idx,
                                  float* C_vals)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows)
        return;

    int start_A = A_row_ptr[row];
    int end_A   = A_row_ptr[row + 1];

    printf("\n entered kernel further for row %d from %d to %d",
           row,
           start_A,
           end_A);


    for (int i = start_A; i < end_A; i++) {
        int   col_A = A_col_idx[i];
        float val_A = A_vals[i];

        int start_B = B_row_ptr[col_A];
        int end_B   = B_row_ptr[col_A + 1];


        for (int j = start_B; j < end_B; j++) {
            int   col_B = B_col_idx[j];
            float val_B = B_vals[j];

            ::atomicAdd(&C_vals[C_row_ptr[row] + col_B], val_A * val_B);
            printf("\n New value of C at %d is %f",
                   C_row_ptr[row] + col_B,
                   C_vals[C_row_ptr[row] + col_B]);
        }
    }
}

CSR csrMultiply(CSR& A, CSR& B)
{
    int *  d_A_row_ptr, *d_A_col_idx, *d_B_row_ptr, *d_B_col_idx;
    float *d_A_vals, *d_B_vals;


    cudaMallocManaged(&d_A_row_ptr, (A.num_rows + 1) * sizeof(int));
    cudaMallocManaged(&d_A_col_idx, A.non_zeros * sizeof(int));
    cudaMallocManaged(&d_A_vals, A.non_zeros * sizeof(float));
    cudaMallocManaged(&d_B_row_ptr, (B.num_rows + 1) * sizeof(int));
    cudaMallocManaged(&d_B_col_idx, B.non_zeros * sizeof(int));
    cudaMallocManaged(&d_B_vals, B.non_zeros * sizeof(float));


    cudaMemcpy(d_A_row_ptr,
               A.row_ptr,
               A.num_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_col_idx,
               A.value_ptr,
               A.non_zeros * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_vals,
               A.data_ptr,
               A.non_zeros * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_row_ptr,
               B.row_ptr,
               B.num_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_col_idx,
               B.value_ptr,
               B.non_zeros * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_vals,
               B.data_ptr,
               B.non_zeros * sizeof(float),
               cudaMemcpyHostToDevice);

    CSR    C;
    int *  d_C_row_ptr = C.row_ptr, *d_C_col_idx = C.value_ptr;
    float* d_C_vals = C.data_ptr;

    C.num_rows = (A.num_rows + 1);


    cudaMallocManaged(&d_C_row_ptr, (A.num_rows + 1) * sizeof(int));
    cudaMallocManaged(&d_C_col_idx, A.num_rows * B.num_rows * sizeof(int));
    cudaMallocManaged(&d_C_vals, A.num_rows * B.num_rows * sizeof(float));
    cudaDeviceSynchronize();

    // std::cout << "\n Number of rows in A: " << A.num_rows;
    // A.printCSR();
    /* std::cout << "\nNumber of rows in B: " << B.num_rows;
    B.printCSR();*/

    dim3 blockDim(256);
    dim3 gridDim((A.num_rows + blockDim.x - 1) / blockDim.x);


    csrMultiplyKernel<<<gridDim, blockDim>>>(d_A_row_ptr,
                                             d_A_col_idx,
                                             d_A_vals,
                                             A.num_rows,
                                             d_B_row_ptr,
                                             d_B_col_idx,
                                             d_B_vals,
                                             B.num_rows,
                                             d_C_row_ptr,
                                             d_C_col_idx,
                                             d_C_vals);

    cudaDeviceSynchronize();
    C.row_ptr   = d_C_row_ptr;
    C.value_ptr = d_C_col_idx;
    C.data_ptr  = d_C_vals;

    // std::cout << "First row: " << C.row_ptr[0];
    /*
    printf("\nCSR Array: \n");
    for (int i = 0; i < C.num_rows; ++i) {
        printf("row_ptr[%d] = %d\n", i, d_C_row_ptr[i]);
        printf("add %d values\n", d_C_row_ptr[i + 1] - d_C_row_ptr[i]);
        for (int q = d_C_row_ptr[i]; q < d_C_row_ptr[i + 1]; q++) {
            printf("vertex %d value: %f\n", d_C_col_idx[q], d_C_vals[q]);
        }
    }
    */
    // C.printCSR();


    /*
    cudaMemcpy(C.row_ptr,
               d_C_row_ptr,
               (A.num_rows + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(C.value_ptr,
               d_C_col_idx,
               A.num_rows * B.num_rows * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(C.data_ptr,
               d_C_vals,
               A.num_rows * B.num_rows * sizeof(float),
               cudaMemcpyDeviceToHost);
               */

    cudaFree(d_A_row_ptr);
    cudaFree(d_A_col_idx);
    cudaFree(d_A_vals);
    cudaFree(d_B_row_ptr);
    cudaFree(d_B_col_idx);
    cudaFree(d_B_vals);

    return C;
}


CSR multiplyCSR(int    A_rows,
                int    A_cols,
                int    B_cols,
                int*   d_A_rowPtr,
                int*   d_A_colIdx,
                float* d_A_values,
                int    nnzA,
                int*   d_B_rowPtr,
                int*   d_B_colIdx,
                float* d_B_values,
                int    nnzB,
                int    transpose = 0)
{
    // Create cuSPARSE handle
    cusparseHandle_t handle;
    CUSPARSE_ERROR(cusparseCreate(&handle));

    // Create sparse matrix descriptors
    cusparseSpMatDescr_t matA, matB, matC;
    CUSPARSE_ERROR(cusparseCreateCsr(&matA,
                                     A_rows,
                                     A_cols,
                                     nnzA,
                                     d_A_rowPtr,
                                     d_A_colIdx,
                                     d_A_values,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));

    CUSPARSE_ERROR(cusparseCreateCsr(&matB,
                                     A_cols,
                                     B_cols,
                                     nnzB,
                                     d_B_rowPtr,
                                     d_B_colIdx,
                                     d_B_values,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));

    // Create an empty descriptor for matC
    CUSPARSE_ERROR(cusparseCreateCsr(&matC,
                                     A_rows,
                                     B_cols,
                                     0,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));

    // Allocate workspace buffer for SpGEMM
    float                 alpha = 1.0f, beta = 0.0f;
    cusparseSpGEMMDescr_t spgemmDesc;
    CUSPARSE_ERROR(cusparseSpGEMM_createDescr(&spgemmDesc));

    // PHASE 1: Work estimation
    size_t bufferSize1 = 0;
    void*  dBuffer1    = nullptr;
    // MAKE THIS DO THE TRANSPOSE, DONT TRANSPOSE EXPLICITLY

    auto operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
    if (transpose == 1)
        operation = CUSPARSE_OPERATION_TRANSPOSE;
    ;
    CUSPARSE_ERROR(
        cusparseSpGEMM_workEstimation(handle,
                                      operation,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      matA,
                                      matB,
                                      &beta,
                                      matC,
                                      CUDA_R_32F,
                                      CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc,
                                      &bufferSize1,
                                      nullptr));
    CUDA_ERROR(cudaMalloc(&dBuffer1, bufferSize1));

    // Execute work estimation
    CUSPARSE_ERROR(
        cusparseSpGEMM_workEstimation(handle,
                                      operation,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      matA,
                                      matB,
                                      &beta,
                                      matC,
                                      CUDA_R_32F,
                                      CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc,
                                      &bufferSize1,
                                      dBuffer1));

    // PHASE 2: Compute non-zero pattern of C
    size_t bufferSize2 = 0;
    void*  dBuffer2    = nullptr;
    CUSPARSE_ERROR(cusparseSpGEMM_compute(handle,
                                          operation,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          matA,
                                          matB,
                                          &beta,
                                          matC,
                                          CUDA_R_32F,
                                          CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc,
                                          &bufferSize2,
                                          nullptr));
    CUDA_ERROR(cudaMalloc(&dBuffer2, bufferSize2));

    // Execute non-zero pattern computation
    CUSPARSE_ERROR(cusparseSpGEMM_compute(handle,
                                          operation,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          matA,
                                          matB,
                                          &beta,
                                          matC,
                                          CUDA_R_32F,
                                          CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc,
                                          &bufferSize2,
                                          dBuffer2));

    // Get the size of matrix C
    int64_t C_rows, C_cols, nnzC;
    CUSPARSE_ERROR(cusparseSpMatGetSize(matC, &C_rows, &C_cols, &nnzC));

    /*
    std::cout << "Matrix C dimensions:" << std::endl;
    std::cout << "Rows: " << C_rows << std::endl;
    std::cout << "Cols: " << C_cols << std::endl;
    std::cout << "Non-zero elements: " << nnzC << std::endl;
    */
    // Allocate memory for matrix C
    int*   d_C_rowPtr;
    int*   d_C_colIdx;
    float* d_C_values;
    CUDA_ERROR(cudaMalloc(&d_C_rowPtr, (A_rows + 1) * sizeof(int)));
    CUDA_ERROR(cudaMalloc(&d_C_colIdx, nnzC * sizeof(int)));
    CUDA_ERROR(cudaMalloc(&d_C_values, nnzC * sizeof(float)));

    // Set pointers for matrix C
    CUSPARSE_ERROR(
        cusparseCsrSetPointers(matC, d_C_rowPtr, d_C_colIdx, d_C_values));

    // PHASE 3: Compute actual values
    CUSPARSE_ERROR(cusparseSpGEMM_copy(handle,
                                       operation,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha,
                                       matA,
                                       matB,
                                       &beta,
                                       matC,
                                       CUDA_R_32F,
                                       CUSPARSE_SPGEMM_DEFAULT,
                                       spgemmDesc));

    // Synchronize to ensure completion
    CUDA_ERROR(cudaDeviceSynchronize());

    // First, copy the computed data to host to filter zeros
    int*   h_C_rowPtr = new int[A_rows + 1];
    int*   h_C_colIdx = new int[nnzC];
    float* h_C_values = new float[nnzC];

    CUDA_ERROR(cudaMemcpy(h_C_rowPtr,
                          d_C_rowPtr,
                          (A_rows + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(
        h_C_colIdx, d_C_colIdx, nnzC * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(
        h_C_values, d_C_values, nnzC * sizeof(float), cudaMemcpyDeviceToHost));

    // Count actual non-zeros and create filtered arrays
    const float        ZERO_THRESHOLD = 1e-6f;
    std::vector<int>   filtered_rowPtr(A_rows + 1, 0);
    std::vector<int>   filtered_colIdx;
    std::vector<float> filtered_values;
    filtered_colIdx.reserve(nnzC);
    filtered_values.reserve(nnzC);

    // Process first row pointer
    filtered_rowPtr[0] = 0;

    // Filter out zeros and build new CSR structure
    int actual_nnz = 0;
    for (int i = 0; i < A_rows; i++) {
        int row_start = h_C_rowPtr[i];
        int row_end   = h_C_rowPtr[i + 1];

        for (int j = row_start; j < row_end; j++) {
            if (std::abs(h_C_values[j]) > ZERO_THRESHOLD ||
                h_C_values[j] != 0.0f) {
                filtered_colIdx.push_back(h_C_colIdx[j]);
                filtered_values.push_back(h_C_values[j]);
                actual_nnz++;
            }
        }
        filtered_rowPtr[i + 1] = actual_nnz;
    }

    // Create new CSR object
    CSR result;
    result.num_rows = A_rows;

    // Allocate new memory with correct sizes
    result.non_zeros = actual_nnz;
    cudaMallocManaged(&result.row_ptr, (A_rows + 1) * sizeof(int));
    cudaMallocManaged(&result.value_ptr, actual_nnz * sizeof(int));
    cudaMallocManaged(&result.data_ptr, actual_nnz * sizeof(float));

    // Copy filtered data to result CSR
    cudaMemcpy(result.row_ptr,
               filtered_rowPtr.data(),
               (A_rows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(result.value_ptr,
               filtered_colIdx.data(),
               actual_nnz * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(result.data_ptr,
               filtered_values.data(),
               actual_nnz * sizeof(float),
               cudaMemcpyHostToDevice);


    // Print the results
    /*
    std::cout << "\nMatrix C in CSR format:" << std::endl;
    std::cout << "Row Pointers: ";
    for (int i = 0; i <= A_rows; i++) {
        std::cout << h_C_rowPtr[i] << " ";
    }
    std::cout << "\n\nColumn Indices: ";
    for (int64_t i = 0; i < nnzC; i++) {
        std::cout << h_C_colIdx[i] << " ";
    }
    std::cout << "\n\nValues: ";
    for (int64_t i = 0; i < nnzC; i++) {
        std::cout << h_C_values[i] << " ";
    }
    std::cout << std::endl;


    // Print in matrix form for better visualization
    std::cout << "\nMatrix C in readable format:" << std::endl;
    for (int i = 0; i < A_rows; i++) {
        int row_start = h_C_rowPtr[i];
        int row_end   = h_C_rowPtr[i + 1];

        // Print each row
        for (int j = 0; j < B_cols; j++) {
            bool found = false;
            // Look for column j in current row
            for (int k = row_start; k < row_end; k++) {
                if (h_C_colIdx[k] == j) {
                    std::cout << std::setw(8) << std::fixed
                              << std::setprecision(2) << h_C_values[k] << " ";
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << std::setw(8) << "0.00"
                          << " ";
            }
        }
        std::cout << std::endl;
    }*/


    // Cleanup host memory
    delete[] h_C_rowPtr;
    delete[] h_C_colIdx;
    delete[] h_C_values;

    // Cleanup device memory
    CUSPARSE_ERROR(cusparseDestroySpMat(matA));
    CUSPARSE_ERROR(cusparseDestroySpMat(matB));
    CUSPARSE_ERROR(cusparseDestroySpMat(matC));
    CUSPARSE_ERROR(cusparseSpGEMM_destroyDescr(spgemmDesc));
    CUSPARSE_ERROR(cusparseDestroy(handle));

    CUDA_ERROR(cudaFree(dBuffer1));
    CUDA_ERROR(cudaFree(dBuffer2));
    CUDA_ERROR(cudaFree(d_C_rowPtr));
    CUDA_ERROR(cudaFree(d_C_colIdx));
    CUDA_ERROR(cudaFree(d_C_values));


    return result;
}

}  // namespace rxmesh