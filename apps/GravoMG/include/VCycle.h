#pragma once


//set up equations
// lhs, x, rhs
// set up the v cycle to solve it

#include "include/GMGCSR.h"

struct VectorCSR
{
    float* vector;
    int    n;
    VectorCSR(int number_of_elements)
    {
        n = number_of_elements;
        cudaMallocManaged(&vector, sizeof(float) * n);
    }
};
struct VectorCSR3D
{
    float* vector;
    int    n;  // number of 3D points (actual vector length is n*3)
    VectorCSR3D(){}
    VectorCSR3D(int number_of_elements)
    {
        n = number_of_elements;
        cudaMallocManaged(
            &vector, sizeof(float) * n * 3); 
    }

    ~VectorCSR3D()
    {
        cudaFree(vector);
    }
};

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/transform.h>



struct GaussJacobiUpdate
{
    const int*   row_ptr;
    const int*   value_ptr;
    const float* data_ptr;
    const float* b;
    const float* x_old;
    float*       x_new;

    GaussJacobiUpdate(const int*   row_ptr,
                      const int*   value_ptr,
                      const float* data_ptr,
                      const float* b,
                      const float* x_old,
                      float*       x_new)
        : row_ptr(row_ptr),
          value_ptr(value_ptr),
          data_ptr(data_ptr),
          b(b),
          x_old(x_old),
          x_new(x_new)
    {
    }

    __device__ void operator()(int i)
    {
        float sum  = 0.0f;
        float diag = 1.0f;

        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            int   col = value_ptr[j];  
            float val = data_ptr[j];   

            if (col == i) {
                diag = val; 
            } else {
                sum += val * x_old[col]; 
            }
        }
        x_new[i] = (b[i] - sum) / diag;
    }
};

// Gauss-Jacobi solver function
void gauss_jacobi_CSR(const CSR&                    A,
                      float* vec_x,
                      float* vec_b,
                      int                           max_iter)
{

    int                          N = A.num_rows;
    thrust::device_ptr<float>    x_ptr(vec_x);
    thrust::device_vector<float> x(x_ptr, x_ptr + N);
    thrust::device_ptr<float>    b_ptr(vec_b);
    thrust::device_vector<float> b(b_ptr, b_ptr + N);


    int                          n = A.num_rows;
    thrust::device_vector<float> x_new(n, 0.0f);

    for (int iter = 0; iter < max_iter; ++iter) {
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(n),
            GaussJacobiUpdate(A.row_ptr,
                              A.value_ptr,
                              A.data_ptr,
                              thrust::raw_pointer_cast(b.data()),
                              thrust::raw_pointer_cast(x.data()),
                              thrust::raw_pointer_cast(x_new.data())));

        // Swap x and x_new for the next iteration
        thrust::swap(x, x_new);
    }
}

//kernel for multiplication
// y = Av
// where A is represented via row_ptr, col_idx and values (pointers)
__global__ void csr_spmv(const int*   row_ptr,
                         const int*   col_idx,
                         const float* values,
                         const float* v, 
                         float*       y,
                         int          m)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m)
        return; 

    float sum       = 0.0f;
    int   row_start = row_ptr[row];
    int   row_end   = row_ptr[row + 1];

    for (int j = row_start; j < row_end; j++) {
        sum += values[j] * v[col_idx[j]];
    }
    y[row] = sum;
}

void SpMV_CSR(const int*   row_ptr,
              const int*   col_idx,
              const float* values,
              const float* v,
              float*       y,
              int          m)
{

    int block_size = 256;
    int grid_size  = (m + block_size - 1) / block_size;  // Ensure full coverage

    csr_spmv<<<grid_size, block_size>>>(row_ptr, col_idx, values, v, y, m);

    cudaDeviceSynchronize();
}

__global__ void vec_subtract(const float* b, const float* Av, float* R, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m)
        return;
    R[i] = b[i] - Av[i];
}

void Compute_R(const int*   row_ptr,
               const int*   col_idx,
               const float* values,
               const float* v,
               const float* b,
               float*       R,
               int          m)
{
    float* Av;
    cudaMalloc(&Av, m * sizeof(float));

    int block_size = 256;
    int grid_size  = (m + block_size - 1) / block_size;

    // Compute Av
    csr_spmv<<<grid_size, block_size>>>(row_ptr, col_idx, values, v, Av, m);

    //R = b - Av
    vec_subtract<<<grid_size, block_size>>>(b, Av, R, m);
    cudaFree(Av);
}
__global__ void csr_spmv_3d(const int*   row_ptr,
                            const int*   col_idx,
                            const float* values,
                            const float* v,
                            float*       y,
                            int          m)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m)
        return;

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;

    int row_start = row_ptr[row];
    int row_end   = row_ptr[row + 1];

    for (int j = row_start; j < row_end; j++) {
        int   col = col_idx[j];
        float val = values[j];

        sum_x += val * v[col * 3];
        sum_y += val * v[col * 3 + 1];
        sum_z += val * v[col * 3 + 2];
    }

    y[row * 3]     = sum_x;
    y[row * 3 + 1] = sum_y;
    y[row * 3 + 2] = sum_z;
}

// SpMV wrapper
void SpMV_CSR_3D(const int*   row_ptr,
                 const int*   col_idx,
                 const float* values,
                 const float* v,
                 float*       y,
                 int          m)
{
    int block_size = 256;
    int grid_size  = (m + block_size - 1) / block_size;
    csr_spmv_3d<<<grid_size, block_size>>>(row_ptr, col_idx, values, v, y, m);
    cudaDeviceSynchronize();
}

__global__ void vec_subtract_3d(const float* b,
                                const float* Av,
                                float*       R,
                                int          m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m)
        return;

    int idx    = i * 3;
    R[idx]     = b[idx] - Av[idx];
    R[idx + 1] = b[idx + 1] - Av[idx + 1];
    R[idx + 2] = b[idx + 2] - Av[idx + 2];
}

void Compute_R_3D(const CSR& A, const float* v, const float* b, float* R, int m)
{
    float* Av;
    cudaMalloc(&Av, m * 3 * sizeof(float));

    SpMV_CSR_3D(A.row_ptr, A.value_ptr, A.data_ptr, v, Av, m);

    int block_size = 256;
    int grid_size  = (m + block_size - 1) / block_size;
    vec_subtract_3d<<<grid_size, block_size>>>(b, Av, R, m);

    cudaFree(Av);
    cudaDeviceSynchronize();
}

struct GaussJacobiUpdate3D
{
    const int*   row_ptr;
    const int*   value_ptr;
    const float* data_ptr;
    const float* b;
    const float* x_old;
    float*       x_new;

    GaussJacobiUpdate3D(const int*   row_ptr,
                        const int*   value_ptr,
                        const float* data_ptr,
                        const float* b,
                        const float* x_old,
                        float*       x_new)
        : row_ptr(row_ptr),
          value_ptr(value_ptr),
          data_ptr(data_ptr),
          b(b),
          x_old(x_old),
          x_new(x_new)
    {
    }

    __device__ void operator()(int i)
    {
        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;
        float diag  = 1.0f;

        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            int   col = value_ptr[j];
            float val = data_ptr[j];

            if (col == i) {
                diag = val;
            } else {
                sum_x += val * x_old[col * 3];
                sum_y += val * x_old[col * 3 + 1];
                sum_z += val * x_old[col * 3 + 2];
            }
        }

        x_new[i * 3]     = (b[i * 3] - sum_x) / diag;
        x_new[i * 3 + 1] = (b[i * 3 + 1] - sum_y) / diag;
        x_new[i * 3 + 2] = (b[i * 3 + 2] - sum_z) / diag;
    }
};

void gauss_jacobi_CSR_3D(const CSR& A, float* vec_x, float* vec_b, int max_iter)
{
    int N = A.num_rows;
    thrust::device_ptr<float> x(vec_x);
    thrust::device_ptr<float> b(vec_b);
    float* x_new_raw;
    cudaMalloc(&x_new_raw, N * 3 * sizeof(float));
    thrust::device_ptr<float> x_new(x_new_raw);

    for (int iter = 0; iter < max_iter; ++iter) {
        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(N),
                         GaussJacobiUpdate3D(A.row_ptr,
                                             A.value_ptr,
                                             A.data_ptr,
                                             thrust::raw_pointer_cast(b),
                                             thrust::raw_pointer_cast(x),
                                             thrust::raw_pointer_cast(x_new)));

        // Swap the raw pointers
        float* temp = thrust::raw_pointer_cast(x);
        x     = thrust::device_ptr<float>(thrust::raw_pointer_cast(x_new));
        x_new = thrust::device_ptr<float>(temp);
    }

    if (thrust::raw_pointer_cast(x) != vec_x) {
        cudaMemcpy(vec_x,
                   thrust::raw_pointer_cast(x),
                   N * 3 * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }

    cudaFree(x_new_raw);
    cudaDeviceSynchronize();
}

class GMGVCycle
{
   public:
    int              pre_relax_iterations  = 2;
    int              post_relax_iterations = 2;
    int              max_number_of_levels;
    std::vector<int> numberOfSamplesPerLevel;

    std::vector<CSR>       prolongationOperators;
    std::vector<CSR>       LHS;
    VectorCSR3D RHS;
    VectorCSR3D                 X;  // final solution

    void VCycle(CSR& A, VectorCSR3D& f, VectorCSR3D& v, int currentLevel)
    {
        // Pre-smoothing
        gauss_jacobi_CSR_3D(A, v.vector, f.vector, pre_relax_iterations);

        // R = f - Av
        VectorCSR3D R(A.num_rows);
        Compute_R_3D(A, v.vector, f.vector, R.vector, A.num_rows);

        // Restrict the residual 
        VectorCSR3D restricted_residual(
            prolongationOperators[currentLevel].num_rows / 8);
        CSR transposeProlongation(
            transposeCSR(prolongationOperators[currentLevel],
                         prolongationOperators[currentLevel].num_rows / 8));

        SpMV_CSR_3D(transposeProlongation.row_ptr,
                    transposeProlongation.value_ptr,
                    transposeProlongation.data_ptr,
                    R.vector,
                    restricted_residual.vector,
                    transposeProlongation.num_rows);

        //correction vector 
        VectorCSR3D coarse_correction(
            prolongationOperators[currentLevel].num_rows / 8);

        if (currentLevel < max_number_of_levels - 1) {
            // next level
            VCycle(LHS[currentLevel + 1],
                   restricted_residual,
                   coarse_correction,
                   currentLevel + 1);
        } else {
            // Direct solve 
            gauss_jacobi_CSR_3D(
                A, coarse_correction.vector, restricted_residual.vector, 50);
        }

        // Prolongate
        VectorCSR3D fine_correction(A.num_rows);
        SpMV_CSR_3D(prolongationOperators[currentLevel].row_ptr,
                    prolongationOperators[currentLevel].value_ptr,
                    prolongationOperators[currentLevel].data_ptr,
                    coarse_correction.vector,
                    fine_correction.vector,
                    A.num_rows);

        // Add correction 
        for (int i = 0; i < A.num_rows; i++) {
            v.vector[i] += fine_correction.vector[i];
        }

        // Post-smoothing
        gauss_jacobi_CSR_3D(A, v.vector, f.vector, post_relax_iterations);
    }

    GMGVCycle(){}
    GMGVCycle(int initialNumberOfRows) : X(initialNumberOfRows)
    {

    }

    ~GMGVCycle(){}
};