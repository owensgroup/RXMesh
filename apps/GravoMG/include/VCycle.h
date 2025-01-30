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
                diag = val;  //diagonal value
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
        return;  // Bounds check

    float sum       = 0.0f;
    int   row_start = row_ptr[row];
    int   row_end   = row_ptr[row + 1];

    // Compute dot product for this row
    for (int j = row_start; j < row_end; j++) {
        sum += values[j] * v[col_idx[j]];
    }

    // Store result
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

    // Allocate GPU memory for intermediate Av
    float* Av;
    cudaMalloc(&Av, m * sizeof(float));

    int block_size = 256;
    int grid_size  = (m + block_size - 1) / block_size;

    // Step 1: Compute Av
    csr_spmv<<<grid_size, block_size>>>(row_ptr, col_idx, values, v, Av, m);

    // Step 2: Compute R = b - Av
    vec_subtract<<<grid_size, block_size>>>(b, Av, R, m);

    // Free temporary memory
    cudaFree(Av);
}



class GMGVCycle
{
   public:
    int              pre_relax_iterations  = 2;
    int              post_relax_iterations = 2;
    int              number_of_levels;
    int              max_number_of_levels;
    std::vector<int> numberOfSamplesPerLevel;

    std::vector<CSR> prolongationOperators;
    std::vector<CSR> LHS;
    float* RHS;
    float* X;

    void VCycle(CSR A, VectorCSR f, VectorCSR v, int currentLevel)
    {
        // pre-relaxation
        gauss_jacobi_CSR(A, v.vector, f.vector, pre_relax_iterations);

        // find residual

        VectorCSR R(A.num_rows);
        Compute_R(A.row_ptr,
                  A.value_ptr,
                  A.data_ptr,
                  v.vector,
                  f.vector,
                  R.vector,
                  A.num_rows);

        // coarsen residual

        VectorCSR y(prolongationOperators[currentLevel].num_rows);
        SpMV_CSR(prolongationOperators[currentLevel].row_ptr,
                 prolongationOperators[currentLevel].value_ptr,
                 prolongationOperators[currentLevel].data_ptr,
                 R.vector,
                 y.vector,
                 prolongationOperators[currentLevel].num_rows);

        if(currentLevel<max_number_of_levels) {
            VCycle(LHS[currentLevel + 1], y, v, currentLevel + 1);
        } else {
            //direct solve
            //return direct solve
        }
      
        // prolongate

        // post smooth

        // reutrn post smoothed vector

    }


};