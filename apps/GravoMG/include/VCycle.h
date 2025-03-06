#pragma once

#include "include/GMGCSR.h"

/**
 * \brief Class for holding n 3D vectors
 */
struct VectorCSR3D
{
    float* vector;
    int    n;  // number of 3D points (actual vector length is n*3)
    VectorCSR3D(){}
    VectorCSR3D(int number_of_elements)
    {
        assert(number_of_elements > 0);
        n = number_of_elements;
        cudaMallocManaged(&vector, sizeof(float) * n * 3);

        reset();
    }
    void reset()
    {
        // TODO:  make this faster, a O(n) used this many times may not be preferable
        for (int i = 0; i < n*3; i++)
            vector[i] = 0;
    }
    void print()
    {
        for (int i = 0; i < n; i++) {
            std::cout << vector[3 * i] << " ";
            std::cout << vector[3 * i + 1] << " ";
            std::cout << vector[3 * i + 2] << " ";
        }
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

/**
 * @brief  Wrapper for cuda implementation of Matrix Vector multiplication
 * @param  Row_ptr Row pointer for CSR matrix
 * @param  col_idx Column pointer for CSR matrix
 * @param  values Value pointer for CSR matrix
 * @param  v vector v
 * @param  y new resulting vector y
 * @param  m number of rows in CSR matrix
 */
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
        float sum_x    = 0.0f;
        float sum_y    = 0.0f;
        float sum_z    = 0.0f;
        float diag     = 0.0f;
        bool  has_diag = false;

        // Loop over non-zero elements in the row
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            int   col = value_ptr[j];  // Column index of non-zero element
            float val = data_ptr[j];   // Value of the non-zero element
            
            if (col == i) {
                // If it's a diagonal element, store its value
                diag     = val;
                has_diag = true;
            }
            else {
                // Sum non-diagonal elements
                sum_x += val * x_old[col * 3];
                sum_y += val * x_old[col * 3 + 1];
                sum_z += val * x_old[col * 3 + 2];
            }
        }
        //float w = 0.85;
        // If the diagonal was found, perform the update
        if (!has_diag)
            printf("\n%d does not have diagonal", i);
        assert(has_diag);
        if (has_diag && abs(diag) > 10e-8f) {
            x_new[i * 3]     =(b[i * 3] - sum_x) / diag;
            x_new[i * 3 + 1] =(b[i * 3 + 1] - sum_y) / diag;
            x_new[i * 3 + 2] =(b[i * 3 + 2] - sum_z) / diag;
            //if (i==0)
            //printf("\n %d %f", i, x_new[i * 3]);

            /*printf("\n thread= %d val= %f, %f, %f",
                   i,
                   x_new[i * 3],
                   x_new[i * 3 + 1],
                   x_new[i * 3 + 2]);*/

        } else {
            x_new[i * 3]     = x_old[i * 3];  
            x_new[i * 3 + 1] = x_old[i * 3 + 1];
            x_new[i * 3 + 2] = x_old[i * 3 + 2];
        }
    }
};

/**
 * \brief Parallel gauss jacobi implementation to solve Ax=b
 * \param A 
 * \param vec_x 
 * \param vec_b 
 * \param max_iter number of iterations
 */
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
        cudaMemcpy(vec_x,
                   thrust::raw_pointer_cast(x),
                   N * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    cudaFree(x_new_raw);
    cudaDeviceSynchronize();
}


/**
 * @brief After computing prolongation operators, this class is used to solve the given Lx=b equation, given certain parameters
 * for the V cycle
 */
class GMGVCycle
{
   public:
    int              pre_relax_iterations  = 2;
    int              post_relax_iterations = 2;
    int              max_number_of_levels;
    std::vector<int> numberOfSamplesPerLevel;
    float            omega=0.5;
    int            directSolveIterations=5;
    float            ratio                 = 8;
    int              numberOfCycles       = 2;

    std::vector<CSR>       prolongationOperators;
    std::vector<CSR>       prolongationOperatorsTransposed;
    std::vector<CSR>       LHS;
    VectorCSR3D RHS;
    VectorCSR3D                 X;  // final solution

    
    /**
     * @brief Solve the equation Av=f using V Cycle
     * @param A the LHS 
     * @param f The RHS
     * @param v The variables we are solving for
     * @param currentLevel The current level of the multigrid V cycle, this usually starts from 0
     */
    void VCycle(CSR& A, VectorCSR3D& f, VectorCSR3D& v, int currentLevel)
    {
        //printf("\n=== Level %d ===", currentLevel);
        //printf("\nGrid size: %d", A.num_rows);

        //// Debug: Check input vector norms
        //float f_norm = 0.0f, v_norm = 0.0f;
        //for (int i = 0; i < A.num_rows * 3; i++) {
        //    f_norm += f.vector[i] * f.vector[i];
        //    v_norm += v.vector[i] * v.vector[i];
        //}
        //printf("\nInput norms - f: %e, v: %e", sqrt(f_norm), sqrt(v_norm));

        // Pre-smoothing
        //printf("\nPre-smoothing...");
        gauss_jacobi_CSR_3D(A, v.vector, f.vector, pre_relax_iterations);

        // Calculate residual on current grid
        VectorCSR3D R(A.num_rows);
        //printf("\nCalculating residual...");
        Compute_R_3D(A, v.vector, f.vector, R.vector, A.num_rows);

        // Create coarse grid vectors
        int coarse_size = prolongationOperators[currentLevel].num_rows / ratio;
        VectorCSR3D restricted_residual(coarse_size);
        VectorCSR3D coarse_correction(coarse_size);

        // Restrict the residual
        //printf("\nRestricting residual...");
        CSR transposeProlongation = prolongationOperatorsTransposed[currentLevel];
            //transposeCSR(prolongationOperators[currentLevel]);

        //if (currentLevel>=1)
        //transposeProlongation.printCSR();

        // Debug: Check restriction operator
        /*float rest_norm = 0.0f;
        for (int i = 0; i < transposeProlongation.non_zeros; i++) {
            rest_norm += transposeProlongation.data_ptr[i] *
                         transposeProlongation.data_ptr[i];
        }*/
        //printf("\nRestriction operator norm: %e", sqrt(rest_norm));

        SpMV_CSR_3D(transposeProlongation.row_ptr,
                    transposeProlongation.value_ptr,
                    transposeProlongation.data_ptr,
                    R.vector,
                    restricted_residual.vector,
                    transposeProlongation.num_rows);


       

        // Debug: Check restricted residual
        /*float rr_norm = 0.0f;
        for (int i = 0; i < coarse_size * 3; i++) {
            rr_norm +=
                restricted_residual.vector[i] * restricted_residual.vector[i];
        }
        printf("\nRestricted residual norm: %e", sqrt(rr_norm));*/

        if (currentLevel < max_number_of_levels - 1) {
            VCycle(LHS[currentLevel+1],
                   restricted_residual,
                   coarse_correction,
                   currentLevel + 1);
        }
        else {
            //printf("\nPerforming direct solve...");
            // Initialize coarse correction to zero
            for (int i = 0; i < coarse_size * 3; i++) {
                coarse_correction.vector[i] = 0.0f;
            }

            assert(restricted_residual.n == coarse_correction.n);

            // std::cout << "\nRestricted residual number of rows: "<<restricted_residual.n;
            //restricted_residual.print();

            //std::cout << "\nNumber of rows in A : " << LHS[currentLevel + 1].num_rows;

          /*  SparseMatrix<float> A_mat(LHS[currentLevel + 1].num_rows,
                                    LHS[currentLevel + 1].num_rows,
                                    LHS[currentLevel + 1].non_zeros,
                                    LHS[currentLevel+1].row_ptr,
                                    LHS[currentLevel+1].value_ptr,
                                    LHS[currentLevel + 1].data_ptr,
                                    LHS[currentLevel + 1].row_ptr,
                                    LHS[currentLevel + 1].value_ptr,
                                    LHS[currentLevel + 1].data_ptr);
            DenseMatrix<float>  B_mat(restricted_residual.n, 3,restricted_residual.vector,restricted_residual.vector);
            
            auto X_mat =
                std::make_shared<DenseMatrix<float>>(coarse_correction.n,
                                                     3,
                                                     coarse_correction.vector,
                                                     coarse_correction.vector);

            A_mat.pre_solve(Solver::CHOL);
            A_mat.solve(B_mat, *X_mat);
*/

            
            gauss_jacobi_CSR_3D(LHS[currentLevel+1],
                                coarse_correction.vector,
                                restricted_residual.vector,
                                directSolveIterations);
        }

        // Debug: Check coarse correction
        //float cc_norm = 0.0f;
        //for (int i = 0; i < coarse_size * 3; i++) {
        //    cc_norm +=
        //        coarse_correction.vector[i] * coarse_correction.vector[i];
        //    if (std::isnan(coarse_correction.vector[i])) {
        //        //printf(
        //          //  "\nWARNING: NaN detected in coarse correction at index %d",
        //           // i);
        //    }
        //}
        //printf("\nCoarse correction norm: %e", sqrt(cc_norm));

        // Prolongate
        VectorCSR3D fine_correction(A.num_rows);
        SpMV_CSR_3D(prolongationOperators[currentLevel].row_ptr,
                    prolongationOperators[currentLevel].value_ptr,
                    prolongationOperators[currentLevel].data_ptr,
                    coarse_correction.vector,
                    fine_correction.vector,
                    prolongationOperators[currentLevel].num_rows);

        // Debug: Check fine correction
        //float fc_norm = 0.0f;
        //for (int i = 0; i < A.num_rows * 3; i++) {
        //    fc_norm += fine_correction.vector[i] * fine_correction.vector[i];
        //}
        //printf("\nFine correction norm: %e", sqrt(fc_norm));

        //// Add correction with relaxation factor
        //for (int i = 0; i < A.num_rows * 3; i++) {
        //    v.vector[i] += omega * fine_correction.vector[i];
        //}

        // Post-smoothing
        //printf("\nPost-smoothing...");
        gauss_jacobi_CSR_3D(A, v.vector, f.vector, post_relax_iterations);

        //printf("\n=== Completed Level %d ===\n", currentLevel);
    }

    /**
     * @brief Solve the system using GMG and Vcycle
     */
    void solve()
    {
        //X.reset();
        for (int i = 0; i < numberOfCycles; i++)
            VCycle(LHS[0], RHS, X, 0);
    }

    /**
     * \brief Function to set prolongation operators, as well as calculate restriction operators
     * \param prolongationOperatorCSR the prolongation operators calculated beforehand
     */
    void SetProlongationOperators(std::vector<CSR> &prolongationOperatorCSR)
    {
        prolongationOperators = prolongationOperatorCSR;
        for (int i=0;i<prolongationOperators.size();i++) {
            prolongationOperatorsTransposed.push_back(
                transposeCSR(prolongationOperatorCSR[i]));
        }
    }

    GMGVCycle(){}
    GMGVCycle(int initialNumberOfRows) : X(initialNumberOfRows)
    {

    }

    ~GMGVCycle(){}
};

void constructLHS(CSR               A_csr,
                  std::vector<CSR>  prolongationOperatorCSR,
                  std::vector<CSR>  prolongationOperatorCSRTranspose,
                  std::vector<CSR>& equationsPerLevel,
                  int               numberOfLevels,
                  int               numberOfSamples,
                  float             ratio)
{
    int currentNumberOfSamples = numberOfSamples;

    CSR result = A_csr;

    // make all the equations for each level

    for (int i = 0; i < numberOfLevels - 1; i++) {
        
        result = multiplyCSR(result.num_rows,
                             result.num_rows,
                             currentNumberOfSamples,
                             result.row_ptr,
                             result.value_ptr,
                             result.data_ptr,
                             result.non_zeros,
                             prolongationOperatorCSR[i].row_ptr,
                             prolongationOperatorCSR[i].value_ptr,
                             prolongationOperatorCSR[i].data_ptr,
                             prolongationOperatorCSR[i].non_zeros);
                             
        /*result = multiplyCSR(result.num_rows,
                             result.num_rows,
                             currentNumberOfSamples,
                             result.row_ptr,
                             result.value_ptr,
                             result.data_ptr,
                             result.non_zeros,
                             prolongationOperatorCSR[i].row_ptr,
                             prolongationOperatorCSR[i].value_ptr,
                             prolongationOperatorCSR[i].data_ptr,
                             prolongationOperatorCSR[i].non_zeros,
                            1);*/

        
        CSR transposeOperator = prolongationOperatorCSRTranspose[i];
           // transposeCSR(prolongationOperatorCSR[i]);

        result = multiplyCSR(transposeOperator.num_rows,
                             prolongationOperatorCSR[i].num_rows,
                             numberOfSamples,
                             transposeOperator.row_ptr,
                             transposeOperator.value_ptr,
                             transposeOperator.data_ptr,
                             transposeOperator.non_zeros,
                             result.row_ptr,
                             result.value_ptr,
                             result.data_ptr,
                             result.non_zeros);

        equationsPerLevel.push_back(result);

        currentNumberOfSamples /= ratio;
        //std::cout << "Equation level " << i << "\n\n";
        //prolongationOperatorCSR[i].printCSR();
        //std::cout << "\n TRANSPOSE : \n";
        //transposeOperator.printCSR();
        //result.printCSR();

    }
    //prolongationOperatorCSR[numberOfLevels-2].printCSR();

}