/**
 * \brief Class for holding n 3D vectors
 */

struct VectorCSR3D
{
    float* vector;
    int    n;  // number of 3D points (actual vector length is n*3)

    VectorCSR3D()
    {
    }
    VectorCSR3D(int number_of_elements)
    {
        assert(number_of_elements > 0);
        n = number_of_elements;
        CUDA_ERROR(cudaMallocManaged(&vector, sizeof(float) * n * 3));

        reset();
    }
    void reset()
    {


        // TODO:  make this faster, a O(n) used this many times may not be
        // preferable
        for (int i = 0; i < n * 3; i++)
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
        GPU_FREE(vector);
    }
};