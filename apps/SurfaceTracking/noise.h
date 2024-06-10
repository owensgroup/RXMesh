#pragma once

#include "rxmesh/types.h"

template <class S, class T>
constexpr __host__ __device__ inline S lerp(const S& value0,
                                            const S& value1,
                                            T        f)
{
    return (1 - f) * value0 + f * value1;
}

template <class S, class T>
constexpr __host__ __device__ inline S bilerp(const S& v00,
                                              const S& v10,
                                              const S& v01,
                                              const S& v11,
                                              T        fx,
                                              T        fy)
{
    return lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy);
}


template <class S, class T>
constexpr __host__ __device__ inline S trilerp(const S& v000,
                                               const S& v100,
                                               const S& v010,
                                               const S& v110,
                                               const S& v001,
                                               const S& v101,
                                               const S& v011,
                                               const S& v111,
                                               T        fx,
                                               T        fy,
                                               T        fz)
{
    return lerp(bilerp(v000, v100, v010, v110, fx, fy),
                bilerp(v001, v101, v011, v111, fx, fy),
                fz);
}

/**
 * @brief returns repeatable stateless pseudo-random number in [0,1]
 * Transforms even the sequence 0,1,2,3,... into reasonably good random
 * numbers
 *  Challenge: improve on this in speed and "randomness"! This seems to pass
 * several statistical tests, and is a bijective map (of 32-bit unsigned ints)
 */
constexpr inline unsigned int randhash(unsigned int seed)
{
    unsigned int i = (seed ^ 12345391u) * 2654435769u;
    i ^= (i << 6) ^ (i >> 26);
    i *= 2654435769u;
    i += (i << 5) ^ (i >> 12);
    return i;
}


/**
 * @brief returns repeatable stateless pseudo-random number in [a,b]
 */
template <typename T>
constexpr inline T randhash(unsigned int seed, T a, T b)
{
    return (b - a) * T(randhash(seed)) / (T)UINT_MAX + a;
}


template <typename T>
rxmesh::vec3<T> sample_sphere(unsigned int& seed)
{
    rxmesh::vec3<T> v;

    T m2;

    do {
        for (unsigned int i = 0; i < 3; ++i) {
            v[i] = randhash(seed++, T(-1), T(1));
        }
        m2 = glm::length2(v);
    } while (m2 > 1 || m2 == 0);

    return v / std::sqrt(m2);
}

template <typename T, uint32_t n = 128>
struct FlowNoise3
{
    __host__ FlowNoise3(unsigned int seed = 171717)
        : noise_lengthscale(1.5), noise_gain(1.3), delta_x(1e-4)
    {
        for (unsigned int i = 0; i < n; ++i) {
            h_basis[i] = sample_sphere<T>(seed);
            h_perm[i]  = i;
        }

        for (unsigned int i = 1; i < n; ++i) {
            int j = randhash(seed++);

            j = j % (i + 1);

            std::swap(h_perm[i], h_perm[j]);
        }
        using namespace rxmesh;

        CUDA_ERROR(cudaMalloc((void**)&d_basis, n * sizeof(vec3<T>)));
        CUDA_ERROR(cudaMalloc((void**)&d_perm, n * sizeof(int)));
        CUDA_ERROR(cudaMemcpy(
            d_basis, h_basis, n * sizeof(vec3<T>), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(
            d_perm, h_perm, n * sizeof(int), cudaMemcpyHostToDevice));
    }

    __host__ void free()
    {
        using namespace rxmesh;
        GPU_FREE(d_basis);
        GPU_FREE(d_perm);
    }


    constexpr __device__ __host__ T operator()(T x, T y, T z) const
    {
        T floorx = std::floor(x);
        T floory = std::floor(y);
        T floorz = std::floor(z);

        int i = (int)floorx;
        int j = (int)floory;
        int k = (int)floorz;

        const rxmesh::vec3<T>& n000 = basis(hash_index(i, j, k));
        const rxmesh::vec3<T>& n100 = basis(hash_index(i + 1, j, k));
        const rxmesh::vec3<T>& n010 = basis(hash_index(i, j + 1, k));
        const rxmesh::vec3<T>& n110 = basis(hash_index(i + 1, j + 1, k));
        const rxmesh::vec3<T>& n001 = basis(hash_index(i, j, k + 1));
        const rxmesh::vec3<T>& n101 = basis(hash_index(i + 1, j, k + 1));
        const rxmesh::vec3<T>& n011 = basis(hash_index(i, j + 1, k + 1));
        const rxmesh::vec3<T>& n111 = basis(hash_index(i + 1, j + 1, k + 1));

        T fx = x - floorx, fy = y - floory, fz = z - floorz;
        T sx = fx * fx * fx * (10 - fx * (15 - fx * 6)),
          sy = fy * fy * fy * (10 - fy * (15 - fy * 6)),
          sz = fz * fz * fz * (10 - fz * (15 - fz * 6));
        return trilerp(
            fx * n000[0] + fy * n000[1] + fz * n000[2],
            (fx - 1) * n100[0] + fy * n100[1] + fz * n100[2],
            fx * n010[0] + (fy - 1) * n010[1] + fz * n010[2],
            (fx - 1) * n110[0] + (fy - 1) * n110[1] + fz * n110[2],
            fx * n001[0] + fy * n001[1] + (fz - 1) * n001[2],
            (fx - 1) * n101[0] + fy * n101[1] + (fz - 1) * n101[2],
            fx * n011[0] + (fy - 1) * n011[1] + (fz - 1) * n011[2],
            (fx - 1) * n111[0] + (fy - 1) * n111[1] + (fz - 1) * n111[2],
            sx,
            sy,
            sz);
    }


    T noise_lengthscale;
    T noise_gain;
    T delta_x;  // used for finite difference approximations of curl

    // private:
    constexpr __device__ __host__ unsigned int hash_index(int i,
                                                          int j,
                                                          int k) const
    {
        return perm((perm((perm(i % n) + j) % n) + k) % n);
    }

    constexpr __device__ __host__ const rxmesh::vec3<T>& basis(
        unsigned int h) const
    {
        assert(h < n);
#ifdef __CUDA_ARCH__
        return d_basis[h];
#else
        return h_basis[h];
#endif
    }

    constexpr __device__ __host__ int perm(unsigned int h) const
    {
        assert(h < n);
#ifdef __CUDA_ARCH__
        return d_perm[h];
#else
        return h_perm[h];
#endif
    }

    rxmesh::vec3<T>  h_basis[n];
    rxmesh::vec3<T>* d_basis;
    int              h_perm[n];
    int*             d_perm;
};
