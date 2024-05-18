#pragma once
#include <limits>

#include "cub/device/device_scan.cuh"
#include "rxmesh/util/macros.h"

template <typename T>
struct CostHistogram
{
    __device__ __host__ CostHistogram()                           = default;
    __device__ __host__ CostHistogram(const CostHistogram& other) = default;
    __device__ __host__ CostHistogram(CostHistogram&&)            = default;
    __device__ __host__ CostHistogram& operator=(const CostHistogram&) =
        default;
    __device__ __host__ CostHistogram& operator=(CostHistogram&&) = default;
    __device__                         __host__ ~CostHistogram()  = default;

    CostHistogram(int num)
        : num_bins(num),
          d_bins(nullptr),
          d_min_max_edge_cost(nullptr),
          d_scan_temp_storage(nullptr),
          temp_storage_bytes(0)
    {
        using namespace rxmesh;
        CUDA_ERROR(cudaMalloc((void**)&d_bins, num_bins * sizeof(int)));
        cub::DeviceScan::InclusiveSum(
            d_scan_temp_storage, temp_storage_bytes, d_bins, num_bins);
        CUDA_ERROR(
            cudaMalloc((void**)&d_scan_temp_storage, temp_storage_bytes));
        CUDA_ERROR(cudaMalloc((void**)&d_min_max_edge_cost, 2 * sizeof(T)));
    }

    __host__ void scan()
    {
        cub::DeviceScan::InclusiveSum(
            d_scan_temp_storage, temp_storage_bytes, d_bins, num_bins);
    }

    __host__ void free()
    {
        using namespace rxmesh;
        CUDA_ERROR(cudaFree(d_min_max_edge_cost));
        CUDA_ERROR(cudaFree(d_bins));
        CUDA_ERROR(cudaFree(d_scan_temp_storage));
    }

    __host__ void init()
    {
        using namespace rxmesh;
        CUDA_ERROR(cudaMemset(d_bins, 0, num_bins * sizeof(int)));
        T min_max_init[2] = {std::numeric_limits<T>::max(),
                             std::numeric_limits<T>::lowest()};
        CUDA_ERROR(cudaMemcpy(d_min_max_edge_cost,
                              min_max_init,
                              2 * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    __device__ __host__ __inline__ const T* min_value() const
    {
        return d_min_max_edge_cost;
    }

    __device__ __host__ __inline__ T* min_value()
    {
        return d_min_max_edge_cost;
    }

    __device__ __host__ __inline__ const T* max_value() const
    {
        return d_min_max_edge_cost + 1;
    }

    __device__ __host__ __inline__ T* max_value()
    {
        return d_min_max_edge_cost + 1;
    }

    __device__ __host__ __inline__ int bin_id(const T cost) const
    {
        const T min_v = min_value()[0];
        const T max_v = max_value()[0];

        const int id =
            std::floor(((cost - min_v) * num_bins) / (max_v - min_v));

        return std::min(std::max(0, id), num_bins - 1);
    }

    __device__ __inline__ void insert(const T value)
    {
        const int id = bin_id(value);
        assert(id >= 0);
        ::atomicAdd(d_bins + id, 1);
    }

    __device__ __inline__ int get_bin(T value) const
    {
        int id = bin_id(value);
        assert(id >= 0);
        return d_bins[id];
    }

    __device__ __inline__ bool below_threshold(T value, T threshold) const
    {

        if (get_bin(value) <= threshold) {
            return true;
        }
        std::pair<int, int> bucket = get_bounds(value);

        if (bucket.first <= threshold && bucket.second > threshold) {
            return true;
        }

        return false;
    }

    __device__ __inline__ std::pair<int, int> get_bounds(T value) const
    {
        int id = bin_id(value);
        assert(id >= 0);
        int high_val = d_bins[id];
        int low_val  = (id == 0) ? 0 : d_bins[id - 1];
        return {low_val, high_val};
    }

    __host__ void print()
    {
        using namespace rxmesh;

        T h_min_max[2];
        CUDA_ERROR(cudaMemcpy(h_min_max,
                              d_min_max_edge_cost,
                              2 * sizeof(T),
                              cudaMemcpyDeviceToHost));

        printf("\n*** Histogram min= %f, max= %f", h_min_max[0], h_min_max[1]);

        std::vector<int> h_bins(num_bins);
        CUDA_ERROR(cudaMemcpy(h_bins.data(),
                              d_bins,
                              num_bins * sizeof(int),
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < h_bins.size(); ++i) {
            printf("\n h_bins[%d]= %d", i, h_bins[i]);
        }
        printf("\n");
    }


    int    num_bins;
    T*     d_min_max_edge_cost;
    int*   d_bins;
    void*  d_scan_temp_storage;
    size_t temp_storage_bytes;
};