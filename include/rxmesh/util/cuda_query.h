#pragma once
#include <cuda_runtime_api.h>
#include "rxmesh/kernels/get_arch.cuh"
#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"

namespace rxmesh {
inline int convert_SMV_to_cores(int major, int minor)
{
    // Taken from Nvidia helper_cuda.h to get the number of SM and cuda cores
    // Defines for GPU Architecture types (using the SM version to determine the
    // # of cores per SM
    typedef struct
    {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m =
                 // SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},  // Kepler Generation (SM 3.0) GK10x class
        {0x32, 192},  // Kepler Generation (SM 3.2) GK10x class
        {0x35, 192},  // Kepler Generation (SM 3.5) GK11x class
        {0x37, 192},  // Kepler Generation (SM 3.7) GK21x class
        {0x50, 128},  // Maxwell Generation (SM 5.0) GM10x class
        {0x52, 128},  // Maxwell Generation (SM 5.2) GM20x class
        {0x53, 128},  // Maxwell Generation (SM 5.3) GM20x class
        {0x60, 64},   // Pascal Generation (SM 6.0) GP100 class
        {0x61, 128},  // Pascal Generation (SM 6.1) GP10x class
        {0x62, 128},  // Pascal Generation (SM 6.2) GP10x class
        {0x70, 64},   // Volta Generation (SM 7.0) GV100 class
        {0x72, 64},
        {0x75, 64},
        {0x80, 64},
        {0x86, 128},
        {0x87, 128},
        {0x89, 128},
        {0x90, 128},
        {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }

    // If we don't find the values, we default use the previous one to run
    // properly
    RXMESH_WARN(
        "MapSMtoCores for SM {}.{} is undefined.  Default to use {} Cores/SM\n",
        major,
        minor,
        nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

inline cudaDeviceProp cuda_query(const int dev)
{

    // Various query about the device we are using
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        RXMESH_ERROR(
            "cuda_query() device count = 0 i.e., there is not"
            " a CUDA-supported GPU!!!");
    }

    CUDA_ERROR(cudaSetDevice(dev));
    cudaDeviceProp dev_prop;

    CUDA_ERROR(cudaGetDeviceProperties(&dev_prop, dev));


    RXMESH_INFO("Total number of device: {}", deviceCount);
    RXMESH_INFO("Using device Number: {}", dev);

    RXMESH_INFO("Device name: {}", dev_prop.name);
    RXMESH_INFO(
        "Compute Capability: {}.{}", (int)dev_prop.major, (int)dev_prop.minor);
    RXMESH_INFO("Total amount of global memory (MB): {0:.1f}",
                (float)dev_prop.totalGlobalMem / 1048576.0f);
    RXMESH_INFO("{} Multiprocessors, {} CUDA Cores/MP: {} CUDA Cores",
                dev_prop.multiProcessorCount,
                convert_SMV_to_cores(dev_prop.major, dev_prop.minor),
                convert_SMV_to_cores(dev_prop.major, dev_prop.minor) *
                    dev_prop.multiProcessorCount);
    RXMESH_INFO("Maximum # blocks per SM: {}",
                dev_prop.maxBlocksPerMultiProcessor);
    RXMESH_INFO("ECC support: {}",
                (dev_prop.ECCEnabled ? "Enabled" : "Disabled"));
    RXMESH_INFO("GPU Max Clock rate: {0:.1f} MHz ({1:.2f} GHz)",
                dev_prop.clockRate * 1e-3f,
                dev_prop.clockRate * 1e-6f);
    RXMESH_INFO("Memory Clock rate: {0:.1f} Mhz",
                dev_prop.memoryClockRate * 1e-3f);
    RXMESH_INFO("Memory Bus Width:  {}-bit", dev_prop.memoryBusWidth);
    const double maxBW = 2.0 * dev_prop.memoryClockRate *
                         (dev_prop.memoryBusWidth / 8.0) / 1.0E6;
    RXMESH_INFO("Peak Memory Bandwidth: {0:f}(GB/s)", maxBW);
    RXMESH_INFO("Kernels compiled for compute capability: {}", cuda_arch());

    int cusolver_major = -1, cusolver_minor = -1, cusolver_patch = -1;
    CUSOLVER_ERROR(cusolverGetProperty(MAJOR_VERSION, &cusolver_major));
    CUSOLVER_ERROR(cusolverGetProperty(MINOR_VERSION, &cusolver_minor));
    CUSOLVER_ERROR(cusolverGetProperty(PATCH_LEVEL, &cusolver_patch));
    RXMESH_INFO("Using cuSolver Version {}.{}.{}",
                cusolver_major,
                cusolver_minor,
                cusolver_patch);


    int cusparse_major = -1, cusparse_minor = -1, cusparse_patch = -1;
    CUSPARSE_ERROR(cusparseGetProperty(MAJOR_VERSION, &cusparse_major));
    CUSPARSE_ERROR(cusparseGetProperty(MINOR_VERSION, &cusparse_minor));
    CUSPARSE_ERROR(cusparseGetProperty(PATCH_LEVEL, &cusparse_patch));
    RXMESH_INFO("Using cuSparse Version {}.{}.{}",
                cusparse_major,
                cusparse_minor,
                cusparse_patch);

    int cublas_major = -1, cublas_minor = -1, cublas_patch = -1;
    CUBLAS_ERROR(cublasGetProperty(MAJOR_VERSION, &cublas_major));
    CUBLAS_ERROR(cublasGetProperty(MINOR_VERSION, &cublas_minor));
    CUBLAS_ERROR(cublasGetProperty(PATCH_LEVEL, &cublas_patch));
    RXMESH_INFO("Using cuBlas Version {}.{}.{}",
                cublas_major,
                cublas_minor,
                cublas_patch);


    if (!dev_prop.managedMemory) {
        RXMESH_ERROR(
            "The selected device does not support CUDA unified memory");
        exit(EXIT_FAILURE);
    }

    return dev_prop;
}
}  // namespace rxmesh