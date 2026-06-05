#pragma once
// RXMesh CUDA->HIP compatibility shim.
//
// On AMD (USE_HIP / __HIP_PLATFORM_AMD__) this aliases the CUDA spellings RXMesh
// uses to their HIP equivalents and pulls in the HIP runtime, cooperative
// groups, and the hip* math libraries. On NVIDIA it is a no-op that includes the
// CUDA runtime. This is the only file that knows about HIP; everywhere else the
// sources keep their plain CUDA spelling.
//
// Keep host libc decls (<cstring>/<cstdlib>) ahead of <hip/hip_runtime.h>: inside
// a .cu compiled as HIP, memcpy/memset can otherwise bind to HIP's __device__
// overloads and the host code fails to compile.

#include <cstdlib>
#include <cstring>

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

// __grid_constant__ is a CUDA kernel-parameter hint (constant-bank placement)
// with no HIP equivalent; drop it on HIP (the parameter is passed normally).
#ifndef __grid_constant__
#define __grid_constant__
#endif

// ---- runtime: types ----
#define cudaError_t                        hipError_t
#define cudaSuccess                        hipSuccess
#define cudaStream_t                       hipStream_t
#define cudaEvent_t                        hipEvent_t
#define cudaDeviceProp                     hipDeviceProp_t
#define cudaFuncAttributes                 hipFuncAttributes
#define cudaDataType                       hipDataType
#define cudaDataType_t                     hipDataType
#define cudaMemcpyKind                     hipMemcpyKind

// ---- runtime: enums ----
#define cudaMemcpyHostToDevice             hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost             hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice           hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost               hipMemcpyHostToHost
#define cudaMemcpyDefault                  hipMemcpyDefault
#define cudaFuncAttributeMaxDynamicSharedMemorySize \
    hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncCachePreferShared          hipFuncCachePreferShared

// data-type enums used by the dense/sparse matrix generic API + cuda_type()
#define CUDA_R_32F                         HIP_R_32F
#define CUDA_R_64F                         HIP_R_64F
#define CUDA_C_32F                         HIP_C_32F
#define CUDA_C_64F                         HIP_C_64F
#define CUDA_R_8I                          HIP_R_8I
#define CUDA_R_8U                          HIP_R_8U
#define CUDA_R_16I                         HIP_R_16I
#define CUDA_R_16U                         HIP_R_16U
#define CUDA_R_32I                         HIP_R_32I
#define CUDA_R_32U                         HIP_R_32U
#define CUDA_R_64I                         HIP_R_64I
#define CUDA_R_64U                         HIP_R_64U

// ---- runtime: memory / device / stream ----
#define cudaMalloc                         hipMalloc
#define cudaMallocManaged                  hipMallocManaged
#define cudaFree                           hipFree
#define cudaMemcpy                         hipMemcpy
#define cudaMemcpyAsync                    hipMemcpyAsync
#define cudaMemset                         hipMemset
#define cudaMemsetAsync                    hipMemsetAsync
#define cudaMemGetInfo                     hipMemGetInfo
#define cudaDeviceSynchronize              hipDeviceSynchronize
#define cudaDeviceReset                    hipDeviceReset
#define cudaStreamSynchronize              hipStreamSynchronize
#define cudaStreamCreate                   hipStreamCreate
#define cudaStreamDestroy                  hipStreamDestroy
#define cudaGetDevice                      hipGetDevice
#define cudaSetDevice                      hipSetDevice
#define cudaGetDeviceCount                 hipGetDeviceCount
#define cudaGetDeviceProperties            hipGetDeviceProperties
#define cudaDeviceGetAttribute             hipDeviceGetAttribute
#define cudaGetLastError                   hipGetLastError
#define cudaGetErrorString                 hipGetErrorString
#define cudaRuntimeGetVersion              hipRuntimeGetVersion
#define cudaDriverGetVersion               hipDriverGetVersion

// ---- runtime: events / profiler / occupancy / func attrs ----
#define cudaEventCreate                    hipEventCreate
#define cudaEventDestroy                   hipEventDestroy
#define cudaEventRecord                    hipEventRecord
#define cudaEventSynchronize               hipEventSynchronize
#define cudaEventElapsedTime               hipEventElapsedTime
#define cudaProfilerStart                  hipProfilerStart
#define cudaProfilerStop                   hipProfilerStop
#define cudaFuncSetAttribute               hipFuncSetAttribute
#define cudaFuncGetAttributes              hipFuncGetAttributes
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor \
    hipOccupancyMaxActiveBlocksPerMultiprocessor

// ---- CUB -> hipCUB ----
// cub::Foo -> hipcub::Foo, and <cub/...> include paths -> <hipcub/...>.
#define cub                                hipcub

#else  // CUDA

#include <cuda_runtime.h>

#endif
