#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>
#include "rxmesh/util/log.h"

namespace rxmesh {

typedef uint8_t    flag_t;
constexpr uint32_t PATCH_SIZE = 512;

// TRANSPOSE_ITEM_PER_THREAD
constexpr uint32_t TRANSPOSE_ITEM_PER_THREAD = 11;

// used for integer rounding
#define DIVIDE_UP(num, divisor) (num + divisor - 1) / (divisor)

// assuming a 32-bit index
#define FULL_MASK 0xffffffff

// unsigned 32-bit
#define INVALID32 0xFFFFFFFFu

// unsigned 16-bit
#define INVALID16 0xFFFFu

// unsigned 8-bit
#define INVALID8 0xFFu

// assuming a 32-bit index
#define SPECIAL 0xFFFFFFFE

// 32
#define WARPSIZE 32u


// http://www.decompile.com/cpp/faq/file_and_line_error_string.htm
#define STRINGIFY(x) TOSTRING(x)
#define TOSTRING(x) #x

// CUDA_ERROR
inline void HandleError(cudaError_t err, const char* file, int line)
{
    // Error handling micro, wrap it around function whenever possible
    if (err != cudaSuccess) {
        Log::get_logger()->error("Line {} File {}", line, file);
        Log::get_logger()->error("CUDA ERROR: {}", cudaGetErrorString(err));
#ifdef _WIN32
        system("pause");
#else
        exit(EXIT_FAILURE);
#endif
    }
}
#define CUDA_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// GPU_FREE
#define GPU_FREE(ptr)              \
    if (ptr != nullptr) {          \
        CUDA_ERROR(cudaFree(ptr)); \
        ptr = nullptr;             \
    }


}  // namespace rxmesh