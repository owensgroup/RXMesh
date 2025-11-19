#pragma once
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t _err = (expr);                                               \
    if (_err != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__          \
                << " in " << #expr << " : " << cudaGetErrorString(_err)      \
                << " (" << static_cast<int>(_err) << ")\n";                  \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

inline void handle_thrust_error(
    const char* expr,
    const char* file,
    int         line,
    const thrust::system_error& e)
{
    std::cerr << "Thrust error at " << file << ":" << line
              << " in " << expr << "\n"
              << "  what()      : " << e.what() << "\n";

    // Try to extract underlying cudaError_t if this is a CUDA backend error
    auto code = e.code();
    if (code.category() == thrust::system::cuda_category()) {
        auto cuda_err = static_cast<cudaError_t>(code.value());
        std::cerr << "  cuda error  : " << cudaGetErrorString(cuda_err)
                  << " (" << static_cast<int>(cuda_err) << ")\n";
    } else {
        std::cerr << "  error code  : " << code.message()
                  << " (" << code.value() << ")\n";
    }

    std::exit(EXIT_FAILURE); // or throw; or your own error policy
}

#define THRUST_CALL(expr)                               \
  do {                                                  \
    try {                                               \
      expr;                                             \
    } catch (const thrust::system_error& e) {           \
      handle_thrust_error(#expr, __FILE__, __LINE__, e);\
    }                                                   \
  } while (0)
