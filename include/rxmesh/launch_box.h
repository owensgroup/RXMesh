#pragma once
#include <stdint.h>

namespace rxmesh {

/**
 * @brief Stores different parameters needed to launch kernels i.e., number of
 * CUDA blocks and threads, dynamic shared memory. These parameters are meant to
 * be calculated by RXMeshStatic and then used by the user to launch kernels
 */
template <uint32_t blockThreads>
struct LaunchBox
{
    uint32_t       blocks, num_registers_per_thread;
    size_t         smem_bytes_dyn, smem_bytes_static;
    size_t         local_mem_per_thread;
    const uint32_t num_threads = blockThreads;
};
}  // namespace rxmesh