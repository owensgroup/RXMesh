#pragma once
#include <stdint.h>

namespace rxmesh {

template <uint32_t blockThreads>
struct LaunchBox
{
    uint32_t       blocks, num_registers_per_thread;
    size_t         smem_bytes_dyn, smem_bytes_static;
    const uint32_t num_threads = blockThreads;
};
}  // namespace rxmesh