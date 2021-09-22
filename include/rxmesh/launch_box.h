#include <stdint.h>

namespace rxmesh {

template <uint32_t blockThreads>
struct LaunchBox
{
    uint32_t blocks, smem_bytes_dyn, smem_bytes_static,
        expected_output_per_block;
    const uint32_t num_threads = blockThreads;
};
}  // namespace rxmesh