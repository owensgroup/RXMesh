#pragma once

namespace rxmesh {
namespace detail {
/**
 * @brief The dimension of a (dense) block in the sparse matrix. In most cases
 * the block size is just 1x1 which represent a single non-zero value. But in,
 * e.g., Hessians, the blocks could be kxk.
 */
struct BlockDim
{
    int x, y;

    BlockDim() : x(1), y(1)
    {
    }

    BlockDim(int x_, int y_) : x(x_), y(y_)
    {
    }
};
}  // namespace detail
}  // namespace rxmesh