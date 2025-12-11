#pragma once

namespace rxmesh {
/**
 * @brief The dimension of a (dense) block in the sparse matrix. In most cases
 * the block size is just 1x1 which represent a single non-zero value. But in,
 * e.g., Hessians, the blocks could be kxk.
 */
struct BlockShape
{
    int x, y;

    BlockShape() : x(1), y(1)
    {
    }

    BlockShape(int x_, int y_) : x(x_), y(y_)
    {
    }
};

}  // namespace rxmesh