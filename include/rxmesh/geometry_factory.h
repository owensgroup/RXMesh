#pragma once
#include <stdint.h>
#include <string>
#include "rxmesh/types.h"
#include "rxmesh/util/macros.h"

namespace rxmesh {

/**
 * @brief create a 2D plane along the x-z plane
 */
template <typename T>
void create_plane(std::vector<std::vector<T>>&        verts,
                  std::vector<std::vector<uint32_t>>& tris,
                  uint32_t                            nx,
                  uint32_t                            ny,
                  int                                 plane      = 1,
                  T                                   dx         = 1.0,
                  const vec3<T>                       low_corner = {0, 0, 0})
{
    verts.clear();
    tris.clear();

    for (uint32_t i = 0; i < ny; i++) {
        for (uint32_t j = 0; j < nx; j++) {

            T x, y, z;


            if (plane == 0) {
                x = 0.0 + low_corner[0];
                y = dx * j + low_corner[1];
                z = dx * i + low_corner[2];
            } else if (plane == 1) {
                x = dx * j + low_corner[0];
                y = 0.0 + low_corner[1];
                z = dx * i + low_corner[2];
            } else if (plane == 2) {
                x = dx * j + low_corner[0];
                y = dx * i + low_corner[1];
                z = 0.0 + low_corner[2];
            }


            std::vector<T> pt({x, y, z});

            verts.push_back(pt);
        }
    }

    for (uint32_t i = 0; i < ny - 1; i++) {
        for (uint32_t j = 0; j < nx - 1; j++) {
            uint32_t idx = i * (nx) + j;

            std::vector<uint32_t> t0({idx, idx + nx, idx + 1});

            std::vector<uint32_t> t1({idx + 1, idx + nx, idx + nx + 1});

            tris.push_back(t0);
            tris.push_back(t1);
        }
    }
}

}  // namespace rxmesh