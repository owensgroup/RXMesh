#pragma once
#include <stdint.h>
#include <string>
#include <vector>
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
                  int                                 plane = 1,
                  T                                   dx    = 1.0,
                  bool          with_cross_diagonal         = false,
                  const vec3<T> low_corner                  = {0, 0, 0})
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

            uint32_t a = idx;
            uint32_t b = idx + nx;
            uint32_t c = idx + 1;
            uint32_t d = idx + nx + 1;

            std::vector<uint32_t> t0({a, b, c});

            std::vector<uint32_t> t1({c, b, d});


            tris.push_back(t0);
            tris.push_back(t1);

            if (with_cross_diagonal) {
                std::vector<uint32_t> t2({a, d, c});
                tris.push_back(t2);
            }
        }
    }
}

/**
 * @brief create a box filled with tets
 */
template <typename T>
void create_tet_box(std::vector<std::vector<T>>&        verts,
                    std::vector<std::vector<uint32_t>>& tets,
                    uint32_t                            nx,
                    uint32_t                            ny,
                    uint32_t                            nz,
                    T                                   dx         = T(1),
                    const vec3<T>                       low_corner = {0, 0, 0})
{
    verts.clear();
    tets.clear();

    verts.reserve(static_cast<std::size_t>(nx) * ny * nz);
    if (nx > 1 && ny > 1 && nz > 1) {
        tets.reserve(static_cast<std::size_t>(6) * (nx - 1) * (ny - 1) *
                     (nz - 1));
    }

    for (uint32_t k = 0; k < nz; ++k) {
        for (uint32_t j = 0; j < ny; ++j) {
            for (uint32_t i = 0; i < nx; ++i) {
                verts.push_back({dx * T(i) + low_corner[0],
                                 dx * T(j) + low_corner[1],
                                 dx * T(k) + low_corner[2]});
            }
        }
    }

    auto vertex_id = [=](uint32_t i, uint32_t j, uint32_t k) {
        return i + nx * (j + ny * k);
    };

    for (uint32_t k = 0; k + 1 < nz; ++k) {
        for (uint32_t j = 0; j + 1 < ny; ++j) {
            for (uint32_t i = 0; i + 1 < nx; ++i) {
                const uint32_t a = vertex_id(i, j, k);
                const uint32_t b = vertex_id(i + 1, j, k);
                const uint32_t c = vertex_id(i, j + 1, k);
                const uint32_t d = vertex_id(i + 1, j + 1, k);
                const uint32_t e = vertex_id(i, j, k + 1);
                const uint32_t f = vertex_id(i + 1, j, k + 1);
                const uint32_t g = vertex_id(i, j + 1, k + 1);
                const uint32_t h = vertex_id(i + 1, j + 1, k + 1);

                tets.push_back({a, b, d, h});
                tets.push_back({a, d, c, h});
                tets.push_back({a, c, g, h});
                tets.push_back({a, g, e, h});
                tets.push_back({a, e, f, h});
                tets.push_back({a, f, b, h});
            }
        }
    }
}

}  // namespace rxmesh
