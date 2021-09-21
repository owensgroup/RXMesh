#pragma once
#include <vector>
#include "rxmesh/util/math.h"
#include "rxmesh/util/report.h"

template <typename T>
inline void vertex_normal_ref(const std::vector<std::vector<uint32_t>>& Faces,
                              const std::vector<std::vector<T>>&        Verts,
                              std::vector<T>& vertex_normal)
{
    uint32_t num_vertices = Verts.size();
    uint32_t num_faces    = Faces.size();

    memset((void*)vertex_normal.data(), 0, vertex_normal.size() * sizeof(T));

    T        edge_len[3];
    uint32_t v[3];
    T        fn[3];

    for (uint32_t f = 0; f < num_faces; ++f) {
        v[0] = Faces[f][0];
        v[1] = Faces[f][1];
        v[2] = Faces[f][2];

        RXMESH::cross_product(Verts[v[1]][0] - Verts[v[0]][0],
                              Verts[v[1]][1] - Verts[v[0]][1],
                              Verts[v[1]][2] - Verts[v[0]][2],
                              Verts[v[2]][0] - Verts[v[0]][0],
                              Verts[v[2]][1] - Verts[v[0]][1],
                              Verts[v[2]][2] - Verts[v[0]][2],
                              fn[0],
                              fn[1],
                              fn[2]);

        edge_len[0] = RXMESH::l2_norm_sq(Verts[v[0]][0],
                                         Verts[v[0]][1],
                                         Verts[v[0]][2],
                                         Verts[v[1]][0],
                                         Verts[v[1]][1],
                                         Verts[v[1]][2]);  // v0-v1

        edge_len[1] = RXMESH::l2_norm_sq(Verts[v[1]][0],
                                         Verts[v[1]][1],
                                         Verts[v[1]][2],
                                         Verts[v[2]][0],
                                         Verts[v[2]][1],
                                         Verts[v[2]][2]);  // v1-v2

        edge_len[2] = RXMESH::l2_norm_sq(Verts[v[2]][0],
                                         Verts[v[2]][1],
                                         Verts[v[2]][2],
                                         Verts[v[0]][0],
                                         Verts[v[0]][1],
                                         Verts[v[0]][2]);  // v2-v0


        for (uint32_t i = 0; i < 3; ++i) {
            uint32_t k    = (i + 2) % 3;
            uint32_t base = 3 * v[i];

            for (uint32_t l = 0; l < 3; ++l) {
                vertex_normal[base + l] += fn[l] / (edge_len[i] + edge_len[k]);
            }
        }
    }

    /*for (T v = 0; v < num_vertices; ++v) {
        T base = 3 * v;
        normalize_vector(vertex_normal[base], vertex_normal[base + 1],
            vertex_normal[base + 2]);
    }*/
}