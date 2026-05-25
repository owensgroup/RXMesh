#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <rapidobj/rapidobj.hpp>

#include "rxmesh/util/log.h"

/**
 * @brief Read an input mesh from obj file format
 * @tparam DataT coordinates type (float/double)
 * @tparam IndexT indices type
 * @param file_name path to the obj file
 * @param vertices 3d vertices (3*#vertices)
 * @param faces face index to the Vert array (3*#faces)
 * @param tex texture coordinates (2*#texture coordinates)
 * @param face_tex faces index to the tex array (3*#faces)
 * @param normals face normal coordinates (3*#normal)
 * @param face_normal faces index to the Normals array (3*faces)
 * @return true if reading the file is successful
 */
template <typename DataT, typename IndexT>
bool import_obj(const std::string                 file_name,
                std::vector<std::vector<DataT>>&  vertices,
                std::vector<std::vector<IndexT>>& faces,
                std::vector<std::vector<DataT>>&  tex,
                std::vector<std::vector<IndexT>>& face_tex,
                std::vector<std::vector<DataT>>&  normals,
                std::vector<std::vector<IndexT>>& face_normal,
                bool                              append)
{
    RXMESH_INFO("Reading {}", file_name);

    const std::size_t vertex_offset = append ? vertices.size() : 0;
    const std::size_t tex_offset    = append ? tex.size() : 0;
    const std::size_t normal_offset = append ? normals.size() : 0;

    if (!append) {
        vertices.clear();
        faces.clear();
        tex.clear();
        face_tex.clear();
        normals.clear();
        face_normal.clear();
    }

    rapidobj::Result result = rapidobj::ParseFile(
        std::filesystem::path(file_name), rapidobj::MaterialLibrary::Ignore());

    if (result.error) {
        RXMESH_ERROR("importOBJ() could not read {}: {}",
                     file_name,
                     result.error.code.message());
        return false;
    }

    const auto& positions = result.attributes.positions;
    const auto& texcoords = result.attributes.texcoords;
    const auto& norms     = result.attributes.normals;

    if (positions.size() % 3 != 0) {
        RXMESH_ERROR("importOBJ() malformed vertex position array in {}",
                     file_name);
        return false;
    }
    if (texcoords.size() % 2 != 0) {
        RXMESH_ERROR("importOBJ() malformed texture coordinate array in {}",
                     file_name);
        return false;
    }
    if (norms.size() % 3 != 0) {
        RXMESH_ERROR("importOBJ() malformed normal array in {}", file_name);
        return false;
    }

    vertices.reserve(vertex_offset + positions.size() / 3);
    for (std::size_t i = 0; i < positions.size(); i += 3) {
        vertices.push_back({static_cast<DataT>(positions[i]),
                            static_cast<DataT>(positions[i + 1]),
                            static_cast<DataT>(positions[i + 2])});
    }

    tex.reserve(tex_offset + texcoords.size() / 2);
    for (std::size_t i = 0; i < texcoords.size(); i += 2) {
        tex.push_back({static_cast<DataT>(texcoords[i]),
                       static_cast<DataT>(texcoords[i + 1])});
    }

    normals.reserve(normal_offset + norms.size() / 3);
    for (std::size_t i = 0; i < norms.size(); i += 3) {
        normals.push_back({static_cast<DataT>(norms[i]),
                           static_cast<DataT>(norms[i + 1]),
                           static_cast<DataT>(norms[i + 2])});
    }

    std::size_t total_faces = 0;
    for (const auto& shape : result.shapes) {
        total_faces += shape.mesh.num_face_vertices.size();
    }
    faces.reserve(faces.size() + total_faces);
    face_tex.reserve(face_tex.size() + total_faces);
    face_normal.reserve(face_normal.size() + total_faces);

    auto offset_index = [&](int               index,
                            std::size_t       offset,
                            const char* const kind) -> IndexT {
        if (index < 0) {
            RXMESH_ERROR("importOBJ() face has missing {} index in {}",
                         kind,
                         file_name);
            return static_cast<IndexT>(0);
        }
        return static_cast<IndexT>(static_cast<std::size_t>(index) + offset);
    };

    for (const auto& shape : result.shapes) {
        std::size_t index_offset = 0;
        for (std::size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const std::size_t face_degree =
                static_cast<std::size_t>(shape.mesh.num_face_vertices[f]);

            std::size_t num_tex_indices    = 0;
            std::size_t num_normal_indices = 0;
            for (std::size_t i = 0; i < face_degree; ++i) {
                const rapidobj::Index& idx =
                    shape.mesh.indices[index_offset + i];
                num_tex_indices += idx.texcoord_index >= 0 ? 1 : 0;
                num_normal_indices += idx.normal_index >= 0 ? 1 : 0;
            }

            if (num_tex_indices != 0 && num_tex_indices != face_degree) {
                RXMESH_ERROR(
                    "importOBJ() face has incomplete texture indices in {}",
                    file_name);
                return false;
            }
            if (num_normal_indices != 0 && num_normal_indices != face_degree) {
                RXMESH_ERROR(
                    "importOBJ() face has incomplete normal indices in {}",
                    file_name);
                return false;
            }

            std::vector<IndexT> face;
            std::vector<IndexT> face_t;
            std::vector<IndexT> face_n;
            face.reserve(face_degree);
            if (num_tex_indices == face_degree) {
                face_t.reserve(face_degree);
            }
            if (num_normal_indices == face_degree) {
                face_n.reserve(face_degree);
            }

            for (std::size_t i = 0; i < face_degree; ++i) {
                const rapidobj::Index& idx =
                    shape.mesh.indices[index_offset + i];
                if (idx.position_index < 0) {
                    RXMESH_ERROR(
                        "importOBJ() face has missing vertex index in {}",
                        file_name);
                    return false;
                }
                face.push_back(offset_index(
                    idx.position_index, vertex_offset, "vertex"));
                if (num_tex_indices == face_degree) {
                    face_t.push_back(
                        offset_index(idx.texcoord_index, tex_offset, "texture"));
                }
                if (num_normal_indices == face_degree) {
                    face_n.push_back(offset_index(
                        idx.normal_index, normal_offset, "normal"));
                }
            }

            faces.push_back(std::move(face));
            face_tex.push_back(std::move(face_t));
            face_normal.push_back(std::move(face_n));
            index_offset += face_degree;
        }
    }

    RXMESH_INFO("import_obj() #vertices= {} ", vertices.size());
    RXMESH_INFO("import_obj() #faces= {} ", faces.size());
    RXMESH_INFO("import_obj() #tex= {} ", tex.size());
    RXMESH_INFO("import_obj() #face_tex= {} ", face_tex.size());
    RXMESH_INFO("import_obj() #normals = {} ", normals.size());
    RXMESH_INFO("import_obj() #face_normal= {} ", face_normal.size());

    return true;
}

/**
 * @brief Read an input mesh from obj file format
 * @tparam DataT coordinates type (float/double)
 * @tparam IndexT indices type
 * @param file_name path to the obj file
 * @param vertices 3d vertices (3*#vertices)
 * @param faces face index to the Vert array (3*#faces)
 * @return true if reading the file is successful
 */
template <typename DataT, typename IndexT>
bool import_obj(const std::string                 file_name,
                std::vector<std::vector<DataT>>&  vertices,
                std::vector<std::vector<IndexT>>& faces,
                bool                              append = false)
{
    std::vector<std::vector<DataT>>  tex;
    std::vector<std::vector<IndexT>> face_tex;
    std::vector<std::vector<DataT>>  normals;
    std::vector<std::vector<IndexT>> face_normal;

    return import_obj(file_name,
                      vertices,
                      faces,
                      tex,
                      face_tex,
                      normals,
                      face_normal,
                      append);
}
