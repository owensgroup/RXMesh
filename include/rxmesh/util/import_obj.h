#pragma once

#include <string>
#include <vector>
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
                std::vector<std::vector<IndexT>>& face_normal)
{

    FILE* Objfile = fopen(file_name.c_str(), "r");
    if (NULL == Objfile) {
        RXMESH_ERROR("importOBJ() can not open {}", file_name);
        return false;
    } else {
        RXMESH_INFO("Reading {}", file_name);
    }


    // make sure everything is clean
    vertices.clear();
    faces.clear();
    tex.clear();
    face_tex.clear();
    normals.clear();
    face_normal.clear();

    constexpr uint32_t max_line_length = 2048;
    char               line[max_line_length];
    uint32_t           lineNum = 1;
    while (fgets(line, max_line_length, Objfile) != NULL) {

        char type[max_line_length];

        if (sscanf(line, "%s", type) == 1) {
            // read only the first letter of the line

            char* l = &line[strlen(type)];  // next thing after the type
            if (strcmp(type, "v") == 0) {
                // vertex
                std::istringstream ls(&line[1]);
                std::vector<DataT> vert{std::istream_iterator<DataT>(ls),
                                        std::istream_iterator<DataT>()};
                if (vert.size() < 3) {
                    // vertex has less than coordinates
                    RXMESH_ERROR(
                        "importOBJ() vertex has less than 3 "
                        "coordinates Line[{}]\n",
                        lineNum);
                    fclose(Objfile);
                    return false;
                }
                vertices.push_back(vert);
            } else if (strcmp(type, "vn") == 0) {
                // normal
                DataT    x[3];
                uint32_t count = sscanf(l, "%f %f %f\n", &x[0], &x[1], &x[2]);

                if (count != 3) {
                    RXMESH_ERROR(
                        "importOBJ() normals does not have 3 coordinates "
                        "Line[{}]\n",
                        lineNum);
                    fclose(Objfile);
                    return false;
                }
                std::vector<DataT> normal_v(3);
                normal_v[0] = x[0];
                normal_v[1] = x[1];
                normal_v[2] = x[2];
                normals.push_back(normal_v);
            } else if (strcmp(type, "vt") == 0) {
                // texture
                DataT    x[3];
                uint32_t count = sscanf(l, "%f %f %f\n", &x[0], &x[1], &x[2]);

                if (count != 2 && count != 3) {
                    RXMESH_ERROR(
                        "importOBJ() texture coordinates are not 2 "
                        "or 3 coordinates Line[{}]",
                        lineNum);
                    fclose(Objfile);
                    return false;
                }
                std::vector<DataT> tx(count);
                for (uint32_t i = 0; i < count; i++) {
                    tx[i] = x[i];
                }
                tex.push_back(tx);
            } else if (strcmp(type, "f") == 0) {
                // face (read vert id, norm id, tex id)

                // const auto & shift = [&vertices](const int i)->int{return i<0
                // ? i+vertices.size():i-1;}; const auto & shift_t =
                // [&Tex](const int i)->int{return i<0 ? i+Tex.size():i-1;};
                // const auto & shift_n = [&normals ](const int i)->int{return
                // i<0 ? i+normals .size():i-1;};

                std::vector<IndexT> f;
                std::vector<IndexT> ft;
                std::vector<IndexT> fn;
                char                word[max_line_length];
                uint32_t            offset;
                while (sscanf(l, "%s%n", word, &offset) == 1) {
                    l += offset;
                    long int i, it, in;
                    if (sscanf(word, "%ld/%ld/%ld", &i, &it, &in) == 3) {
                        // face, norm, tex
                        f.push_back(i < 0 ? i + vertices.size() : i - 1);
                        ft.push_back(i < 0 ? i + tex.size() : i - 1);
                        fn.push_back(i < 0 ? i + normals.size() : i - 1);
                    } else if (sscanf(word, "%ld/%ld", &i, &it) == 2) {
                        // face, tex
                        f.push_back(i < 0 ? i + vertices.size() : i - 1);
                        ft.push_back(i < 0 ? i + tex.size() : i - 1);
                    } else if (sscanf(word, "%ld", &i) == 1) {
                        // face
                        f.push_back(i < 0 ? i + vertices.size() : i - 1);
                    } else {
                        RXMESH_ERROR(
                            "importOBJ() face has wrong format Line[{}]",
                            lineNum);
                        fclose(Objfile);
                        return false;
                    }
                }
                if ((f.size() > 0 && fn.size() == 0 && ft.size() == 0) ||
                    (f.size() > 0 && fn.size() == f.size() && ft.size() == 0) ||
                    (f.size() > 0 && fn.size() == 0 && ft.size() == f.size()) ||
                    (f.size() > 0 && fn.size() == f.size() &&
                     ft.size() == f.size())) {

                    faces.push_back(f);
                    face_tex.push_back(ft);
                    face_normal.push_back(fn);
                } else {
                    RXMESH_ERROR("importOBJ() face has wrong format Line[{}]",
                                 lineNum);
                    fclose(Objfile);
                    return false;
                }
            } else if (strlen(type) >= 1 &&
                       (type[0] == '#' || type[0] == 'g' || type[0] == 'l' ||
                        type[0] == 's' || strcmp("usemtl", type) == 0 ||
                        strcmp("mtllib", type) == 0)) {
                // materials, comments, groups, shading, or lines -> do nothing

            } else {
                // others
                RXMESH_ERROR(
                    "importOBJ() invalid Line[{}] File[{}]\n", lineNum, line);
                fclose(Objfile);
                return false;
            }

        } else {
            // empty line
        }
        lineNum++;
    }
    fclose(Objfile);


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
                std::vector<std::vector<IndexT>>& faces)
{

    std::vector<std::vector<DataT>>  tex;
    std::vector<std::vector<IndexT>> face_tex;
    std::vector<std::vector<DataT>>  normals;
    std::vector<std::vector<IndexT>> face_normal;

    return import_obj(
        file_name, vertices, faces, tex, face_tex, normals, face_normal);
}