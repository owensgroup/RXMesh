#pragma once

#include <string>
#include <vector>

#include "rxmesh/util/log.h"

#ifndef MAX_LINE_LENGTH
#define MAX_LINE_LENGTH 2048
#endif


// Read and input mesh from obj file format
// Input: path to the obj file
// Output: Verts = 3d vertices (Num vertices X 3)
//        Faces = faces index to the Vert array (Num facex X 3)
//        Tex = Tex coordinates (Num texture coordinates X 2)
//        Faces = faces index to the Tex array (Num facex X 3)
//        Normals = faces index to the Tex array (Num normals X 3)
//        Faces = faces index to the Normals array (Num facex X 3)

template <typename DATA_T, typename INDEX_T>
bool import_obj(const std::string                  fileName,
                std::vector<std::vector<DATA_T>>&  Verts,
                std::vector<std::vector<INDEX_T>>& Faces,
                std::vector<std::vector<DATA_T>>&  Tex,
                std::vector<std::vector<INDEX_T>>& FacesTex,
                std::vector<std::vector<DATA_T>>&  Normal,
                std::vector<std::vector<INDEX_T>>& FacesNormal,
                bool                               quite = false)
{

    FILE* Objfile = fopen(fileName.c_str(), "r");
    if (NULL == Objfile) {
        RXMESH_ERROR("importOBJ() can not open {}", fileName);
        return false;
    } else {
        if (!quite) {
            RXMESH_TRACE("Reading {}", fileName);
        }
    }


    // make sure everything is clean
    Verts.clear();
    Faces.clear();
    Tex.clear();
    FacesTex.clear();
    Normal.clear();
    FacesNormal.clear();

    char     line[MAX_LINE_LENGTH];
    uint32_t lineNum = 1;
    while (fgets(line, MAX_LINE_LENGTH, Objfile) != NULL) {

        char type[MAX_LINE_LENGTH];

        if (sscanf(line, "%s", type) == 1) {
            // read only the first letter of the line

            char* l = &line[strlen(type)];  // next thing after the type
            if (strcmp(type, "v") == 0) {
                // vertex
                std::istringstream  ls(&line[1]);
                std::vector<DATA_T> vert{std::istream_iterator<DATA_T>(ls),
                                         std::istream_iterator<DATA_T>()};
                if (vert.size() < 3) {
                    // vertex has less than coordinates
                    RXMESH_ERROR(
                        "importOBJ() vertex has less than 3 "
                        "coordinates Line[{}]\n",
                        lineNum);
                    fclose(Objfile);
                    return false;
                }
                Verts.push_back(vert);
            } else if (strcmp(type, "vn") == 0) {
                // normal
                DATA_T   x[3];
                uint32_t count = sscanf(l, "%f %f %f\n", &x[0], &x[1], &x[2]);

                if (count != 3) {
                    RXMESH_ERROR(
                        "importOBJ() normal has less than 3 "
                        "coordinates Line[{}]\n",
                        lineNum);
                    fclose(Objfile);
                    return false;
                }
                std::vector<DATA_T> normal_v(3);
                normal_v[0] = x[0];
                normal_v[1] = x[1];
                normal_v[2] = x[2];
                Normal.push_back(normal_v);
            } else if (strcmp(type, "vt") == 0) {
                // texture
                DATA_T   x[3];
                uint32_t count = sscanf(l, "%f %f %f\n", &x[0], &x[1], &x[2]);

                if (count != 2 && count != 3) {
                    RXMESH_ERROR(
                        "importOBJ() texture coordinates are not 2 "
                        "or 3 coordinates Line[{}]",
                        lineNum);
                    fclose(Objfile);
                    return false;
                }
                std::vector<DATA_T> tex(count);
                for (uint32_t i = 0; i < count; i++) {
                    tex[i] = x[i];
                }
                Tex.push_back(tex);
            } else if (strcmp(type, "f") == 0) {
                // face (read vert id, norm id, tex id)

                // const auto & shift = [&Verts](const int i)->int{return i<0 ?
                // i+Verts.size():i-1;}; const auto & shift_t = [&Tex](const int
                // i)->int{return i<0 ? i+Tex.size():i-1;}; const auto & shift_n
                // = [&Normal](const int i)->int{return i<0 ?
                // i+Normal.size():i-1;};

                std::vector<INDEX_T> f;
                std::vector<INDEX_T> ft;
                std::vector<INDEX_T> fn;
                char                 word[MAX_LINE_LENGTH];
                uint32_t             offset;
                while (sscanf(l, "%s%n", word, &offset) == 1) {
                    l += offset;
                    long int i, it, in;
                    if (sscanf(word, "%ld/%ld/%ld", &i, &it, &in) == 3) {
                        // face, norm, tex
                        f.push_back(i < 0 ? i + Verts.size() : i - 1);
                        ft.push_back(i < 0 ? i + Tex.size() : i - 1);
                        fn.push_back(i < 0 ? i + Normal.size() : i - 1);
                    } else if (sscanf(word, "%ld/%ld", &i, &it) == 2) {
                        // face, tex
                        f.push_back(i < 0 ? i + Verts.size() : i - 1);
                        ft.push_back(i < 0 ? i + Tex.size() : i - 1);
                    } else if (sscanf(word, "%ld", &i) == 1) {
                        // face
                        f.push_back(i < 0 ? i + Verts.size() : i - 1);
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

                    Faces.push_back(f);
                    FacesTex.push_back(ft);
                    FacesNormal.push_back(fn);
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

    if (!quite) {
        RXMESH_TRACE("import_obj() #Verts= {} ", Verts.size());
        RXMESH_TRACE("import_obj() #Faces= {} ", Faces.size());
        RXMESH_TRACE("import_obj() #Tex= {} ", Tex.size());
        RXMESH_TRACE("import_obj() #FacesTex= {} ", FacesTex.size());
        RXMESH_TRACE("import_obj() #Normal= {} ", Normal.size());
        RXMESH_TRACE("import_obj() #FacesNormal= {} ", FacesNormal.size());
    }
    return true;
}


template <typename DATA_T, typename INDEX_T>
bool import_obj(const std::string                  fileName,
                std::vector<std::vector<DATA_T>>&  Verts,
                std::vector<std::vector<INDEX_T>>& Faces,
                bool                               quite = false)
{

    std::vector<std::vector<DATA_T>>  Tex;
    std::vector<std::vector<INDEX_T>> FacesTex;
    std::vector<std::vector<DATA_T>>  Normal;
    std::vector<std::vector<INDEX_T>> FacesNormal;

    return import_obj(
        fileName, Verts, Faces, Tex, FacesTex, Normal, FacesNormal, quite);
}