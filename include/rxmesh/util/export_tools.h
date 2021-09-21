#pragma once

#include <stdint.h>
#include <fstream>
#include <map>
#include "rxmesh/util/log.h"

template <typename CubeX, typename CubeY, typename CubeZ>
void export_as_cubes(std::string    filename,
                     const uint32_t num_cubes,
                     const float    cube_len,
                     CubeX          funX,
                     CubeY          funY,
                     CubeZ          funZ)
{
    // draw samples as cubes in obj file
    filename = STRINGIFY(OUTPUT_DIR) + filename;
    std::fstream file(filename.c_str(), std::ios::out);
    file.precision(15);


    file << "#vertices " << num_cubes * 8 << std::endl;
    file << "#faces " << num_cubes * 6 << std::endl;

    float half_len = cube_len / 2.0f;

    int total_num_vertices = 0;
    for (uint32_t v = 0; v < num_cubes; ++v) {
        double lowerLeft_X, lowerLeft_Y, lowerLeft_Z;
        double center_X, center_Y, center_Z;
        lowerLeft_X = funX(v);
        lowerLeft_Y = funY(v);
        lowerLeft_Z = funZ(v);

        center_X = funX(v);
        center_Y = funY(v);
        center_Z = funZ(v);

        file << "v " << center_X - half_len << " " << center_Y - half_len << " "
             << center_Z - half_len << std::endl;  // 1
        file << "v " << center_X + half_len << " " << center_Y - half_len << " "
             << center_Z - half_len << std::endl;  // 2
        file << "v " << center_X + half_len << " " << center_Y + half_len << " "
             << center_Z - half_len << std::endl;  // 3
        file << "v " << center_X - half_len << " " << center_Y + half_len << " "
             << center_Z - half_len << std::endl;  // 4

        file << "v " << center_X - half_len << " " << center_Y - half_len << " "
             << center_Z + half_len << std::endl;  // 5
        file << "v " << center_X + half_len << " " << center_Y - half_len << " "
             << center_Z + half_len << std::endl;  // 6
        file << "v " << center_X + half_len << " " << center_Y + half_len << " "
             << center_Z + half_len << std::endl;  // 7
        file << "v " << center_X - half_len << " " << center_Y + half_len << " "
             << center_Z + half_len << std::endl;  // 8


        file << "f " << total_num_vertices + 1 << " " << total_num_vertices + 2
             << " " << total_num_vertices + 3 << " " << total_num_vertices + 4
             << std::endl;
        file << "f " << total_num_vertices + 1 << " " << total_num_vertices + 2
             << " " << total_num_vertices + 6 << " " << total_num_vertices + 5
             << std::endl;
        file << "f " << total_num_vertices + 1 << " " << total_num_vertices + 5
             << " " << total_num_vertices + 8 << " " << total_num_vertices + 4
             << std::endl;
        file << "f " << total_num_vertices + 7 << " " << total_num_vertices + 3
             << " " << total_num_vertices + 4 << " " << total_num_vertices + 8
             << std::endl;
        file << "f " << total_num_vertices + 7 << " " << total_num_vertices + 6
             << " " << total_num_vertices + 2 << " " << total_num_vertices + 3
             << std::endl;
        file << "f " << total_num_vertices + 7 << " " << total_num_vertices + 6
             << " " << total_num_vertices + 5 << " " << total_num_vertices + 8
             << std::endl;

        total_num_vertices += 8;
    }
    file.close();
}

template <typename CubeX, typename CubeY, typename CubeZ, typename AttT>
void export_as_cubes_VTK(std::string    filename,
                         const uint32_t num_cubes,
                         const float    cube_len,
                         const AttT*    vertex_att,
                         CubeX          funX,
                         CubeY          funY,
                         CubeZ          funZ,
                         const uint32_t num_att,
                         bool           randomize  = 1,
                         float*         randomness = (float*)nullptr)
{

    // draw samples as cubes with attibutes/colors in vtk file
    // funX, funY, and funZ should return the sample
    // vertex_att is an array of lengh num_cubes with the attibures for each
    // point

    filename = STRINGIFY(OUTPUT_DIR) + filename;
    std::fstream file_vtk(filename.c_str(), std::ios::out);
    file_vtk.precision(15);

    file_vtk << "# vtk DataFile Version 2.0" << std::endl;
    file_vtk << "Voxel Grid" << std::endl;
    file_vtk << "ASCII" << std::endl;
    file_vtk << "DATASET UNSTRUCTURED_GRID" << std::endl;
    file_vtk << "POINTS " << num_cubes * 8 << " float" << std::endl;

    float half_len = cube_len / 2.0f;

    for (uint32_t v = 0; v < num_cubes; ++v) {
        double lowerLeft_X = funX(v) - half_len;
        double lowerLeft_Y = funY(v) - half_len;
        double lowerLeft_Z = funZ(v) - half_len;

        file_vtk << lowerLeft_X << " " << lowerLeft_Y << " " << lowerLeft_Z
                 << std::endl;  // 0
        file_vtk << lowerLeft_X + cube_len << " " << lowerLeft_Y << " "
                 << lowerLeft_Z << std::endl;  // 1
        file_vtk << lowerLeft_X + cube_len << " " << lowerLeft_Y << " "
                 << lowerLeft_Z + cube_len << std::endl;  // 2
        file_vtk << lowerLeft_X << " " << lowerLeft_Y << " "
                 << lowerLeft_Z + cube_len << std::endl;  // 3

        file_vtk << lowerLeft_X << " " << lowerLeft_Y + cube_len << " "
                 << lowerLeft_Z << std::endl;  // 4
        file_vtk << lowerLeft_X + cube_len << " " << lowerLeft_Y + cube_len
                 << " " << lowerLeft_Z << std::endl;  // 5
        file_vtk << lowerLeft_X + cube_len << " " << lowerLeft_Y + cube_len
                 << " " << lowerLeft_Z + cube_len << std::endl;  // 6
        file_vtk << lowerLeft_X << " " << lowerLeft_Y + cube_len << " "
                 << lowerLeft_Z + cube_len << std::endl;  // 7
    }

    file_vtk << "CELLS " << num_cubes << " " << num_cubes * 9 << std::endl;
    for (uint32_t v = 0; v < num_cubes * 8; v += 8) {
        file_vtk << "8 ";
        for (uint32_t i = 0; i < 8; ++i) {
            file_vtk << v + i << " ";
        }
        file_vtk << std::endl;
    }

    file_vtk << "CELL_TYPES " << num_cubes << std::endl;
    for (uint32_t v = 0; v < num_cubes; ++v) {
        file_vtk << 12 << std::endl;
    }

    file_vtk << "POINT_DATA " << num_cubes * 8 << std::endl;
    file_vtk << "SCALARS scalars float 1" << std::endl;
    file_vtk << "LOOKUP_TABLE default" << std::endl;

    if (randomize && randomness == (float*)nullptr) {
        randomness = (float*)malloc(num_att * sizeof(float));
        for (uint32_t i = 0; i < num_att; i++) {
            randomness[i] = 1.0f;
        }
    }

    for (uint32_t v = 0; v < num_cubes; ++v) {
        float val = static_cast<float>(vertex_att[v]);

        if (randomize) {
            if (fabs(randomness[vertex_att[v]] - 1.0f) < 0.0001f) {
                randomness[vertex_att[v]] *= float(rand()) / float(RAND_MAX);
            }
            val = randomness[vertex_att[v]];
        }


        for (uint32_t i = 0; i < 8; ++i) {
            file_vtk << val << std::endl;
        }
    }

    file_vtk.close();
}


template <typename T_d, typename T>
void export_obj(const std::vector<std::vector<T>>&   Faces,
                const std::vector<std::vector<T_d>>& Verts,
                std::string                          filename,
                bool                                 default_folder = true)
{
    if (default_folder) {
        filename = STRINGIFY(OUTPUT_DIR) + filename;
    }

    RXMESH_TRACE(" Exporting to {}", filename);

    // std::ofstream file;
    FILE* file;
    file = fopen(filename.c_str(), "w");
    if (file == NULL) {
        RXMESH_ERROR("export_obj() can not open {}", filename);
    }
    // file.open(filename);
    // file.precision(16);

    // write vertices
    for (uint32_t v = 0; v < Verts.size(); v++) {
        // file << "v  ";
        fprintf(file, "v  ");
        for (uint32_t iv = 0; iv < Verts[v].size(); iv++) {
            // file << Verts[v][iv] << "  ";
            fprintf(file, "%f ", Verts[v][iv]);
        }
        // file << std::endl;
        fprintf(file, "\n");
    }

    // write faces
    for (uint32_t f = 0; f < Faces.size(); f++) {
        // file << "f  ";
        fprintf(file, "f  ");
        for (uint32_t fi = 0; fi < Faces[f].size(); fi++) {
            // file << Faces[f][fi] + 1 << "		";
            fprintf(file, "%u ", Faces[f][fi] + 1);
        }
        // file << std::endl;
        fprintf(file, "\n");
    }
    fclose(file);
    // file.close();
}


template <typename T, typename dataT, typename attrT>
void export_attribute_VTK(
    std::string                            filename,
    const std::vector<std::vector<T>>&     fvn,
    const std::vector<std::vector<dataT>>& Verts,
    bool per_face,  // is the attribute per-face? if not, then it is per-vertex
    const attrT* face_att,
    const attrT* vertex_att,
    bool         randomize = false)
{
    // write vtk files such that each patch has a different color

    filename = STRINGIFY(OUTPUT_DIR) + filename;

    RXMESH_TRACE(" Exporting to {}", filename);

    std::fstream file_vtk(filename, std::ios::out);
    file_vtk << "# vtk DataFile Version 2.5" << std::endl;
    file_vtk << "Unstructured Grid" << std::endl;
    file_vtk << "ASCII" << std::endl;
    file_vtk << "DATASET UNSTRUCTURED_GRID" << std::endl;
    file_vtk << "POINTS " << Verts.size() << " double" << std::endl;
    for (T ipoint = 0; ipoint < Verts.size(); ipoint++) {
        file_vtk << Verts[ipoint][0] << " " << Verts[ipoint][1] << " "
                 << Verts[ipoint][2] << std::endl;
    }

    T num_faces = fvn.size();

    T num_entry = num_faces * (3 + 1);  // assume all triangles

    file_vtk << "CELLS " << num_faces << " " << num_entry << std::endl;
    std::vector<T> face_vert;
    for (T f = 0; f < num_faces; f++) {
        face_vert = fvn[f];
        file_vtk << 3 << " ";
        for (T fi = 0; fi < 3 /*face_vert.size()*/; fi++) {
            file_vtk << face_vert[fi] << "  ";
        }
        file_vtk << std::endl;
    }

    file_vtk << "CELL_TYPES  " << num_faces << std::endl;
    for (T f = 0; f < num_faces; f++) {
        file_vtk << "7" << std::endl;
    }

    std::map<attrT, double> rand_map;
    auto randomize_func = [&rand_map](const attrT* att, const T id) {
        typename std::map<attrT, double>::iterator rand_map_it =
            rand_map.find(att[id]);
        if (rand_map_it != rand_map.end()) {
            return rand_map[att[id]];
        } else {
            double val        = double(rand()) / double(RAND_MAX);
            rand_map[att[id]] = val;
            return val;
        }
    };

    if (per_face) {
        // face attribute
        file_vtk << "CELL_DATA  " << num_faces << std::endl;
        file_vtk << "SCALARS cell_scalars float  " << 1 << std::endl;
        file_vtk << "LOOKUP_TABLE default" << std::endl;
        for (T f = 0; f < num_faces; f++) {
            double val = double(face_att[f]);
            if (randomize) {
                val = randomize_func(face_att, f);
            }
            file_vtk << val << std::endl;
        }
    } else {
        // vertex attribute
        file_vtk << "POINT_DATA  " << Verts.size() << std::endl;
        file_vtk << "SCALARS rad double " << 1 << std::endl;
        file_vtk << "LOOKUP_TABLE default" << std::endl;
        for (T v = 0; v < Verts.size(); v++) {
            double val = double(vertex_att[v]);
            if (randomize) {
                val = randomize_func(vertex_att, v);
            }
            file_vtk << val << std::endl;
        }
    }
    file_vtk.close();
}


template <typename T, typename dataT>
inline void export_face_list(std::string                            filename,
                             const std::vector<std::vector<T>>&     fvn,
                             const std::vector<std::vector<dataT>>& Verts,
                             const T  face_list_size,
                             const T* face_list)
{

    filename = STRINGIFY(OUTPUT_DIR) + filename;

    RXMESH_TRACE(" Exporting to {}", filename);

    std::fstream file(filename, std::ios::out);
    T            total_v = 0;
    for (T j = 0; j < face_list_size; j++) {
        T face = face_list[j];

        for (T k = 0; k < 3; k++) {
            file << "v " << Verts[fvn[face][k]][0] << " "
                 << Verts[fvn[face][k]][1] << " " << Verts[fvn[face][k]][2]
                 << std::endl;
        }
        file << "f  ";
        for (T k = 0; k < 3; k++) {
            file << total_v + 1 << " ";
            total_v++;
        }
        file << std::endl;
    }
    file.close();
}