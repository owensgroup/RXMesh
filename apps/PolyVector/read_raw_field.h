#pragma once

#include <fstream>

#include "rxmesh/rxmesh_static.h"

template <typename FAttrT>
int inline read_raw_field(const std::string&    file_name,
                          rxmesh::RXMeshStatic& rx,
                          FAttrT&               attr)
{
    using namespace rxmesh;

    using T = typename FAttrT::Type;

    int N;

    try {
        std::ifstream f(file_name);
        if (!f.is_open()) {
            return -1;
        }
        int num_f;

        f >> N;
        f >> num_f;

        if (attr.rows() != num_f) {
            RXMESH_ERROR(
                "read_raw_field() mismatch between the number of elements ({}) "
                "in the file ({}) and the RXMesh attribute type/number of "
                "elements ({})",
                num_f,
                file_name,
                attr.rows());
            return -1;
        }

        if (attr.cols() != 3 * N) {
            RXMESH_ERROR(
                "read_raw_field() mismatch between the number of components "
                "({}) in the file ({}) and the number of attributes in the "
                "input/output RXMesh attribute",
                N,
                file_name,
                attr.cols());
            return -1;
        }

        Eigen::MatrixX<T> raw_field(num_f, 3 * N);

        for (int i = 0; i < num_f; i++) {
            for (int j = 0; j < 3 * N; j++) {
                f >> raw_field(i, j);
            }
        }
        f.close();

        rx.for_each_face(HOST, [&](const FaceHandle fh) {
            int f_global_id = rx.map_to_global(fh);

            assert(f_global_id <= num_f);

            for (int j = 0; j < 3 * N; ++j) {
                attr(fh, j) = raw_field(f_global_id, j);
            }
        });

        attr.move(HOST, DEVICE);
        return N;
    } catch (std::exception e) {
        return -1;
    }
}
