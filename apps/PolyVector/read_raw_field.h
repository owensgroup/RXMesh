#pragma once

#include <fstream>

#include "rxmesh/rxmesh_static.h"

template <typename FAttrT>
int inline read_raw_field(const std::string&    file_name,
                          rxmesh::RXMeshStatic& rx,
                          FAttrT&               attr,
                          const FAttrT&         B1,
                          const FAttrT&         B2)
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

        if (attr.cols() != N) {
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

            Eigen::Vector3<T> s0(raw_field(f_global_id, 0),
                                 raw_field(f_global_id, 1),
                                 raw_field(f_global_id, 2));
            Eigen::Vector3<T> s1(raw_field(f_global_id, 3),
                                 raw_field(f_global_id, 4),
                                 raw_field(f_global_id, 5));

            Eigen::Vector3<T> b1 = B1.template to_eigen<3>(fh);
            Eigen::Vector3<T> b2 = B2.template to_eigen<3>(fh);

            attr(fh, 0) = b1.dot(s0);
            attr(fh, 1) = b2.dot(s0);
            attr(fh, 2) = b1.dot(s1);
            attr(fh, 3) = b2.dot(s1);
        });

        attr.move(HOST, DEVICE);
        return N;
    } catch (std::exception e) {
        return -1;
    }
}
