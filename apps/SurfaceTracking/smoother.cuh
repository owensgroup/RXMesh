#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

template <typename T>
using Vec3 = glm::vec<3, T, glm::defaultp>;

#include <Eigen/Dense>

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"


template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    null_space_smooth_vertex(const rxmesh::Context                 context,
                             const rxmesh::VertexAttribute<int8_t> is_vertex_bd,
                             const rxmesh::VertexAttribute<T> current_position,
                             rxmesh::VertexAttribute<T>       new_position)
{

    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    auto smooth = [&](VertexHandle vh, VertexIterator& iter) {
        const Vec3<T> v(current_position(vh, 0),
                        current_position(vh, 1),
                        current_position(vh, 2));

        if (is_vertex_bd(vh) == 1) {
            new_position(vh, 0) = v[0];
            new_position(vh, 1) = v[1];
            new_position(vh, 2) = v[2];
            return;
        }

        VertexHandle qh = iter.back();

        Vec3<T> q(current_position(qh, 0),
                  current_position(qh, 1),
                  current_position(qh, 2));

        Eigen::Matrix<T, 3, 3> A;
        A << 0, 0, 0, 0, 0, 0, 0, 0, 0;


        // for each triangle q-v-r where v is the vertex we want to compute its
        // displacement
        for (uint32_t i = 0; i < iter.size(); ++i) {

            const VertexHandle rh = iter[i];

            const Vec3<T> r(current_position(rh, 0),
                            current_position(rh, 1),
                            current_position(rh, 2));

            // triangle normal
            const Vec3<T> c = glm::cross(q - v, r - v);

            assert(glm::length(c) >= std::numeric_limits<T>::min());

            const Vec3<T> n = glm::normalize(c);

            // triangle area
            const T area = T(0.5) * glm::length(c);

            qh = rh;
            q  = r;

            for (int j = 0; j < 3; ++j) {
                A(0, j) += n[0] * area * n[j];
                A(1, j) += n[1] * area * n[j];
                A(2, j) += n[2] * area * n[j];
            }
        }

        // eigen decomp
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> eigen_solver(A);
        assert(eigen_solver.info() == Eigen::Success);

        T eigenvalues[3];
        eigenvalues[0] = eigen_solver.eigenvalues().real()[0];
        eigenvalues[1] = eigen_solver.eigenvalues().real()[1];
        eigenvalues[2] = eigen_solver.eigenvalues().real()[2];

        // compute basis for null space
        Eigen::Matrix<T, 3, 3> tt;
        tt << 0, 0, 0, 0, 0, 0, 0, 0, 0;
        for (uint32_t i = 0; i < 3; ++i) {
            if (eigenvalues[i] < G_EIGENVALUE_RANK_RATIO * eigenvalues[2]) {
                tt(i, 0) = A(0, i);
                tt(i, 1) = A(1, i);
                tt(i, 2) = A(2, i);
            }
        }

        Eigen::Matrix<T, 3, 3> null_space_projection;
        null_space_projection << 0, 0, 0, 0, 0, 0, 0, 0, 0;
        for (uint32_t row = 0; row < 3; ++row) {
            for (uint32_t col = 0; col < 3; ++col) {
                for (size_t i = 0; i < 3; ++i) {
                    null_space_projection(row, col) += tt(i, row) * tt(i, col);
                }
            }
        }

        Eigen::Matrix<T, 3, 1> t;  // displacement
        t << 0, 0, 0;
        T sum_areas = 0;


        // for each triangle p-v-r where v is the vertex we want to compute its
        // displacement
        VertexHandle ph = iter.back();

        Vec3<T> p(current_position(ph, 0),
                  current_position(ph, 1),
                  current_position(ph, 2));

        for (uint32_t i = 0; i < iter.size(); ++i) {

            const VertexHandle rh = iter[i];

            const Vec3<T> r(current_position(rh, 0),
                            current_position(rh, 1),
                            current_position(rh, 2));

            // triangle normal
            const Vec3<T> c = glm::cross(p - v, r - v);

            assert(glm::length(c) >= std::numeric_limits<T>::min());

            // triangle area
            const T area = T(0.5) * glm::length(c);

            // centriod
            constexpr T third = T(1) / T(3);

            Vec3<T> center = (third * (v + p + r)) - v;

            sum_areas += area;

            t(0) += area * center[0];
            t(1) += area * center[1];
            t(2) += area * center[2];


            ph = rh;
            p  = r;
        }

        t = null_space_projection * t;
        t /= sum_areas;

        new_position(vh, 0) = v[0] + t(0);
        new_position(vh, 1) = v[1] + t(1);
        new_position(vh, 2) = v[2] + t(2);
    };

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, smooth, true);
}
