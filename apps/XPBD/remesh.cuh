#pragma once

#include "rxmesh/query.cuh"

#include "svd.cuh"

template <typename T = float>
using vec2 = glm::vec<2, T, glm::defaultp>;

template <typename T = float>
using mat2x2 = glm::mat<2, 2, T, glm::defaultp>;

template <typename T = float>
using mat3x2 = glm::mat<3, 2, T, glm::defaultp>;

template <typename T = float>
using mat2x3 = glm::mat<2, 3, T, glm::defaultp>;

template <typename T>
__device__ vec3<T> normal(const vec3<T>& x0,
                          const vec3<T>& x1,
                          const vec3<T>& x2)
{
    return glm::normalize(glm::cross(x1 - x0, x2 - x0));
}

template <typename T>
mat3x3<T> local_base(const vec3<T>& n)
{
    vec3<T> u =
        (glm::dot(n, vec3<T>(1, 0, 0)) > glm::dot(n, vec3<T>(0, 0, 1))) ?
            vec3<T>(0, 0, 1) :
            vec3<T>(1, 0, 0);

    u = glm::normalize(u - glm::dot(u, n) * n);

    vec3<T> v = glm::cross(n, u);

    return mat3x3<T>(u, v, n);
}


template <typename T>
vec2<T> perp(const vec2<T>& u)
{
    return vec2<T>(-u[1], u[0]);
}

template <typename T>
mat2x2<T> projected_curvature(const vec3<T>&   m0,
                              const vec3<T>&   m1,
                              const vec3<T>&   m2,
                              const mat2x2<T>& base,
                              const T&         area)
{
    mat2x2<T> S;

    for (int e = 0; e < 3; e++) {
        vec2<T> e_mat;
        if (e == 0) {
            e_mat = base * (m2 - m1);
        } else if (e == 1) {
            e_mat = base * (m0 - m2);
        } else if (e == 2) {
            e_mat = base * (m1 - m0);
        }

        vec2<T> t_mat = perp(glm::normalize(e_mat));

        // TODO compute dihedral angle using world coordinates
        //
        //  double theta = dihedral_angle<s>(face->adje[e]);
        T theta = 0;


        S -=
            0.5f * theta * glm::length(e_mat) * glm::outerProduct(t_mat, t_mat);
    }
    S /= area;
    return S;
}


template <uint32_t blockThreads, typename T>
void __global__ compute_face_sizing(const Context      context,
                                    VertexAttribute<T> w_coord,
                                    VertexAttribute<T> m_coord,
                                    VertexAttribute<T> v_normal,
                                    FaceAttribute<T>   face_sizing,
                                    const T            remeshing_size_min)
{

    auto calc_sizing = [&](const FaceHandle& fh, const VertexIterator& iter) {
        const VertexHandle v0 = iter[0];
        const VertexHandle v1 = iter[1];
        const VertexHandle v2 = iter[2];

        // material space
        const vec3<T> m0(m_coord(v0, 0), m_coord(v0, 1), m_coord(v0, 2));
        const vec3<T> m1(m_coord(v1, 0), m_coord(v1, 1), m_coord(v1, 2));
        const vec3<T> m2(m_coord(v2, 0), m_coord(v2, 1), m_coord(v2, 2));

        // world space
        const vec3<T> w0(w_coord(v0, 0), w_coord(v0, 1), w_coord(v0, 2));
        const vec3<T> w1(w_coord(v1, 0), w_coord(v1, 1), w_coord(v1, 2));
        const vec3<T> w2(w_coord(v2, 0), w_coord(v2, 1), w_coord(v2, 2));

        // local normal
        const vec3<T> fn = normal(m0, m1, m2);

        const vec3<T> d0 = m1 - m0;
        const vec3<T> d1 = m2 - m0;
        const vec3<T> d2 = glm::cross(d0, d1);

        // compute_ms_data
        //
        //  face area
        const T area = 0.5 * glm::length(d2);

        mat3x3<T> Dm, invDm;  // finite element matrix

        mat3x3<T> Dm3(d0, d1, d2 / (2.f * area));

        // TODO
        //  face->m = face->material ? face->a * face->material->density : 0;

        if (area < std::numeric_limits<T>::epsilon()) {
            invDm = mat3x3<T>(0);
        } else {
            invDm = glm::inverse(Dm3);

            // clamp
            const T clamp = 1000.f / remeshing_size_min;
            SVD<T>  svd   = singular_value_decomposition(invDm);
            for (int i = 0; i < 3; i++) {
                if (svd.S[i][i] > clamp) {
                    svd.S[i][i] = clamp;
                }
            }
            invDm = svd.U * svd.S * glm::transpose(svd.V);
        }


        const mat3x3<T> base = local_base(fn);

        const mat3x2<T> UV(base.col(0), base.col(1));

        const mat2x3<T> UVt = UV.t();


        mat2x2<T> sw1 = projected_curvature(m0, m1, m2, UVt, area);

        //mat3x3<T> sw2 =
        //    derivative(v_normal(v0), v_normal(v1), v_normal(v2), vec3<T>(0), face);

        mat3x3<T> f_sizing;
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, calc_sizing);
}


template <uint32_t blockThreads, typename T>
void __global__ compute_vertex_normal(const Context      context,
                                      VertexAttribute<T> w_coord,
                                      VertexAttribute<T> v_normal)
{
    // TODO make sure to init v_normal with zero

    auto calc_vn = [&](const FaceHandle& f, const VertexIterator& v) {
        const vec3<T> w0(w_coord(v[0], 0), w_coord(v[0], 1), w_coord(v[0], 2));
        const vec3<T> w1(w_coord(v[1], 0), w_coord(v[1], 1), w_coord(v[1], 2));
        const vec3<T> w2(w_coord(v[2], 0), w_coord(v[2], 1), w_coord(v[2], 2));

        const vec3<T> e1 = w1 - w0;
        const vec3<T> e2 = w2 - w0;

        vec3<T> n =
            glm::cross(e1, e2) / (2 * glm::length2(e1) * glm::length2(e2));

        for (int i = 0; i < 3; ++i) {  // v
            for (int j = 0; j < 3; ++j) {
                ::atomicAdd(&w_coord(v[i], j), n[j]);
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, calc_vn);
}
