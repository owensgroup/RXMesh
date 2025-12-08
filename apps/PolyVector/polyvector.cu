#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_vector_problem.h"
#include "rxmesh/diff/newton_solver.h"

#include "read_raw_field.h"

using namespace rxmesh;
using T = float;

struct AlgoSetting
{
    T       w_smooth              = 1.0;
    const T w_smooth_decay        = 0.8;
    const T w_polycurl            = 100.0;
    const T w_polyquotient        = 10.0;
    const T w_close_unconstrained = 1e-3;
    const T w_close_constrained   = 100.0;
    const T w_barrier             = 0.1;
    const T s_barrier             = 0.9;
};

template <typename FAttrT>
int inline viz_field(RXMeshStatic& rx, const int N, FAttrT& attr)
{
    assert(N == 4);

    int M = N / 2;

    Eigen::MatrixX<T> vec0(attr.rows(), M);
    Eigen::MatrixX<T> vec1(attr.rows(), M);

    /*rx.for_each_face(HOST, [&](const FaceHandle fh) {
        int f_global_id = rx.map_to_global(fh);

        assert(f_global_id <= num_f);

        for (int j = 0; j < 3 * M; ++j) {

            attr(fh, j) = raw_field(f_global_id, j);
        }
    });*/
}


template <typename FAttrT, typename VAttrT>
void inline compute_local_basis(RXMeshStatic& rx,
                                const VAttrT& v,
                                FAttrT&       b1,
                                FAttrT&       b2)
{
    rx.run_query_kernel<Op::FV, 256>(
        [=] __device__(const FaceHandle& fh, const VertexIterator& vv) mutable {
            Eigen::Vector3<T> x0 = v.template to_eigen<3>(vv[0]);
            Eigen::Vector3<T> x1 = v.template to_eigen<3>(vv[1]);
            Eigen::Vector3<T> x2 = v.template to_eigen<3>(vv[2]);

            Eigen::Vector3<T> v1 = (x1 - x0).normalized();
            Eigen::Vector3<T> t  = x2 - x0;
            Eigen::Vector3<T> v3 = v1.cross(t).normalized();  // face normal
            Eigen::Vector3<T> v2 = v1.cross(v3).normalized();

            v2 = -v2;

            b1.from_eigen(fh, v1);


            b2.from_eigen(fh, v2);
        });
}

template <typename EAttrT, typename VAttrT>
void inline compute_per_edge_transport_term(RXMeshStatic& rx,
                                            const VAttrT& v,
                                            EAttrT&       e_f_conj,
                                            EAttrT&       e_g_conj,
                                            EAttrT&       t_fg_4,
                                            EAttrT&       t_fg_2)
{
    rx.run_query_kernel<Op::EVDiamond, 256>(
        [=] __device__(const EdgeHandle& eh, const VertexIterator& vv) mutable {
            // v0, v2 are from-to vertices
            // v1, v3 are the two opposite vertices

            // 1st triangle (f): v0, v1, v2
            // 2nd triangle (g): v0, v2, v3

            auto compute_basis = [&](const Eigen::Vector3<T>& x0,
                                     const Eigen::Vector3<T>& x1,
                                     const Eigen::Vector3<T>& x2,
                                     Eigen::Vector3<T>&       f_b1,
                                     Eigen::Vector3<T>&       f_b2) {
                Eigen::Vector3<T> v1 = (x1 - x0).normalized();
                Eigen::Vector3<T> t  = x2 - x0;
                Eigen::Vector3<T> v3 = v1.cross(t).normalized();  // face normal
                Eigen::Vector3<T> v2 = v1.cross(v3).normalized();
                v2                   = -v2;

                f_b1 = v1;
                f_b2 = v2;
            };

            Eigen::Vector3<T> x0 = v.template to_eigen<3>(vv[0]);
            Eigen::Vector3<T> x1 = v.template to_eigen<3>(vv[1]);
            Eigen::Vector3<T> x2 = v.template to_eigen<3>(vv[2]);
            Eigen::Vector3<T> x3 = v.template to_eigen<3>(vv[3]);

            // face g's basis
            Eigen::Vector3<T> f_b1;
            Eigen::Vector3<T> f_b2;
            compute_basis(x0, x1, x2, f_b1, f_b2);

            // face g's basis
            Eigen::Vector3<T> g_b1;
            Eigen::Vector3<T> g_b2;
            compute_basis(x0, x2, x3, f_b1, f_b2);


            Eigen::Vector3<T> e = (x2 - x0).normalized();


            cuComplex f_conj = make_cuComplex(f_b1.dot(e), f_b2.dot(e));
            cuComplex g_conj = make_cuComplex(g_b1.dot(e), g_b2.dot(e));

            e_f_conj(eh) = cuConjf(f_conj);
            e_g_conj(eh) = cuConjf(g_conj);

            cuComplex fg = cuCdivf(f_conj, g_conj);

            cuComplex fg_2 = cuCmulf(fg, fg);
            cuComplex fg_4 = cuCmulf(fg_2, fg_2);

            t_fg_2(eh) = fg_2;
            t_fg_4(eh) = fg_4;
        });
}


int main(int argc, char** argv)
{
    rx_init(0);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "cheburashka.obj");

    if (!rx.is_closed()) {
        RXMESH_ERROR("The input mesh is not closed mesh!");
        return EXIT_FAILURE;
    }

    if (!rx.is_edge_manifold()) {
        RXMESH_ERROR("The input mesh is not edge manifold!");
        return EXIT_FAILURE;
    }

    const int N = 4;

    AlgoSetting algo_settings;

    // input field
    auto x_init = *rx.add_face_attribute<T>("xInit", 3 * N);
    x_init.reset(0, LOCATION_ALL);

    if (read_raw_field(
            STRINGIFY(INPUT_DIR) "cheburashka.rawfield", rx, x_init) != N) {
        RXMESH_ERROR("Failed reading the input rawfield file!");
        return EXIT_FAILURE;
    }

    auto x = *rx.add_face_attribute<T>("x", 3 * N);
    x.copy_from(x_init, DEVICE, DEVICE);

    auto x_prev = *rx.add_face_attribute<T>("x_prev", 3 * N);
    x_prev.copy_from(x_init, DEVICE, DEVICE);

    auto x_constr = *rx.add_face_attribute<T>("x_constr", 3 * N);
    x_constr.copy_from(x_init, DEVICE, DEVICE);

    // soft constraints only face 0
    const FaceHandle constr_face(0, 0);

    // input coordinates
    auto v = *rx.get_input_vertex_coordinates();

    //// local basis
    // auto b1 = *rx.add_face_attribute<T>("B1", 3);
    // auto b2 = *rx.add_face_attribute<T>("B2", 3);

    // compute_local_basis(rx, v, b1, b2);
    // b1.move(DEVICE, HOST);
    // b2.move(DEVICE, HOST);


    // constant transport terms for polynomial coefficients per edge
    auto e_f_conj = *rx.add_edge_attribute<cuComplex>("e_f_conj", 1);
    auto e_g_conj = *rx.add_edge_attribute<cuComplex>("e_g_conj", 1);
    auto t_fg_4   = *rx.add_edge_attribute<cuComplex>("t_fg_4", 1);
    auto t_fg_2   = *rx.add_edge_attribute<cuComplex>("t_fg_2", 1);

    compute_per_edge_transport_term(rx, v, e_f_conj, e_g_conj, t_fg_4, t_fg_2);


#if USE_POLYSCOPE
    rx.get_polyscope_mesh()->addFaceVectorQuantity("xInit", x_init);
    // rx.get_polyscope_mesh()->addFaceVectorQuantity("B1", b1);
    // rx.get_polyscope_mesh()->addFaceVectorQuantity("B2", b2);
    polyscope::show();
#endif
}