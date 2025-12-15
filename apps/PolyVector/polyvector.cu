#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_vector_problem.h"
#include "rxmesh/diff/newton_solver.h"

#include "read_raw_field.h"

#include "thrust/complex.h"

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

    // cheburashka
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "cheburashka.obj");

    if (!rx.is_closed()) {
        RXMESH_ERROR("The input mesh is not closed mesh!");
        return EXIT_FAILURE;
    }

    if (!rx.is_edge_manifold()) {
        RXMESH_ERROR("The input mesh is not edge manifold!");
        return EXIT_FAILURE;
    }

    constexpr int N = 4;

    AlgoSetting algo_settings;

    // input coordinates
    auto v = *rx.get_input_vertex_coordinates();

    // local basis
    auto b1 = *rx.add_face_attribute<T>("B1", 3);
    auto b2 = *rx.add_face_attribute<T>("B2", 3);

    compute_local_basis(rx, v, b1, b2);
    b1.move(DEVICE, HOST);
    b2.move(DEVICE, HOST);

    // input field
    auto x_init = *rx.add_face_attribute<T>("xInit", N);
    x_init.reset(0, LOCATION_ALL);

    if (read_raw_field(
            STRINGIFY(INPUT_DIR) "cheburashka.rawfield", rx, x_init, b1, b2) !=
        N) {
        RXMESH_ERROR("Failed reading the input rawfield file!");
        return EXIT_FAILURE;
    }

    auto x = *rx.add_face_attribute<T>("x", N);
    x.copy_from(x_init, DEVICE, DEVICE);

    auto x_prev = *rx.add_face_attribute<T>("x_prev", N);
    x_prev.copy_from(x_init, DEVICE, DEVICE);

    auto x_constr = *rx.add_face_attribute<T>("x_constr", N);
    x_constr.copy_from(x_init, DEVICE, DEVICE);

    // soft constraints only face 0
    const FaceHandle constr_face(0, 0);


    // constant transport terms for polynomial coefficients per edge
    // TODO replace cuComplex with thrust::complex
    auto e_f_conj = *rx.add_edge_attribute<cuComplex>("e_f_conj", 1);
    auto e_g_conj = *rx.add_edge_attribute<cuComplex>("e_g_conj", 1);
    auto t_fg_4   = *rx.add_edge_attribute<cuComplex>("t_fg_4", 1);
    auto t_fg_2   = *rx.add_edge_attribute<cuComplex>("t_fg_2", 1);

    compute_per_edge_transport_term(rx, v, e_f_conj, e_g_conj, t_fg_4, t_fg_2);


    using ProblemT = DiffVectorProblem<T, N, FaceHandle>;

    ProblemT problem(rx);

    problem.add_term<Op::EF, 7>(
        [=] __device__(const auto& eh, const auto& iter, auto& objective) {
            assert(iter.size() == 2);

            using ActiveT = ACTIVE_TYPE(eh);

            Eigen::Vector<ActiveT, 7> res;

            // two faces incident to the edge eh
            FaceHandle f_idx = iter[0];
            FaceHandle g_idx = iter[1];

            // 4 variables in face f
            Eigen::Vector4<ActiveT> vars_f =
                iter_val<ActiveT, 4>(eh, iter, objective, 0);

            // 4 variables in face g
            Eigen::Vector4<ActiveT> vars_g =
                iter_val<ActiveT, 4>(eh, iter, objective, 1);


            thrust::complex<ActiveT> alpha_f(vars_f[0], vars_f[1]);
            thrust::complex<ActiveT> beta_f(vars_f[2], vars_f[3]);
            thrust::complex<ActiveT> alpha_g(vars_g[0], vars_g[1]);
            thrust::complex<ActiveT> beta_g(vars_g[2], vars_g[3]);

            // Smoothness term:
            // Compare complex coefficients of smoothness polynomial across edge
            // [Diamanti 2015, Eq. 17, 18]
            thrust::complex<ActiveT> C_f_0 = sqr(alpha_f) * sqr(beta_f);
            thrust::complex<ActiveT> C_g_0 = sqr(alpha_g) * sqr(beta_g);
            thrust::complex<ActiveT> C_f_2 = -(sqr(alpha_f) + sqr(beta_f));
            thrust::complex<ActiveT> C_g_2 = -(sqr(alpha_g) + sqr(beta_g));

            thrust::complex<ActiveT> fg_4(t_fg_4(eh).x, t_fg_4(eh).y);
            thrust::complex<ActiveT> fg_2(t_fg_2(eh).x, t_fg_2(eh).y);

            thrust::complex<ActiveT> C_0_residual = C_f_0 * fg_4 - C_g_0;
            thrust::complex<ActiveT> C_2_residual = C_f_2 * fg_2 - C_g_2;

            T w_smooth_sqrt = sqrt(algo_settings.w_smooth);

            res[0] = w_smooth_sqrt * C_0_residual.real();
            res[1] = w_smooth_sqrt * C_0_residual.imag();
            res[2] = w_smooth_sqrt * C_2_residual.real();
            res[3] = w_smooth_sqrt * C_2_residual.imag();


            // PolyCurl term:
            // Compare real coefficients of polycurl polynomial across edge
            // [Diamanti 2015, Eq. 11, 19]
            thrust::complex<ActiveT> f_conj(e_f_conj(eh).x, e_f_conj(eh).y);
            thrust::complex<ActiveT> g_conj(e_g_conj(eh).x, e_g_conj(eh).y);

            ActiveT af_ef = (alpha_f * f_conj).real();
            ActiveT ag_eg = (alpha_g * g_conj).real();
            ActiveT bf_ef = (beta_f * f_conj).real();
            ActiveT bg_eg = (beta_g * g_conj).real();
            ActiveT c_f_0 = sqr(af_ef) * sqr(bf_ef);
            ActiveT c_g_0 = sqr(ag_eg) * sqr(bg_eg);
            ActiveT c_f_2 = -(sqr(af_ef) + sqr(bf_ef));
            ActiveT c_g_2 = -(sqr(ag_eg) + sqr(bg_eg));
            res[4]        = algo_settings.w_polycurl * (c_f_0 - c_g_0);
            res[5]        = sqrt(algo_settings.w_polycurl) * (c_f_2 - c_g_2);

            // PolyQuotient term:
            // Compare real coefficients of polyquotient terms across edge
            // [Diamanti 2015, Eq. 21] There are typos in Eq. 14 and 21, where +
            // should be -.
            ActiveT q1 = (sqr(bf_ef) - sqr(af_ef)) * ag_eg * bg_eg;
            ActiveT q2 = (sqr(bg_eg) - sqr(ag_eg)) * af_ef * bf_ef;
            res[6]     = sqrt(algo_settings.w_polyquotient) * (q1 - q2);

            return res;
        });


    auto ctx = rx.get_context();

    problem.add_term<Op::F, 5>([=] __device__(const auto& fh, auto& objective) {
        using ActiveT = ACTIVE_TYPE(fh);

        uint32_t id = ctx.linear_id(fh);

        Eigen::Vector<ActiveT, 5> res;

        // Get 2D vectors (alpha, beta) in local basis of face f
        Eigen::Vector4<ActiveT> vars = iter_val<ActiveT, 4>(fh, objective);

        thrust::complex<ActiveT> alpha(vars[0], vars[1]);
        thrust::complex<ActiveT> beta(vars[2], vars[3]);

        // Closeness term:
        // Either soft penalty towards constraint or towards previous iteration.
        // Get reference vectors (alpha_ref, beta_ref).
        // Either from x_constr or from x_prev.
        Eigen::Matrix<T, N, 1> x_ref;
        if (id == 0) {
            x_ref = x_constr.template to_eigen<N>(fh);
        } else {
            x_ref = x_prev.template to_eigen<N>(fh);
        }

        thrust::complex<T> alpha_ref(x_ref[0], x_ref[1]);

        thrust::complex<T> beta_ref(x_ref[2], x_ref[3]);

        T w_close_sqrt;
        if (id == 0) {
            T w_close_sqrt = sqrt(algo_settings.w_close_constrained);
        } else {
            T w_close_sqrt = sqrt(algo_settings.w_close_unconstrained);
        }

        res[0] = w_close_sqrt * (alpha.real() - alpha_ref.real());
        res[1] = w_close_sqrt * (alpha.imag() - alpha_ref.imag());
        res[2] = w_close_sqrt * (beta.real() - beta_ref.real());
        res[3] = w_close_sqrt * (beta.imag() - beta_ref.imag());


        // Barrier term:
        // Ensure convex angle between alpha and beta
        ActiveT barrier_x = (beta * thrust::conj(alpha)).imag();

        ActiveT barrier = 0.0;

        if (barrier_x <= 0.0) {
            using PassiveT = PassiveType<ActiveT>;

            barrier = ActiveT(std::numeric_limits<PassiveT>::max());
        } else if (barrier_x < algo_settings.s_barrier) {
            ActiveT b =
                barrier_x * sqr(barrier_x) /
                    (algo_settings.s_barrier * sqr(algo_settings.s_barrier)) -
                3.0 * sqr(barrier_x) / sqr(algo_settings.s_barrier) +
                3.0 * barrier_x / algo_settings.s_barrier;
            barrier = 1.0 / b - 1.0;
        }
        res[4] = sqrt(algo_settings.w_barrier) * barrier;

        return res;
    });


    problem.prep_eval();

    // problem.jac->to_file("jj");

    problem.eval_terms();

#if USE_POLYSCOPE
    rx.get_polyscope_mesh()->addFaceVectorQuantity2D("xInit", x_init);
    // rx.get_polyscope_mesh()->addFaceVectorQuantity("B1", b1);
    // rx.get_polyscope_mesh()->addFaceVectorQuantity("B2", b2);
    polyscope::show();
#endif
}