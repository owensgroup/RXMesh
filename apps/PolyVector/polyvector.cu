// Integrable PolyVector Fields
// https://igl.ethz.ch/projects/integrable/integrable-polyvector-fields.pdf
// Major part of this code is taken from TinyAD

#include <CLI/CLI.hpp>
#include <cstdlib>

#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_vector_problem.h"
#include "rxmesh/diff/gauss_newton_solver.h"
#include "rxmesh/util/log.h"

#include "raw_field_io.h"

#include "thrust/complex.h"

#include <nvtx3/nvToolsExt.h>

using namespace rxmesh;
using T = float;

template <typename FAttrT, typename EAttrT>
void inline viz_curl(RXMeshStatic& rx,
                     std::string   name,
                     const FAttrT& x,
                     const EAttrT& e_f_conj,
                     const EAttrT& e_g_conj,
                     const FAttrT& b1,
                     const FAttrT& b2,
                     const double  curl_min = 1e-6,
                     const double  curl_max = 0.02)
{
    auto vcolor = *rx.add_vertex_attribute<T>(name + "v", 1);
    vcolor.reset(DEVICE, 0);

    auto ecurl = *rx.add_edge_attribute<T>(name + "e", 1);

    auto xa3d = *rx.add_face_attribute<T>(name + "_alpha", 3);
    auto xb3d = *rx.add_face_attribute<T>(name + "_beta", 3);

    auto ctx = rx.get_context();

    rx.run_query_kernel<Op::EF, 256>(
        [=] __device__(const EdgeHandle& eh, FaceIterator& iter) mutable {
            FaceHandle f(iter[0]), g(iter[1]);

            if (ctx.linear_id_fast(f) < ctx.linear_id_fast(g)) {
                f = iter[1];
                g = iter[0];
            }

            thrust::complex<T> alpha_f(x(f, 0), x(f, 1));
            thrust::complex<T> beta_f(x(f, 2), x(f, 3));

            thrust::complex<T> alpha_g(x(g, 0), x(g, 1));
            thrust::complex<T> beta_g(x(g, 2), x(g, 3));

            T af_ef = (alpha_f * e_f_conj(eh)).real();
            T ag_eg = (alpha_g * e_g_conj(eh)).real();
            T bf_ef = (beta_f * e_f_conj(eh)).real();
            T bg_eg = (beta_g * e_g_conj(eh)).real();
            T c_f_0 = sqr(af_ef) * sqr(bf_ef);
            T c_g_0 = sqr(ag_eg) * sqr(bg_eg);
            T c_f_2 = -(sqr(af_ef) + sqr(bf_ef));
            T c_g_2 = -(sqr(ag_eg) + sqr(bg_eg));

            ecurl(eh) = sqr(c_f_0 - c_g_0) + sqr(c_f_2 - c_g_2);
        });

    rx.run_query_kernel<Op::EV, 256>(
        [=] __device__(const EdgeHandle& eh, const VertexIterator& vv) mutable {
            T curl = ecurl(eh);

            ::atomicAdd(&vcolor(vv[0]), curl);

            ::atomicAdd(&vcolor(vv[1]), curl);
        });

    rx.run_query_kernel<Op::VV, 256>(
        [=] __device__(const VertexHandle&   vh,
                       const VertexIterator& vv) mutable {
            T curl = vcolor(vh);

            curl /= vv.size();

            // curl = (log(curl) - log(curl_min)) / (log(curl_max) -
            // log(curl_min));
            //
            // curl = min(max(curl, 0.0), 1.0);

            vcolor(vh) = curl;
        });

    rx.for_each_face(HOST, [&](const FaceHandle fh) {
        T ax = x(fh, 0);
        T ay = x(fh, 1);
        T bx = x(fh, 2);
        T by = x(fh, 3);

        Eigen::Vector3<T> f_b1 = b1.template to_eigen<3>(fh);
        Eigen::Vector3<T> f_b2 = b2.template to_eigen<3>(fh);

        Eigen::Vector3<T> a3 = ax * f_b1 + ay * f_b2;
        Eigen::Vector3<T> b3 = bx * f_b1 + by * f_b2;

        xa3d.from_eigen(fh, a3);

        xb3d.from_eigen(fh, b3);
    });

    vcolor.move(DEVICE, HOST);

    rx.get_polyscope_mesh()
        ->addVertexScalarQuantity(name, vcolor)
        ->setMapRange({curl_min, curl_max});

    rx.get_polyscope_mesh()->addFaceVectorQuantity(name + "_alpha", xa3d);
    rx.get_polyscope_mesh()->addFaceVectorQuantity(name + "_beta", xb3d);

    rx.remove_attribute(name + "v");
    rx.remove_attribute(name + "e");
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

template <typename EAttrT, typename VAttrT, typename FAttrT>
void inline compute_per_edge_transport_term(RXMeshStatic& rx,
                                            const VAttrT& v,
                                            const FAttrT& b1,
                                            const FAttrT& b2,
                                            EAttrT&       e_f_conj,
                                            EAttrT&       e_g_conj)
{
    // compute the edge normalized edge length and (hijack) store it in e_f_conj
    // and e_g_conj, so we can use it in the next step
    rx.run_query_kernel<Op::EV, 256>(
        [=] __device__(const EdgeHandle& eh, const VertexIterator& vv) mutable {
            Eigen::Vector3<T> x0 = v.template to_eigen<3>(vv[0]);
            Eigen::Vector3<T> x2 = v.template to_eigen<3>(vv[1]);

            Eigen::Vector3<T> e = (x2 - x0).normalized();
            e_f_conj(eh)        = thrust::complex<T>(e[0], e[1]);
            e_g_conj(eh)        = thrust::complex<T>(e[2], 0);
        });

    auto ctx = rx.get_context();

    rx.run_query_kernel<Op::EF, 256>(
        [=] __device__(const EdgeHandle& eh, FaceIterator& iter) mutable {
            FaceHandle f(iter[0]), g(iter[1]);

            if (ctx.linear_id_fast(f) < ctx.linear_id_fast(g)) {
                f = iter[1];
                g = iter[0];
            }

            Eigen::Vector3<T> f_b1 = b1.template to_eigen<3>(f);
            Eigen::Vector3<T> f_b2 = b2.template to_eigen<3>(f);

            Eigen::Vector3<T> g_b1 = b1.template to_eigen<3>(g);
            Eigen::Vector3<T> g_b2 = b2.template to_eigen<3>(g);

            Eigen::Vector3<T> e(
                e_f_conj(eh).real(), e_f_conj(eh).imag(), e_g_conj(eh).real());

            e_f_conj(eh) =
                thrust::conj(thrust::complex<T>(f_b1.dot(e), f_b2.dot(e)));

            e_g_conj(eh) =
                thrust::conj(thrust::complex<T>(g_b1.dot(e), g_b2.dot(e)));
        });

    e_f_conj.move(DEVICE, HOST);
    e_g_conj.move(DEVICE, HOST);
}

int main(int argc, char** argv)
{
    CLI::App app{"PolyVector - Integrable PolyVector Fields"};

    std::string base_mesh_name = STRINGIFY(INPUT_DIR) "cheburashka";
    int         max_iters      = 60;
    int         cg_max_iters   = 50;
    int         device_id      = 0;

    app.add_option("-i,--input",
                   base_mesh_name,
                   "Base mesh name (without .obj extension)")
        ->default_val(std::string(STRINGIFY(INPUT_DIR) "cheburashka"));

    app.add_option("-m,--max_iters", max_iters, "Maximum number of iterations")
        ->default_val(60);

    app.add_option(
           "--cg_max_iters", cg_max_iters, "Maximum number of CG iterations")
        ->default_val(50);

    app.add_option("-d,--device_id", device_id, "GPU device ID")
        ->default_val(0);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    rx_init(device_id);

    RXMESH_INFO("base_mesh_name= {}", base_mesh_name);
    RXMESH_INFO("max_iters= {}", max_iters);
    RXMESH_INFO("cg_max_iters= {}", cg_max_iters);
    RXMESH_INFO("device_id= {}", device_id);

    // cheburashka
    RXMeshStatic rx(base_mesh_name + ".obj");

    if (!rx.is_closed()) {
        RXMESH_ERROR("The input mesh is not closed mesh!");
        return EXIT_FAILURE;
    }

    if (!rx.is_edge_manifold()) {
        RXMESH_ERROR("The input mesh is not edge manifold!");
        return EXIT_FAILURE;
    }

    constexpr int N = 4;

    const T w_smooth_decay             = 0.8;
    const T w_polycurl                 = 100.0;
    const T w_polyquotient             = 10.0;
    const T w_close_unconstrained_sqrt = std::sqrt(1e-3);
    const T w_close_constrained_sqrt   = 10.0;
    const T w_barrier                  = 0.1;
    const T s_barrier                  = 0.9;
    T       step_size                  = 0.1;


    DenseMatrix<T> w_smooth(1, 1, LOCATION_ALL);
    w_smooth(0) = 1.0;
    w_smooth.move(HOST, DEVICE);

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

    if (read_raw_field(base_mesh_name + ".rawfield", rx, x_init, b1, b2) != N) {
        RXMESH_ERROR("Failed reading the input rawfield file!");
        return EXIT_FAILURE;
    }

    auto x_prev = *rx.add_face_attribute<T>("x_prev", N);
    x_prev.copy_from(x_init, DEVICE, DEVICE);

    auto x_constr = *rx.add_face_attribute<T>("x_constr", N);
    x_constr.copy_from(x_init, DEVICE, DEVICE);

    using ProblemT = DiffVectorProblem<T, N, FaceHandle>;
    ProblemT problem(rx);

    // used in line search
    auto x_new = *rx.add_attribute_like("x_new", *problem.objective);

#ifdef USE_CUDSS
    using SolverT =
        cuDSSCholeskySolver<SparseMatrix<T>, ProblemT::DenseMatT::OrderT>;

    // using SolverT = CGSolver<T, ProblemT::DenseMatT::OrderT>;
#else
    using SolverT =
        CholeskySolver<SparseMatrix<T>, ProblemT::DenseMatT::OrderT>;
#endif

    GaussNetwtonSolver<T, N, FaceHandle, SolverT> solver(problem);

    // constant transport terms for polynomial coefficients per edge
    auto e_f_conj = *rx.add_edge_attribute<thrust::complex<T>>("e_f_conj", 1);
    auto e_g_conj = *rx.add_edge_attribute<thrust::complex<T>>("e_g_conj", 1);

    compute_per_edge_transport_term(rx, v, b1, b2, e_f_conj, e_g_conj);

    auto ctx = rx.get_context();

    problem.add_term<Op::EF, 7>(
        [=] __device__(const auto& eh, const auto& iter, auto& objective) {
            assert(iter.size() == 2);

            using ActiveT = ACTIVE_TYPE(eh);

            Eigen::Vector<ActiveT, 7> res;

            // face f is the one with highest id (to make it consistent with
            //  how we calculated the basis)
            //
            //  4 variables in face f and g
            Eigen::Vector4<ActiveT> vars_f, vars_g;
            if (ctx.linear_id_fast(iter[0]) > ctx.linear_id_fast(iter[1])) {
                vars_f = iter_val<ActiveT, 4>(eh, iter, objective, 0);
                vars_g = iter_val<ActiveT, 4>(eh, iter, objective, 1);
            } else {
                vars_f = iter_val<ActiveT, 4>(eh, iter, objective, 1);
                vars_g = iter_val<ActiveT, 4>(eh, iter, objective, 0);
            }

            thrust::complex<T> f_conj = e_f_conj(eh);
            thrust::complex<T> g_conj = e_g_conj(eh);

            thrust::complex<T> t_fg = f_conj / g_conj;

            thrust::complex<T> fg_2 = sqr(t_fg);
            thrust::complex<T> fg_4 = sqr(fg_2);


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


            thrust::complex<ActiveT> C_0_residual = C_f_0 * fg_4 - C_g_0;
            thrust::complex<ActiveT> C_2_residual = C_f_2 * fg_2 - C_g_2;

            T w_smooth_sqrt = sqrt(w_smooth(0));

            res[0] = w_smooth_sqrt * C_0_residual.real();
            res[1] = w_smooth_sqrt * C_0_residual.imag();
            res[2] = w_smooth_sqrt * C_2_residual.real();
            res[3] = w_smooth_sqrt * C_2_residual.imag();


            // PolyCurl term:
            // Compare real coefficients of polycurl polynomial across edge
            // [Diamanti 2015, Eq. 11, 19]

            ActiveT af_ef = (alpha_f * f_conj).real();
            ActiveT ag_eg = (alpha_g * g_conj).real();
            ActiveT bf_ef = (beta_f * f_conj).real();
            ActiveT bg_eg = (beta_g * g_conj).real();
            ActiveT c_f_0 = sqr(af_ef) * sqr(bf_ef);
            ActiveT c_g_0 = sqr(ag_eg) * sqr(bg_eg);
            ActiveT c_f_2 = -(sqr(af_ef) + sqr(bf_ef));
            ActiveT c_g_2 = -(sqr(ag_eg) + sqr(bg_eg));
            res[4]        = w_polycurl * (c_f_0 - c_g_0);
            res[5]        = sqrt(w_polycurl) * (c_f_2 - c_g_2);

            // PolyQuotient term:
            // Compare real coefficients of polyquotient terms across edge
            // [Diamanti 2015, Eq. 21] There are typos in Eq. 14 and 21, where +
            // should be -.
            ActiveT q1 = (sqr(bf_ef) - sqr(af_ef)) * ag_eg * bg_eg;
            ActiveT q2 = (sqr(bg_eg) - sqr(ag_eg)) * af_ef * bf_ef;
            res[6]     = sqrt(w_polyquotient) * (q1 - q2);

            return res;
        });


    problem.add_term<Op::F, 5>([=] __device__(const auto& fh, auto& objective) {
        using ActiveT = ACTIVE_TYPE(fh);

        uint32_t id = ctx.linear_id_fast(fh);

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


        T w_close_sqrt = 0;
        if (id == 0) {
            w_close_sqrt = w_close_constrained_sqrt;
        } else {
            w_close_sqrt = w_close_unconstrained_sqrt;
        }


        ActiveT alpha_real = alpha.real() - alpha_ref.real();
        ActiveT alpha_imag = alpha.imag() - alpha_ref.imag();
        ActiveT beta_real  = beta.real() - beta_ref.real();
        ActiveT beta_imag  = beta.imag() - beta_ref.imag();

        res[0] = w_close_sqrt * alpha_real;
        res[1] = w_close_sqrt * alpha_imag;
        res[2] = w_close_sqrt * beta_real;
        res[3] = w_close_sqrt * beta_imag;


        // Barrier term:
        // Ensure convex angle between alpha and beta
        ActiveT barrier_x = (beta * thrust::conj(alpha)).imag();

        ActiveT barrier = 0.0;

        if (barrier_x <= 0.0) {
            using PassiveT = PassiveType<ActiveT>;

            barrier = ActiveT(std::numeric_limits<PassiveT>::max());
        } else if (barrier_x < s_barrier) {
            ActiveT b =
                barrier_x * sqr(barrier_x) / (s_barrier * sqr(s_barrier)) -
                3.0 * sqr(barrier_x) / sqr(s_barrier) +
                3.0 * barrier_x / s_barrier;
            barrier = 1.0 / b - 1.0;
        }
        res[4] = sqrt(w_barrier) * barrier;

        return res;
    });

    problem.objective->copy_from(x_init, LOCATION_ALL, LOCATION_ALL);

    Timers<GPUTimer> timer;
    timer.add("Total");
    timer.add("Diff");
    timer.add("LineSearch");
    timer.add("LinearSolver");
    timer.add("problem.prep_eval");
    timer.add("solver.prep_solver");

    // nvtxRangePushA("problem.prep_eval");
    //  nvtxRangePop();
    timer.start("problem.prep_eval");
    problem.prep_eval();
    timer.stop("problem.prep_eval");


    timer.start("solver.prep_solver");
    solver.prep_solver(cg_max_iters);
    timer.stop("solver.prep_solver");

    timer.start("Total");
    for (int iter = 0; iter < max_iters; ++iter) {
        // Compute derivatives and Gauss-Newton direction

        // compute g = -J^T r
        // the -1.0 is used here so we don't need to scale things again
        // in the gauss-newton which solves (J^T J).dir = -J^T r
        timer.start("Diff");
        problem.eval_terms_sum_of_squares(-1.0);
        timer.stop("Diff");

        T f = problem.get_current_loss();
        RXMESH_INFO("Iteration: {}, Energy: {}", iter, f);

        // compute new direction
        timer.start("LinearSolver");
        solver.compute_direction();
        timer.stop("LinearSolver");


        timer.start("LineSearch");
        // Line search
        while (true) {
            // take step
            rx.for_each_face(
                DEVICE,
                [x_new   = x_new,
                 obj     = *problem.objective,
                 dir     = solver.dir,
                 sz      = step_size,
                 n_faces = rx.get_num_faces()] __device__(const FaceHandle
                                                              fh) mutable {
                    dir.reshape(n_faces, N);
                    for (int i = 0; i < N; ++i) {
                        x_new(fh, i) = obj(fh, i) + sz * dir(fh, i);
                    }
                });

            // eval current energy
            problem.eval_terms_passive(&x_new);

            T f_new = problem.get_current_loss();

            if (f_new < f) {
                // Line search success. Increase step size in next iteration.
                step_size *= 2.0;
                break;
            } else {
                // Decrease step size and try again.
                step_size *= 0.5;
                if (step_size < 1e-10) {
                    RXMESH_ERROR("Line search failed");
                }
            }
        }

        rx.for_each_face(DEVICE,
                         [x0  = x_prev,
                          obj = *problem.objective,
                          x1  = x_new,
                          N   = N] __device__(const FaceHandle fh) mutable {
                             for (int i = 0; i < N; ++i) {
                                 x0(fh, i)  = obj(fh, i);
                                 obj(fh, i) = x1(fh, i);
                             }
                         });

        timer.stop("LineSearch");

        // x_prev.copy_from(*problem.objective, DEVICE, DEVICE);
        // problem.objective->copy_from(x_new, DEVICE, DEVICE);


        // Decay smoothness term after every 5 iters
        if ((iter + 1) % 5 == 0) {
            w_smooth.multiply(w_smooth_decay);
        }
    }
    timer.stop("Total");

    RXMESH_INFO("PolyVector: iterations= {}, time={} (ms), time/iter (ms) {}",
                max_iters,
                timer.elapsed_millis("Total"),
                timer.elapsed_millis("Total") / max_iters);

    RXMESH_INFO("problem.prep_eval (ms)= {}, solver.prep_solvers (ms) = {}",
                timer.elapsed_millis("problem.prep_eval"),
                timer.elapsed_millis("solver.prep_solver"));

    RXMESH_INFO("Diff (ms)= {}, Line Search (ms) = {}, Linear Solver (ms) = {}",
                timer.elapsed_millis("Diff"),
                timer.elapsed_millis("LineSearch"),
                timer.elapsed_millis("LinearSolver"));

    RXMESH_INFO(
        "Diff/iter (ms)= {}, Line Search/iter (ms) = {}, Linear Solver/iter "
        "(ms) = {}",
        timer.elapsed_millis("Diff") / max_iters,
        timer.elapsed_millis("LineSearch") / max_iters,
        timer.elapsed_millis("LinearSolver") / max_iters);

#if USE_POLYSCOPE
    problem.objective->move(DEVICE, HOST);

    viz_curl(rx, "xInit", x_init, e_f_conj, e_g_conj, b1, b2);
    viz_curl(rx, "x", *problem.objective, e_f_conj, e_g_conj, b1, b2);

    rx.get_polyscope_mesh()->addFaceVectorQuantity("B1", b1);
    rx.get_polyscope_mesh()->addFaceVectorQuantity("B2", b2);
    polyscope::show();
#endif


    // std::string mesh_file_name =
    //     base_mesh_name.substr(base_mesh_name.find_last_of("/\\") + 1);
    //
    // rx.export_obj("rx_" + mesh_file_name + ".obj", v);
    // Eigen::write_text("rx_" + mesh_file_name + "_x_init.dat",
    //                   x_init.to_matrix()->to_eigen_copy());
    // Eigen::write_text("rx_" + mesh_file_name + "_b1.dat",
    //                   b1.to_matrix()->to_eigen_copy());
    // Eigen::write_text("rx_" + mesh_file_name + "_b2.dat",
    //                   b2.to_matrix()->to_eigen_copy());
    // Eigen::write_text("rx_" + mesh_file_name + "_x.dat",
    //                   problem.objective->to_matrix()->to_eigen_copy());


    solver.release();
    w_smooth.release();

    return 0;
}