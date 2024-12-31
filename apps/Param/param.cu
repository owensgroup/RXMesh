#include "rxmesh/matrix/sparse_matrix.cuh"

#include "rxmesh/diff/hessian_sparse_matrix.h"
#include "rxmesh/diff/scalar.h"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/algo/tutte_embedding.h"
#include "rxmesh/diff/hessian_projection.h"
#include "rxmesh/util/inverse.h"


#include <Eigen/SparseLU>

using namespace rxmesh;

template <typename T, int VariableDim, int blockThreads, bool Active>
__global__ static void calc_param_loss(
    const Context                               context,
    const FaceAttribute<Eigen::Matrix<T, 2, 2>> rest_shape,
    const VertexAttribute<T>                    uv,
    DenseMatrix<T, Eigen::RowMajor>             grad,
    HessianSparseMatrix<T, VariableDim>         hess,
    FaceAttribute<T>                            f_obj_func,
    bool                                        project_hessian)
{
    using namespace rxmesh;

    constexpr int ElementValence = 3;
    constexpr int NElements      = VariableDim * ElementValence;
    using ScalarT                = Scalar<T, NElements, true>;
    using ActiveT                = std::conditional_t<Active, ScalarT, T>;

    auto func = [&](const FaceHandle& fh, const VertexIterator& iter) {
        assert(iter[0].is_valid() && iter[1].is_valid() && iter[2].is_valid());

        // uv
        Eigen::Vector2<ActiveT> a;
        Eigen::Vector2<ActiveT> b;
        Eigen::Vector2<ActiveT> c;


        if constexpr (Active) {
            a[0].val     = uv(iter[0], 0);
            a[0].grad[0] = 1;

            a[1].val     = uv(iter[0], 1);
            a[1].grad[1] = 1;

            b[0].val     = uv(iter[1], 0);
            b[0].grad[2] = 1;

            b[1].val     = uv(iter[1], 1);
            b[1].grad[3] = 1;

            c[0].val     = uv(iter[2], 0);
            c[0].grad[4] = 1;

            c[1].val     = uv(iter[2], 1);
            c[1].grad[5] = 1;
        } else {
            a[0] = uv(iter[0], 0);
            a[1] = uv(iter[0], 1);
            b[0] = uv(iter[1], 0);
            b[1] = uv(iter[1], 1);
            c[0] = uv(iter[2], 0);
            c[1] = uv(iter[2], 1);
        }

        // Triangle flipped?
        Eigen::Matrix<ActiveT, 2, 2> M = col_mat(b - a, c - a);

        if (M.determinant() <= 0.0) {
            // assert(false);
            //  TODO
            //  return (ScalarT)INFINITY;
        }

        // Get constant 2D rest shape and area of triangle t
        const Eigen::Matrix<T, 2, 2> Mr = rest_shape(fh);

        const T A = T(0.5) * Mr.determinant();

        // Compute symmetric Dirichlet energy
        Eigen::Matrix<ActiveT, 2, 2> J = M * Mr.inverse();

        ActiveT res = A * (J.squaredNorm() + J.inverse().squaredNorm());


        if constexpr (Active) {
            // function
            f_obj_func(fh) = res.val;

            // gradient
            for (uint16_t vertex = 0; vertex < iter.size(); ++vertex) {
                for (int local = 0; local < VariableDim; ++local) {
                    ::atomicAdd(
                        &grad(iter[vertex], local),
                        res.grad[index_mapping<VariableDim>(vertex, local)]);
                }
            }

            // project hessian to PD matrix
            if (project_hessian) {
                project_positive_definite(res.Hess);
            }

            // hessian
            for (int vertex_i = 0; vertex_i < iter.size(); ++vertex_i) {
                const VertexHandle vi = iter[vertex_i];

                for (int vertex_j = 0; vertex_j < iter.size(); ++vertex_j) {
                    const VertexHandle vj = iter[vertex_j];

                    for (int local_i = 0; local_i < VariableDim; ++local_i) {

                        for (int local_j = 0; local_j < VariableDim;
                             ++local_j) {

                            ::atomicAdd(&hess(vi, vj, local_i, local_j),
                                        res.Hess(index_mapping<VariableDim>(
                                                     vertex_i, local_i),
                                                 index_mapping<VariableDim>(
                                                     vertex_j, local_j)));
                        }
                    }
                }
            }
        } else {
            f_obj_func(fh) = res;
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);

    ShmemAllocator shrd_alloc;

    query.dispatch<Op::FV>(block, shrd_alloc, func);
}

template <typename T>
bool armijo_condition(const T                                f_curr,
                      const T                                f_new,
                      const T                                s,
                      const DenseMatrix<T, Eigen::RowMajor>& dir,
                      const DenseMatrix<T, Eigen::RowMajor>& grad,
                      const T                                armijo_const)
{
    return f_new <= f_curr + armijo_const * s * dir.dot(grad);
}


template <int VariableDim, typename T>
void line_search(const RXMeshStatic&                          rx,
                 const T                                      current_f,
                 const DenseMatrix<T, Eigen::RowMajor>&       dir,
                 VertexAttribute<T>&                          sol,
                 VertexAttribute<T>&                          sol_temp,
                 DenseMatrix<T, Eigen::RowMajor>&             grad,
                 HessianSparseMatrix<T, VariableDim>&         hess,
                 const FaceAttribute<Eigen::Matrix<T, 2, 2>>& rest_shape,
                 FaceAttribute<T>&                            f_obj_func,
                 FaceReduceHandle<T>&                         reducer,
                 const T                                      s_max     = 1.0,
                 const T                                      shrink    = 0.8,
                 const int                                    max_iters = 64,
                 const T armijo_const                                   = 1e-4)
{
    // we are going to keep trying to update sol_temp until we reach solution
    // we are satisfied with, then we will copy it to sol. If no good solution
    // found, then sol will not be updated.
    using namespace rxmesh;

    assert(dir.rows() == grad.rows());
    assert(dir.cols() == grad.cols());
    assert(sol.rows() == grad.rows());
    assert(sol.cols() == grad.cols());
    assert(sol.rows() == sol_temp.rows());
    assert(sol.cols() == sol_temp.cols());

    assert(s_max > 0.0);


    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> lb;

    rx.prepare_launch_box(
        {Op::FV},
        lb,
        (void*)calc_param_loss<T, VariableDim, blockThreads, false>);

    const bool try_one = s_max > 1.0;

    T s = s_max;

    bool update = false;

    for (int i = 0; i < max_iters; ++i) {

        rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
            for (int j = 0; j < sol_temp.get_num_attributes(); ++j) {
                sol_temp(vh, j) = sol(vh, j) + s * dir(vh, j);
            }
        });

        rx.run_kernel(lb,
                      calc_param_loss<T, VariableDim, blockThreads, false>,
                      rest_shape,
                      sol_temp,
                      grad,
                      hess,
                      f_obj_func,
                      true);

        T f_new = reducer.reduce(f_obj_func, cub::Sum(), 0);


        if (armijo_condition(current_f, f_new, s, dir, grad, armijo_const)) {
            update = true;
            break;
        }

        if (try_one && s > 1.0 && s * shrink < 1.0) {
            s = 1.0;
        } else {
            s *= shrink;
        }
    }

    if (update) {
        rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
            for (int j = 0; j < sol_temp.get_num_attributes(); ++j) {
                sol(vh, j) = sol_temp(vh, j);
            }
        });
    }
}

int main(int argc, char** argv)
{
    Log::init();

    using T = float;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bunnyhead.obj");

    // dimension of the variable we want to optimize per vertex (i.e., uv)
    constexpr int VariableDim = 2;

    VertexAttribute<T> uv = *rx.add_vertex_attribute<T>("uv", VariableDim);

    VertexAttribute<T> uv_temp = *rx.add_vertex_attribute_like("uv_temp", uv);

    // auto grad = *rx.add_vertex_attribute<T>("uvGradPos", 2);

    DenseMatrix<T, Eigen::RowMajor> grad(rx, rx.get_num_vertices(), 2);

    DenseMatrix<T, Eigen::RowMajor> dir(rx, rx.get_num_vertices(), 2);
    dir.reset(0, LOCATION_ALL);


    auto coordinates = *rx.get_input_vertex_coordinates();

    // VertexAttribute<T> coordinates_t = *rx.add_vertex_attribute<T>("coords",
    // 3); rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
    //     coordinates_t(vh, 0) = coordinates(vh, 0);
    //     coordinates_t(vh, 1) = coordinates(vh, 1);
    //     coordinates_t(vh, 2) = coordinates(vh, 2);
    // });
    // coordinates_t.move(HOST, DEVICE);

    auto f_obj_func = *rx.add_face_attribute<T>("fObjFunc", 1);

    HessianSparseMatrix<T, VariableDim> hess(rx);
    hess.reset(0.f, LOCATION_ALL);

    // TODO this is a AoS and should be converted into SoA
    auto rest_shape =
        *rx.add_face_attribute<Eigen::Matrix<T, 2, 2>>("fRestShape", 1);

    tutte_embedding(rx, coordinates, uv);

    // uv(VertexHandle(0, 0), 0) = 1;
    // uv(VertexHandle(0, 0), 1) = 0;
    //
    // uv(VertexHandle(0, 1), 0) = 0.25;
    // uv(VertexHandle(0, 1), 1) = -0.25;
    //
    // uv(VertexHandle(0, 2), 0) = -1.83697e-16;
    // uv(VertexHandle(0, 2), 1) = -1;
    //
    // uv(VertexHandle(0, 3), 0) = -0.166667;
    // uv(VertexHandle(0, 3), 1) = -0.166667;
    //
    // uv(VertexHandle(0, 4), 0) = -1;
    // uv(VertexHandle(0, 4), 1) = 1.22465e-16;
    //
    // uv(VertexHandle(0, 5), 0) = -0.25;
    // uv(VertexHandle(0, 5), 1) = 0.25;
    //
    // uv(VertexHandle(0, 6), 0) = 6.12323e-17;
    // uv(VertexHandle(0, 6), 1) = 1;
    //
    // uv(VertexHandle(0, 7), 0) = 0.166667;
    // uv(VertexHandle(0, 7), 1) = 0.166667;
    // uv.move(HOST, DEVICE);

    // rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
    //     coordinates(vh, 0) = uv(vh, 0);
    //     coordinates(vh, 1) = uv(vh, 1);
    //     coordinates(vh, 2) = 0;
    // });
    // rx.get_polyscope_mesh()->updateVertexPositions(coordinates);
    // polyscope::show();

    // rx.get_polyscope_mesh()->addVertexParameterizationQuantity("uv_tutte",
    // uv);


    constexpr uint32_t blockThreads = 256;

    int num_iterations = 100;


    rx.run_query_kernel<Op::FV, blockThreads>(
        [=] __device__(const FaceHandle& fh, const VertexIterator& iter) {
            const VertexHandle v0 = iter[0];
            const VertexHandle v1 = iter[1];
            const VertexHandle v2 = iter[2];

            assert(v0.is_valid() && v1.is_valid() && v2.is_valid());

            // 3d position
            Eigen::Vector3<T> ar_3d = coordinates.to_eigen<3>(v0);
            Eigen::Vector3<T> br_3d = coordinates.to_eigen<3>(v1);
            Eigen::Vector3<T> cr_3d = coordinates.to_eigen<3>(v2);

            // Local 2D coordinate system
            Eigen::Vector3<T> n  = (br_3d - ar_3d).cross(cr_3d - ar_3d);
            Eigen::Vector3<T> b1 = (br_3d - ar_3d).normalized();
            Eigen::Vector3<T> b2 = n.cross(b1).normalized();

            // Express a, b, c in local 2D coordiante system
            Eigen::Vector2<T> ar_2d(T(0.0), T(0.0));
            Eigen::Vector2<T> br_2d((br_3d - ar_3d).dot(b1), T(0.0));
            Eigen::Vector2<T> cr_2d((cr_3d - ar_3d).dot(b1),
                                    (cr_3d - ar_3d).dot(b2));

            // Save 2-by-2 matrix with edge vectors as colums
            Eigen::Matrix<T, 2, 2> fout = col_mat(br_2d - ar_2d, cr_2d - ar_2d);

            rest_shape(fh) = fout;
        });

    // rx.run_kernel<blockThreads>(
    //     {Op::FV}, calc_rest_shape<T, blockThreads>, coordinates, rest_shape);

    CUDA_ERROR(cudaDeviceSynchronize());


    LaunchBox<blockThreads> lb;
    rx.prepare_launch_box(
        {Op::FV},
        lb,
        (void*)calc_param_loss<T, VariableDim, blockThreads, true>);

    FaceReduceHandle<T> reducer(f_obj_func);

    GPUTimer timer;
    timer.start();


    Solver solver = Solver::LU;

    // TODO
    // hess.pre_solve(rx, Solver::CHOL);

    T convergence_eps = 1e-2;

    int iter;
    for (iter = 0; iter < num_iterations; ++iter) {

        // 1) calcu objective function
        rx.run_kernel(lb,
                      calc_param_loss<T, VariableDim, blockThreads, true>,
                      rest_shape,
                      uv,
                      grad,
                      hess,
                      f_obj_func,
                      true);


        T f = reducer.reduce(f_obj_func, cub::Sum(), 0);

        if (iter % 10 == 0) {
            RXMESH_INFO("Iteration= {}: Energy = {}", iter, f);
        }

        // 2) direction newton
        // TODO we should refactor (or at least analyze_pattern) once
        grad.multiply(T(-1.f));


        if (solver == Solver::CHOL || solver == Solver::QR) {
            hess.solve(grad.data(), dir.data(), solver, PermuteMethod::NSTDIS);
        } else if (solver == Solver::LU) {
            grad.move(DEVICE, HOST);
            hess.move(DEVICE, HOST);
            hess.solve(
                grad.data(HOST), dir.data(HOST), solver, PermuteMethod::NSTDIS);
            dir.move(HOST, DEVICE);
        }

        // 3) newton decrement
        if (0.5f * grad.dot(dir) < convergence_eps) {
            break;
        }

        // 4) line search
        line_search(rx,
                    f,
                    dir,
                    uv,
                    uv_temp,
                    grad,
                    hess,
                    rest_shape,
                    f_obj_func,
                    reducer);


        // 5) reset
        grad.reset(0, DEVICE);
        hess.reset(0, DEVICE);
    }

    timer.stop();


    RXMESH_INFO(
        "Paramterization RXMesh: iterations ={}, time= {} (ms), "
        "timer/iteration= {} ms/iter",
        iter,
        timer.elapsed_millis(),
        timer.elapsed_millis() / float(num_iterations));

#if USE_POLYSCOPE
    uv.move(DEVICE, HOST);
    // rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
    //     coordinates(vh, 0) = uv(vh, 0);
    //     coordinates(vh, 1) = uv(vh, 1);
    //     coordinates(vh, 2) = 0;
    // });
    // rx.get_polyscope_mesh()->updateVertexPositions(coordinates);

    rx.get_polyscope_mesh()->addVertexParameterizationQuantity("uv_opt", uv);
    polyscope::show();
#endif
}