#pragma once
#include "../common/openmesh_report.h"
#include "../common/openmesh_trimesh.h"
#include "mcf_util.h"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/util/timer.h"
#include "rxmesh/util/vector.h"

/**
 * axpy3()
 */
template <typename T>
void axpy3(const rxmesh::RXMeshAttribute<T>& X,
           rxmesh::Vector<3, T>              alpha,
           rxmesh::Vector<3, T>              beta,
           rxmesh::RXMeshAttribute<T>&       Y,
           const int                         num_omp_threads)
{
    // Y = beta*Y + alpha*X

    int size = static_cast<int>(X.get_num_mesh_elements());
#pragma omp parallel for schedule(static) num_threads(num_omp_threads)
    for (int i = 0; i < size; ++i) {
        Y(i, 0) *= beta[0];
        Y(i, 1) *= beta[1];
        Y(i, 2) *= beta[2];

        Y(i, 0) += alpha[0] * X(i, 0);
        Y(i, 1) += alpha[1] * X(i, 1);
        Y(i, 2) += alpha[2] * X(i, 2);
    }
}

/**
 * dot3()
 */
template <typename T>
void dot3(const rxmesh::RXMeshAttribute<T>& A,
          const rxmesh::RXMeshAttribute<T>& B,
          rxmesh::Vector<3, T>&             res,
          const int                         num_omp_threads)
{
    // creating temp variables because variable in 'reduction' clause/directive
    // cannot have reference type

    T   x_sum(0), y_sum(0), z_sum(0);
    int size = static_cast<int>(A.get_num_mesh_elements());
#pragma omp parallel for schedule(static) num_threads(num_omp_threads) reduction(+ : x_sum,y_sum,z_sum)
    for (int i = 0; i < size; ++i) {
        x_sum += A(i, 0) * B(i, 0);
        y_sum += A(i, 1) * B(i, 1);
        z_sum += A(i, 2) * B(i, 2);
    }

    res[0] = x_sum;
    res[1] = y_sum;
    res[2] = z_sum;
}

/**
 * partial_voronoi_area()
 */
template <typename T>
T partial_voronoi_area(const int      p_id,  // center
                       const int      q_id,  // before center
                       const int      r_id,  // after center
                       const TriMesh& mesh)
{
    // compute partial Voronoi area of the center vertex that is associated with
    // the triangle p->q->r (oriented ccw)

    TriMesh::VertexIter p_it = mesh.vertices_begin() + p_id;
    TriMesh::VertexIter q_it = mesh.vertices_begin() + q_id;
    TriMesh::VertexIter r_it = mesh.vertices_begin() + r_id;

    assert((*p_it).idx() == p_id);
    assert((*q_it).idx() == q_id);
    assert((*r_it).idx() == r_id);

    const rxmesh::Vector<3, T> p(
        mesh.point(*p_it)[0], mesh.point(*p_it)[1], mesh.point(*p_it)[2]);
    const rxmesh::Vector<3, T> q(
        mesh.point(*q_it)[0], mesh.point(*q_it)[1], mesh.point(*q_it)[2]);
    const rxmesh::Vector<3, T> r(
        mesh.point(*r_it)[0], mesh.point(*r_it)[1], mesh.point(*r_it)[2]);

    return partial_voronoi_area(p, q, r);
}

/**
 * edge_cotan_weight()
 */

template <typename T>
T edge_cotan_weight(const int      p_id,
                    const int      r_id,
                    const int      q_id,
                    const int      s_id,
                    const TriMesh& mesh)
{
    // Get the edge weight between the two verteices p-r where
    // q and s composes the diamond around p-r

    TriMesh::VertexIter p_it = mesh.vertices_begin() + p_id;
    TriMesh::VertexIter r_it = mesh.vertices_begin() + r_id;
    TriMesh::VertexIter q_it = mesh.vertices_begin() + q_id;
    TriMesh::VertexIter s_it = mesh.vertices_begin() + s_id;

    const rxmesh::Vector<3, T> p(
        mesh.point(*p_it)[0], mesh.point(*p_it)[1], mesh.point(*p_it)[2]);
    const rxmesh::Vector<3, T> r(
        mesh.point(*r_it)[0], mesh.point(*r_it)[1], mesh.point(*r_it)[2]);
    const rxmesh::Vector<3, T> q(
        mesh.point(*q_it)[0], mesh.point(*q_it)[1], mesh.point(*q_it)[2]);
    const rxmesh::Vector<3, T> s(
        mesh.point(*s_it)[0], mesh.point(*s_it)[1], mesh.point(*s_it)[2]);

    return edge_cotan_weight(p, r, q, s);
}


template <typename T>
void mcf_matvec(TriMesh&                          mesh,
                const rxmesh::RXMeshAttribute<T>& in,
                rxmesh::RXMeshAttribute<T>&       out,
                const int                         num_omp_threads)
{
    // Matrix vector multiplication operation based on uniform Laplacian weight
    // defined in Equation 7 in Implicit Fairing of Irregular Meshes using
    // Diffusion and Curvature Flow paper

    // Ideally we should compute the vertex weight first in one loop over the
    // one-ring and then do another loop to do the matvect operation. We choose
    // to optimize this by saving one loop and incrementally compute the vertex
    // weight. Note the vertex weight in case of uniform Laplace is the valence
    // inversed, otherwise it is 0.5/voronoi_area. We build this voronoi_area
    // incrementally which makes the code looks a bit ugly.

    // To compute the vertex cotan weight, we use the following configuration
    // where P is the center vertex we want to compute vertex weight for.
    // Looping over P's one ring should gives q->r->s.
    /*       r
          /  |  \
         /   |   \
        s    |    q
         \   |   /
           \ |  /
             p
    */

#pragma omp parallel for schedule(static) num_threads(num_omp_threads)
    for (int p_id = 0; p_id < int(mesh.n_vertices()); ++p_id) {
        TriMesh::VertexIter p_iter = mesh.vertices_begin() + p_id;

        // Off-diagonal entries
        rxmesh::Vector<3, T> x(T(0));
        T                    sum_e_weight(0);

        // vertex weight
        T v_weight(0);

        // The last vertex in the one ring
        TriMesh::VertexVertexIter q_iter = mesh.vv_iter(*p_iter);
        --q_iter;
        assert(q_iter.is_valid());

        // the second vertex in the one ring
        TriMesh::VertexVertexIter s_iter = mesh.vv_iter(*p_iter);
        ++s_iter;
        assert(s_iter.is_valid());

        for (TriMesh::VertexVertexIter r_iter = mesh.vv_iter(*p_iter);
             r_iter.is_valid();
             ++r_iter) {

            int r_id = (*r_iter).idx();


            T e_weight = 0;
            if (Arg.use_uniform_laplace) {
                e_weight = 1;
            } else {
                e_weight = std::max(
                    T(0.0),
                    edge_cotan_weight<T>(
                        p_id, r_id, (*q_iter).idx(), (*s_iter).idx(), mesh));
                ++s_iter;
            }

            e_weight *= static_cast<T>(Arg.time_step);
            sum_e_weight += e_weight;

            x[0] -= e_weight * in(r_id, 0);
            x[1] -= e_weight * in(r_id, 1);
            x[2] -= e_weight * in(r_id, 2);

            if (Arg.use_uniform_laplace) {
                ++v_weight;
            } else {
                T tri_area =
                    partial_voronoi_area<T>(p_id, (*q_iter).idx(), r_id, mesh);

                v_weight += (tri_area > 0) ? tri_area : 0;

                q_iter++;
                assert(q_iter == r_iter);
            }
        }

        // Diagonal entry
        if (Arg.use_uniform_laplace) {
            v_weight = 1.0 / v_weight;
        } else {
            v_weight = 0.5 / v_weight;
        }

        assert(!std::isnan(v_weight));
        assert(!std::isinf(v_weight));

        T diag       = ((1.0 / v_weight) + sum_e_weight);
        out(p_id, 0) = x[0] + diag * in(p_id, 0);
        out(p_id, 1) = x[1] + diag * in(p_id, 1);
        out(p_id, 2) = x[2] + diag * in(p_id, 2);
    }
}


/**
 * cg()
 */
template <typename T>
void cg(TriMesh&                    mesh,
        rxmesh::RXMeshAttribute<T>& X,
        rxmesh::RXMeshAttribute<T>& B,
        rxmesh::RXMeshAttribute<T>& R,
        rxmesh::RXMeshAttribute<T>& P,
        rxmesh::RXMeshAttribute<T>& S,
        uint32_t&                   num_cg_iter_taken,
        rxmesh::Vector<3, T>&       start_residual,
        rxmesh::Vector<3, T>&       stop_residual,
        const int                   num_omp_threads)
{
    // CG solver. Solve for the three coordinates simultaneously

    // s = Ax
    mcf_matvec(mesh, X, S, num_omp_threads);


    // r = b - s = b - Ax
    // p = r
#pragma omp parallel for schedule(static) num_threads(num_omp_threads)
    for (int i = 0; i < int(mesh.n_vertices()); ++i) {
        R(i, 0) = B(i, 0) - S(i, 0);
        R(i, 1) = B(i, 1) - S(i, 1);
        R(i, 2) = B(i, 2) - S(i, 2);

        P(i, 0) = R(i, 0);
        P(i, 1) = R(i, 1);
        P(i, 2) = R(i, 2);
    }

    // delta_new = <r,r>
    rxmesh::Vector<3, T> delta_new;
    dot3(R, R, delta_new, num_omp_threads);

    // delta_0 = delta_new
    const rxmesh::Vector<3, T> delta_0(delta_new);

    start_residual = delta_0;
    const rxmesh::Vector<3, T> ones(1);
    uint32_t                   iter = 0;
    while (iter < Arg.max_num_cg_iter) {
        // s = Ap
        mcf_matvec(mesh, P, S, num_omp_threads);

        // alpha = delta_new / <s,p>
        rxmesh::Vector<3, T> alpha;
        dot3(S, P, alpha, num_omp_threads);
        alpha = delta_new / alpha;


        // x =  x + alpha*p
        axpy3(P, alpha, ones, X, num_omp_threads);

        // r = r - alpha*s
        axpy3(S, -alpha, ones, R, num_omp_threads);

        // delta_old = delta_new
        rxmesh::Vector<3, T> delta_old(delta_new);

        // delta_new = <r,r>
        dot3(R, R, delta_new, num_omp_threads);

        // beta = delta_new/delta_old
        rxmesh::Vector<3, T> beta(delta_new / delta_old);

        // exit if error is getting too low across three coordinates
        if (delta_new[0] < Arg.cg_tolerance * Arg.cg_tolerance * delta_0[0] &&
            delta_new[1] < Arg.cg_tolerance * Arg.cg_tolerance * delta_0[1] &&
            delta_new[2] < Arg.cg_tolerance * Arg.cg_tolerance * delta_0[2]) {
            break;
        }

        // p = beta*p + r
        axpy3(R, ones, beta, P, num_omp_threads);

        ++iter;
    }
    num_cg_iter_taken = iter;
    stop_residual     = delta_new;
}

/**
 * implicit_smoothing()
 */
template <typename T>
void implicit_smoothing(TriMesh&                    mesh,
                        rxmesh::RXMeshAttribute<T>& X,
                        uint32_t&                   num_cg_iter_taken,
                        float&                      time,
                        rxmesh::Vector<3, T>&       start_residual,
                        rxmesh::Vector<3, T>&       stop_residual,
                        const int                   num_omp_threads)
{

    for (TriMesh::VertexIter v_it = mesh.vertices_begin();
         v_it != mesh.vertices_end();
         ++v_it) {
        ASSERT_FALSE(mesh.is_boundary(*v_it))
            << "OpenMesh MCF only takes watertight/closed mesh without "
               "boundaries";
    }

    // CG containers
    rxmesh::RXMeshAttribute<T> B, R, P, S;

    X.init(mesh.n_vertices(), 3u, rxmesh::HOST);
    X.reset(0.0, rxmesh::HOST);

    S.init(mesh.n_vertices(), 3u, rxmesh::HOST);
    S.reset(0.0, rxmesh::HOST);

    P.init(mesh.n_vertices(), 3u, rxmesh::HOST);
    P.reset(0.0, rxmesh::HOST);

    R.init(mesh.n_vertices(), 3u, rxmesh::HOST);
    R.reset(0.0, rxmesh::HOST);

    B.init(mesh.n_vertices(), 3u, rxmesh::HOST);
    B.reset(0.0, rxmesh::HOST);

#pragma omp parallel for
    for (uint32_t v_id = 0; v_id < mesh.n_vertices(); ++v_id) {
        TriMesh::VertexIter v_iter = mesh.vertices_begin() + v_id;

        // LHS
        X(v_id, 0) = mesh.point(*v_iter)[0];
        X(v_id, 1) = mesh.point(*v_iter)[1];
        X(v_id, 2) = mesh.point(*v_iter)[2];

        // RHS
        T v_weight = 1;

        if (Arg.use_uniform_laplace) {
            v_weight = static_cast<T>(mesh.valence(*v_iter));
        }
        // will fix it later for cotan weight

        B(v_id, 0) = X(v_id, 0) * v_weight;
        B(v_id, 1) = X(v_id, 1) * v_weight;
        B(v_id, 2) = X(v_id, 2) * v_weight;
    }

    if (!Arg.use_uniform_laplace) {
        // fix RHS (B)
#pragma omp parallel for
        for (int v_id = 0; v_id < int(mesh.n_vertices()); ++v_id) {
            TriMesh::VertexIter v_iter = mesh.vertices_begin() + v_id;

            T v_weight(0);

            TriMesh::VertexVertexIter q_iter = mesh.vv_iter(*v_iter);
            --q_iter;
            assert(q_iter.is_valid());

            for (TriMesh::VertexVertexIter vv_iter = mesh.vv_iter(*v_iter);
                 vv_iter.is_valid();
                 ++vv_iter) {

                T tri_area = partial_voronoi_area<T>(
                    v_id, (*q_iter).idx(), (*vv_iter).idx(), mesh);

                v_weight += (tri_area > 0) ? tri_area : 0;

                q_iter++;
                assert(q_iter == vv_iter);
            }
            v_weight   = 0.5 / v_weight;
            B(v_id, 0) = X(v_id, 0) / v_weight;
            B(v_id, 1) = X(v_id, 1) / v_weight;
            B(v_id, 2) = X(v_id, 2) / v_weight;
        }
    }

    num_cg_iter_taken = 0;

    // solve
    rxmesh::CPUTimer timer;
    timer.start();

    cg(mesh,
       X,
       B,
       R,
       P,
       S,
       num_cg_iter_taken,
       start_residual,
       stop_residual,
       num_omp_threads);

    timer.stop();
    time = timer.elapsed_millis();
}

template <typename T>
void mcf_openmesh(const int                   num_omp_threads,
                  TriMesh&                    input_mesh,
                  rxmesh::RXMeshAttribute<T>& smoothed_coord)
{
    // Report
    OpenMeshReport report("MCF_OpenMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.system();
    report.model_data(Arg.obj_file_name, input_mesh);
    std::string method =
        "OpenMesh " + std::to_string(num_omp_threads) + " Core";
    report.add_member("method", method);
    std::string order = "default";
    if (Arg.shuffle) {
        order = "shuffle";
    } else if (Arg.sort) {
        order = "sorted";
    }
    report.add_member("input_order", order);
    report.add_member("time_step", Arg.time_step);
    report.add_member("cg_tolerance", Arg.cg_tolerance);
    report.add_member("use_uniform_laplace", Arg.use_uniform_laplace);
    report.add_member("max_num_cg_iter", Arg.max_num_cg_iter);


    // implicit smoothing
    uint32_t             num_cg_iter_taken = 0;
    float                time              = 0;
    rxmesh::Vector<3, T> start_residual;
    rxmesh::Vector<3, T> stop_residual;

    implicit_smoothing(input_mesh,
                       smoothed_coord,
                       num_cg_iter_taken,
                       time,
                       start_residual,
                       stop_residual,
                       num_omp_threads);

    RXMESH_TRACE(
        "mcf_openmesh() took {} (ms) and {} iterations (i.e., {} ms/iter) ",
        time,
        num_cg_iter_taken,
        time / float(num_cg_iter_taken));


    // write output
    //#pragma omp parallel for
    //    for (int v_id = 0; v_id < int(input_mesh.n_vertices()); ++v_id) {
    //        TriMesh::VertexIter v_iter = input_mesh.vertices_begin() + v_id;
    //        input_mesh.point(*v_iter)[0] = smoothed_coord(v_id, 0);
    //        input_mesh.point(*v_iter)[1] = smoothed_coord(v_id, 1);
    //        input_mesh.point(*v_iter)[2] = smoothed_coord(v_id, 2);
    //    }
    //    std::string fn = STRINGIFY(OUTPUT_DIR) "mcf_openmesh.obj";
    //    if (!OpenMesh::IO::write_mesh(input_mesh, fn)) {
    //        RXMESH_WARN("OpenMesh cannot write mesh to file {}", fn);
    //    }

    // Finalize report
    report.add_member("start_residual", to_string(start_residual));
    report.add_member("end_residual", to_string(stop_residual));
    report.add_member("num_cg_iter_taken", num_cg_iter_taken);
    report.add_member("total_time (ms)", time);
    rxmesh::TestData td;
    td.test_name   = "MCF";
    td.num_threads = num_omp_threads;
    td.time_ms.push_back(time / float(num_cg_iter_taken));
    td.passed.push_back(true);
    report.add_test(td);
    report.write(
        Arg.output_folder + "/openmesh",
        "MCF_OpenMesh_" + rxmesh::extract_file_name(Arg.obj_file_name));
}