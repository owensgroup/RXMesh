// Reference Implementation
// https://github.com/patr-schm/TinyAD-Examples/blob/main/apps/manifold_optimization.cc
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"

using namespace rxmesh;

enum class Direction
{
    Default   = 0,
    Equator   = 1,
    NorthPole = 2,
};

std::string direction_name(Direction dir)
{
    switch (dir) {
        case Direction::Default:
            return "default";
            break;
        case Direction::Equator:
            return "equator";
            break;
        case Direction::NorthPole:
            return "northpole";
            break;
        default:
            return "unknown";
            break;
    }
}

template <typename T>
void add_mesh_to_polyscope(RXMeshStatic&       rx,
                           VertexAttribute<T>& v,
                           std::string         name)
{
    if (v.get_num_attributes() == 3) {
        polyscope::registerSurfaceMesh(name, v, rx.get_polyscope_mesh()->faces);
    } else {
        auto v3 = *rx.add_vertex_attribute<T>(name, 3);

        rx.for_each_vertex(HOST, [&](const VertexHandle h) {
            v3(h, 0) = v(h, 0);
            v3(h, 1) = v(h, 1);
            v3(h, 2) = 0;
        });

        polyscope::registerSurfaceMesh(
            name, v3, rx.get_polyscope_mesh()->faces);

        rx.remove_attribute(name);
    }
}

template <typename T>
__host__ __device__ Eigen::Vector3<T> any_tangent(const Eigen::Vector3<T>& _p)
{
    // Compute an arbitrary tangent vector of the sphere at position p.
    //
    // Find coordinate axis spanning the largest angle with _p.
    // Return cross product that of axis with _p
    Eigen::Vector3<T> tang;

    T min_dot2 = std::numeric_limits<T>::max();

    Eigen::Vector3<T> list[3] = {Eigen::Vector3<T>(1.0, 0.0, 0.0),
                                 Eigen::Vector3<T>(0.0, 1.0, 0.0),
                                 Eigen::Vector3<T>(0.0, 0.0, 1.0)};

    for (const Eigen::Vector3<T>& ax : list) {
        T dot2 = _p.dot(ax);
        dot2 *= dot2;
        if (dot2 < min_dot2) {
            min_dot2 = dot2;
            tang     = ax.cross(_p).normalized();
        }
    }

    return tang;
}

template <typename T>
void compute_local_bases(const RXMeshStatic&       rx,
                         const VertexAttribute<T>& S,
                         VertexAttribute<T>&       B1,
                         VertexAttribute<T>&       B2)
{
    // Compute an orthonormal tangent space basis of the sphere at each vertex.

    rx.for_each_vertex(DEVICE,
                       [S, B1, B2] __device__(const VertexHandle vh) mutable {
                           Eigen::Vector3<T> s = S.template to_eigen<3>(vh);

                           Eigen::Vector3<T> b1 = any_tangent(s);

                           Eigen::Vector3<T> b2 = s.cross(b1);

                           B1.from_eigen(vh, b1);
                           B2.from_eigen(vh, b2);
                       });
}

template <typename U, typename T>
__host__ __device__ Eigen::Vector3<U> retract(Eigen::Vector2<U>&        v_tang,
                                              const VertexHandle&       vh,
                                              const VertexAttribute<T>& S,
                                              const VertexAttribute<T>& B1,
                                              const VertexAttribute<T>& B2)
{
    // Retraction operator: map from a local tangent space to the
    // sphere.

    // Evaluate target point in 3D ambient space and project to
    // sphere via normalization.

    const Eigen::Vector3<T> s  = S.template to_eigen<3>(vh);
    const Eigen::Vector3<T> b1 = B1.template to_eigen<3>(vh);
    const Eigen::Vector3<T> b2 = B2.template to_eigen<3>(vh);

    const Eigen::Vector3<U> ret =
        (s + v_tang[0] * b1 + v_tang[1] * b2).normalized().eval();

    return ret;
}

template <typename T>
void manifold_optimization(RXMeshStatic&                          rx,
                           const std::vector<std::vector<float>>& init_s,
                           const Direction                        dir)
{
    constexpr int VariableDim = 2;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;


    ProblemT problem(rx);

    using HessMatT = typename ProblemT::HessMatT;

    LUSolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(&problem.hess);

    NetwtonSolver newton_solver(problem, &solver);

    auto S  = *rx.add_vertex_attribute<T>(init_s, "S");
    auto B1 = *rx.add_vertex_attribute<T>("B1", 3);
    auto B2 = *rx.add_vertex_attribute<T>("B2", 3);


    compute_local_bases(rx, S, B1, B2);


    // add energy term
    problem.template add_term<Op::FV, true>([=] __device__(const auto& fh,
                                                           const auto& iter,
                                                           auto&       obj) {
        // fh is a face handle
        // iter is an iterator over fh's vertices


        assert(iter[0].is_valid() && iter[1].is_valid() && iter[2].is_valid());

        assert(iter.size() == 3);

        using ActiveT = ACTIVE_TYPE(fh);

        // tangent vectors at the triangle three vertices (a,b,c)
        Eigen::Vector2<ActiveT> a_tang = iter_val<ActiveT, 2>(fh, iter, obj, 0);
        Eigen::Vector2<ActiveT> b_tang = iter_val<ActiveT, 2>(fh, iter, obj, 1);
        Eigen::Vector2<ActiveT> c_tang = iter_val<ActiveT, 2>(fh, iter, obj, 2);


        // Retract 2D tangent vectors to 3D points on the sphere.
        Eigen::Vector3<ActiveT> a_mani = retract(a_tang, iter[0], S, B1, B2);
        Eigen::Vector3<ActiveT> b_mani = retract(b_tang, iter[1], S, B1, B2);
        Eigen::Vector3<ActiveT> c_mani = retract(c_tang, iter[2], S, B1, B2);


        // Objective: injectivity barrier + Dirichlet energy
        ActiveT volume =
            (1.0 / 6.0) * col_mat(a_mani, b_mani, c_mani).determinant();

        if (volume <= 0.0) {
            using PassiveT = PassiveType<ActiveT>;
            return ActiveT(std::numeric_limits<PassiveT>::max());
        }

        ActiveT E = -0.1 * log(volume);
        E += (a_mani - b_mani).squaredNorm() + (b_mani - c_mani).squaredNorm() +
             (c_mani - a_mani).squaredNorm();

        if (dir == Direction::Equator) {
            E += sqr(a_mani.y()) + sqr(b_mani.y()) + sqr(c_mani.y());
        } else if (dir == Direction::NorthPole) {
            E += sqr(1.0 - a_mani.y()) + sqr(1.0 - b_mani.y()) +
                 sqr(1.0 - c_mani.y());
        }

        return E;
    });


    T convergence_eps = 1e-1;

    int num_iterations = 1000;
    int iter;

    GPUTimer timer;
    timer.start();

    for (iter = 0; iter < num_iterations; ++iter) {

        problem.objective->reset(0, DEVICE);

        problem.eval_terms();


        T f = problem.get_current_loss();
        RXMESH_INFO("Iteration= {}: Energy = {}", iter, f);


        newton_solver.newton_direction();


        if (0.5f * problem.grad.dot(newton_solver.dir) < convergence_eps) {
            break;
        }


        newton_solver.line_search();


        // Re-center local bases
        rx.for_each_vertex(DEVICE,
                           [S, B1, B2, obj = *problem.objective] __device__(
                               const VertexHandle h) mutable {
                               Eigen::Vector2<T> v =
                                   obj.template to_eigen<2>(h);
                               Eigen::Vector3<T> s = retract(v, h, S, B1, B2);
                               S.from_eigen(h, s);
                           });

        compute_local_bases(rx, S, B1, B2);
    }

    timer.stop();


    RXMESH_INFO(
        "Manifold Optimization: iterations ={}, time= {} (ms), "
        "timer/iteration= {} ms/iter",
        iter,
        timer.elapsed_millis(),
        timer.elapsed_millis() / float(num_iterations));


    S.move(DEVICE, HOST);

    rx.get_polyscope_mesh()->addVertexParameterizationQuantity(
        direction_name(dir), S);

    add_mesh_to_polyscope(rx, S, direction_name(dir));

    polyscope::show();
}

int main(int argc, char** argv)
{
    Log::init(spdlog::level::info);

    using T = float;

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "giraffe.obj");

    std::vector<std::vector<uint32_t>> fv;
    std::vector<std::vector<float>>    init_s;
    import_obj(STRINGIFY(INPUT_DIR) "giraffe_embedding.obj", init_s, fv);

    if (rx.get_num_faces() != fv.size()) {
        RXMESH_ERROR(
            "The input mesh and initial embedding have different number of "
            "faces");
    }

    if (rx.get_num_vertices() != init_s.size()) {
        RXMESH_ERROR(
            "The input mesh and initial embedding have different number of "
            "vertices");
    }

    manifold_optimization<T>(rx, init_s, Direction::Default);
}