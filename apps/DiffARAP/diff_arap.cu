#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.h"

#include "rxmesh/util/svd3_cuda.h"

#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"

#include "arap_kernels.h"

using namespace rxmesh;

template <typename T>
void arap(RXMeshStatic& rx)
{
    constexpr int VariableDim = 3;

    constexpr uint32_t blockThreads = 256;

    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    using HessMatT = typename ProblemT::HessMatT;

    ProblemT problem(rx);

    CholeskySolver<HessMatT, ProblemT::DenseMatT::OrderT> solver(&problem.hess);

    NetwtonSolver newton_solver(problem, &solver);

    // stays same across computation
    auto P = *rx.get_input_vertex_coordinates();

    // deformed vertex position that change every iteration
    auto& P_prime = *problem.objective;
    P_prime.copy_from(P, DEVICE, DEVICE);

    // vertex constraints where
    //  0 means free
    //  1 means user-displaced
    //  2 means fixed
    auto constraints = *rx.add_vertex_attribute<T>("Constraints", 1);

    // rotation matrix as a very attribute where every vertex has 3x3 matrix
    auto rotations = *rx.add_vertex_attribute<T>("RotationMatrix", 9);

    // weights matrix
    SparseMatrix<float> weight_matrix(rx);
    weight_matrix.reset(0.f, LOCATION_ALL);

    // set constraints
    const vec3<float> sphere_center(0.1818329, -0.99023, 0.325066);
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
        const vec3<float> p(P(vh, 0), P(vh, 1), P(vh, 2));

        // fix the bottom
        if (p[2] < -0.63) {
            constraints(vh) = 2;
        }

        // move the jaw
        if (glm::distance(p, sphere_center) < 0.1) {
            constraints(vh) = 1;
        }
    });

#if USE_POLYSCOPE
    // move constraints to the host and add it to Polyscope
    constraints.move(DEVICE, HOST);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("constraintsV",
                                                     constraints);
#endif


    // obtain cotangent weight matrix
    rx.run_kernel<blockThreads>({Op::EVDiamond},
                                calc_edge_weights_mat<float, blockThreads>,
                                P,
                                weight_matrix);


    // energy term
    //   add energy term
    problem.template add_term<Op::VV, true>(
        [=] __device__(const auto& vh, const auto& iter, auto& objective) {
            using ActiveT = ACTIVE_TYPE(vh);

            // p_prime
            Eigen::Vector3<ActiveT> pi_prime =
                iter_val<ActiveT, 3>(vh, objective);

            // p
            Eigen::Vector3<T> pi = P.to_eigen<3>(vh);

            // r
            Eigen::Matrix3f ri = Eigen::Matrix3f::Zero(3, 3);
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    ri(i, j) = rotations(vh, i * 3 + j);
                }
            }


            ActiveT res;
            return res;
        });


    T   convergence_eps = 1e-2;
    int newton_max_iter = 10;

    int iterations = 1;

    float       t    = 0;
    bool        flag = false;
    vec3<float> start(0.0f, 0.2f, 0.0f);
    vec3<float> end(0.0f, -0.2f, 0.0f);
    vec3<float> displacement(0.0f, 0.0f, 0.0f);

    auto polyscope_callback = [&]() mutable {
        t += flag ? -0.5f : 0.5f;

        flag = (t < 0 || t > 1.0f) ? !flag : flag;

        displacement = (1 - t) * start + (t)*end;

        // apply user deformation
        rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
            if (constraints(vh) == 1) {
                P_prime(vh, 0) += displacement[0];
                P_prime(vh, 1) += displacement[1];
                P_prime(vh, 2) += displacement[2];
            }
        });


        // process step
        for (int i = 0; i < iterations; i++) {
            // solver for rotation
            rx.run_kernel<blockThreads>(
                {Op::VV},
                calculate_rotation_matrix<float, blockThreads>,
                P,
                P_prime,
                rotations,
                weight_matrix);

            // solve for position via Newton

            for (int iter = 0; iter < newton_max_iter; ++iter) {
                // evaluate energy terms
                problem.eval_terms();

                // get the current value of the loss function
                T f = problem.get_current_loss();
                RXMESH_INFO(
                    "Iter {} =, Newton Iter= {}: Energy = {}", i, iter, f);

                // direction newton
                newton_solver.newton_direction();


                // newton decrement
                if (0.5f * problem.grad.dot(newton_solver.dir) <
                    convergence_eps) {
                    break;
                }

                // line search
                newton_solver.line_search();
            }
        }

        // move mat to the host
        P_prime.move(DEVICE, HOST);

#if USE_POLYSCOPE
        rx.get_polyscope_mesh()->updateVertexPositions(P_prime);
#endif
    };

#ifdef USE_POLYSCOPE
    polyscope::view::upDir         = polyscope::UpDir::ZUp;
    polyscope::state::userCallback = polyscope_callback;
    polyscope::show();

#endif

    weight_matrix.release();
}

int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

    if (!rx.is_closed()) {
        RXMESH_ERROR("Input mesh should be closed without boundaries");
        return EXIT_FAILURE;
    }


    arap<float>(rx);


    return 0;
}