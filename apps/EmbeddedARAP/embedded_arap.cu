#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_vector_problem.h"
#include "rxmesh/diff/gauss_newton_solver.h"

using namespace rxmesh;

template <typename T>
void arap(RXMeshStatic& rx)
{
    constexpr int VariableDim = 12;

    constexpr uint32_t blockThreads = 256;

    T w_fit_sqrt = std::sqrt(3.0);
    T w_reg_sqrt = std::sqrt(12.0);
    T w_rot_sqrt = std::sqrt(5.0);

    using ProblemT = DiffVectorProblem<T, VariableDim, VertexHandle>;
    ProblemT problem(rx);

#ifdef USE_CUDSS
    // using SolverT =
    //     cuDSSCholeskySolver<SparseMatrix<T>, ProblemT::DenseMatT::OrderT>;

    using SolverT = PCGSolver<T, ProblemT::DenseMatT::OrderT>;
#else
    using SolverT =
        CholeskySolver<SparseMatrix<T>, ProblemT::DenseMatT::OrderT>;
#endif

    GaussNetwtonSolver<T, VariableDim, VertexHandle, SolverT> solver(problem);

    auto Urshape = *rx.get_input_vertex_coordinates();

    // combined deformed vertex position and rotation matrix
    // 0-2 are the offset
    // 3-11 are the 9 entries of the rotation matrix row-wise, i.e.,
    // col_0 = [c3, c6, c9]
    auto& opt_var = *problem.objective;

    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) mutable {
        for (int i = 0; i < VariableDim; ++i) {
            opt_var(vh, i) = 0;
        }

        opt_var(vh, 3)  = 1;
        opt_var(vh, 7)  = 1;
        opt_var(vh, 11) = 1;
    });

    // vertex constraints where
    //  0 means free
    //  1 means user-displaced
    //  2 means fixed
    auto constraints = *rx.add_vertex_attribute<int>("Constraints", 1);
    constraints.reset(0, LOCATION_ALL);

    // target offset for user-displaced vertices
    auto cn = *rx.add_vertex_attribute<T>("cn", 3);
    cn.reset(0, LOCATION_ALL);

    // set constraints
    const vec3<float> sphere_center(0.1818329, -0.99023, 0.325066);
    rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
        const vec3<float> p(Urshape(vh, 0), Urshape(vh, 1), Urshape(vh, 2));

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

    // E_fit + E_rot
    problem.template add_term<Op::V, 9>([=] __device__(const auto& vh,
                                                       auto&       obj) {
        using ActiveT = ACTIVE_TYPE(vh);

        // vertex combined position and rotation matrix variable
        Eigen::Vector<ActiveT, 12> o_r = iter_val<ActiveT, 12>(vh, obj);

        Eigen::Vector<ActiveT, 9> ret;

        // E_fit energy, i.e., offset of constraints vertices should a) for
        // fixed vertices should be minimized, b) for user-displaced
        // vertices, the  difference between the user-defined displacement
        // and the offset should be minimized, c) for free vertices, this is
        // just zero
        if (constraints(vh) == 0) {
            // free vertices do not contribute to this energy
            ret[0] = 0;
            ret[1] = 0;
            ret[2] = 0;
        } else if (constraints(vh) == 2) {
            // fixed point should stay fixed, i.e., its offset should be
            // minimized
            ret[0] = o_r[0];
            ret[1] = o_r[1];
            ret[2] = o_r[2];
        } else if (constraints(vh) == 1) {
            // user-displaced points
            ret[0] = o_r[0] - cn(vh, 0);
            ret[1] = o_r[1] - cn(vh, 1);
            ret[2] = o_r[2] - cn(vh, 2);
        }
        ret[0] *= w_fit_sqrt;
        ret[1] *= w_fit_sqrt;
        ret[2] *= w_fit_sqrt;


        // E_rot, i.e., soft rotation constraints

        //(c0 c1)
        ret[3] =
            w_rot_sqrt * (o_r[3] * o_r[4] + o_r[6] * o_r[7] + o_r[9] * o_r[10]);

        //(c0 c2)
        ret[4] =
            w_rot_sqrt * (o_r[3] * o_r[5] + o_r[6] * o_r[8] + o_r[9] * o_r[11]);

        //(c1 c2)
        ret[5] = w_rot_sqrt *
                 (o_r[4] * o_r[5] + o_r[7] * o_r[8] + o_r[10] * o_r[11]);

        //(c0 c0) - 1
        ret[6] = w_rot_sqrt *
                 (o_r[3] * o_r[3] + o_r[6] * o_r[6] + o_r[9] * o_r[9] - 1);

        //(c1 c1) - 1
        ret[7] = w_rot_sqrt *
                 (o_r[4] * o_r[4] + o_r[7] * o_r[7] + o_r[10] * o_r[10] - 1);

        //(c2 c2) - 1
        ret[8] = w_rot_sqrt *
                 (o_r[5] * o_r[5] + o_r[8] * o_r[8] + o_r[11] * o_r[11] - 1);


        return ret;
    });


    // E_reg
    problem.template add_term<Op::EV, 3>([=] __device__(const auto& eh,
                                                        const auto& iter,
                                                        auto&       obj) {
        using ActiveT = ACTIVE_TYPE(eh);


        Eigen::Vector<ActiveT, 12> v0 = iter_val<ActiveT, 12>(eh, iter, obj, 0);
        Eigen::Vector<ActiveT, 12> v1 = iter_val<ActiveT, 12>(eh, iter, obj, 1);


        // Rest edge
        Eigen::Vector<T, 3> u0  = Urshape.to_eigen<3>(iter[0]);
        Eigen::Vector<T, 3> u1  = Urshape.to_eigen<3>(iter[1]);
        T                   dux = u1[0] - u0[0];
        T                   duy = u1[1] - u0[1];
        T                   duz = u1[2] - u0[2];

        ActiveT rdu_x = v0[3] * dux + v0[4] * duy + v0[5] * duz;
        ActiveT rdu_y = v0[6] * dux + v0[7] * duy + v0[8] * duz;
        ActiveT rdu_z = v0[9] * dux + v0[10] * duy + v0[11] * duz;

        Eigen::Matrix<ActiveT, 3, 1> res;
        res[0] = (v1[0] - v0[0]) - rdu_x;
        res[1] = (v1[1] - v0[1]) - rdu_y;
        res[2] = (v1[2] - v0[2]) - rdu_z;

        // dux    = u0[0] - u1[0];
        // duy    = u0[1] - u1[1];
        // duz    = u0[2] - u1[2];
        // res[3] = (v0[0] - v1[0]) - rdu_x;
        // res[4] = (v0[1] - v1[1]) - rdu_y;
        // res[5] = (v0[2] - v1[2]) - rdu_z;

        return w_reg_sqrt * res;
    });

    Timers<GPUTimer> timer;
    timer.add("Step");
    timer.add("LinearSolver");
    timer.add("Diff");


    problem.prep_eval();
    solver.prep_solver(5);

    //problem.jac->to_file("arap_jac");

    float       t    = 0;
    bool        flag = false;
    vec3<float> start(0.0f, 0.2f, 0.0f);
    vec3<float> end(0.0f, -0.2f, 0.0f);
    vec3<float> displacement(0.0f, 0.0f, 0.0f);

    const int num_verts = rx.get_num_vertices();

    int  num_steps  = 0;
    bool is_running = false;

    auto polyscope_callback = [&]() mutable {
        bool step_once = false;

        if (ImGui::Button("Start")) {
            is_running = true;
        }

        ImGui::SameLine();
        if (ImGui::Button("Step")) {
            step_once = true;
        }


        if (step_once || is_running) {
            t += flag ? -0.5f : 0.5f;

            flag = (t < 0 || t > 1.0f) ? !flag : flag;

            displacement = (1 - t) * start + (t)*end;

            // apply user deformation
            rx.for_each_vertex(DEVICE, [=] __device__(const VertexHandle& vh) {
                if (constraints(vh) == 1) {
                    cn(vh, 0) = displacement[0];
                    cn(vh, 1) = displacement[1];
                    cn(vh, 2) = displacement[2];
                } else {
                    cn(vh, 0) = 0;
                    cn(vh, 1) = 0;
                    cn(vh, 2) = 0;
                }
            });

            // process step
            num_steps++;

            timer.start("Step");

            // compute g = -J^T r
            // the -1.0 is used here so we don't need to scale things again
            // in the gauss-newton which solves (J^T J).dir = -J^T r
            timer.start("Diff");
            problem.eval_terms_sum_of_squares(-1.0);
            timer.stop("Diff");

            //problem.jac->move(DEVICE, HOST);
            //problem.jac->to_file("arap_jac"+std::to_string(num_steps));

            // get the current value of the loss function
            T f = problem.get_current_loss();
            RXMESH_INFO("Step: {}, Energy: {}", num_steps, f);


            // direction newton
            timer.start("LinearSolver");
            solver.compute_direction();
            timer.stop("LinearSolver");

            // take a step
            rx.for_each_vertex(
                DEVICE,
                [p = opt_var, dir = solver.dir, n = num_verts] __device__(
                    const VertexHandle& vh) mutable {
                    dir.reshape(n, VariableDim);
                    for (int i = 0; i < VariableDim; ++i) {
                        p(vh, i) = p(vh, i) + dir(vh, i);
                    }
                });

            timer.stop("Step");
        }

#if USE_POLYSCOPE
        // repurpose cn to be the new position (for gui only)
        rx.for_each_vertex(DEVICE, [=] __device__(auto vh) {
            for (int i = 0; i < 3; ++i) {
                cn(vh, i) = Urshape(vh, i) + opt_var(vh, i);
            }
        });
        cn.move(DEVICE, HOST);
        rx.get_polyscope_mesh()->updateVertexPositions(cn);
#endif
        if (ImGui::Button("Export")) {
            rx.export_obj(std::to_string(num_steps) + ".obj", cn);
        }

        ImGui::SameLine();
        if (ImGui::Button("Pause")) {
            is_running = false;
        }
    };

#ifdef USE_POLYSCOPE
    polyscope::view::upDir         = polyscope::UpDir::ZUp;
    polyscope::state::userCallback = polyscope_callback;
    polyscope::show();
#endif

    RXMESH_INFO(
        "DiffArap: #V= {}, #Steps ={}, time= {} (ms), time/step= {} ms/iter",
        rx.get_num_vertices(),
        num_steps,
        timer.elapsed_millis("Step"),
        timer.elapsed_millis("Step") / float(num_steps));

    RXMESH_INFO("LinearSolver {} (ms), Diff {} (ms)",
                timer.elapsed_millis("LinearSolver"),
                timer.elapsed_millis("Diff"));

    RXMESH_INFO("LinearSolver/step {} (ms), Diff/step {} (ms)",
                timer.elapsed_millis("LinearSolver") / float(num_steps),
                timer.elapsed_millis("Diff") / float(num_steps));
}

int main(int argc, char** argv)
{
    rx_init(0);

    std::string filename = STRINGIFY(INPUT_DIR) "dragon_simp.obj";

    if (argc > 1) {
        filename = std::string(argv[1]);
    }

    RXMeshStatic rx(filename);

    arap<float>(rx);

    return 0;
}