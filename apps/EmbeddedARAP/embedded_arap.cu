#include "rxmesh/rxmesh_static.h"

#include "rxmesh/diff/diff_vector_problem.h"

using namespace rxmesh;

template <typename T>
void arap(RXMeshStatic& rx)
{
    constexpr int VariableDim = 12;

    constexpr uint32_t blockThreads = 256;

    T w_fit_sqrt = 1;
    T w_reg_sqrt = 1;
    T w_rot_sqrt = 1;

    using ProblemT = DiffVectorProblem<T, VariableDim, VertexHandle>;
    ProblemT problem(rx);

    auto Urshape = *rx.get_input_vertex_coordinates();

    // deformed vertex position that change every iteration
    auto& offset = *problem.objective;
    offset.copy_from(Urshape, DEVICE, DEVICE);

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

    // E_fit energy, i.e., offset of constraints vertices should a) for fixed
    // vertices should be minimized, b) for user-displaced vertices, the
    // difference between the user-defined displacement and the offset should be
    // minimized, c) for free vertices, this is just zero
    problem.template add_term<Op::V, 3, 0, 3>(
        [=] __device__(const auto& vh, auto& obj) {
            using ActiveT = ACTIVE_TYPE(vh);

            Eigen::Vector<ActiveT, 3> res;

            if (constraints(vh) != 2) {
                // fixed point should stay fixed
                res = iter_val<ActiveT, 3>(vh, obj, 0, 3);
            }

            if (constraints(vh) != 1) {
                // user-displaced points
                Eigen::Vector<ActiveT, 3> oi = iter_val<ActiveT, 3>(vh, obj);
                Eigen::Vector<ActiveT, 3> ci = cn.template to_eigen<3>(vh);

                res = oi - ci;
            }

            return w_fit_sqrt * res;
        });


    // E_rot, i.e., soft rotation constraints
    problem.template add_term<Op::V, 6, 3, 12>([=] __device__(const auto vh,
                                                              auto&      obj) {
        using ActiveT = ACTIVE_TYPE(vh);

        Eigen::Vector<ActiveT, 6> res;

        Eigen::Vector<ActiveT, 9> rot = iter_val<ActiveT, 9>(vh, obj, 3, 12);

        Eigen::Vector<ActiveT, 3> c0(rot[0], rot[1], rot[2]);
        Eigen::Vector<ActiveT, 3> c1(rot[3], rot[4], rot[5]);
        Eigen::Vector<ActiveT, 3> c2(rot[6], rot[7], rot[8]);

        res[0] = c0.dot(c1);
        res[1] = c1.dot(c2);
        res[2] = c2.dot(c0);
        res[3] = c0.dot(c0) - 1;
        res[4] = c1.dot(c1) - 1;
        res[5] = c2.dot(c2) - 1;


        return w_rot_sqrt * res;
    });


    // E_reg
    problem.template add_term<Op::EV, 3, 0, 12>(
        [=] __device__(const auto& eh, const auto& iter, auto& obj) {
            using ActiveT = ACTIVE_TYPE(eh);

            // first vertex variables
            Eigen::Vector<ActiveT, 12> var0 =
                iter_val<ActiveT, 12>(eh, iter, obj, 0, 0, 12);

            // second vertex variables
            Eigen::Vector<ActiveT, 12> var1 =
                iter_val<ActiveT, 12>(eh, iter, obj, 1, 0, 12);

            // first vertex position
            Eigen::Vector<ActiveT, 3> o0(var0[0], var0[1], var0[2]);

            // second vertex position
            Eigen::Vector<ActiveT, 3> o1(var1[0], var1[1], var1[2]);

            // first vertex rotation matrix
            Eigen::Vector<ActiveT, 3>    c0_v0(var0[3], var0[4], var0[5]);
            Eigen::Vector<ActiveT, 3>    c1_v0(var0[6], var0[7], var0[8]);
            Eigen::Vector<ActiveT, 3>    c2_v0(var0[9], var0[10], var0[11]);
            Eigen::Matrix<ActiveT, 3, 3> r0;
            r0 << c0_v0, c1_v0, c2_v0;

            // second vertex rotation matrix
            Eigen::Vector<ActiveT, 3>    c0_v1(var1[3], var1[4], var1[5]);
            Eigen::Vector<ActiveT, 3>    c1_v1(var1[6], var1[7], var1[8]);
            Eigen::Vector<ActiveT, 3>    c2_v1(var1[9], var1[10], var1[11]);
            Eigen::Matrix<ActiveT, 3, 3> r1;
            r1 << c0_v1, c1_v1, c2_v1;

            // first and second vertex rest position
            Eigen::Vector<T, 3> u0 = Urshape.to_eigen<3>(iter[0]);
            Eigen::Vector<T, 3> u1 = Urshape.to_eigen<3>(iter[1]);
            Eigen::Vector<T, 3> du = u1 - u0;


            Eigen::Vector<ActiveT, 3> res = (o1 - o0) - (r0 * du);


            return w_rot_sqrt * res;
        });

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
        for (int i = 0; i < iterations; i++) {


            // solve for position via Newton
            // for (int iter = 0; iter < newton_max_iter; ++iter) {
            //    // evaluate energy terms
            //    problem.eval_terms();
            //
            //    // get the current value of the loss function
            //    T f = problem.get_current_loss();
            //    RXMESH_INFO(
            //        "Iter {} =, Newton Iter= {}: Energy = {}", i, iter, f);
            //
            //    // apply bc
            //    newton_solver.apply_bc(bc);
            //
            //    // direction newton
            //    newton_solver.compute_direction();
            //
            //
            //    // newton decrement
            //    if (0.5f * problem.grad.dot(newton_solver.dir) <
            //        convergence_eps) {
            //        break;
            //    }
            //
            //    // line search
            //    newton_solver.line_search();
            //}
        }


#if USE_POLYSCOPE
        // repurpose cn to be the new position (for gui only)
        rx.for_each_vertex(DEVICE, [=] __device__(auto vh) {
            for (int i = 0; i < 3; ++i) {
                cn(vh, i) = Urshape(vh, i) + offset(vh, i);
            }
        });
        cn.move(DEVICE, HOST);
        rx.get_polyscope_mesh()->updateVertexPositions(cn);
#endif
    };

#ifdef USE_POLYSCOPE
    polyscope::view::upDir         = polyscope::UpDir::ZUp;
    polyscope::state::userCallback = polyscope_callback;
    polyscope::show();

#endif
}

int main(int argc, char** argv)
{
    rx_init(0);

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