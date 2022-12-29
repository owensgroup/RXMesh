// Reference
// https://github.com/taichi-dev/meshtaichi/blob/main/xpbd_cloth/solver.py

#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <uint32_t blockThreads>
void __global__ init_edges(const Context                context,
                           const VertexAttribute<float> x,
                           const EdgeAttribute<float>   rest_len)
{
    auto calc_rest_len = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        auto v0 = iter[0];
        auto v1 = iter[1];

        const Vector3f x0(x(v0, 0), x(v0, 1), x(v0, 2));
        const Vector3f x1(x(v1, 0), x(v1, 1), x(v1, 2));

        rest_len(eh, 0) = (x0 - x1).norm();
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    query.dispatch<Op::EV>(
        block, calc_rest_len, [](EdgeHandle) { return true; }, false);
}


template <uint32_t blockThreads>
void __global__ solve_stretch(const Context                context,
                              VertexAttribute<float>       dp,
                              EdgeAttribute<float>         la_s,
                              const VertexAttribute<float> invM,
                              const VertexAttribute<float> new_x,
                              const EdgeAttribute<float>   rest_len,
                              const bool                   XPBD,
                              const float                  stretch_compliance,
                              const float                  stretch_relaxation,
                              const float                  dt)
{
    auto solve = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        auto v0 = iter[0];
        auto v1 = iter[1];

        const Vector3f x0(new_x(v0, 0), new_x(v0, 1), new_x(v0, 2));
        const Vector3f x1(new_x(v1, 0), new_x(v1, 1), new_x(v1, 2));

        const float w1(invM(v0, 0)), w2(invM(v1, 0));

        if (w1 + w2 > 0.f) {
            Vector3f    n = x0 - x1;
            const float d = n.norm();
            Vector3f    dpp(0.f, 0.f, 0.f);
            const float constraint = (d - rest_len(eh, 0));

            n.normalize();
            if (XPBD) {
                const float compliance = stretch_compliance / (dt * dt);

                const float d_lambda =
                    -(constraint + compliance * la_s(eh, 0)) /
                    (w1 + w2 + compliance) * stretch_relaxation;

                for (int i = 0; i < 3; ++i) {
                    dpp[i] = d_lambda * n[i];
                }
                la_s(eh, 0) += d_lambda;

            } else {
                for (int i = 0; i < 3; ++i) {
                    dpp[i] =
                        -constraint / (w1 + w2) * n[i] * stretch_relaxation;
                }
            }

            for (int i = 0; i < 3; ++i) {
                ::atomicAdd(&dp(v0, i), dpp[i] * w1);
                ::atomicAdd(&dp(v1, i), -(dpp[i] * w2));
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    query.dispatch<Op::EV>(
        block, solve, [](EdgeHandle) { return true; }, false);
}


int main(int argc, char** argv)
{
    Log::init();

#if USE_POLYSCOPE
    polyscope::view::upDir                             = polyscope::UpDir::ZUp;
    polyscope::options::groundPlaneHeightFactor        = 0.2;
    polyscope::options::openImGuiWindowForUserCallback = false;
#endif
    // set device
    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "cloth.obj");

    constexpr uint32_t blockThreads = 256;

    // XPBD paramters
    const float    frame_dt = 1e-2;
    const float    dt       = 5e-4;
    const Vector3f gravity(0.f, 0.f, -15.0f);
    const uint32_t rest_iter          = 5;
    const float    stretch_relaxation = 0.3;
    const float    bending_relaxation = 0.2;
    const float    stretch_compliance = 1e-7;
    const float    bending_compliance = 1e-6;
    const float    mass               = 1.0;
    const bool     XPBD               = true;

    // fixtures paramters
    const float    box_len  = 0.01;
    const Vector3f boxes[4] = {{0.25, 0.75, 0.75},
                               {0.75, 0.75, 0.75},
                               {0.25, 0.25, 0.75},
                               {0.75, 0.25, 0.75}};

    // mesh data
    auto x     = rx.get_input_vertex_coordinates();
    auto new_x = rx.add_vertex_attribute<float>("new_x", 3);
    auto v     = rx.add_vertex_attribute<float>("v", 3);
    auto invM  = rx.add_vertex_attribute<float>("invM", 1);
    auto dp    = rx.add_vertex_attribute<float>("dp", 3);

    auto rest_len = rx.add_edge_attribute<float>("rest_len", 1);
    auto la_s     = rx.add_edge_attribute<float>("la_s", 1);
    auto la_b     = rx.add_edge_attribute<float>("la_b", 1);


    // initialize
    rx.for_each_vertex(DEVICE,
                       [mass, boxes, box_len, invM = *invM, x = *x] __device__(
                           VertexHandle vh) {
                           invM(vh, 0) = mass;
                           Vector3f v(x(vh, 0), x(vh, 1), x(vh, 2));
                           float    eps = std::numeric_limits<float>::epsilon();
                           if ((v - boxes[0]).norm2() - box_len < eps ||
                               (v - boxes[1]).norm2() - box_len < eps ||
                               (v - boxes[2]).norm2() - box_len < eps ||
                               (v - boxes[3]).norm2() - box_len < eps) {
                               invM(vh, 0) = 0;
                           }
                       });

    LaunchBox<blockThreads> init_edges_lb;
    LaunchBox<blockThreads> solve_stretch_lb;

    rx.prepare_launch_box(
        {Op::EV}, init_edges_lb, (void*)init_edges<blockThreads>);

    rx.prepare_launch_box(
        {Op::EV}, solve_stretch_lb, (void*)solve_stretch<blockThreads>);

    init_edges<blockThreads>
        <<<init_edges_lb.blocks,
           init_edges_lb.num_threads,
           init_edges_lb.smem_bytes_dyn>>>(rx.get_context(), *x, *rest_len);


    // solve
    auto polyscope_callback = [&]() mutable {
        float frame_time_left = frame_dt;
        while (frame_time_left > 0.0) {
            float dt0 = std::min(dt, frame_time_left);
            frame_time_left -= dt0;

            // applyExtForce
            rx.for_each_vertex(DEVICE,
                               [dt0,
                                gravity,
                                invM  = *invM,
                                v     = *v,
                                new_x = *new_x,
                                x     = *x] __device__(VertexHandle vh) {
                                   if (invM(vh, 0) > 0.0) {
                                       v(vh, 0) += gravity[0] * dt0;
                                       v(vh, 1) += gravity[1] * dt0;
                                       v(vh, 2) += gravity[2] * dt0;
                                   }
                                   new_x(vh, 0) = x(vh, 0) + v(vh, 0) * dt0;
                                   new_x(vh, 1) = x(vh, 1) + v(vh, 1) * dt0;
                                   new_x(vh, 2) = x(vh, 2) + v(vh, 2) * dt0;
                               });

            if (XPBD) {
                la_s->reset(0.0, DEVICE);
                la_b->reset(0.0, DEVICE);
            }

            for (uint32_t iter = 0; iter < rest_iter; ++iter) {
                // preSolve
                dp->reset(0, DEVICE);

                // solveStretch
                solve_stretch<blockThreads>
                    <<<solve_stretch_lb.blocks,
                       solve_stretch_lb.num_threads,
                       solve_stretch_lb.smem_bytes_dyn>>>(rx.get_context(),
                                                          *dp,
                                                          *la_s,
                                                          *invM,
                                                          *new_x,
                                                          *rest_len,
                                                          XPBD,
                                                          stretch_compliance,
                                                          stretch_relaxation,
                                                          dt0);
                // TODO
                //  solveBending(dt0);

                // postSolve
                rx.for_each_vertex(
                    DEVICE,
                    [dp = *dp, new_x = *new_x] __device__(VertexHandle vh) {
                        new_x(vh, 0) += dp(vh, 0);
                        new_x(vh, 1) += dp(vh, 1);
                        new_x(vh, 2) += dp(vh, 2);
                    });
            }

            // update;
            rx.for_each_vertex(
                DEVICE,
                [dt0, invM = *invM, v = *v, new_x = *new_x, x = *x] __device__(
                    VertexHandle vh) {
                    if (invM(vh, 0) <= 0.0) {
                        new_x(vh, 0) = x(vh, 0);
                        new_x(vh, 1) = x(vh, 1);
                        new_x(vh, 2) = x(vh, 2);
                    } else {
                        v(vh, 0) = (new_x(vh, 0) - x(vh, 0)) / dt0;
                        v(vh, 1) = (new_x(vh, 1) - x(vh, 1)) / dt0;
                        v(vh, 2) = (new_x(vh, 2) - x(vh, 2)) / dt0;

                        x(vh, 0) = new_x(vh, 0);
                        x(vh, 1) = new_x(vh, 1);
                        x(vh, 2) = new_x(vh, 2);
                    }
                });
        }

        x->move(DEVICE, HOST);
        rx.get_polyscope_mesh()->updateVertexPositions(*x);
    };

#if USE_POLYSCOPE
    polyscope::state::userCallback = polyscope_callback;
    polyscope::show();
#endif
}