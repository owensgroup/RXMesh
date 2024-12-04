// Reference
// https://github.com/taichi-dev/meshtaichi/blob/main/xpbd_cloth/solver.py

#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include "imgui.h"
#include "polyscope/polyscope.h"

#include "rxmesh/geometry_factory.h"

using namespace rxmesh;

template <uint32_t blockThreads>
void __global__ init_edges(const Context                context,
                           const VertexAttribute<float> x,
                           const EdgeAttribute<float>   rest_len)
{
    auto calc_rest_len = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        auto v0 = iter[0];
        auto v1 = iter[1];

        const glm::fvec3 x0 = x.to_glm<3>(v0);
        const glm::fvec3 x1 = x.to_glm<3>(v1);

        rest_len(eh, 0) = glm::length(x0 - x1);
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, calc_rest_len);
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
                              const float                  dt2)
{
    auto solve = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        auto v1 = iter[0];
        auto v2 = iter[1];

        const glm::fvec3 x1 = new_x.to_glm<3>(v1);
        const glm::fvec3 x2 = new_x.to_glm<3>(v2);

        const float w1(invM(v1, 0)), w2(invM(v2, 0));

        if (w1 + w2 > 0.f) {
            glm::fvec3  n = x1 - x2;
            const float d = glm::length(n);
            glm::fvec3  dpp(0.f, 0.f, 0.f);
            const float constraint = (d - rest_len(eh, 0));

            n = glm::normalize(n);
            if (XPBD) {
                const float compliance = stretch_compliance / dt2;

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
                ::atomicAdd(&dp(v1, i), dpp[i] * w1);
                ::atomicAdd(&dp(v2, i), -(dpp[i] * w2));
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, solve);
}

template <uint32_t blockThreads>
void __global__ solve_bending(const Context                context,
                              VertexAttribute<float>       dp,
                              EdgeAttribute<float>         la_b,
                              const VertexAttribute<float> invM,
                              const VertexAttribute<float> new_x,
                              const bool                   XPBD,
                              const float                  bending_compliance,
                              const float                  bending_relaxation,
                              const float                  dt2)
{
    auto solve = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        // iter[0] and iter[2] are the edge two vertices
        // iter[1] and iter[3] are the two opposite vertices

        auto v1 = iter[0];
        auto v2 = iter[2];

        auto v3 = iter[1];
        auto v4 = iter[3];

        if (v3.is_valid() && v4.is_valid()) {
            const float w1(invM(v1, 0)), w2(invM(v2, 0)), w3(invM(v3, 0)),
                w4(invM(v4, 0));
            if (w1 + w2 + w3 + w4 > 0.f) {
                glm::fvec3 p2(new_x(v2, 0) - new_x(v1, 0),
                              new_x(v2, 1) - new_x(v1, 1),
                              new_x(v2, 2) - new_x(v1, 2));
                glm::fvec3 p3(new_x(v3, 0) - new_x(v1, 0),
                              new_x(v3, 1) - new_x(v1, 1),
                              new_x(v3, 2) - new_x(v1, 2));
                glm::fvec3 p4(new_x(v4, 0) - new_x(v1, 0),
                              new_x(v4, 1) - new_x(v1, 1),
                              new_x(v4, 2) - new_x(v1, 2));

                float l23 = glm::length(glm::cross(p2, p3));
                float l24 = glm::length(glm::cross(p2, p4));
                if (l23 < 1e-8) {
                    l23 = 1.f;
                }
                if (l24 < 1e-8) {
                    l24 = 1.f;
                }
                glm::fvec3 n1 = glm::cross(p2, p3);
                n1 /= l23;
                glm::fvec3 n2 = glm::cross(p2, p4);
                n2 /= l24;

                // clamp(dot(n1, n2), -1., 1.)
                float d = std::max(1.f, std::min(dot(n1, n2), -1.f));

                glm::fvec3 q3 = (cross(p2, n2) + cross(n1, p2) * d) / l23;
                glm::fvec3 q4 = (cross(p2, n1) + cross(n2, p2) * d) / l24;
                glm::fvec3 q2 = -(cross(p3, n2) + cross(n1, p3) * d) / l23 -
                                (cross(p4, n1) + cross(n2, p4) * d) / l24;
                glm::fvec3 q1 = -q2 - q3 - q4;

                float sum_wq = w1 * glm::length2(q1) + w2 * glm::length2(q2) +
                               w3 * glm::length2(q3) + w4 * glm::length2(q4);
                float constraint = acos(d) - acos(-1.);

                if (XPBD) {
                    float compliance = bending_compliance / dt2;
                    float d_lambda = -(constraint + compliance * la_b(eh, 0)) /
                                     (sum_wq + compliance) * bending_relaxation;

                    constraint = sqrt(1 - d * d) * d_lambda;
                    la_b(eh, 0) += d_lambda;
                } else {
                    constraint = -sqrt(1 - d * d) * constraint /
                                 (sum_wq + 1e-7) * bending_relaxation;
                }
                for (int i = 0; i < 3; ++i) {
                    ::atomicAdd(&dp(v1, i), w1 * constraint * q1[i]);
                    ::atomicAdd(&dp(v2, i), w2 * constraint * q2[i]);
                    ::atomicAdd(&dp(v3, i), w3 * constraint * q3[i]);
                    ::atomicAdd(&dp(v4, i), w4 * constraint * q4[i]);
                }
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, solve);
}

template <uint32_t blockThreads, bool XPBD>
void __global__ solve_stretch_and_bending(const Context                context,
                                          VertexAttribute<float>       dp,
                                          EdgeAttribute<float>         la_s,
                                          EdgeAttribute<float>         la_b,
                                          const VertexAttribute<float> invM,
                                          const VertexAttribute<float> new_x,
                                          const EdgeAttribute<float>   rest_len,
                                          const float stretch_compliance,
                                          const float stretch_relaxation,
                                          const float bending_compliance,
                                          const float bending_relaxation,
                                          const float dt2_inv)
{
    auto solve = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        // iter[0] and iter[2] are the edge two vertices
        // iter[1] and iter[3] are the two opposite vertices

        auto v1 = iter[0];
        auto v2 = iter[2];

        auto v3 = iter[1];
        auto v4 = iter[3];

        float v1_st[3] = {0, 0, 0};
        float v2_st[3] = {0, 0, 0};

        const float w1(invM(v1, 0)), w2(invM(v2, 0));

        const glm::fvec3 x1 = new_x.to_glm<3>(v1);
        const glm::fvec3 x2 = new_x.to_glm<3>(v2);

        // stretch term (for v1 and v2)
        if (w1 + w2 > 0.f) {
            glm::fvec3  n = x1 - x2;
            const float d = glm::length(n);
            glm::fvec3  dpp(0.f, 0.f, 0.f);
            const float constraint = (d - rest_len(eh, 0));

            n = glm::normalize(n);
            if constexpr (XPBD) {
                const float compliance = stretch_compliance * dt2_inv;

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
                v1_st[i] = dpp[i] * w1;
                v2_st[i] = -(dpp[i] * w2);
            }
        }

        // bending term (for v1, v2, v3, and v4)
        if (v3.is_valid() && v4.is_valid()) {
            const float w3(invM(v3, 0)), w4(invM(v4, 0));

            const glm::fvec3 x3 = new_x.to_glm<3>(v3);
            const glm::fvec3 x4 = new_x.to_glm<3>(v4);


            if (w1 + w2 + w3 + w4 > 0.f) {

                glm::fvec3 p2 = x2 - x1;
                glm::fvec3 p3 = x3 - x1;
                glm::fvec3 p4 = x4 - x1;

                float l23 = glm::length(glm::cross(p2, p3));
                float l24 = glm::length(glm::cross(p2, p4));
                if (l23 < 1e-8) {
                    l23 = 1.f;
                }
                if (l24 < 1e-8) {
                    l24 = 1.f;
                }
                glm::fvec3 n1 = glm::cross(p2, p3);
                n1 /= l23;
                glm::fvec3 n2 = glm::cross(p2, p4);
                n2 /= l24;

                // clamp(dot(n1, n2), -1., 1.)
                float d = std::max(1.f, std::min(dot(n1, n2), -1.f));

                glm::fvec3 q3 = (cross(p2, n2) + cross(n1, p2) * d) / l23;
                glm::fvec3 q4 = (cross(p2, n1) + cross(n2, p2) * d) / l24;
                glm::fvec3 q2 = -(cross(p3, n2) + cross(n1, p3) * d) / l23 -
                                (cross(p4, n1) + cross(n2, p4) * d) / l24;
                glm::fvec3 q1 = -q2 - q3 - q4;

                float sum_wq = w1 * glm::length2(q1) + w2 * glm::length2(q2) +
                               w3 * glm::length2(q3) + w4 * glm::length2(q4);
                float constraint = acos(d) - acos(-1.);

                if constexpr (XPBD) {
                    float compliance = bending_compliance * dt2_inv;
                    float d_lambda = -(constraint + compliance * la_b(eh, 0)) /
                                     (sum_wq + compliance) * bending_relaxation;

                    constraint = sqrt(1 - d * d) * d_lambda;
                    la_b(eh, 0) += d_lambda;
                } else {
                    constraint = -sqrt(1 - d * d) * constraint /
                                 (sum_wq + 1e-7) * bending_relaxation;
                }
                for (int i = 0; i < 3; ++i) {
                    ::atomicAdd(&dp(v1, i), w1 * constraint * q1[i] + v1_st[i]);
                    ::atomicAdd(&dp(v2, i), w2 * constraint * q2[i] + v2_st[i]);
                    ::atomicAdd(&dp(v3, i), w3 * constraint * q3[i]);
                    ::atomicAdd(&dp(v4, i), w4 * constraint * q4[i]);
                }
            }
        } else {
            for (int i = 0; i < 3; ++i) {
                ::atomicAdd(&dp(v1, i), v1_st[i]);
                ::atomicAdd(&dp(v2, i), v2_st[i]);
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, solve);
}

int main(int argc, char** argv)
{
    Log::init();

#if USE_POLYSCOPE
    polyscope::view::upDir                             = polyscope::UpDir::ZUp;
    polyscope::options::groundPlaneHeightFactor        = 1.5;
    polyscope::options::openImGuiWindowForUserCallback = false;
#endif
    // set device
    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "cloth.obj");

    // std::vector<std::vector<float>>    verts;
    // std::vector<std::vector<uint32_t>> fv;
    // const int                          nnn = 540;
    // const float                        dxx = 1.0f / float(nnn);
    // rxmesh::create_plane(verts, fv, nnn, nnn, 2, dxx);
    // RXMeshStatic rx(fv);
    // rx.add_vertex_coordinates(verts, "Coords");

    // scale mesh info unit bounding box
    rx.scale({0.f, 0.f, 0.f}, {1.f, 1.f, 1.f});


    constexpr uint32_t blockThreads = 320;

    // XPBD paramters
    const float      frame_dt = 1e-2;
    const float      dt       = 5e-4;
    const glm::fvec3 gravity(0.f, 0.f, -15.0f);
    const uint32_t   rest_iter          = 5;
    const float      stretch_relaxation = 0.3;
    const float      bending_relaxation = 0.2;
    const float      stretch_compliance = 1e-7;
    const float      bending_compliance = 1e-6;
    const float      mass               = 1.0;
    constexpr bool   XPBD               = true;

    // fixtures paramters
    const glm::fvec4 fixure_spheres[4] = {{0.f, 1.f, 0.f, 0.004},
                                          {1.f, 1.f, 0.f, 0.004},
                                          {0.f, 0.f, 0.f, 0.004},
                                          {1.f, 0.f, 0.f, 0.004}};

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
    rx.for_each_vertex(
        DEVICE,
        [mass, fixure_spheres, invM = *invM, x = *x] __device__(
            VertexHandle vh) {
            invM(vh, 0) = mass;
            glm::fvec4 v(x(vh, 0), x(vh, 1), x(vh, 2), 0.f);
            float      eps = std::numeric_limits<float>::epsilon();
            if (glm::length2(v - fixure_spheres[0]) - fixure_spheres[0][3] <
                    eps ||
                glm::length2(v - fixure_spheres[1]) - fixure_spheres[1][3] <
                    eps ||
                glm::length2(v - fixure_spheres[2]) - fixure_spheres[2][3] <
                    eps ||
                glm::length2(v - fixure_spheres[3]) - fixure_spheres[3][3] <
                    eps) {
                invM(vh, 0) = 0;
            }
        });


    LaunchBox<blockThreads> solve_lb;

    LaunchBox<blockThreads> solve_stretch_lb;
    LaunchBox<blockThreads> solve_bending_lb;

    rx.prepare_launch_box(
        {Op::EV}, solve_stretch_lb, (void*)solve_stretch<blockThreads>);

    rx.prepare_launch_box(
        {Op::EVDiamond}, solve_bending_lb, (void*)solve_bending<blockThreads>);

    rx.prepare_launch_box({Op::EVDiamond},
                          solve_lb,
                          (void*)solve_stretch_and_bending<blockThreads, XPBD>);

    // init edges
    rx.run_kernel<blockThreads>(
        {Op::EV}, init_edges<blockThreads>, *x, *rest_len);

    int frame      = 0;
    int max_frames = 100;

    bool  test = false;
    float mean(0.f);
    float mean2(0.f);


    // solve
    bool started = false;

    float total_time = 0;

    auto polyscope_callback = [&]() mutable {
        if (ImGui::Button("Start Simulation") || started) {
            started = true;

            GPUTimer timer;
            timer.start();
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
                    // rx.run_kernel(solve_stretch_lb,
                    //              solve_stretch<blockThreads>,
                    //              *dp,
                    //              *la_s,
                    //              *invM,
                    //              *new_x,
                    //              *rest_len,
                    //              XPBD,
                    //              stretch_compliance,
                    //              stretch_relaxation,
                    //              dt0 * dt0);
                    //
                    //
                    ////  solveBending
                    // rx.run_kernel(solve_bending_lb,
                    //               solve_bending<blockThreads>,
                    //               *dp,
                    //               *la_b,
                    //               *invM,
                    //               *new_x,
                    //               XPBD,
                    //               bending_compliance,
                    //               bending_relaxation,
                    //               dt0 * dt0);

                    // solve Stretch and bending
                    rx.run_kernel(solve_lb,
                                  solve_stretch_and_bending<blockThreads, XPBD>,
                                  *dp,
                                  *la_b,
                                  *la_s,
                                  *invM,
                                  *new_x,
                                  *rest_len,
                                  stretch_compliance,
                                  stretch_relaxation,
                                  bending_compliance,
                                  bending_relaxation,
                                  1.0f / (dt0 * dt0));

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
                    [dt0,
                     invM  = *invM,
                     v     = *v,
                     new_x = *new_x,
                     x     = *x] __device__(VertexHandle vh) {
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

            timer.stop();
            RXMESH_INFO(
                "Frame {}, time= {}(ms)", frame, timer.elapsed_millis());
            total_time += timer.elapsed_millis();

#if USE_POLYSCOPE
            x->move(DEVICE, HOST);
            rx.get_polyscope_mesh()->updateVertexPositions(*x);
#endif
            frame++;
            if (test) {
                if (frame == 99) {
                    rx.for_each_vertex(HOST, [&](VertexHandle vh) {
                        for (int i = 0; i < 3; ++i) {
                            mean += (*x)(vh, i);
                            mean2 += (*x)(vh, i) * (*x)(vh, i);
                        }
                    });
                    mean /= (3.f * rx.get_num_vertices());
                    mean2 /= (3.f * rx.get_num_vertices());
                }
            }
            // if (frame >= max_frames) {
            //     RXMESH_INFO("fps = {}", (frame * 100.f) / total_time);
            //     exit(0);
            // }
        }
    };

    // started = true;
    // while (true) {
    //     polyscope_callback();
    // }

#if USE_POLYSCOPE
    polyscope::state::userCallback = polyscope_callback;
    polyscope::show();
#endif

    if (test) {
        RXMESH_INFO("mean= {}, mean2= {}", mean, mean2);
    }
}