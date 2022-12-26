// Reference
// https://github.com/taichi-dev/meshtaichi/blob/main/xpbd_cloth/solver.py

#include "rxmesh/rxmesh_static.h"

int main(int argc, char** argv)
{
    using namespace rxmesh;
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
    const float    box_len  = 0.013;
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

    // solve
    auto solve = [&]() mutable {
        float    frame_time_left = frame_dt;
        uint32_t substep         = 0;
        while (frame_time_left > 0.0) {
            substep += 1;
            float dt0 = std::min(dt, frame_time_left);
            frame_time_left -= dt0;

            // applyExtForce(dt0);
            if (XPBD) {
                la_s->reset(0.0, DEVICE);
                la_b->reset(0.0, DEVICE);
            }

            for (uint32_t iter = 0; iter < rest_iter; ++iter) {
                // preSolve();
                // solveStretch(dt0);
                // solveBending(dt0);
                // postSolve(1.0);
            }

            // update(dt0);
        }

        // x->move(DEVICE, HOST);
        // rx.get_polyscope_mesh()->updateVertexPositions(*x);
    };

#if USE_POLYSCOPE
    polyscope::state::userCallback = solve;
    polyscope::show();
#endif
}