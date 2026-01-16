#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;


template <typename T>
void draw_collision_sphere_polyscope(
    const Eigen::Vector3<T>& center,
    T                        radius,
    const std::string& sphere_obj_path = STRINGIFY(INPUT_DIR) "sphere3.obj")
{
    const std::string ps_name = "collision_sphere";

    std::vector<std::vector<float>>    V;
    std::vector<std::vector<uint32_t>> F;
    import_obj(sphere_obj_path, V, F);


    T mx = 0, my = 0, mz = 0;
    for (auto& p : V) {
        mx += T(p[0]);
        my += T(p[1]);
        mz += T(p[2]);
    }
    mx /= T(V.size());
    my /= T(V.size());
    mz /= T(V.size());

    T r0 = 0;
    for (auto& p : V) {
        T x = T(p[0]) - mx;
        T y = T(p[1]) - my;
        T z = T(p[2]) - mz;
        r0  = std::max(r0, std::sqrt(x * x + y * y + z * z));
    }

    T s = radius / r0;
        
    for (auto& p : V) {
        p[0] = (T(p[0]) - mx) * s + center.x();
        p[1] = (T(p[1]) - my) * s + center.y();
        p[2] = (T(p[2]) - mz) * s + center.z();        
    }

    polyscope::registerSurfaceMesh(ps_name, V, F);
}

template <typename FunT, typename VAttrT>
void draw(RXMeshStatic& rx,
          VAttrT&       x_tilde,
          VAttrT&       velocity,
          FunT&         step_forward,
          int&          time_step,
          int           max_time_steps)
{
#if USE_POLYSCOPE
    polyscope::options::groundPlaneHeightFactor = 0.37;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::Tile;

    bool is_running = false;
    bool is_export  = false;

    auto ps_callback = [&]() mutable {
        auto step_and_update = [&]() {
            step_forward();
            x_tilde.move(DEVICE, HOST);
            velocity.move(DEVICE, HOST);
            auto vel = rx.get_polyscope_mesh()->addVertexVectorQuantity(
                "Velocity", velocity);
            rx.get_polyscope_mesh()->updateVertexPositions(x_tilde);
        };
        if (ImGui::Button("Step")) {
            step_and_update();
        }

        ImGui::SameLine();
        if (ImGui::Button("Start")) {
            is_running = true;
        }

        ImGui::SameLine();
        if (ImGui::Button("Pause") || time_step == max_time_steps) {
            is_running = false;
        }

        ImGui::SameLine();
        if (is_export) {
            rx.export_obj(std::to_string(time_step) + ".obj", x_tilde);
        }

        if (ImGui::Button("Export")) {
            is_export = !is_export;
        }
        if (is_running) {
            step_and_update();
        }
    };

    polyscope::state::userCallback = ps_callback;
    polyscope::show();
#endif
}