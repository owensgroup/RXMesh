#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename FunT, typename VAttrT>
void draw(RXMeshStatic& rx,
          VAttrT&       x_tilde,
          VAttrT&       velocity,
          FunT&         step_forward,
          int&          step)
{
    polyscope::options::groundPlaneHeightFactor = 0.37;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::Tile;

    bool is_running = false;

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
        if (ImGui::Button("Pause")) {
            is_running = false;
        }

        ImGui::SameLine();
        if (ImGui::Button("Export")) {
            rx.export_obj("MS_" + std::to_string(step) + ".obj", x_tilde);
        }

        if (is_running) {
            step_and_update();
        }
    };

    polyscope::state::userCallback = ps_callback;
    polyscope::show();
}