#pragma once

#include "rxmesh/rxmesh_static.h"

using namespace rxmesh;

template <typename FunT, typename VAttrT, typename ProblemT>
void draw(RXMeshStatic& rx,
          ProblemT&     problem,
          VAttrT&       x,
          VAttrT&       velocity,
          FunT&         step_forward,
          int&          step)
{
#if USE_POLYSCOPE

    polyscope::options::groundPlaneHeightFactor = 0.37;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::Tile;

    bool is_running = false;

    auto dir  = *rx.add_vertex_attribute<float>("Dir", 3);
    auto grad = *rx.add_vertex_attribute<float>("Grad", 3);

    auto ps_callback = [&]() mutable {
        auto step_and_update = [&]() {
            step_forward();
            x.move(DEVICE, HOST);
            velocity.move(DEVICE, HOST);
            auto vel = rx.get_polyscope_mesh()->addVertexVectorQuantity(
                "Velocity", velocity);

            rx.get_polyscope_mesh()->updateVertexPositions(x);
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
        if (ImGui::Button("ExportObj")) {
            rx.export_obj("NeoHookean_" + std::to_string(step) + ".obj", x);
        }

        ImGui::SameLine();
        if (ImGui::Button("ExportHess")) {
            problem.hess->to_file("NeoHookean_hess_" + std::to_string(step));
        }

        if (is_running) {
            step_and_update();
        }
    };

    polyscope::state::userCallback = ps_callback;
    polyscope::show();
#endif
}