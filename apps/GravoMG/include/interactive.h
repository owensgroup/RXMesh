#pragma once
#include <imgui.h>

#include "rxmesh/rxmesh_static.h"

#include "VCycle.h"

#ifdef USE_POLYSCOPE
void render_output_mesh(VectorCSR3D X, rxmesh::RXMeshStatic& rx)
{
    std::vector<std::array<double, 3>> vertexMeshPositions;
    vertexMeshPositions.resize(X.n);

    for (int i = 0; i < X.n; i++) {
        vertexMeshPositions[i] = {
            X.vector[3 * i], X.vector[3 * i + 1], X.vector[3 * i + 2]};
    }
    polyscope::registerSurfaceMesh(
        "output mesh", vertexMeshPositions, rx.get_polyscope_mesh()->faces);
}

 template <typename T>
 void render_output_mesh(rxmesh::DenseMatrix<T>& X, rxmesh::RXMeshStatic& rx)
{
     std::vector<std::array<double, 3>> vertexMeshPositions;
     vertexMeshPositions.resize(X.rows());

     for (int i = 0; i < X.rows(); i++) {
         vertexMeshPositions[i] = {X(i, 0), X(i, 1), X(i, 2)};
     }
     polyscope::registerSurfaceMesh(
         "output mesh", vertexMeshPositions, rx.get_polyscope_mesh()->faces);
 }


void interactive_menu(GMGVCycle& gmg, rxmesh::RXMeshStatic& rx)
{
    Timers<GPUTimer> timers;
    timers.add("solve");

    auto polyscope_callback = [&]() mutable {
        ImGui::Begin("GMG Parameters");

        ImGui::InputInt("Number of Levels", &gmg.max_number_of_levels);
        ImGui::InputInt("Number of V cycles", &gmg.numberOfCycles);
        ImGui::InputInt("Number of pre solve smoothing iterations",
                        &gmg.pre_relax_iterations);
        ImGui::InputInt("Number of post solve smoothing iterations",
                        &gmg.post_relax_iterations);
        ImGui::InputInt("Number of direct solve iterations",
                        &gmg.directSolveIterations);
        ImGui::SliderFloat("Omega", &gmg.omega, 0.0, 1.0);


        if (ImGui::Button("Run V Cycles again")) {
            RXMESH_TRACE("==== NEW SOLVE INITIATED====");
            // gmg.X.reset();

            timers.start("solve");
            gmg.solve();
            timers.stop("solve");

            RXMESH_TRACE("Solving time: {}", timers.elapsed_millis("solve"));

            render_output_mesh(gmg.X, rx);
        }

        ImGui::End();
    };

    polyscope::state::userCallback = polyscope_callback;

    polyscope::show();
}
#endif