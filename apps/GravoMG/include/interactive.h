#pragma once
#include <imgui.h>


/*

void renderOutputMesh(VectorCSR3D X, RXMeshStatic &rx)
{
    std::vector<std::array<double, 3>> vertexMeshPositions;
    vertexMeshPositions.resize(X.n);

    for (int i = 0; i < X.n; i++) {
        vertexMeshPositions[i] = {X.vector[3 * i],
                                  X.vector[3 * i + 1],
                                  X.vector[3 * i + 2]};
    }
     polyscope::registerSurfaceMesh("output mesh",
                                  vertexMeshPositions,
                                rx.get_polyscope_mesh()->faces);
}

void interactiveMenu(GMGVCycle &gmg, RXMeshStatic &rx)
{
    

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
            std::cout
                << "\n---------------NEW SOLVE INITIATED--------------------\n";
            gmg.X.reset();
            gmg.solve();
            renderOutputMesh(gmg.X, rx);
        }

        ImGui::End();
    };

    polyscope::state::userCallback = polyscope_callback;

}
*/