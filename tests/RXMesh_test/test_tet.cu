#include "gtest/gtest.h"

#include "rxmesh/util/MshLoader.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/log.h"

#include "polyscope/volume_mesh.h"

TEST(Util, Tet)
{
    using namespace rxmesh;

    MshLoader mshload(STRINGIFY(INPUT_DIR) "car.msh");

    RXMESH_INFO("#Nodes = {}", mshload.get_nodes().size());    

    //polyscope::registerTetMesh("my mesh", V, T);
}