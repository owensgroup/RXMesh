#include "gtest/gtest.h"

#include "rxmesh/util/MshLoader.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/log.h"

#if USE_POLYSCOPE
#include "polyscope/volume_mesh.h"
#endif

TEST(Util, Tet)
{
    using namespace rxmesh;

    MshLoader mshload(STRINGIFY(INPUT_DIR) "car.msh");

    RXMESH_INFO("#Nodes = {}", mshload.get_nodes().size());    

    //polyscope::registerTetMesh("my mesh", V, T);
}