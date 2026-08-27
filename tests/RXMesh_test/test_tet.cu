#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <vector>

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/MshLoader.h"
#include "rxmesh/util/macros.h"

TEST(Util, Tet)
{
    std::vector<std::vector<rx_coord_t>> vertices;
    std::vector<std::vector<uint32_t>>   tets;

    EXPECT_EQ(rxmesh::load_msh(STRINGIFY(INPUT_DIR) "car.msh", vertices, tets),
              rxmesh::MeshKind::Tet);
    ASSERT_EQ(vertices.size(), 2927);
    ASSERT_EQ(tets.size(), 11843);
    EXPECT_EQ(tets.front(), (std::vector<uint32_t>{1766, 2094, 1446, 297}));
    EXPECT_EQ(tets.back(), (std::vector<uint32_t>{87, 2779, 126, 2859}));
}