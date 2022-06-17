#include "gtest/gtest.h"

#include "rxmesh/lp_pair.cuh"

TEST(RXMesh, LP_Pair)
{
    using namespace rxmesh;

    uint16_t local_id             = 5;
    uint16_t local_id_owner_patch = 10;
    uint8_t  patch                = 7;
    LPPair   pair(local_id, local_id_owner_patch, patch);
    EXPECT_EQ(pair.local_id(), local_id);
    EXPECT_EQ(pair.local_id_in_owner_patch(), local_id_owner_patch);
    EXPECT_EQ(pair.patch_stash_id(), patch);
}