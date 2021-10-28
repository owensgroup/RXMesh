#include "gtest/gtest.h"
#include "rxmesh/util/vector.h"

TEST(RXMesh, Vector)
{
    using namespace rxmesh;

    // constrctors
    Vector3f v0(0.5f);
    EXPECT_TRUE(v0[0] == 0.5f && v0[1] == 0.5f && v0[2] == 0.5f);

    Vector3f v1(v0);
    EXPECT_TRUE(v1[0] == 0.5f && v1[1] == 0.5f && v1[2] == 0.5f);

    Vector3i v2(10, 20, 30);
    EXPECT_TRUE(v2[0] == 10 && v2[1] == 20 && v2[2] == 30);

    Vector4ui c = Vector4ui::constant(5);
    EXPECT_TRUE(c[0] == 5 && c[1] == 5 && c[2] == 5 && c[3] == 5);

    Vector2s z = Vector2s::zero();
    EXPECT_TRUE(z[0] == 0 && z[1] == 0);

    // assignment
    z[0] = 10;
    z[1] = 20;
    EXPECT_TRUE(z[0] == 10 && z[1] == 20);

    // neg
    Vector2s neg_z = -z;
    EXPECT_TRUE(neg_z[0] == -10 && neg_z[1] == -20);

    // sum
    auto sum = z + z;
    EXPECT_TRUE(sum[0] == 20 && sum[1] == 40);

    z += z;
    EXPECT_TRUE(z[0] == 20 && z[1] == 40);

    // diff
    auto diff = neg_z - z;
    EXPECT_TRUE(diff[0] == -30 && diff[1] == -60);

    neg_z -= z;
    EXPECT_TRUE(neg_z[0] == -30 && neg_z[1] == -60);

    // mul
    auto mul = z * z;
    EXPECT_TRUE(mul[0] == 20 * 20 && mul[1] == 40 * 40);

    z *= z;
    EXPECT_TRUE(z[0] == 20 * 20 && z[1] == 40 * 40);

    // division
    auto div = mul / z;
    EXPECT_TRUE(div[0] == 1 && div[1] == 1);

    v0 /= 0.2f;
    EXPECT_TRUE(v0[0] == 2.5f && v0[1] == 2.5f && v0[2] == 2.5f);

    // equality
    EXPECT_TRUE(mul == z);

    // std::cout << "mul= " << mul << " v0= " << v0;

    // norm
    EXPECT_TRUE(v0.norm2() == 2.5 * 2.5 * 3);

    // sum
    EXPECT_EQ(neg_z.sum(), -90);

    // max
    EXPECT_EQ(neg_z.max(), -30);

    // min
    EXPECT_EQ(neg_z.min(), -60);

    // normalize
    normalize(v0);
    EXPECT_NEAR(norm(v0), 1, 0.001);
}
