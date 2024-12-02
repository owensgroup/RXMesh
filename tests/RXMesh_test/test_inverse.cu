#include "gtest/gtest.h"

#include <Eigen/Dense>

#include "rxmesh/util/inverse.h"


TEST(Diff, Inverse2x2)
{
    using namespace rxmesh;

    Eigen::Matrix2f m2x2;
    m2x2 << 4, 5,  //
        8, 10;

    Eigen::Matrix2f m2x2_inv = inverse(m2x2);

    EXPECT_TRUE(m2x2_inv == m2x2.inverse());
}


TEST(Diff, Inverse3x3)
{
    using namespace rxmesh;

    Eigen::Matrix3f m3x3;
    m3x3 << 4, 5, 7,  //
        8, 10, 9,     //
        88, 13, 99;

    Eigen::Matrix3f m3x3_inv = inverse(m3x3);

    EXPECT_TRUE(m3x3_inv == m3x3.inverse());
}


TEST(Diff, Inverse4x4)
{
    using namespace rxmesh;

    Eigen::Matrix4f m4x4;
    // m4x4 << 1, 2, 3, 4,  //
    //     5, 6, 7, 8,      //
    //     9, 7, 6, 6,      //
    //     7, 5, 3, 8;

    m4x4 << 0.1, 0.4354, 0.2343, 0.4139,       //
        0.4631, 0.713954, 0.1319, 0.00146379,  //
        9, 7, 6, 6,                            //
        0.84, 0.3493, 0.7430, 0.001;

    Eigen::Matrix4f m4x4_inv = inverse(m4x4);

    Eigen::Matrix4f m4x4_eigen_inv = m4x4.inverse();

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(m4x4_inv(i, j), m4x4_eigen_inv(i, j), 1e-6);
        }
    }
}