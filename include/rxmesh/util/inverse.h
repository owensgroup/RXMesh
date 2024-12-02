#pragma once

#include <cuda.h>
#include "rxmesh/types.h"

#include <Eigen/Dense>

namespace rxmesh {
/**
 * @brief since Eigen matrix inverse is buggy on the device (it results into
 * "unspecified launch failure"), we implement our own device-compatible matrix
 * inverse for small typical matrix sizes (i.e., 2x2, 3x3, 4x4) that takes eigen
 * matrix as an input. This implementation where taken from glm
 * Note: we can not glm::inverse() size it is limited to floating-point type
 * while we are interested in rxmesh::Scalar type
 * @tparam T the floating point type of the matrix
 */
template <typename T>
__device__ __host__ __inline__ Eigen::Matrix<T, 2, 2> inverse(
    const Eigen::Matrix<T, 2, 2>& m)
{

    T OneOverDeterminant =
        static_cast<T>(1) / (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1));

    Eigen::Matrix<T, 2, 2> ret;
    ret << m(1, 1) * OneOverDeterminant,  //
        -m(0, 1) * OneOverDeterminant,    //
        -m(1, 0) * OneOverDeterminant,    //
        m(0, 0) * OneOverDeterminant;

    return ret;
}

template <typename T>
__device__ __host__ __inline__ Eigen::Matrix<T, 3, 3> inverse(
    const Eigen::Matrix<T, 3, 3>& m)
{
    Eigen::Matrix<T, 3, 3> ret;


    T OneOverDeterminant =
        static_cast<T>(1) / (m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) -
                             m(1, 0) * (m(0, 1) * m(2, 2) - m(2, 1) * m(0, 2)) +
                             m(2, 0) * (m(0, 1) * m(1, 2) - m(1, 1) * m(0, 2)));


    ret(0, 0) = (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2));
    ret(1, 0) = -(m(1, 0) * m(2, 2) - m(2, 0) * m(1, 2));
    ret(2, 0) = (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1));
    ret(0, 1) = -(m(0, 1) * m(2, 2) - m(2, 1) * m(0, 2));
    ret(1, 1) = (m(0, 0) * m(2, 2) - m(2, 0) * m(0, 2));
    ret(2, 1) = -(m(0, 0) * m(2, 1) - m(2, 0) * m(0, 1));
    ret(0, 2) = (m(0, 1) * m(1, 2) - m(1, 1) * m(0, 2));
    ret(1, 2) = -(m(0, 0) * m(1, 2) - m(1, 0) * m(0, 2));
    ret(2, 2) = (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1));

    ret *= OneOverDeterminant;

    return ret;
}

template <typename T>
__device__ __host__ __inline__ Eigen::Matrix<T, 4, 4> inverse(
    const Eigen::Matrix<T, 4, 4>& m)
{

    T Coef00 = m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2);
    T Coef02 = m(2, 1) * m(3, 3) - m(2, 3) * m(3, 1);
    T Coef03 = m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1);

    T Coef04 = m(1, 2) * m(3, 3) - m(1, 3) * m(3, 2);
    T Coef06 = m(1, 1) * m(3, 3) - m(1, 3) * m(3, 1);
    T Coef07 = m(1, 1) * m(3, 2) - m(1, 2) * m(3, 1);

    T Coef08 = m(1, 2) * m(2, 3) - m(1, 3) * m(2, 2);
    T Coef10 = m(1, 1) * m(2, 3) - m(1, 3) * m(2, 1);
    T Coef11 = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);

    T Coef12 = m(0, 2) * m(3, 3) - m(0, 3) * m(3, 2);
    T Coef14 = m(0, 1) * m(3, 3) - m(0, 3) * m(3, 1);
    T Coef15 = m(0, 1) * m(3, 2) - m(0, 2) * m(3, 1);

    T Coef16 = m(0, 2) * m(2, 3) - m(0, 3) * m(2, 2);
    T Coef18 = m(0, 1) * m(2, 3) - m(0, 3) * m(2, 1);
    T Coef19 = m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1);

    T Coef20 = m(0, 2) * m(1, 3) - m(0, 3) * m(1, 2);
    T Coef22 = m(0, 1) * m(1, 3) - m(0, 3) * m(1, 1);
    T Coef23 = m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1);

    Eigen::Vector<T, 4> Fac0(Coef00, Coef00, Coef02, Coef03);
    Eigen::Vector<T, 4> Fac1(Coef04, Coef04, Coef06, Coef07);
    Eigen::Vector<T, 4> Fac2(Coef08, Coef08, Coef10, Coef11);
    Eigen::Vector<T, 4> Fac3(Coef12, Coef12, Coef14, Coef15);
    Eigen::Vector<T, 4> Fac4(Coef16, Coef16, Coef18, Coef19);
    Eigen::Vector<T, 4> Fac5(Coef20, Coef20, Coef22, Coef23);

    Eigen::Vector<T, 4> Vec0(m(0, 1), m(0, 0), m(0, 0), m(0, 0));
    Eigen::Vector<T, 4> Vec1(m(1, 1), m(1, 0), m(1, 0), m(1, 0));
    Eigen::Vector<T, 4> Vec2(m(2, 1), m(2, 0), m(2, 0), m(2, 0));
    Eigen::Vector<T, 4> Vec3(m(3, 1), m(3, 0), m(3, 0), m(3, 0));

    Eigen::Vector<T, 4> Inv0 = Vec1.cwiseProduct(Fac0) -
                               Vec2.cwiseProduct(Fac1) +
                               Vec3.cwiseProduct(Fac2);
    Eigen::Vector<T, 4> Inv1 = Vec0.cwiseProduct(Fac0) -
                               Vec2.cwiseProduct(Fac3) +
                               Vec3.cwiseProduct(Fac4);
    Eigen::Vector<T, 4> Inv2 = Vec0.cwiseProduct(Fac1) -
                               Vec1.cwiseProduct(Fac3) +
                               Vec3.cwiseProduct(Fac5);
    Eigen::Vector<T, 4> Inv3 = Vec0.cwiseProduct(Fac2) -
                               Vec1.cwiseProduct(Fac4) +
                               Vec2.cwiseProduct(Fac5);

    Eigen::Vector<T, 4> SignA(+1, -1, +1, -1);
    Eigen::Vector<T, 4> SignB(-1, +1, -1, +1);

    Eigen::Matrix<T, 4, 4> ret;
    ret.col(0) = Inv0.cwiseProduct(SignA);
    ret.col(1) = Inv1.cwiseProduct(SignB);
    ret.col(2) = Inv2.cwiseProduct(SignA);
    ret.col(3) = Inv3.cwiseProduct(SignB);


    Eigen::Vector<T, 4> Row0(ret(0, 0), ret(1, 0), ret(2, 0), ret(3, 0));

    Eigen::Vector<T, 4> Dot0(m(0, 0) * Row0[0],
                             m(0, 1) * Row0[1],
                             m(0, 2) * Row0[2],
                             m(0, 3) * Row0[3]);

    T Dot1 = Dot0.sum();

    T OneOverDeterminant = static_cast<T>(1) / Dot1;

    ret *= OneOverDeterminant;

    return ret;
}
}  // namespace rxmesh