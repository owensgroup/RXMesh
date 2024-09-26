#include "gtest/gtest.h"

#include "rxmesh/diff/scalar.h"

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/inverse.h"


#define RX_ASSERT_NEAR(val1, val2, eps, d_err)                           \
    if (abs(val1 - val2) > eps) {                                        \
        printf("\n val1= %.17g, val2= %.17g, eps= %.17g, diff= %.17g\n", \
               double(val1),                                             \
               double(val2),                                             \
               double(eps),                                              \
               abs(val1 - val2));                                        \
        d_err[0]++;                                                      \
    }

#define RX_ASSERT_TRUE(exp, d_err) \
    if (!(exp)) {                  \
        d_err[0]++;                \
    }

#define RX_ASSERT_FALSE(exp, d_err) \
    if ((exp)) {                    \
        d_err[0]++;                 \
    }


template <typename T, bool WithHessian>
__inline__ __device__ void test_unary_minus(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x + 2 at x = 1
    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = -a;

    RX_ASSERT_NEAR(f.val, -a.val, eps, d_err);
    RX_ASSERT_NEAR(f.grad(0), -a.grad(0), eps, d_err);

    if constexpr (WithHessian) {
        RX_ASSERT_NEAR(f.Hess(0, 0), -a.Hess(0, 0), eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_sqrt(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x + 2 at x=1
    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = sqrt(a);

    RX_ASSERT_NEAR(f.val, 2.0, eps, d_err);
    RX_ASSERT_NEAR(f.grad(0), (3.0 / 4.0), eps, d_err);

    if constexpr (WithHessian) {
        RX_ASSERT_NEAR(f.Hess(0, 0), (7.0 / 32.0), eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_sqr(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real2 = Scalar<T, 2, WithHessian>;

    Real2 x(4.0, 0);
    Real2 y(6.0, 1);
    Real2 a = x * x + 7.0 * y * y - 8.0 * x + 2 * y;

    Real2 a_sqr = sqr(a);
    Real2 a_pow = pow(a, 2);
    Real2 aa    = a * a;

    RX_ASSERT_NEAR(a_sqr.val, a_pow.val, eps, d_err);
    RX_ASSERT_NEAR(a_sqr.val, aa.val, eps, d_err);
    RX_ASSERT_NEAR(a_sqr.grad(0), a_pow.grad(0), eps, d_err);
    RX_ASSERT_NEAR(a_sqr.grad(1), a_pow.grad(1), eps, d_err);
    RX_ASSERT_NEAR(a_sqr.grad(0), aa.grad(0), eps, d_err);
    RX_ASSERT_NEAR(a_sqr.grad(1), aa.grad(1), eps, d_err);

    if constexpr (WithHessian) {
        RX_ASSERT_TRUE(is_sym(a_sqr.Hess, eps), d_err);
        RX_ASSERT_TRUE(is_sym(a_pow.Hess, eps), d_err);
        RX_ASSERT_TRUE(is_sym(aa.Hess, eps), d_err);
        RX_ASSERT_NEAR(a_sqr.Hess(0, 0), a_pow.Hess(0, 0), eps, d_err);
        RX_ASSERT_NEAR(a_sqr.Hess(0, 1), a_pow.Hess(0, 1), eps, d_err);
        RX_ASSERT_NEAR(a_sqr.Hess(1, 0), a_pow.Hess(1, 0), eps, d_err);
        RX_ASSERT_NEAR(a_sqr.Hess(1, 1), a_pow.Hess(1, 1), eps, d_err);
        RX_ASSERT_NEAR(a_sqr.Hess(0, 0), aa.Hess(0, 0), eps, d_err);
        RX_ASSERT_NEAR(a_sqr.Hess(0, 1), aa.Hess(0, 1), eps, d_err);
        RX_ASSERT_NEAR(a_sqr.Hess(1, 0), aa.Hess(1, 0), eps, d_err);
        RX_ASSERT_NEAR(a_sqr.Hess(1, 1), aa.Hess(1, 1), eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_fabs(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    {  // a(x) = x^3 at x = 1
        const Real1 a = Real1::known_derivatives(1.0, 3.0, 6.0);
        const Real1 f = fabs(a);
        RX_ASSERT_NEAR(f.val, 1.0, eps, d_err);
        RX_ASSERT_NEAR(f.grad(0), 3.0, eps, d_err);
        if constexpr (WithHessian) {
            RX_ASSERT_NEAR(f.Hess(0, 0), 6.0, eps, d_err);
            RX_ASSERT_TRUE(is_sym(f.Hess, eps), d_err);
        }
    }

    {  // a(x) = x^3 at x = -1
        const Real1 a = Real1::known_derivatives(-1.0, 3.0, -6.0);
        const Real1 f = fabs(a);
        RX_ASSERT_NEAR(f.val, 1.0, eps, d_err);
        RX_ASSERT_NEAR(f.grad(0), -3.0, eps, d_err);
        if constexpr (WithHessian) {
            RX_ASSERT_NEAR(f.Hess(0, 0), 6.0, eps, d_err);
            RX_ASSERT_TRUE(is_sym(f.Hess, eps), d_err);
        }
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_abs(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    {  // a(x) = x^3 at x = 1
        const Real1 a = Real1::known_derivatives(1.0, 3.0, 6.0);
        const Real1 f = abs(a);
        RX_ASSERT_NEAR(f.val, 1.0, eps, d_err);
        RX_ASSERT_NEAR(f.grad(0), 3.0, eps, d_err);
        if constexpr (WithHessian) {
            RX_ASSERT_NEAR(f.Hess(0, 0), 6.0, eps, d_err);
            RX_ASSERT_TRUE(is_sym(f.Hess, eps), d_err);
        }
    }

    {  // a(x) = x^3 at x = -1
        const Real1 a = Real1::known_derivatives(-1.0, 3.0, -6.0);
        const Real1 f = abs(a);
        RX_ASSERT_NEAR(f.val, 1.0, eps, d_err);
        RX_ASSERT_NEAR(f.grad(0), -3.0, eps, d_err);
        if constexpr (WithHessian) {
            RX_ASSERT_NEAR(f.Hess(0, 0), 6.0, eps, d_err);
            RX_ASSERT_TRUE(is_sym(f.Hess, eps), d_err);
        }
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_exp(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x + 2 at x=1
    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = exp(a);
    RX_ASSERT_NEAR(f.val, std::exp(T(4.0)), eps, d_err);
    const T g = T(3.0) * exp(T(4.0));
    RX_ASSERT_NEAR(f.grad(0), g, eps, d_err);
    if constexpr (WithHessian) {
        const T h = T(11.0) * std::exp(T(4.0));
        RX_ASSERT_NEAR(f.Hess(0, 0), h, eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_log(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x + 2 at x=1
    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = log(a);
    RX_ASSERT_NEAR(f.val, T(2.0) * std::log(T(2.0)), eps, d_err);
    const T g = T(3.0) / T(4.0);
    RX_ASSERT_NEAR(f.grad(0), g, eps, d_err);
    if constexpr (WithHessian) {
        const T h = T(-1.0) / T(16.0);
        RX_ASSERT_NEAR(f.Hess(0, 0), h, eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_log2(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x + 2 at x=1
    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = log2(a);
    RX_ASSERT_NEAR(f.val, 2.0, eps, d_err);
    const T g = 3.0 / 4.0 / std::log(2.0);
    RX_ASSERT_NEAR(f.grad(0), g, eps, d_err);
    if constexpr (WithHessian) {
        const T h = -1.0 / 16.0 / std::log(2.0);
        RX_ASSERT_NEAR(f.Hess(0, 0), h, eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_log10(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x + 2 at x=1
    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = log10(a);
    RX_ASSERT_NEAR(f.val, std::log10(T(4.0)), eps, d_err);
    const T g = 3.0 / 4.0 / std::log(10.0);
    RX_ASSERT_NEAR(f.grad(0), g, eps, d_err);
    if constexpr (WithHessian) {
        const T h = -1.0 / 16.0 / std::log(10.0);
        RX_ASSERT_NEAR(f.Hess(0, 0), h, eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_sin(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x + 2 at x=1
    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = sin(a);
    RX_ASSERT_NEAR(f.val, std::sin(T(4.0)), eps, d_err);

    const T g = T(3.0) * std::cos(T(4.0));
    RX_ASSERT_NEAR(f.grad(0), g, eps, d_err);
    if constexpr (WithHessian) {
        const T h = T(2.0) * std::cos(T(4.0)) - T(9.0) * std::sin(T(4.0));
        RX_ASSERT_NEAR(f.Hess(0, 0), h, eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_cos(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x + 2 at x=1
    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = cos(a);
    RX_ASSERT_NEAR(f.val, std::cos(T(4.0)), eps, d_err);

    const T g = T(-3.0) * std::sin(T(4.0));
    RX_ASSERT_NEAR(f.grad(0), g, eps, d_err);
    if constexpr (WithHessian) {
        const T h = T(-2.0) * std::sin(T(4.0)) - T(9.0) * std::cos(T(4.0));
        RX_ASSERT_NEAR(f.Hess(0, 0), h, eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_tan(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x + 2 at x=1
    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = tan(a);
    RX_ASSERT_NEAR(f.val, std::tan(T(4.0)), eps, d_err);

    const T g = T(3.0) / sqr(std::cos(T(4.0)));
    RX_ASSERT_NEAR(f.grad(0), g, eps, d_err);
    if constexpr (WithHessian) {
        const T h = T(4.0) * (T(1.0) + T(9.0) * std::tan(T(4.0))) /
                    (T(1.0) + std::cos(T(8.0)));
        RX_ASSERT_NEAR(f.Hess(0, 0), h, eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_asin(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x - 1.5 at x=1
    const Real1 a = Real1::known_derivatives(0.5, 3.0, 2.0);
    const Real1 f = asin(a);
    RX_ASSERT_NEAR(f.val, std::asin(T(0.5)), eps, d_err);

    const T g = 3.4641;
    RX_ASSERT_NEAR(f.grad(0), g, 1e-4, d_err);
    if constexpr (WithHessian) {
        const T h = 9.2376;
        RX_ASSERT_NEAR(f.Hess(0, 0), h, 1e-4, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_acos(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x - 1.5 at x=1
    const Real1 a = Real1::known_derivatives(0.5, 3.0, 2.0);
    const Real1 f = acos(a);
    RX_ASSERT_NEAR(f.val, std::acos(T(0.5)), eps, d_err);

    const T g = -3.4641;
    RX_ASSERT_NEAR(f.grad(0), g, 1e-4, d_err);
    if constexpr (WithHessian) {
        const T h = -9.2376;
        RX_ASSERT_NEAR(f.Hess(0, 0), h, 1e-4, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_atan(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x - 1.5 at x=1
    const Real1 a = Real1::known_derivatives(0.5, 3.0, 2.0);
    const Real1 f = atan(a);
    RX_ASSERT_NEAR(f.val, std::atan(T(0.5)), eps, d_err);

    const T g = 2.4;
    RX_ASSERT_NEAR(f.grad(0), g, eps, d_err);
    if constexpr (WithHessian) {
        const T h = -4.16;
        RX_ASSERT_NEAR(f.Hess(0, 0), h, eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_sinh(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x + 2 at x=1
    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = sinh(a);
    RX_ASSERT_NEAR(f.val, std::sinh(T(4.0)), eps, d_err);

    const T g = 3.0 * std::cosh(4.0);
    RX_ASSERT_NEAR(f.grad(0), g, eps, d_err);
    if constexpr (WithHessian) {
        const T h = 9.0 * std::sinh(4.0) + 2.0 * std::cosh(4.0);
        RX_ASSERT_NEAR(f.Hess(0, 0), h, eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_cosh(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x + 2 at x=1
    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = cosh(a);
    RX_ASSERT_NEAR(f.val, std::cosh(T(4.0)), eps, d_err);

    const T g = 3.0 * std::sinh(4.0);
    RX_ASSERT_NEAR(f.grad(0), g, eps, d_err);
    if constexpr (WithHessian) {
        const T h = 2.0 * std::sinh(4.0) + 9.0 * std::cosh(4.0);
        RX_ASSERT_NEAR(f.Hess(0, 0), h, eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_tanh(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x + 2 at x=1
    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = tanh(a);
    RX_ASSERT_NEAR(f.val, std::tanh(T(4.0)), eps, d_err);

    const T g = 3.0 / sqr(std::cosh(4.0));
    RX_ASSERT_NEAR(f.grad(0), g, eps, d_err);
    if constexpr (WithHessian) {
        const T h = 2.0 * (1.0 - 9.0 * std::sinh(4.0) / std::cosh(4.0)) /
                    (sqr(std::cosh(4.0)));
        RX_ASSERT_NEAR(f.Hess(0, 0), h, eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_asinh(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x - 1.5  at x=1
    const Real1 a = Real1::known_derivatives(0.5, 3.0, 2.0);
    const Real1 f = asinh(a);
    RX_ASSERT_NEAR(f.val, std::asinh(T(0.5)), eps, d_err);

    const T g = 2.68328;
    RX_ASSERT_NEAR(f.grad(0), g, 1e-5, d_err);

    if constexpr (WithHessian) {
        const T h = -1.43108;
        RX_ASSERT_NEAR(f.Hess(0, 0), h, 1e-5, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_acosh(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x + 2 at x=1
    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = acosh(a);
    RX_ASSERT_NEAR(f.val, std::acosh(T(4.0)), eps, d_err);

    const T g = std::sqrt(3.0 / 5.0);
    RX_ASSERT_NEAR(f.grad(0), g, eps, d_err);
    if constexpr (WithHessian) {
        const T h = -2.0 / 5.0 / std::sqrt(T(15));
        RX_ASSERT_NEAR(f.Hess(0, 0), h, eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_atanh(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // a(x) = x^2 + x - 1.5 at x=1
    const Real1 a = Real1::known_derivatives(0.5, 3.0, 2.0);
    const Real1 f = atanh(a);
    RX_ASSERT_NEAR(f.val, std::atanh(T(0.5)), eps, d_err);

    const T g = 4.0;
    RX_ASSERT_NEAR(f.grad(0), g, eps, d_err);
    if constexpr (WithHessian) {
        const T h = 18.6667;
        RX_ASSERT_NEAR(f.Hess(0, 0), h, 1e-4, d_err);
    }
}

template <typename T, bool WithHessian>
__global__ static void test_scalar_unary(int* d_err, T eps = 1e-4)
{


    // unary minus
    test_unary_minus<T, WithHessian>(d_err, eps);

    // sqrt
    test_sqrt<T, WithHessian>(d_err, eps);

    // test sqr
    test_sqr<T, WithHessian>(d_err, eps);

    // test fabs
    test_fabs<T, WithHessian>(d_err, eps);

    // test abs
    test_abs<T, WithHessian>(d_err, eps);

    // test exp
    test_exp<T, WithHessian>(d_err, eps);

    // test log
    test_log<T, WithHessian>(d_err, eps);

    // test log2
    test_log2<T, WithHessian>(d_err, eps);

    // test log10
    test_log10<T, WithHessian>(d_err, eps);

    // test sin
    test_sin<T, WithHessian>(d_err, eps);

    // test cos
    test_cos<T, WithHessian>(d_err, eps);

    // test tan
    test_tan<T, WithHessian>(d_err, eps);

    // test asin
    test_asin<T, WithHessian>(d_err, eps);

    // test acos
    test_acos<T, WithHessian>(d_err, eps);

    // test atan
    test_atan<T, WithHessian>(d_err, eps);

    // test sinh
    test_sinh<T, WithHessian>(d_err, eps);

    // test cosh
    test_cosh<T, WithHessian>(d_err, eps);

    // test tanh
    test_tanh<T, WithHessian>(d_err, eps);

    // test asinh
    test_asinh<T, WithHessian>(d_err, eps);

    // test acosh
    test_acosh<T, WithHessian>(d_err, eps);

    // test atanh
    test_atanh<T, WithHessian>(d_err, eps);
}

template <typename T, bool WithHessian>
__global__ static void test_scalar_constructors(int* d_err, T eps = 1e-9)
{
    using namespace rxmesh;
    using Real2 = Scalar<T, 2, WithHessian>;
    using Real4 = Scalar<T, 4, WithHessian>;

    {
        // make_active()
        const auto v = Real2::make_active({2.0, 4.0});
        RX_ASSERT_NEAR(v[0].val, 2.0, eps, d_err);
        RX_ASSERT_NEAR(v[1].val, 4.0, eps, d_err);

        RX_ASSERT_NEAR(v[0].grad[0], 1.0, eps, d_err);
        RX_ASSERT_NEAR(v[0].grad[1], 0.0, eps, d_err);

        RX_ASSERT_NEAR(v[1].grad[0], 0.0, eps, d_err);
        RX_ASSERT_NEAR(v[1].grad[1], 1.0, eps, d_err);

        RX_ASSERT_TRUE(v[0].Hess.isZero(), d_err);
        RX_ASSERT_TRUE(v[1].Hess.isZero(), d_err);

        // to_passive() vector
        const Eigen::Matrix<T, 2, 1> v_passive  = to_passive(v);
        const Eigen::Matrix<T, 2, 1> v_passive2 = to_passive(v_passive);
        RX_ASSERT_NEAR(v_passive[0], 2.0, eps, d_err);
        RX_ASSERT_NEAR(v_passive[1], 4.0, eps, d_err);
        RX_ASSERT_NEAR(v_passive2[0], 2.0, eps, d_err);
        RX_ASSERT_NEAR(v_passive2[1], 4.0, eps, d_err);
    }

    {

        // to_passive() matrix
        const Eigen::Vector<Real4, 4> v =
            Real4::make_active({1.0, 2.0, 3.0, 4.0});

        Eigen::Matrix<Real4, 2, 2> M;
        M << v[0], v[1], v[2], v[3];
        const Eigen::Matrix2<T> M_passive  = to_passive(M);
        const Eigen::Matrix2<T> M_passive2 = to_passive(M_passive);
        RX_ASSERT_NEAR(M(0, 0).val, M_passive(0, 0), eps, d_err);
        RX_ASSERT_NEAR(M(0, 1).val, M_passive(0, 1), eps, d_err);
        RX_ASSERT_NEAR(M(1, 0).val, M_passive(1, 0), eps, d_err);
        RX_ASSERT_NEAR(M(1, 1).val, M_passive(1, 1), eps, d_err);

        RX_ASSERT_NEAR(M_passive2(0, 0), M_passive(0, 0), eps, d_err);
        RX_ASSERT_NEAR(M_passive2(0, 1), M_passive(0, 1), eps, d_err);
        RX_ASSERT_NEAR(M_passive2(1, 0), M_passive(1, 0), eps, d_err);
        RX_ASSERT_NEAR(M_passive2(1, 1), M_passive(1, 1), eps, d_err);
    }


    {
        // Active variable
        Real2 a(4.0, 0);
        RX_ASSERT_TRUE(a.val == 4.0, d_err);
        RX_ASSERT_TRUE(a.grad[0] == 1.0, d_err);
        RX_ASSERT_TRUE(a.grad[1] == 0.0, d_err);
        RX_ASSERT_TRUE(a.Hess.isZero(), d_err);

        // Passive variable
        Real2 b(5.0);
        RX_ASSERT_TRUE(b.val == 5.0, d_err);
        RX_ASSERT_TRUE(b.grad.isZero(), d_err);
        RX_ASSERT_TRUE(b.Hess.isZero(), d_err);

        // Copy constructor
        const auto a2(a);
        RX_ASSERT_TRUE(a.val == a2.val, d_err);
        RX_ASSERT_TRUE(a.grad == a2.grad, d_err);
        RX_ASSERT_TRUE(a.Hess == a2.Hess, d_err);

        // Assignment operator
        const auto b2 = b;
        RX_ASSERT_TRUE(b.val == b2.val, d_err);
        RX_ASSERT_TRUE(b.grad == b2.grad, d_err);
        RX_ASSERT_TRUE(b.Hess == b2.Hess, d_err);
    }
}

template <typename T, bool WithHessian>
__global__ static void test_is_nan_is_inf(int* d_err, T eps = 1e-9)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;
    const Real1 a(0.0);
    const Real1 b(INFINITY);
    const Real1 c(-INFINITY);
    const Real1 d(NAN);

    RX_ASSERT_TRUE(!isnan(a), d_err);
    RX_ASSERT_TRUE(!isnan(b), d_err);
    RX_ASSERT_TRUE(!isnan(c), d_err);
    RX_ASSERT_TRUE(isnan(d), d_err);

    RX_ASSERT_TRUE(!isinf(a), d_err);
    RX_ASSERT_TRUE(isinf(b), d_err);
    RX_ASSERT_TRUE(isinf(c), d_err);
    RX_ASSERT_TRUE(!isinf(d), d_err);

    RX_ASSERT_TRUE(isfinite(a), d_err);
    RX_ASSERT_TRUE(!isfinite(b), d_err);
    RX_ASSERT_TRUE(!isfinite(c), d_err);
    RX_ASSERT_TRUE(isfinite(d), d_err);
}

template <typename T, bool WithHessian>
__global__ static void test_comparsion(int* d_err, T eps = 1e-9)
{
    using namespace rxmesh;
    using Real1   = Scalar<T, 1, WithHessian>;
    const Real1 a = Real1::known_derivatives(1.0, 1.0, 4.0);
    const Real1 b = Real1::known_derivatives(1.0, 2.0, 8.0);
    const Real1 c = Real1::known_derivatives(2.0, 2.0, 8.0);

    RX_ASSERT_TRUE(a == b, d_err);
    RX_ASSERT_TRUE(b == a, d_err);
    RX_ASSERT_TRUE(a != c, d_err);
    RX_ASSERT_TRUE(c != a, d_err);
    RX_ASSERT_TRUE(b != c, d_err);
    RX_ASSERT_TRUE(c != b, d_err);

    RX_ASSERT_FALSE(a < b, d_err);
    RX_ASSERT_FALSE(b < a, d_err);
    RX_ASSERT_TRUE(a < c, d_err);
    RX_ASSERT_FALSE(c < a, d_err);
    RX_ASSERT_TRUE(b < c, d_err);
    RX_ASSERT_FALSE(c < b, d_err);

    RX_ASSERT_TRUE(a <= b, d_err);
    RX_ASSERT_TRUE(b <= a, d_err);
    RX_ASSERT_TRUE(a <= c, d_err);
    RX_ASSERT_FALSE(c <= a, d_err);
    RX_ASSERT_TRUE(b <= c, d_err);
    RX_ASSERT_FALSE(c <= b, d_err);

    RX_ASSERT_FALSE(a > b, d_err);
    RX_ASSERT_FALSE(b > a, d_err);
    RX_ASSERT_FALSE(a > c, d_err);
    RX_ASSERT_TRUE(c > a, d_err);
    RX_ASSERT_FALSE(b > c, d_err);
    RX_ASSERT_TRUE(c > b, d_err);

    RX_ASSERT_TRUE(a >= b, d_err);
    RX_ASSERT_TRUE(b >= a, d_err);
    RX_ASSERT_FALSE(a >= c, d_err);
    RX_ASSERT_TRUE(c >= a, d_err);
    RX_ASSERT_FALSE(b >= c, d_err);
    RX_ASSERT_TRUE(c >= b, d_err);

    // Test double overloads
    RX_ASSERT_TRUE(a == 1.0, d_err);
    RX_ASSERT_FALSE(a == 2.0, d_err);
    RX_ASSERT_FALSE(a != 1.0, d_err);
    RX_ASSERT_TRUE(a != 2.0, d_err);
    RX_ASSERT_FALSE(a < 1.0, d_err);
    RX_ASSERT_TRUE(a < 2.0, d_err);
    RX_ASSERT_TRUE(a <= 1.0, d_err);
    RX_ASSERT_TRUE(a <= 2.0, d_err);
    RX_ASSERT_FALSE(a > 1.0, d_err);
    RX_ASSERT_FALSE(a > 2.0, d_err);
    RX_ASSERT_TRUE(a >= 1.0, d_err);
    RX_ASSERT_FALSE(a >= 2.0, d_err);
}

template <typename T, bool WithHessian>
__global__ static void test_min_max(int* d_err, T eps = 1e-9)
{
    using namespace rxmesh;
    using Real1   = Scalar<T, 1, WithHessian>;
    const Real1 a = Real1::known_derivatives(1.0, 2.0, 3.0);
    const Real1 b = Real1::known_derivatives(2.0, 3.0, 4.0);

    RX_ASSERT_TRUE(min(a, b) == a, d_err);
    RX_ASSERT_TRUE(min(a, b).grad == a.grad, d_err);
    RX_ASSERT_TRUE(min(a, b).Hess == a.Hess, d_err);

    RX_ASSERT_TRUE(fmin(a, b) == a, d_err);
    RX_ASSERT_TRUE(fmin(a, b).grad == a.grad, d_err);
    RX_ASSERT_TRUE(fmin(a, b).Hess == a.Hess, d_err);

    RX_ASSERT_TRUE(max(a, b) == b, d_err);
    RX_ASSERT_TRUE(max(a, b).grad == b.grad, d_err);
    RX_ASSERT_TRUE(max(a, b).Hess == b.Hess, d_err);

    RX_ASSERT_TRUE(fmax(a, b) == b, d_err);
    RX_ASSERT_TRUE(fmax(a, b).grad == b.grad, d_err);
    RX_ASSERT_TRUE(fmax(a, b).Hess == b.Hess, d_err);
}

template <typename T, bool WithHessian>
__global__ static void test_quadratic(int* d_err, T eps = 1e-9)
{
    using namespace rxmesh;
    using Real1 = Scalar<T, 1, WithHessian>;

    // f(a) = a^2 + a + 2 at a = 1
    const Real1 a = Real1::make_active(1.0, 0);
    const Real1 f = sqr(a) + a + 2.0;

    RX_ASSERT_NEAR(f.val, 4.0, eps, d_err);
    RX_ASSERT_NEAR(f.grad(0), 3.0, eps, d_err);
    if constexpr (WithHessian) {
        RX_ASSERT_NEAR(f.Hess(0, 0), 2.0, eps, d_err);
    }
}

template <typename T, bool WithHessian>
__global__ static void test_min_quadratic(int* d_err, T eps = 1e-9)
{
    using namespace rxmesh;
    using Real3 = Scalar<T, 3, WithHessian>;

    // Variable vector in R^3
    const Eigen::Vector<Real3, 3> x = Real3::make_active({0.0, 0.0, 0.0});

    // Quadratic function
    const Real3 f = sqr(x[0]) + 2.0 * sqr(x[1]) + 6.0 * sqr(x[2]) + x[0] -
                    2.0 * x[1] + 6.0 * x[2] + 10;

    // Solve for minimum
    typename Real3::HessType f_hess_inv = inverse(f.Hess);

    const Eigen::Vector<T, 3> x_min = -f_hess_inv * f.grad;

    RX_ASSERT_NEAR(x_min.x(), -0.5, eps, d_err);
    RX_ASSERT_NEAR(x_min.y(), 0.5, eps, d_err);
    RX_ASSERT_NEAR(x_min.z(), -0.5, eps, d_err);
}

template <typename T, bool WithHessian>
__global__ static void test_triangle_distortion(int* d_err, T eps = 1e-9)
{
    using namespace rxmesh;
    using Real6 = Scalar<T, 6, WithHessian>;

    // passive rest-state triangle ar, br, cr
    const Eigen::Matrix<T, 2, 1> ar(1.0, 1.0);
    const Eigen::Matrix<T, 2, 1> br(2.0, 1.0);
    const Eigen::Matrix<T, 2, 1> cr(1.0, 2.0);
    const Eigen::Matrix<T, 2, 2> Mr = col_mat(br - ar, cr - ar);

    auto Mr_inv = inverse(Mr);

    // printf("\n **** Mr_inv= ");
    // for (int i = 0; i < 2; ++i) {
    //     printf("\n ");
    //     for (int j = 0; j < 2; ++j) {
    //         printf(" %.9g", Mr_inv(i, j));
    //     }
    // }


    // active variables vector for vertex positions a, b, c

    // clang-format off
    const Eigen::Vector<Real6, 6> x = Real6::make_active({
        10.0, 1.0,
        15.0, 3.0,
        2.0, 2.0,
    });
    // clang-format on

    const Eigen::Matrix<Real6, 2, 1> a(x[0], x[1]);
    const Eigen::Matrix<Real6, 2, 1> b(x[2], x[3]);
    const Eigen::Matrix<Real6, 2, 1> c(x[4], x[5]);
    const Eigen::Matrix<Real6, 2, 2> M = col_mat(b - a, c - a);

    // printf("\n **** M= ");
    // for (int i = 0; i < 2; ++i) {
    //     for (int j = 0; j < 2; ++j) {
    //         printf("\n i= %d, j= %d\n", i, j);
    //         M(i, j).print();
    //     }
    // }

    const Eigen::Matrix<Real6, 2, 2> J = M * Mr_inv;

    // printf("\n **** J= ");
    // for (int i = 0; i < 2; ++i) {
    //     for (int j = 0; j < 2; ++j) {
    //         printf("\n i= %d, j= %d\n", i, j);
    //         J(i, j).print();
    //     }
    // }

    auto J_inv = inverse(J);
    // auto J_inv = J.inverse();

    // printf("\n **** J_inv= ");
    // for (int i = 0; i < 2; ++i) {
    //     for (int j = 0; j < 2; ++j) {
    //         printf("\n i= %d, j= %d\n", i, j);
    //         J_inv(i, j).print();
    //     }
    // }


    const Real6 E = J.squaredNorm() + J_inv.squaredNorm();
    
    //E.print();

    RX_ASSERT_TRUE(is_finite_scalar(E), d_err);
}

TEST(Diff, ScalarUnaryOps)
{
    using namespace rxmesh;

    int* d_err;
    CUDA_ERROR(cudaMalloc((void**)&d_err, sizeof(int)));
    CUDA_ERROR(cudaMemset(d_err, 0, sizeof(int)));

    test_scalar_unary<float, false><<<1, 1>>>(d_err);
    test_scalar_unary<double, false><<<1, 1>>>(d_err);
    test_scalar_unary<float, true><<<1, 1>>>(d_err);
    test_scalar_unary<double, true><<<1, 1>>>(d_err);

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_err;
    CUDA_ERROR(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_err, 0);

    GPU_FREE(d_err);
}

TEST(Diff, ScalarConstructors)
{
    using namespace rxmesh;

    int* d_err;
    CUDA_ERROR(cudaMalloc((void**)&d_err, sizeof(int)));
    CUDA_ERROR(cudaMemset(d_err, 0, sizeof(int)));

    test_scalar_constructors<float, false><<<1, 1>>>(d_err);
    test_scalar_constructors<double, false><<<1, 1>>>(d_err);
    test_scalar_constructors<float, true><<<1, 1>>>(d_err);
    test_scalar_constructors<double, true><<<1, 1>>>(d_err);

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_err;
    CUDA_ERROR(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_err, 0);

    GPU_FREE(d_err);
}

TEST(Diff, ScalarIsNANIsInf)
{
    using namespace rxmesh;

    int* d_err;
    CUDA_ERROR(cudaMalloc((void**)&d_err, sizeof(int)));
    CUDA_ERROR(cudaMemset(d_err, 0, sizeof(int)));

    test_is_nan_is_inf<float, false><<<1, 1>>>(d_err);
    test_is_nan_is_inf<double, false><<<1, 1>>>(d_err);
    test_is_nan_is_inf<float, true><<<1, 1>>>(d_err);
    test_is_nan_is_inf<double, true><<<1, 1>>>(d_err);

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_err;
    CUDA_ERROR(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_err, 0);

    GPU_FREE(d_err);
}

TEST(Diff, ScalarComparison)
{
    using namespace rxmesh;

    int* d_err;
    CUDA_ERROR(cudaMalloc((void**)&d_err, sizeof(int)));
    CUDA_ERROR(cudaMemset(d_err, 0, sizeof(int)));

    test_comparsion<float, false><<<1, 1>>>(d_err);
    test_comparsion<double, false><<<1, 1>>>(d_err);
    test_comparsion<float, true><<<1, 1>>>(d_err);
    test_comparsion<double, true><<<1, 1>>>(d_err);

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_err;
    CUDA_ERROR(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_err, 0);

    GPU_FREE(d_err);
}

TEST(Diff, ScalarMinMax)
{
    using namespace rxmesh;

    int* d_err;
    CUDA_ERROR(cudaMalloc((void**)&d_err, sizeof(int)));
    CUDA_ERROR(cudaMemset(d_err, 0, sizeof(int)));

    test_min_max<float, false><<<1, 1>>>(d_err);
    test_min_max<double, false><<<1, 1>>>(d_err);
    test_min_max<float, true><<<1, 1>>>(d_err);
    test_min_max<double, true><<<1, 1>>>(d_err);

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_err;
    CUDA_ERROR(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_err, 0);

    GPU_FREE(d_err);
}

TEST(Diff, ScalarQuadratic)
{
    using namespace rxmesh;

    int* d_err;
    CUDA_ERROR(cudaMalloc((void**)&d_err, sizeof(int)));
    CUDA_ERROR(cudaMemset(d_err, 0, sizeof(int)));

    test_quadratic<float, false><<<1, 1>>>(d_err);
    test_quadratic<double, false><<<1, 1>>>(d_err);
    test_quadratic<float, true><<<1, 1>>>(d_err);
    test_quadratic<double, true><<<1, 1>>>(d_err);

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_err;
    CUDA_ERROR(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_err, 0);

    GPU_FREE(d_err);
}

TEST(Diff, ScalarMinQuadratic)
{
    using namespace rxmesh;

    int* d_err;
    CUDA_ERROR(cudaMalloc((void**)&d_err, sizeof(int)));
    CUDA_ERROR(cudaMemset(d_err, 0, sizeof(int)));

    test_min_quadratic<float, true><<<1, 1>>>(d_err);
    test_min_quadratic<double, true><<<1, 1>>>(d_err);

    CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_err;
    CUDA_ERROR(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_err, 0);

    GPU_FREE(d_err);
}


TEST(Diff, ScalarTriangleDistortion)
{
    using namespace rxmesh;

    int* d_err;
    CUDA_ERROR(cudaMalloc((void**)&d_err, sizeof(int)));
    CUDA_ERROR(cudaMemset(d_err, 0, sizeof(int)));

    test_triangle_distortion<float, true><<<1, 1>>>(d_err);
    // test_triangle_distortion<double, true><<<1, 1>>>(d_err);

    CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_err;
    CUDA_ERROR(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_err, 0);

    GPU_FREE(d_err);
}