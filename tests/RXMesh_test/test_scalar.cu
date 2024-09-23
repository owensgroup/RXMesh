#include "gtest/gtest.h"

#include "rxmesh/diff/scalar.h"

#include "rxmesh/rxmesh_static.h"

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
    if (!exp) {                    \
        d_err[0]++;                \
    }


template <typename T, bool WithHessian>
__inline__ __device__ void test_unary_minus(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real2 = Scalar<2, T, WithHessian>;

    Real2 x = Real2::Scalar(4.0, 0);
    Real2 y = Real2::Scalar(6.0, 1);
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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
    using Real1 = Scalar<1, T, WithHessian>;

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
__global__ static void test_scalar(int* d_err, T eps = 1e-4)
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


TEST(Diff, Scalar)
{
    using namespace rxmesh;

    int* d_err;
    CUDA_ERROR(cudaMalloc((void**)&d_err, sizeof(int)));
    CUDA_ERROR(cudaMemset(d_err, 0, sizeof(int)));

    test_scalar<float, false><<<1, 1>>>(d_err);
    test_scalar<double, false><<<1, 1>>>(d_err);
    test_scalar<float, true><<<1, 1>>>(d_err);
    test_scalar<double, true><<<1, 1>>>(d_err);

    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int h_err;
    CUDA_ERROR(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_err, 0);

    GPU_FREE(d_err);
}

TEST(Diff, ScalarAttr)
{
    using namespace rxmesh;

    std::string obj_path = STRINGIFY(INPUT_DIR) "sphere3.obj";

    RXMeshStatic rx(obj_path);
}