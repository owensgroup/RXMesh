#include "gtest/gtest.h"

#include "rxmesh/diff/scalar.h"

#include "rxmesh/rxmesh_static.h"

#define RX_ASSERT_NEAR(val1, val2, eps, d_err)                     \
    if (abs(val1 - val2) > eps) {                                  \
        printf("\n val1= %f, val2= %f, eps= %f", val1, val2, eps); \
        d_err[0]++;                                                \
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
    RX_ASSERT_NEAR(f.grad(0), T(3.0) * exp(T(4.0)), eps, d_err);
    if constexpr (WithHessian) {
        RX_ASSERT_NEAR(f.Hess(0, 0), T(11.0) * std::exp(T(4.0)), eps, d_err);
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
    RX_ASSERT_NEAR(f.grad(0), T(3.0) / T(4.0), eps, d_err);
    if constexpr (WithHessian) {
        RX_ASSERT_NEAR(f.Hess(0, 0), T(-1.0) / T(16.0), eps, d_err);
    }
}

template <typename T, bool WithHessian>
__inline__ __device__ void test_log2(int* d_err, T eps)
{
    using namespace rxmesh;
    using Real1 = Scalar<1, T, WithHessian>;

    const Real1 a = Real1::known_derivatives(4.0, 3.0, 2.0);
    const Real1 f = log2(a);
    RX_ASSERT_NEAR(f.val, 2.0, eps, d_err);
    RX_ASSERT_NEAR(f.grad(0), 3.0 / 4.0 / std::log(2.0), eps, d_err);
    if constexpr (WithHessian) {
        RX_ASSERT_NEAR(f.Hess(0, 0), -1.0 / 16.0 / std::log(2.0), eps, d_err);
    }
}

template <typename T, bool WithHessian>
__global__ static void test_scalar(int* d_err, T eps = 1e-6)
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