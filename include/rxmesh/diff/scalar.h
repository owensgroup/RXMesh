/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 *
 * Update: adding support for the Scalar type to run on both host and device
 * Author: Ahmed Mahmoud
 */
#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <cmath>

#include "rxmesh/diff/util.h"

namespace rxmesh {

/**
 * Forward-differentiable scalar type with constructors for passive and active
 * variables. Each scalar carries its gradient and Hessian w.r.t. a variable
 * vector. k: Size of variable vector at compile time. PassiveT: Internal
 * floating point type, e.g. double. WithHessian: Set to false for
 * gradient-only mode.
 */
template <typename PassiveT, int k, bool WithHessian = true>
struct Scalar
{
    // Make template arguments available as members
    static_assert(k >= 1,
                  "We don't support Eigen:Dynamic. Thus, k should be >= 1");
    static constexpr int  k_           = k;
    static constexpr bool WithHessian_ = WithHessian;

    // Determine derivative data types at compile time. Use 0-by-0 if no Hessian
    // required.
    using PassiveType = PassiveT;
    using GradType    = Eigen::Matrix<PassiveT, k, 1>;
    using HessType    = typename std::conditional_t<WithHessian,
                                                 Eigen::Matrix<PassiveT, k, k>,
                                                 Eigen::Matrix<PassiveT, 0, 0>>;

    // ///////////////////////////////////////////////////////////////////////////
    // Scalar constructors
    // ///////////////////////////////////////////////////////////////////////////

    /// Default constructor, copy, move, assignment
    __host__ __device__         Scalar()                      = default;
    __host__ __device__         Scalar(const Scalar& _rhs)    = default;
    __host__ __device__         Scalar(Scalar&& _rhs)         = default;
    __host__ __device__ Scalar& operator=(const Scalar& _rhs) = default;
    __host__ __device__ Scalar& operator=(Scalar&& _rhs)      = default;

    /// Passive variable a.k.a. constant.
    /// Gradient and Hessian are zero.
    __host__ __device__ Scalar(PassiveT _val) : val(_val)
    {
    }

    /// Active variable.
    ///     _idx: index in variable vector
    __host__ __device__ Scalar(PassiveT _val, Eigen::Index _idx) : val(_val)
    {
        assert(_idx >= 0);
        assert(_idx < k);
        grad(_idx) = 1.0;
    }

    /// Initialize scalar with known derivatives
    __host__ __device__ static Scalar known_derivatives(PassiveT        _val,
                                                        const GradType& _grad,
                                                        const HessType& _Hess)
    {
        Scalar res;
        res.val  = _val;
        res.grad = _grad;

        if constexpr (WithHessian)
            res.Hess = _Hess;

        return res;
    }

    /// Initialize scalar with known derivatives (univariate case)
    __host__ __device__ static Scalar known_derivatives(PassiveT _val,
                                                        PassiveT _grad,
                                                        PassiveT _Hess)
    {
        static_assert(k == 1,
                      "Constructor only available for univariate case. Call "
                      "overload with vector-valued arguments.");

        Scalar res;
        res.val  = _val;
        res.grad = GradType::Constant(1, _grad);

        if constexpr (WithHessian)
            res.Hess = HessType::Constant(1, 1, _Hess);

        return res;
    }

    /// Initialize passive variable a.k.a. constant with zero derivatives
    /// it just calls Scalar(_val)
    __host__ __device__ static Scalar make_passive(PassiveT _val)
    {
        return Scalar(_val);
    }

    /// Initialize active variable with derivatives of
    // by calling Scalar(val, idx)
    __host__ __device__ static Scalar make_active(PassiveT     _val,
                                                  Eigen::Index _idx)
    {
        return Scalar(_val, _idx);
    }


    /// Initialize an active variable vector of size k from given values.
    __host__ __device__ static Eigen::Matrix<Scalar, k, 1> make_active(
        const Eigen::Matrix<PassiveT, k, 1>& _passive)
    {
        assert(_passive.size() == k);

        Eigen::Matrix<Scalar, k, 1> active(k);
        for (Eigen::Index i = 0; i < k; ++i)
            active[i] = Scalar(_passive[i], i);

        return active;
    }

    /// Initialize an active variable vector of size k from given values.
    // TODO fix this so that it does not use dynamic
    //__host__ __device__ static Eigen::Matrix<Scalar, k, 1> make_active(
    //    std::initializer_list<PassiveT> _passive)
    //{
    //    return make_active(
    //        Eigen::Map<const Eigen::Matrix<PassiveT, Eigen::Dynamic, 1>>(
    //            _passive.begin(), _passive.size()));
    //}

    // ///////////////////////////////////////////////////////////////////////////
    // Unary operators
    // ///////////////////////////////////////////////////////////////////////////

    /// Apply chain rule to compute f(a(x)) and its derivatives.
    __host__ __device__ static Scalar chain(const PassiveT& val,   // f
                                            const PassiveT& grad,  // df/da
                                            const PassiveT& Hess,  // ddf/daa
                                            const Scalar&   a)
    {
        Scalar res;
        res.val  = val;
        res.grad = grad * a.grad;

        if constexpr (WithHessian)
            res.Hess = Hess * a.grad * a.grad.transpose() + grad * a.Hess;

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ friend Scalar operator-(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        Scalar res;
        res.val  = -a.val;
        res.grad = -a.grad;

        if constexpr (WithHessian)
            res.Hess = -a.Hess;

        return res;
    }

    __host__ __device__ friend Scalar sqrt(const Scalar& a)
    {
        assert(is_finite_scalar(a));
        const PassiveT f = std::sqrt(a.val);
        return chain(f, (PassiveT)0.5 / f, (PassiveT)-0.25 / (f * a.val), a);
    }

    __host__ __device__ friend Scalar sqr(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        Scalar res;
        res.val  = a.val * a.val;
        res.grad = 2.0 * a.val * a.grad;

        if constexpr (WithHessian)
            res.Hess = 2.0 * (a.val * a.Hess + a.grad * a.grad.transpose());

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ friend Scalar pow(const Scalar& a, const int& e)
    {
        assert(is_finite_scalar(a));

        const PassiveT f2 = (PassiveT)std::pow(a.val, e - 2);
        const PassiveT f1 = f2 * a.val;
        const PassiveT f  = f1 * a.val;

        return chain(f, e * f1, e * (e - 1) * f2, a);
    }

    __host__ __device__ Scalar pow(const Scalar& a, const PassiveT& e)
    {
        assert(is_finite_scalar(a));

        const PassiveT f2 = std::pow(a.val, e - (PassiveT)2.0);
        const PassiveT f1 = f2 * a.val;
        const PassiveT f  = f1 * a.val;

        return chain(f, e * f1, e * (e - (PassiveT)1.0) * f2, a);
    }

    __host__ __device__ friend Scalar fabs(const Scalar& a)
    {
        assert(is_finite_scalar(a));
        if (a.val >= 0.0)
            return chain(a.val, 1.0, 0.0, a);
        else
            return chain(-a.val, -1.0, 0.0, a);
    }

    __host__ __device__ friend Scalar abs(const Scalar& a)
    {
        return fabs(a);
    }

    __host__ __device__ friend Scalar exp(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT exp_a = std::exp(a.val);
        return chain(exp_a, exp_a, exp_a, a);
    }

    __host__ __device__ friend Scalar log(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT a_inv = (PassiveT)1.0 / a.val;
        return chain(std::log(a.val), a_inv, -a_inv / a.val, a);
    }

    __host__ __device__ friend Scalar log2(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT a_inv = (PassiveT)1.0 / a.val / (PassiveT)std::log(2.0);
        return chain(std::log2(a.val), a_inv, -a_inv / a.val, a);
    }

    __host__ __device__ friend Scalar log10(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT a_inv = (PassiveT)1.0 / a.val / (PassiveT)std::log(10.0);
        return chain(std::log10(a.val), a_inv, -a_inv / a.val, a);
    }

    __host__ __device__ friend Scalar sin(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT sin_a = std::sin(a.val);
        return chain(sin_a, std::cos(a.val), -sin_a, a);
    }

    __host__ __device__ friend Scalar cos(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT cos_a = std::cos(a.val);
        return chain(cos_a, -std::sin(a.val), -cos_a, a);
    }

    __host__ __device__ friend Scalar tan(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT cos   = std::cos(a.val);
        const PassiveT cos_2 = cos * cos;
        const PassiveT cos_3 = cos_2 * cos;
        return chain(std::tan(a.val),
                     (PassiveT)1.0 / cos_2,
                     (PassiveT)2.0 * std::sin(a.val) / cos_3,
                     a);
    }

    __host__ __device__ friend Scalar asin(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT s      = (PassiveT)1.0 - a.val * a.val;
        const PassiveT s_sqrt = std::sqrt(s);
        return chain(
            std::asin(a.val), (PassiveT)1.0 / s_sqrt, a.val / s_sqrt / s, a);
    }

    __host__ __device__ friend Scalar acos(const Scalar& a)
    {
        assert(is_finite_scalar(a));
        assert(a.val > -1.0);
        assert(a.val < 1.0);

        const PassiveT s      = (PassiveT)1.0 - a.val * a.val;
        const PassiveT s_sqrt = std::sqrt(s);
        assert(is_finite(s));
        assert(is_finite(s_sqrt));
        assert(s > 0.0);

        return chain(
            std::acos(a.val), (PassiveT)-1.0 / s_sqrt, -a.val / s_sqrt / s, a);
    }

    __host__ __device__ friend Scalar atan(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT s = a.val * a.val + (PassiveT)1.0;
        return chain(std::atan(a.val),
                     (PassiveT)1.0 / s,
                     (PassiveT)-2.0 * a.val / s / s,
                     a);
    }

    __host__ __device__ friend Scalar sinh(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT sinh_a = std::sinh(a.val);
        return chain(sinh_a, std::cosh(a.val), sinh_a, a);
    }

    __host__ __device__ friend Scalar cosh(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT cosh_a = std::cosh(a.val);
        return chain(cosh_a, std::sinh(a.val), cosh_a, a);
    }

    __host__ __device__ friend Scalar tanh(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT cosh   = std::cosh(a.val);
        const PassiveT cosh_2 = cosh * cosh;
        const PassiveT cosh_3 = cosh_2 * cosh;
        return chain(std::tanh(a.val),
                     (PassiveT)1.0 / cosh_2,
                     (PassiveT)-2.0 * std::sinh(a.val) / cosh_3,
                     a);
    }

    __host__ __device__ friend Scalar asinh(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT s      = a.val * a.val + (PassiveT)1.0;
        const PassiveT s_sqrt = std::sqrt(s);
        assert(is_finite(s));
        assert(is_finite(s_sqrt));

        return chain(
            std::asinh(a.val), (PassiveT)1.0 / s_sqrt, -a.val / s_sqrt / s, a);
    }

    __host__ __device__ friend Scalar acosh(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT sm      = a.val - (PassiveT)1.0;
        const PassiveT sp      = a.val + (PassiveT)1.0;
        const PassiveT sm_sqrt = std::sqrt(sm);
        const PassiveT sp_sqrt = std::sqrt(sp);
        const PassiveT prod    = sm_sqrt * sp_sqrt;
        assert(is_finite(sm_sqrt));
        assert(is_finite(sp_sqrt));

        return chain(std::acosh(a.val),
                     (PassiveT)1.0 / prod,
                     -a.val / prod / sm / sp,
                     a);
    }

    __host__ __device__ friend Scalar atanh(const Scalar& a)
    {
        assert(is_finite_scalar(a));

        const PassiveT s = (PassiveT)1.0 - a.val * a.val;
        return chain(std::atanh(a.val),
                     (PassiveT)1.0 / s,
                     (PassiveT)2.0 * a.val / s / s,
                     a);
    }

    __host__ __device__ friend bool isnan(const Scalar& a)
    {
        return is_nan(a.val);
    }

    __host__ __device__ friend bool isinf(const Scalar& a)
    {
        return is_inf(a.val);
    }

    __host__ __device__ friend bool isfinite(const Scalar& a)
    {

        return is_finite(a.val);
    }

    // ///////////////////////////////////////////////////////////////////////////
    // Binary operators
    // ///////////////////////////////////////////////////////////////////////////

    __host__ __device__ friend Scalar operator+(const Scalar& a,
                                                const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        Scalar res;
        res.val  = a.val + b.val;
        res.grad = a.grad + b.grad;

        if constexpr (WithHessian)
            res.Hess = a.Hess + b.Hess;

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ friend Scalar operator+(const Scalar&   a,
                                                const PassiveT& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite(b));

        Scalar res = a;
        res.val += b;

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ friend Scalar operator+(const PassiveT& a,
                                                const Scalar&   b)
    {
        assert(is_finite(a));
        assert(is_finite_scalar(b));

        Scalar res = b;
        res.val += a;

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ Scalar& operator+=(const Scalar& b)
    {
        assert(is_finite_scalar(*this));
        assert(is_finite_scalar(b));

        assert(this->grad.size() == b.grad.size());

        this->val += b.val;
        this->grad += b.grad;
        if constexpr (WithHessian)
            this->Hess += b.Hess;

        assert(is_finite_scalar(*this));
        return *this;
    }

    __host__ __device__ Scalar& operator+=(const PassiveT& b)
    {
        assert(is_finite(b));
        assert(is_finite_scalar(*this));

        this->val += b;

        assert(is_finite_scalar(*this));
        return *this;
    }

    __host__ __device__ friend Scalar operator-(const Scalar& a,
                                                const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        assert(a.grad.size() == b.grad.size());


        Scalar res;
        res.val  = a.val - b.val;
        res.grad = a.grad - b.grad;

        if constexpr (WithHessian)
            res.Hess = a.Hess - b.Hess;

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ friend Scalar operator-(const Scalar&   a,
                                                const PassiveT& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite(b));

        Scalar res = a;
        res.val -= b;

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ friend Scalar operator-(const PassiveT& a,
                                                const Scalar&   b)
    {
        assert(is_finite(a));
        assert(is_finite_scalar(b));

        Scalar res;
        res.val  = a - b.val;
        res.grad = -b.grad;

        if constexpr (WithHessian)
            res.Hess = -b.Hess;

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ Scalar& operator-=(const Scalar& b)
    {
        assert(is_finite_scalar(*this));
        assert(is_finite_scalar(b));


        assert(this->grad.size() == b.grad.size());

        this->val -= b.val;
        this->grad -= b.grad;

        if constexpr (WithHessian)
            this->Hess -= b.Hess;

        assert(is_finite_scalar(*this));
        return *this;
    }

    __host__ __device__ Scalar& operator-=(const PassiveT& b)
    {
        assert(is_finite(b));
        assert(is_finite_scalar(*this));

        this->val -= b;

        assert(is_finite_scalar(*this));
        return *this;
    }

    __host__ __device__ friend Scalar operator*(const Scalar& a,
                                                const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        assert(a.grad.size() == b.grad.size());

        Scalar res;
        res.val  = a.val * b.val;
        res.grad = b.val * a.grad + a.val * b.grad;

        // Exploiting symmetry did not yield speedup in some tests
        //        if constexpr (WithHessian)
        //        {
        //            for(Eigen::Index j = 0; j < k; ++j)
        //            {
        //                for(Eigen::Index i = j; i < k; ++i)
        //                {
        //                    res.Hess(i, j) = b.val * a.Hess(i, j) + a.grad[i]
        //                    * b.grad[j] + a.grad[j] * b.grad[i] + a.val *
        //                    b.Hess(i, j);
        //                }
        //            }
        //            res.Hess = res.Hess.template
        //            selfadjointView<Eigen::Lower>();
        //        }

        if constexpr (WithHessian)
            res.Hess = b.val * a.Hess + a.grad * b.grad.transpose() +
                       b.grad * a.grad.transpose() + a.val * b.Hess;

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ friend Scalar operator*(const Scalar&   a,
                                                const PassiveT& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite(b));

        Scalar res = a;
        res.val *= b;
        res.grad *= b;

        if constexpr (WithHessian)
            res.Hess *= b;

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ friend Scalar operator*(const PassiveT& a,
                                                const Scalar&   b)
    {
        assert(is_finite(a));
        assert(is_finite_scalar(b));

        Scalar res = b;
        res.val *= a;
        res.grad *= a;

        if constexpr (WithHessian)
            res.Hess *= a;

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ Scalar& operator*=(const Scalar& b)
    {
        *this = *this * b;
        return *this;
    }

    __host__ __device__ Scalar& operator*=(const PassiveT& b)
    {
        *this = *this * b;
        return *this;
    }

    __host__ __device__ friend Scalar operator/(const Scalar& a,
                                                const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));


        assert(a.grad.size() == b.grad.size());

        Scalar res;
        res.val  = a.val / b.val;
        res.grad = (b.val * a.grad - a.val * b.grad) / (b.val * b.val);

        if constexpr (WithHessian)
            res.Hess = (a.Hess - res.grad * b.grad.transpose() -
                        b.grad * res.grad.transpose() - res.val * b.Hess) /
                       b.val;

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ friend Scalar operator/(const Scalar&   a,
                                                const PassiveT& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite(b));

        Scalar res = a;
        res.val /= b;
        res.grad /= b;

        if constexpr (WithHessian)
            res.Hess /= b;

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ friend Scalar operator/(const PassiveT& a,
                                                const Scalar&   b)
    {
        assert(is_finite(a));
        assert(is_finite_scalar(b));

        Scalar res;
        res.val  = a / b.val;
        res.grad = (-a / (b.val * b.val)) * b.grad;

        if constexpr (WithHessian)
            res.Hess = (-res.grad * b.grad.transpose() -
                        b.grad * res.grad.transpose() - res.val * b.Hess) /
                       b.val;

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ Scalar& operator/=(const Scalar& b)
    {
        *this = *this / b;
        return *this;
    }

    __host__ __device__ Scalar& operator/=(const PassiveT& b)
    {
        *this = *this / b;
        return *this;
    }

    __host__ __device__ friend Scalar atan2(const Scalar& y, const Scalar& x)
    {
        assert(is_finite_scalar(y));
        assert(is_finite_scalar(x));

        assert(y.grad.size() == x.grad.size());

        Scalar res;
        res.val = std::atan2(y.val, x.val);

        const GradType u = x.val * y.grad - y.val * x.grad;
        const PassiveT v = x.val * x.val + y.val * y.val;
        res.grad         = u / v;

        if constexpr (WithHessian) {
            const HessType du = x.val * y.Hess - y.val * x.Hess +
                                y.grad * x.grad.transpose() -
                                x.grad * y.grad.transpose();
            const GradType dv =
                (PassiveT)2.0 * (x.val * x.grad + y.val * y.grad);
            res.Hess = (du - res.grad * dv.transpose()) / v;
        }

        assert(is_finite_scalar(res));
        return res;
    }

    __host__ __device__ friend Scalar hypot(const Scalar& a, const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        assert(a.grad.size() == b.grad.size());

        return sqrt(a * a + b * b);
    }

    __host__ __device__ friend bool operator==(const Scalar& a, const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        return a.val == b.val;
    }

    __host__ __device__ friend bool operator==(const Scalar&   a,
                                               const PassiveT& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite(b));

        return a.val == b;
    }

    __host__ __device__ friend bool operator==(const PassiveT& a,
                                               const Scalar&   b)
    {
        assert(is_finite(a));
        assert(is_finite_scalar(b));

        return a == b.val;
    }

    __host__ __device__ friend bool operator!=(const Scalar& a, const Scalar& b)
    {

        return a.val != b.val;
    }

    __host__ __device__ friend bool operator!=(const Scalar&   a,
                                               const PassiveT& b)
    {

        return a.val != b;
    }

    __host__ __device__ friend bool operator!=(const PassiveT& a,
                                               const Scalar&   b)
    {

        return a != b.val;
    }

    __host__ __device__ friend bool operator<(const Scalar& a, const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        return a.val < b.val;
    }

    __host__ __device__ friend bool operator<(const Scalar&   a,
                                              const PassiveT& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite(b));

        return a.val < b;
    }

    __host__ __device__ friend bool operator<(const PassiveT& a,
                                              const Scalar&   b)
    {
        assert(is_finite(a));
        assert(is_finite_scalar(b));

        return a < b.val;
    }

    __host__ __device__ friend bool operator<=(const Scalar& a, const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        return a.val <= b.val;
    }

    __host__ __device__ friend bool operator<=(const Scalar&   a,
                                               const PassiveT& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite(b));

        return a.val <= b;
    }

    __host__ __device__ friend bool operator<=(const PassiveT& a,
                                               const Scalar&   b)
    {
        assert(is_finite(a));
        assert(is_finite_scalar(b));

        return a <= b.val;
    }

    __host__ __device__ friend bool operator>(const Scalar& a, const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));
        return a.val > b.val;
    }

    __host__ __device__ friend bool operator>(const Scalar&   a,
                                              const PassiveT& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite(b));

        return a.val > b;
    }

    __host__ __device__ friend bool operator>(const PassiveT& a,
                                              const Scalar&   b)
    {
        assert(is_finite(a));
        assert(is_finite_scalar(b));

        return a > b.val;
    }

    __host__ __device__ friend bool operator>=(const Scalar& a, const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        return a.val >= b.val;
    }

    __host__ __device__ friend bool operator>=(const Scalar&   a,
                                               const PassiveT& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite(b));

        return a.val >= b;
    }

    __host__ __device__ friend bool operator>=(const PassiveT& a,
                                               const Scalar&   b)
    {
        assert(is_finite(a));
        assert(is_finite_scalar(b));

        return a >= b.val;
    }

    __host__ __device__ friend Scalar min(const Scalar& a, const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        return (b < a) ? b : a;
    }

    __host__ __device__ friend Scalar fmin(const Scalar& a, const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        return min(a, b);
    }

    __host__ __device__ friend Scalar max(const Scalar& a, const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        return (a < b) ? b : a;
    }

    __host__ __device__ friend Scalar fmax(const Scalar& a, const Scalar& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        return max(a, b);
    }

    // ///////////////////////////////////////////////////////////////////////////
    // std::complex operators (just spell out and differentiate the real case)
    // ///////////////////////////////////////////////////////////////////////////

    /*__host__ __device__ friend std::complex<Scalar> operator+(
        const std::complex<Scalar>& a,
        const std::complex<Scalar>& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        return std::complex<Scalar>(a.real() + b.real(), a.imag() + b.imag());
    }

    __host__ __device__ friend std::complex<Scalar> operator+(
        const std::complex<PassiveT>& a,
        const std::complex<Scalar>&   b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));


        return std::complex<Scalar>(a.real() + b.real(), a.imag() + b.imag());
    }

    __host__ __device__ friend std::complex<Scalar> operator+(
        const std::complex<Scalar>&   a,
        const std::complex<PassiveT>& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));


        return std::complex<Scalar>(a.real() + b.real(), a.imag() + b.imag());
    }

    __host__ __device__ friend std::complex<Scalar> operator-(
        const std::complex<Scalar>& a,
        const std::complex<Scalar>& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));


        return std::complex<Scalar>(a.real() - b.real(), a.imag() - b.imag());
    }

    __host__ __device__ friend std::complex<Scalar> operator-(
        const std::complex<PassiveT>& a,
        const std::complex<Scalar>&   b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        return std::complex<Scalar>(a.real() - b.real(), a.imag() - b.imag());
    }

    __host__ __device__ friend std::complex<Scalar> operator-(
        const std::complex<Scalar>&   a,
        const std::complex<PassiveT>& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        return std::complex<Scalar>(a.real() - b.real(), a.imag() - b.imag());
    }

    __host__ __device__ friend std::complex<Scalar> operator*(
        const std::complex<Scalar>& a,
        const std::complex<Scalar>& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));


        return std::complex<Scalar>(a.real() * b.real() - a.imag() * b.imag(),
                                    a.real() * b.imag() + a.imag() * b.real());
    }

    __host__ __device__ friend std::complex<Scalar> operator*(
        const std::complex<PassiveT>& a,
        const std::complex<Scalar>&   b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        return std::complex<Scalar>(a.real() * b.real() - a.imag() * b.imag(),
                                    a.real() * b.imag() + a.imag() * b.real());
    }

    __host__ __device__ friend std::complex<Scalar> operator*(
        const std::complex<Scalar>&   a,
        const std::complex<PassiveT>& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));


        return std::complex<Scalar>(a.real() * b.real() - a.imag() * b.imag(),
                                    a.real() * b.imag() + a.imag() * b.real());
    }

    __host__ __device__ friend std::complex<Scalar> sqr(
        const std::complex<Scalar>& a)
    {
        assert(is_finite_scalar(a));

        return std::complex<Scalar>(sqr(a.real()) - sqr(a.imag()),
                                    2.0 * a.real() * a.imag());
    }

    __host__ __device__ friend std::complex<Scalar> operator/(
        const std::complex<Scalar>& a,
        const std::complex<Scalar>& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));


        const Scalar denom = b.real() * b.real() + b.imag() * b.imag();
        return std::complex<Scalar>(
            (a.real() * b.real() + a.imag() * b.imag()) / denom,
            (a.imag() * b.real() - a.real() * b.imag()) / denom);
    }

    __host__ __device__ friend std::complex<Scalar> operator/(
        const std::complex<Scalar>&   a,
        const std::complex<PassiveT>& b)
    {
        assert(is_finite_scalar(a));
        assert(is_finite_scalar(b));

        const PassiveT denom = b.real() * b.real() + b.imag() * b.imag();
        return std::complex<Scalar>(
            (a.real() * b.real() + a.imag() * b.imag()) / denom,
            (a.imag() * b.real() - a.real() * b.imag()) / denom);
    }

    __host__ __device__ friend std::complex<Scalar> conj(
        const std::complex<Scalar>& a)
    {
        assert(is_finite_scalar(a));

        return std::complex<Scalar>(a.real(), -a.imag());
    }

    __host__ __device__ friend Scalar abs(const std::complex<Scalar>& a)
    {
        assert(is_finite_scalar(a));

        return hypot(a.real(), a.imag());
    }

    __host__ __device__ friend Scalar arg(const std::complex<Scalar>& a)
    {
        assert(is_finite_scalar(a));

        return atan2(a.imag(), a.real());
    }*/

    // ///////////////////////////////////////////////////////////////////////////
    // Stream Operators
    // ///////////////////////////////////////////////////////////////////////////

    __host__ friend std::ostream& operator<<(std::ostream& s, const Scalar& a)
    {
        s << a.val << std::endl;
        s << "grad: \n" << a.grad << std::endl;
        if constexpr (WithHessian)
            s << "Hess: \n" << a.Hess;
        return s;
    }

    __host__ __device__ void print() const
    {
        printf("\n val: %.9g", val);
        printf("\n grad: ");
        for (int i = 0; i < k; ++i) {
            printf("\n %.9g", grad(i));
        }
        if constexpr (WithHessian) {
            printf("\n Hess: ");
            for (int i = 0; i < k; ++i) {
                printf("\n");
                for (int j = 0; j < k; ++j) {
                    printf("%.9g ", Hess(i, j));
                }
            }
        }
    }

    // ///////////////////////////////////////////////////////////////////////////
    // Data
    // ///////////////////////////////////////////////////////////////////////////

    PassiveT val  = 0.0;  // Scalar value of this (intermediate) variable.
    GradType grad = GradType::Zero(k);  // Gradient (first derivative) of val
                                        // w.r.t. the active variable vector.

    // Hessian (second derivative) of val
    // w.r.t. the active variable vector.
    HessType Hess =
        HessType::Zero((WithHessian ? k : 0), (WithHessian ? k : 0));
};

// ///////////////////////////////////////////////////////////////////////////
// Overloads (Fails to build on windows otherwise)
// ///////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2>
__host__ __device__ T1 pow(const T1& a, const T2& e)
{
    return std::pow(a, e);
}

template <typename PassiveT>
__host__ __device__ PassiveT atan2(const PassiveT& _y, const PassiveT& _x)
{
    return std::atan2(_y, _x);
}

// ///////////////////////////////////////////////////////////////////////////
// Explicit conversion to passive types
// ///////////////////////////////////////////////////////////////////////////

template <int k, typename PassiveT, bool WithHessian>
__host__ __device__ PassiveT
to_passive(const Scalar<PassiveT, k, WithHessian>& a)
{
    return a.val;
}

template <int k, int rows, int cols, typename PassiveT, bool WithHessian>
__host__ __device__ Eigen::Matrix<PassiveT, rows, cols> to_passive(
    const Eigen::Matrix<Scalar<PassiveT, k, WithHessian>, rows, cols>& A)
{
    Eigen::Matrix<PassiveT, rows, cols> A_passive(A.rows(), A.cols());
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        for (Eigen::Index j = 0; j < A.cols(); ++j)
            A_passive(i, j) = A(i, j).val;
    }

    return A_passive;
}

// ///////////////////////////////////////////////////////////////////////////
// Scalar typedefs
// ///////////////////////////////////////////////////////////////////////////

template <int k, bool WithHessian = true>
using Float = Scalar<float, k, WithHessian>;
template <int k, bool WithHessian = true>
using Double = Scalar<double, k, WithHessian>;
template <int k, bool WithHessian = true>
using LongDouble = Scalar<long double, k, WithHessian>;

}  // namespace rxmesh

// ///////////////////////////////////////////////////////////////////////////
// Eigen3 traits
// ///////////////////////////////////////////////////////////////////////////
namespace Eigen {

/**
 * See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
 * and https://eigen.tuxfamily.org/dox/structEigen_1_1NumTraits.html
 */
template <int k, typename PassiveT, bool WithHessian>
struct NumTraits<rxmesh::Scalar<PassiveT, k, WithHessian>> : NumTraits<PassiveT>
{
    typedef rxmesh::Scalar<PassiveT, k, WithHessian> Real;
    typedef rxmesh::Scalar<PassiveT, k, WithHessian> NonInteger;
    typedef rxmesh::Scalar<PassiveT, k, WithHessian> Nested;

    enum
    {
        IsComplex             = 0,
        IsInteger             = 0,
        IsSigned              = 1,
        RequireInitialization = 1,
        ReadCost              = 1,
        AddCost = k == Eigen::Dynamic ? 1 : 1 + k + (WithHessian ? k * k : 0),
        MulCost = k == Eigen::Dynamic ? 1 : 1 + k + (WithHessian ? k * k : 0),
    };
};

/*
 * Let Eigen know that binary operations between rxmesh::Scalar and T are
 * allowed, and that the return type is rxmesh::Scalar.
 */
template <typename BinaryOp, int k, typename PassiveT, bool WithHessian>
struct ScalarBinaryOpTraits<rxmesh::Scalar<PassiveT, k, WithHessian>,
                            PassiveT,
                            BinaryOp>
{
    typedef rxmesh::Scalar<PassiveT, k, WithHessian> ReturnType;
};

template <typename BinaryOp, int k, typename PassiveT, bool WithHessian>
struct ScalarBinaryOpTraits<PassiveT,
                            rxmesh::Scalar<PassiveT, k, WithHessian>,
                            BinaryOp>
{
    typedef rxmesh::Scalar<PassiveT, k, WithHessian> ReturnType;
};

}  // namespace Eigen
