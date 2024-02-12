//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright © 2018–2024 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// Eigen includes
#include <Eigen/Core>

// autodiff includes
#include <autodiff/common/eigen.hpp>
#include <autodiff/common/meta.hpp>
#include <autodiff/reverse/var/var.hpp>

//------------------------------------------------------------------------------
// SUPPORT FOR EIGEN MATRICES AND VECTORS OF VAR
//------------------------------------------------------------------------------
namespace Eigen {

template<typename T>
struct NumTraits;

template<typename T>
struct NumTraits<autodiff::Variable<T>> : NumTraits<T> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
    typedef autodiff::Variable<T> Real;
    typedef autodiff::Variable<T> NonInteger;
    typedef autodiff::Variable<T> Nested;
    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};

template<typename T, typename BinOp>
struct ScalarBinaryOpTraits<autodiff::Variable<T>, T, BinOp>
{
    typedef autodiff::Variable<T> ReturnType;
};

template<typename T, typename BinOp>
struct ScalarBinaryOpTraits<T, autodiff::Variable<T>, BinOp>
{
    typedef autodiff::Variable<T> ReturnType;
};

template<typename T>
struct NumTraits<autodiff::reverse::detail::ExprPtr<T>> : NumTraits<T> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
    typedef autodiff::Variable<T> Real;
    typedef autodiff::Variable<T> NonInteger;
    typedef autodiff::Variable<T> Nested;
    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};

template<typename T, typename BinOp>
struct ScalarBinaryOpTraits<autodiff::reverse::detail::ExprPtr<T>, T, BinOp>
{
    typedef autodiff::Variable<T> ReturnType;
};

template<typename T, typename BinOp>
struct ScalarBinaryOpTraits<T, autodiff::reverse::detail::ExprPtr<T>, BinOp>
{
    typedef autodiff::Variable<T> ReturnType;
};

} // namespace Eigen

namespace autodiff {
namespace reverse {
namespace detail {

template<typename T, int Rows, int MaxRows>
using Vec = Eigen::Matrix<T, Rows, 1, 0, MaxRows, 1>;

template<typename T, int Rows, int Cols, int MaxRows, int MaxCols>
using Mat = Eigen::Matrix<T, Rows, Cols, 0, MaxRows, MaxCols>;

/// Return the gradient vector of variable y with respect to variables x.
template<typename T, typename X>
auto gradient(const Variable<T>& y, Eigen::DenseBase<X>& x)
{
    using U = VariableValueType<T>;

    using ScalarX = typename X::Scalar;
    static_assert(isVariable<ScalarX>, "Argument x is not a vector with Variable<T> (aka var) objects..");

    constexpr auto isVec = X::IsVectorAtCompileTime;
    static_assert(isVec, "Argument x is not a vector.");

    constexpr auto Rows = X::RowsAtCompileTime;
    constexpr auto MaxRows = X::MaxRowsAtCompileTime;

    const auto n = x.size();
    using Gradient = Vec<U, Rows, MaxRows>;
    Gradient g = Gradient::Zero(n);

    for(auto i = 0; i < n; ++i)
        x[i].expr->bind_value(&g[i]);

    y.expr->propagate(1.0);

    for(auto i = 0; i < n; ++i)
        x[i].expr->bind_value(nullptr);

    return g;
}

/// Return the Hessian matrix of variable y with respect to variables x.
template<typename T, typename X, typename GradientVec>
auto hessian(const Variable<T>& y, Eigen::DenseBase<X>& x, GradientVec& g)
{
    using U = VariableValueType<T>;

    using ScalarX = typename X::Scalar;
    static_assert(isVariable<ScalarX>, "Argument x is not a vector with Variable<T> (aka var) objects.");

    using ScalarG = typename GradientVec::Scalar;
    static_assert(std::is_same_v<U, ScalarG>, "Argument g does not have the same arithmetic type as y.");

    constexpr auto Rows = X::RowsAtCompileTime;
    constexpr auto MaxRows = X::MaxRowsAtCompileTime;

    const auto n = x.size();

    // Form a vector containing gradient expressions for each variable
    using ExpressionGradient = Vec<ScalarX, Rows, MaxRows>;
    ExpressionGradient G(n);

    for(auto k = 0; k < n; ++k)
        x[k].expr->bind_expr(&G(k).expr);

    /* Build a full gradient expression in DFS tree traversal, updating
     * gradient expressions when encountering variables
     */
    y.expr->propagatex(constant<T>(1.0));

    for(auto k = 0; k < n; ++k) {
      x[k].expr->bind_expr(nullptr);
    }

    // Read the gradient value from gradient expressions' cached values
    g.resize(n);
    for(auto i = 0; i < n; ++i)
        g[i] = val(G[i]);

    // Form a numeric hessian using the gradient expressions
    using Hessian = Mat<U, Rows, Rows, MaxRows, MaxRows>;
    Hessian H = Hessian::Zero(n, n);
    for(auto i = 0; i < n; ++i)
    {
        for(auto k = 0; k < n; ++k)
            x[k].expr->bind_value(&H(i, k));

        // Propagate a second derivative value calculation down the gradient expression tree for variable i
        G[i].expr->propagate(1.0);

        for(auto k = 0; k < n; ++k)
            x[k].expr->bind_value(nullptr);
    }

    return H;
}

/// Return the Hessian matrix of variable y with respect to variables x.
template<typename T, typename X>
auto hessian(const Variable<T>& y, Eigen::DenseBase<X>& x)
{
    using U = VariableValueType<T>;
    constexpr auto Rows = X::RowsAtCompileTime;
    constexpr auto MaxRows = X::MaxRowsAtCompileTime;
    Vec<U, Rows, MaxRows> g;
    return hessian(y, x, g);
}

} // namespace detail
  //
} // namespace reverse

AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(autodiff::var, var)

using reverse::detail::gradient;
using reverse::detail::hessian;

} // namespace autodiff
