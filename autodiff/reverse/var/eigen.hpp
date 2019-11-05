//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2020 Allan Leal
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
#include <autodiff/reverse/var/var.hpp>
#include <autodiff/common/eigen.hpp>

//------------------------------------------------------------------------------
// SUPPORT FOR EIGEN MATRICES AND VECTORS OF VAR
//------------------------------------------------------------------------------
namespace Eigen {

template<typename T>
struct NumTraits;

template<>
struct NumTraits<autodiff::var> : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
    typedef autodiff::var Real;
    typedef autodiff::var NonInteger;
    typedef autodiff::var Nested;
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

template<typename BinOp>
struct ScalarBinaryOpTraits<autodiff::var, double, BinOp>
{
    typedef autodiff::var ReturnType;
};

template<typename BinOp>
struct ScalarBinaryOpTraits<double, autodiff::var, BinOp>
{
    typedef autodiff::var ReturnType;
};

} // namespace Eigen

namespace autodiff {

AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(var, var);

/// Return the gradient vector of variable y with respect to variables x.
template<typename vars>
auto gradient(const var& y, const vars& x) -> Eigen::VectorXd
{
    const auto n = x.size();
    Eigen::VectorXd dydx(n);
    Derivatives dyd = derivatives(y);
    for(auto i = 0; i < n; ++i)
        dydx[i] = dyd(x[i]);
    return dydx;
}

/// Return the Hessian matrix of variable y with respect to variables x.
template<typename vars>
auto hessian(const var& y, const vars& x) -> Eigen::MatrixXd
{
    const auto n = x.size();
    Eigen::MatrixXd mat(n, n);
    DerivativesX dyd = derivativesx(y);
    for(auto i = 0; i < n; ++i)
    {
        Derivatives d2yd = derivatives(dyd(x[i]));
        for(auto j = i; j < n; ++j) {
            mat(i, j) = mat(j, i) = d2yd(x(j));
        }
    }
    return mat;
}

} // namespace autodiff


