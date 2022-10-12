//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2022 Allan Leal
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
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/utils/gradient.hpp>
#include <autodiff/common/eigen.hpp>

//------------------------------------------------------------------------------
// SUPPORT FOR EIGEN MATRICES AND VECTORS OF DUAL
//------------------------------------------------------------------------------
namespace Eigen {

using namespace autodiff;
using namespace autodiff::detail;

template<typename T>
struct NumTraits;

template<typename T, typename G>
struct NumTraits<autodiff::Dual<T, G>> : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
    typedef autodiff::Dual<T, G> Real;
    typedef autodiff::Dual<T, G> NonInteger;
    typedef autodiff::Dual<T, G> Nested;
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

template<typename T, typename G, typename BinOp>
struct ScalarBinaryOpTraits<Dual<T, G>, NumericType<T>, BinOp>
{
    typedef DualType<Dual<T, G>> ReturnType;
};

template<typename Op, typename R, typename BinOp>
struct ScalarBinaryOpTraits<UnaryExpr<Op, R>, NumericType<UnaryExpr<Op, R>>, BinOp>
{
    typedef DualType<UnaryExpr<Op, R>> ReturnType;
};

template<typename Op, typename L, typename R, typename BinOp>
struct ScalarBinaryOpTraits<BinaryExpr<Op, L, R>, NumericType<BinaryExpr<Op, L, R>>, BinOp>
{
    typedef DualType<BinaryExpr<Op, L, R>> ReturnType;
};

template<typename Op, typename L, typename C, typename R, typename BinOp>
struct ScalarBinaryOpTraits<TernaryExpr<Op, L, C, R>, NumericType<TernaryExpr<Op, L, C, R>>, BinOp>
{
    typedef DualType<TernaryExpr<Op, L, C, R>> ReturnType;
};

template<typename T, typename G, typename BinOp>
struct ScalarBinaryOpTraits<NumericType<T>, Dual<T, G>, BinOp>
{
    typedef DualType<Dual<T, G>> ReturnType;
};

template<typename Op, typename R, typename BinOp>
struct ScalarBinaryOpTraits<NumericType<UnaryExpr<Op, R>>, UnaryExpr<Op, R>, BinOp>
{
    typedef DualType<UnaryExpr<Op, R>> ReturnType;
};

template<typename Op, typename L, typename R, typename BinOp>
struct ScalarBinaryOpTraits<NumericType<BinaryExpr<Op, L, R>>, BinaryExpr<Op, L, R>, BinOp>
{
    typedef DualType<BinaryExpr<Op, L, R>> ReturnType;
};

template<typename Op, typename L, typename C, typename R, typename BinOp>
struct ScalarBinaryOpTraits<NumericType<TernaryExpr<Op, L, C, R>>, TernaryExpr<Op, L, C, R>, BinOp>
{
    typedef DualType<TernaryExpr<Op, L, C, R>> ReturnType;
};

} // namespace Eigen

namespace autodiff {

AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual0th, dual0th);
AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual1st, dual1st);
AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual2nd, dual2nd);
AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual3rd, dual3rd);
AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual4th, dual4th);

AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual, dual)

} // namespace autodiff
