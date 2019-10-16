//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2019 Allan Leal
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
#include <autodiff/forward/common/gradient.hpp>
#include <autodiff/utils/eigenmacros.hpp>

//------------------------------------------------------------------------------
// SUPPORT FOR EIGEN MATRICES AND VECTORS OF DUAL
//------------------------------------------------------------------------------
namespace Eigen {

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
struct ScalarBinaryOpTraits<autodiff::Dual<T, G>, T, BinOp>
{
    typedef autodiff::Dual<T, G> ReturnType;
};

template<typename T, typename G, typename BinOp>
struct ScalarBinaryOpTraits<T, autodiff::Dual<T, G>, BinOp>
{
    typedef autodiff::Dual<T, G> ReturnType;
};

} // namespace Eigen

namespace autodiff {

AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual, dual)

} // namespace autodiff
