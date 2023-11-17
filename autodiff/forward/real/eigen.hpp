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
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/utils/gradient.hpp>
#include <autodiff/common/eigen.hpp>

//------------------------------------------------------------------------------
// SUPPORT FOR EIGEN MATRICES AND VECTORS OF REAL
//------------------------------------------------------------------------------
namespace Eigen {

template<typename T>
struct NumTraits;

template<size_t N, typename T>
struct NumTraits<autodiff::Real<N, T>> : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
    typedef autodiff::Real<N, T> Real;
    typedef autodiff::Real<N, T> NonInteger;
    typedef autodiff::Real<N, T> Nested;
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

template<size_t N, typename T, typename BinOp>
struct ScalarBinaryOpTraits<autodiff::Real<N, T>, T, BinOp>
{
    typedef autodiff::Real<N, T> ReturnType;
};

template<size_t N, typename T, typename BinOp>
struct ScalarBinaryOpTraits<T, autodiff::Real<N, T>, BinOp>
{
    typedef autodiff::Real<N, T> ReturnType;
};

} // namespace Eigen

namespace autodiff {

AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(real0th, real0th);
AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(real1st, real1st);
AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(real2nd, real2nd);
AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(real3rd, real3rd);
AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(real4th, real4th);

AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(real, real)

} // namespace autodiff
