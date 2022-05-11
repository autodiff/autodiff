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

// C++ includes
#include <autodiff/common/meta.hpp>

namespace autodiff {
namespace detail {

/// A trait class used to specify whether a type is arithmetic.
template<typename T>
struct ArithmeticTraits
{
    static constexpr bool isArithmetic = std::is_arithmetic_v<T>;
};

/// A compile-time constant that indicates whether a type is arithmetic.
template<typename T>
constexpr bool isArithmetic = ArithmeticTraits<PlainType<T>>::isArithmetic;

/// An auxiliary template type to indicate NumberTraits has not been defined for a type.
template<typename T>
struct NumericTypeInfoNotDefinedFor { using type = T; };

/// A trait class used to specify whether a type is an autodiff number.
template<typename T>
struct NumberTraits
{
    /// The underlying floating point type of the autodiff number type.
    using NumericType = std::conditional_t<isArithmetic<T>, T, NumericTypeInfoNotDefinedFor<T>>;

    /// The order of the autodiff number type.
    static constexpr auto Order = 0;
};

/// A template alias to get the underlying floating point type of an autodiff number.
template<typename T>
using NumericType = typename NumberTraits<PlainType<T>>::NumericType;

/// A compile-time constant with the order of an autodiff number.
template<typename T>
constexpr auto Order = NumberTraits<PlainType<T>>::Order;

} // namespace detail
} // namespace autodiff
