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

// C++ includes
#include <cstddef>
#include <tuple>
#include <type_traits>

#ifndef AUTODIFF_DEVICE_FUNC
#ifdef AUTODIFF_EIGEN_FOUND
    #include <Eigen/src/Core/util/Macros.h>
    #define AUTODIFF_DEVICE_FUNC EIGEN_DEVICE_FUNC
#else
    #define AUTODIFF_DEVICE_FUNC
#endif
#endif

namespace autodiff {
namespace detail {

template<bool value>
using EnableIf = std::enable_if_t<value>;

template<bool value>
using Requires = std::enable_if_t<value, bool>;

template<typename T>
using PlainType = std::remove_cv_t<std::remove_reference_t<T>>;

template<bool Cond, typename WhenTrue, typename WhenFalse>
using ConditionalType = std::conditional_t<Cond, WhenTrue, WhenFalse>;

template<typename A, typename B>
using CommonType = std::common_type_t<A, B>;

template<typename Fun, typename... Args>
using ReturnType = std::invoke_result_t<Fun, Args...>;

template<typename T>
constexpr bool isConst = std::is_const_v<std::remove_reference_t<T>>;

template<typename T, typename U>
constexpr bool isConvertible = std::is_convertible<PlainType<T>, U>::value;

template<typename A, typename B>
constexpr bool isSame = std::is_same_v<A, B>;

template<typename Tuple>
constexpr auto TupleSize = std::tuple_size_v<std::decay_t<Tuple>>;

template<typename Tuple>
AUTODIFF_DEVICE_FUNC constexpr auto TupleHead(Tuple&& tuple)
{
    return std::get<0>(std::forward<Tuple>(tuple));
}

template<typename Tuple>
AUTODIFF_DEVICE_FUNC constexpr auto TupleTail(Tuple&& tuple)
{
    auto g = [&](auto&&, auto&&... args) constexpr {
        return std::forward_as_tuple(args...);
    };
    return std::apply(g, std::forward<Tuple>(tuple));
}

template<size_t i>
struct Index
{
    constexpr static size_t index = i;
    constexpr operator size_t() const { return index; }
    constexpr operator size_t() { return index; }
};

template<size_t i, size_t ibegin, size_t iend, typename Function>
AUTODIFF_DEVICE_FUNC constexpr auto AuxFor(Function&& f)
{
    if constexpr (i < iend) {
        f(Index<i>{});
        AuxFor<i + 1, ibegin, iend>(std::forward<Function>(f));
    }
}

template<size_t ibegin, size_t iend, typename Function>
AUTODIFF_DEVICE_FUNC constexpr auto For(Function&& f)
{
    AuxFor<ibegin, ibegin, iend>(std::forward<Function>(f));
}

template<size_t iend, typename Function>
AUTODIFF_DEVICE_FUNC constexpr auto For(Function&& f)
{
    For<0, iend>(std::forward<Function>(f));
}

template<size_t i, size_t ibegin, size_t iend, typename Function>
AUTODIFF_DEVICE_FUNC constexpr auto AuxReverseFor(Function&& f)
{
    if constexpr (i < iend)
    {
        AuxReverseFor<i + 1, ibegin, iend>(std::forward<Function>(f));
        f(Index<i>{});
    }
}

template<size_t ibegin, size_t iend, typename Function>
AUTODIFF_DEVICE_FUNC constexpr auto ReverseFor(Function&& f)
{
    AuxReverseFor<ibegin, ibegin, iend>(std::forward<Function>(f));
}

template<size_t iend, typename Function>
AUTODIFF_DEVICE_FUNC constexpr auto ReverseFor(Function&& f)
{
    ReverseFor<0, iend>(std::forward<Function>(f));
}

template<typename Tuple, typename Function>
AUTODIFF_DEVICE_FUNC constexpr auto ForEach(Tuple&& tuple, Function&& f)
{
    constexpr auto N = TupleSize<Tuple>;
    For<N>([&](auto i) constexpr {
        f(std::get<i>(tuple));
    });
    //------------------------------------------------------------
    // ALTERNATIVE IMPLEMENTATION POSSIBLY USEFUL TO KEEP IT HERE
    // auto g = [&](auto&&... args) constexpr {
    //     ( f(std::forward<decltype(args)>(args)), ...);
    // };
    // std::apply(g, std::forward<Tuple>(tuple));
    //------------------------------------------------------------
}

template<typename Tuple1, typename Tuple2, typename Function>
AUTODIFF_DEVICE_FUNC constexpr auto ForEach(Tuple1&& tuple1, Tuple2&& tuple2, Function&& f)
{
    constexpr auto N1 = TupleSize<Tuple1>;
    constexpr auto N2 = TupleSize<Tuple2>;
    static_assert(N1 == N2);
    For<N1>([&](auto i) constexpr {
        f(std::get<i>(tuple1), std::get<i>(tuple2));
    });
}

template<size_t ibegin, size_t iend, typename Function>
AUTODIFF_DEVICE_FUNC constexpr auto Sum(Function&& f)
{
    using ResultType = std::invoke_result_t<Function, Index<ibegin>>;
    ResultType res = {};
    For<ibegin, iend>([&](auto i) constexpr {
        res += f(Index<i>{});
    });
    return res;
}

template<typename Tuple, typename Function>
AUTODIFF_DEVICE_FUNC constexpr auto Reduce(Tuple&& tuple, Function&& f)
{
    using ResultType = std::invoke_result_t<Function, decltype(std::get<0>(tuple))>;
    ResultType res = {};
    ForEach(tuple, [&](auto&& item) constexpr {
        res += f(item);
    });
    return res;
}

} // namespace detail
} // namespace autodiff
