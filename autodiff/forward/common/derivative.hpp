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

// C++ includes
#include <cstddef>

// autodiff includes
#include <autodiff/utils/meta.hpp>

#pragma once

namespace autodiff {
namespace detail {

// template<typename Fun, typename Wrt, typename At, typename Result>
// auto derivative(const Fun& f, const Wrt& wrt, const At& at, Result& u)
// {
//     seed(wrt);
//     u = std::apply(f, at.args);
//     unseed(wrt);
//     return derivative<std::tuple_size_v<Wrt>>(u);
// }

// template<typename Fun, typename Wrt, typename At>
// auto derivative(const Fun& f, const Wrt& wrt, const At& at)
// {
//     using Result = decltype(std::apply(f, at.args));
//     Result u;
//     return derivative(f, wrt, at, u);
// }
// /// Return the directional derivatives of a function along a direction at some point.
// template<typename Fun, typename Along, typename At>
// auto derivatives(const Fun& f, const Along& along, const At& at)
// {
//     seed(std::get<0>(along));
//     auto res = std::apply(f, at.args);
//     unseed(std::get<0>(along));
//     return res;
// }

// template<size_t order=1, typename X, typename T>
// auto seed(X& x, const T& seedval)
// {
//     derivative<order>(x) = static_cast<NumericType<X>>(seedval);
// }


// template<typename Tuple, typename T>
// auto seed(Tuple& nums, const T& seedval)
// {
//     constexpr auto size = std::tuple_size_v<Tuple>;
//     For<size>([&](auto i) constexpr {
//         seednum<i.index + 1>(std::get<i>(nums), seedval);
//     });
// }


// template<typename Tuple>
// auto seed(Tuple& nums)
// {
//     seed(nums, 1.0);
// }

// template<typename Tuple>
// auto unseed(Tuple& nums)
// {
//     seed(nums, 0.0);
// }

template<typename T, typename... Args>
auto seed(const Wrt<Args&...>& wrt, const T& seedval)
{
    // static_assert( ( ... && Args::hasGeneralDerivativeSupport ) );
    constexpr auto size = sizeof...(Args);
    For<size>([&](auto i) constexpr {
        seednum<i.index + 1>(std::get<i>(wrt.args), seedval);
    });
}

// template<typename At, typename... Args>
// auto seed(const Along<Args&...>& along, const At& at)
// {
//     static_assert( ( ... && Args::hasGeneralDerivativeSupport ) );
//     constexpr auto size = sizeof...(Args);
//     For<size>([&](auto i) constexpr {
//         seednum<i.index + 1>(std::get<i>(wrt.args), seedval);
//     });
// }

template<typename... Args>
auto seed(const Wrt<Args&...>& wrt)
{
    seed(wrt, 1.0);
}

template<typename... Args>
auto unseed(const Wrt<Args&...>& wrt)
{
    seed(wrt, 0.0);
}

template<typename Fun, typename Wrt, typename At>
auto derivatives(const Fun& f, const Wrt& wrt, const At& at)
{
    seed(wrt);
    auto u = std::apply(f, at.args);
    unseed(wrt);
    return u;
}

template<size_t order=1, typename Fun, typename Wrt, typename At, typename Result>
auto derivative(const Fun& f, const Wrt& wrt, const At& at, Result& u)
{
    u = derivatives(f, wrt, at);
    return derivative<order>(u);
}

template<size_t order=1, typename Fun, typename Wrt, typename At>
auto derivative(const Fun& f, const Wrt& wrt, const At& at)
{
    auto u = derivatives(f, wrt, at);
    return derivative<order>(u);
}

} // namespace detail

using detail::derivative;
using detail::derivatives;

} // namespace autodiff
