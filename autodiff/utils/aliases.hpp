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

// C++ includes
#include <tuple>

// autodiff includes
#include <autodiff/utils/meta.hpp>

namespace autodiff {
namespace detail {

template<typename... Args>
struct At
{
    std::tuple<Args&...> args;
    constexpr static auto numArgs = sizeof...(Args);
};

template<typename... Args>
struct Wrt
{
    std::tuple<Args...> args;
    constexpr static auto numArgs = sizeof...(Args);
};

template<typename... Args>
struct Along
{
    std::tuple<Args&...> args;
    constexpr static auto numArgs = sizeof...(Args);
};

/// The keyword used to denote the variables *with respect to* the derivative is calculated.
template<typename... Args>
auto wrt(Args&&... args)
{
    return Wrt<Args&&...>{ std::forward_as_tuple(std::forward<Args>(args)...) };
}

/// The keyword used to denote the derivative order *N* and the variable *with respect to* the derivative is calculated.
template<std::size_t N, typename Arg>
auto wrt(Arg&& arg) // TODO: This permits rvalues in wrt (temporaries), which will not have an effect in any variable in the `at` list. && should be replaced by & only
{
    static_assert(N > 0);
    auto head = std::forward_as_tuple(std::forward<Arg>(arg));
    if constexpr (N == 1)
        return head;
    else return std::tuple_cat(head, wrt<N - 1>(arg));
}

/// The keyword used to denote the variables *at* which the derivatives are calculated.
template<typename... Args>
auto at(Args&... args)
{
    return At<Args&...>{ std::forward_as_tuple(args...) };
}

/// The keyword used to denote the direction vector *along* which the derivatives are calculated.
template<typename Arg>
auto along(Arg& arg)
{
    return Along<Arg&>{ std::forward_as_tuple(arg) };
}

/// The keyword used to denote the direction vector *along* which the derivatives are calculated.
template<typename... Args>
auto along(Args&&... args)
{
    return Along<Args...>{ std::forward_as_tuple(args...) };
}

} // namespace detail

using detail::along;
using detail::at;
using detail::wrt;

} // namespace autodiff