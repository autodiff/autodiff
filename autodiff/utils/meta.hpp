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
#include <cstddef>
#include <type_traits>

namespace autodiff {

template<size_t i>
struct Index
{
    constexpr static size_t index = i;
    constexpr operator const size_t() const { return index; }
    constexpr operator size_t() { return index; }
};

namespace aux {

template<size_t i, size_t ibegin, size_t iend, typename Function>
constexpr auto For(Function&& f)
{
    if constexpr (i < iend) {
        f(Index<i>{});
        For<i + 1, ibegin, iend>(std::forward<Function>(f));
    }
}

template<size_t i, size_t ibegin, size_t iend, typename Function>
constexpr auto ReverseFor(Function&& f)
{
    if constexpr (i < iend)
    {
        ReverseFor<i + 1, ibegin, iend>(std::forward<Function>(f));
        f(Index<i>{});
    }
}

} // namespace aux

template<size_t ibegin, size_t iend, typename Function>
constexpr auto For(Function&& f)
{
    aux::For<ibegin, ibegin, iend>(std::forward<Function>(f));
}

template<size_t iend, typename Function>
constexpr auto For(Function&& f)
{
    For<0, iend>(std::forward<Function>(f));
}

template<size_t ibegin, size_t iend, typename Function>
constexpr auto ReverseFor(Function&& f)
{
    aux::ReverseFor<ibegin, ibegin, iend>(std::forward<Function>(f));
}

template<size_t iend, typename Function>
constexpr auto ReverseFor(Function&& f)
{
    ReverseFor<0, iend>(std::forward<Function>(f));
}

template<size_t ibegin, size_t iend, typename Function>
constexpr auto Sum(Function&& f)
{
    using ResultType = std::invoke_result_t<Function, Index<ibegin>>;
    ResultType res = {};
    For<ibegin, iend>([&](auto i) constexpr {
        res += f(Index<i>{});
    });
    return res;
}

template<typename Tuple, typename Function>
constexpr auto ForEach(Tuple&& tuple, Function&& f)
{
    auto g = [&](auto&&... args) constexpr {
        ( f(std::forward<decltype(args)>(args)), ...);
    };
    std::apply(g, std::forward<Tuple>(tuple));
}

template<typename Tuple, typename Function>
constexpr auto Reduce(Tuple&& tuple, Function&& f)
{
    using ResultType = std::invoke_result_t<Function, decltype(std::get<0>(tuple))>;
    ResultType res = {};
    ForEach(tuple, [&](auto&& item) constexpr {
        res += f(item);
    });
    return res;
}

} // namespace autodiff