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
#include <autodiff/utils/aliases.hpp>
#include <autodiff/utils/meta.hpp>

#pragma once

namespace autodiff {
namespace detail {

template<typename... Vars, typename T>
auto seed(const Wrt<Vars&...>& wrt, T&& seedval)
{
    constexpr auto size = sizeof...(Vars);
    For<size>([&](auto i) constexpr {
        seed<i.index + 1>(std::get<i>(wrt.args), seedval);
    });
}

template<typename... Vars>
auto seed(const Wrt<Vars&...>& wrt)
{
    seed(wrt, 1.0);
}

template<typename... Vars>
auto unseed(const Wrt<Vars&...>& wrt)
{
    seed(wrt, 0.0);
}

template<typename... Args, typename... Vecs>
auto seed(const At<Args...>& at, const Along<Vecs...>& along)
{
    static_assert(sizeof...(Args) == sizeof...(Vecs));

    ForEach(at.args, along.args, [&](auto& arg, auto&& dir) constexpr {
        if constexpr (hasSize<decltype(arg)>) {
            static_assert(hasSize<decltype(dir)>);
            assert(arg.size() == dir.size());
            const size_t len = dir.size();
            for(size_t i = 0; i < len; ++i)
                seed<1>(arg[i], dir[i]);
        }
        else seed<1>(arg, dir);
    });
}

template<typename... Args>
auto unseed(const At<Args...>& at)
{
    ForEach(at.args, [&](auto& arg) constexpr {
        if constexpr (hasSize<decltype(arg)>) {
            const size_t len = arg.size();
            for(size_t i = 0; i < len; ++i)
                seed<1>(arg[i], 0.0);
        }
        else seed<1>(arg, 0.0);
    });
}

/// Unpack the derivatives from the result of an @ref eval call into an array.
template<typename Result>
auto derivatives(const Result& result)
{
    if constexpr (hasSize<Result>) // check if the argument is a vector container of dual/real numbers
    {
        const size_t len = result.size(); // the length of the vector containing dual/real numbers
        using NumType = decltype(result[0]); // get the type of the dual/real number
        using T = NumericType<NumType>; // get the numeric/floating point type of the dual/real number
        using Vec = VectorReplaceValueType<Result, T>; // get the type of the vector containing numeric values instead of dual/real numbers (e.g., vector<real> becomes vector<double>, VectorXdual becomes VectorXd, etc.)
        constexpr auto N = 1 + Order<NumType>; // 1 + the order of the dual/real number
        std::array<Vec, N> values; // create an array to store the derivatives stored inside the dual/real number
        For<N>([&](auto i) constexpr {
            values[i].resize(len);
            for(size_t j = 0; j < len; ++j)
                values[i][j] = derivative<i>(result[j]); // get the ith derivative of the jth dual/real number
        });
        return values;
    }
    else // result is then just a dual/real number
    {
        using T = NumericType<Result>; // get the numeric/floating point type of the dual/real result number
        constexpr auto N = 1 + Order<Result>; // 1 + the order of the dual/real result number
        std::array<T, N> values; // create an array to store the derivatives stored inside the dual/real number
        For<N>([&](auto i) constexpr {
            values[i] = derivative<i>(result);
        });
        return values;
    }
}

template<typename Fun, typename... Vars, typename... Args>
auto derivatives(const Fun& f, const Wrt<Vars&...>& wrt, const At<Args&...>& at)
{
    // Seed, evaluate, unseed
    seed(wrt);
    auto u = std::apply(f, at.args);
    unseed(wrt);

    // Store the derivatives in an array
    using T = NumericType<decltype(u)>;
    constexpr auto N = 1 + sizeof...(Vars);
    std::array<T, N> values;
    For<N>([&](auto i) constexpr {
        values[i] = derivative<i>(u);
    });

    return values;
}

template<size_t order=1, typename Fun, typename... Vars, typename... Args, typename Result>
auto derivative(const Fun& f, const Wrt<Vars&...>& wrt, const At<Args&...>& at, Result& u)
{
    u = derivatives(f, wrt, at);
    return derivative<order>(u);
}

template<size_t order=1, typename Fun, typename... Vars, typename... Args>
auto derivative(const Fun& f, const Wrt<Vars&...>& wrt, const At<Args&...>& at)
{
    auto u = derivatives(f, wrt, at);
    return derivative<order>(u);
}

template<typename Fun, typename... Vecs, typename... Args>
auto derivatives(const Fun& f, const Along<Vecs...>& along, const At<Args&...>& at)
{
    // Seed, evaluate, unseed
    seed(at, along);
    auto u = std::apply(f, at.args);
    unseed(at);

    // Store the derivatives in an array
    using T = NumericType<decltype(u)>;
    constexpr auto N = 1 + Order<decltype(u)>;
    std::array<T, N> values;
    For<N>([&](auto i) constexpr {
        values[i] = derivative<i>(u);
    });

    return values;
}

} // namespace detail

using detail::derivative;
using detail::derivatives;

} // namespace autodiff
