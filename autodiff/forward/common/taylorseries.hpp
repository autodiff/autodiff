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
#include <array>

// autodiff includes
#include <autodiff/forward/common/derivative.hpp>
#include <autodiff/utils/aliases.hpp>
#include <autodiff/utils/meta.hpp>

namespace autodiff {
namespace detail {

template<size_t N, typename U>
class TaylorSeries
{
public:
    TaylorSeries() = default;

    explicit TaylorSeries(const std::array<N + 1, U>& data)
    : _data(data)
    {}

    template<typename... Args>
    auto operator()(const Args...& args)
    {
        using T = NumericType<U>;
        auto res = _data[0];
        auto factor = One<T>;
        For<1, N + 1>([](auto&& i) constexpr {
            res += factor * _data[i];
            factor /= static_cast<T>(i + 1);
        });
        return res;
    }

    auto derivative(size_t order)
    {
        return _data[order];
    }

private:
    std::array<N + 1, U> _data;
};

/// Return the gradient of scalar function *f* with respect to some or all variables *x*.
template<typename Fun, typename...Vecs, typename... Args>
auto taylorseries(const Fun& f, const Along<Vecs...>& along, const At<Args&...>& at)
{
    static_assert(Wrt::numArgs);
    static_assert(At::numArgs);

    const size_t n = _wrt_total_length(wrt.args);

    if(n == 0) return {};

    using Result = decltype(std::apply(f, at.args));

    seed(at.args, along.args);

    auto u = std::apply(f, at.args);

    unseed(at.args);

    Eigen::VectorXd g(n);

    size_t offset = 0;


    return g;
}

} // namespace detail

using detail::taylorseries;

} // namespace autodiff

