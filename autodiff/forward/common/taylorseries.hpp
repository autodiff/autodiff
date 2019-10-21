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
#include <autodiff/utils/meta.hpp>

namespace autodiff {
namespace detail {

template<typename Item>
auto _wrt_item_length(const Item& item) -> size_t
{
    if constexpr (hasSize<Item>)
        return item.size();
    else return 1;
}

template<typename... Items>
auto _wrt_total_length(const std::tuple<Items...>& items)
{
    return Reduce(items, [&](auto&& item) constexpr {
        return _wrt_item_length(item);
    });
}

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

template<typename At, typename Along>
auto seed(const At& at, const Along& along)
{
    static_assert(At::numArgs == Along::numArgs);
    constexpr auto M = At::numArgs;

    For<M>([&](auto&& i) constexpr {
        std::get<i.index>(at.args)
    });

    ForEach(wrt.args, [&](auto& item) constexpr
    {
        if constexpr (hasSize<decltype(item)>)
        {
            const size_t len = item.size();
            for(size_t j = 0; j < len; ++j)
            {
                seed(item[j]);
                u = std::apply(f, at.args);
                unseed(item[j]);
                g[offset + j] = derivative<1>(u);
            }
            offset += len;
        }
        else
        {
            seed(item);
            u = std::apply(f, at.args);
            unseed(item);
            g[offset] = derivative<1>(u);
            ++offset;
        }
    });
}

/// Return the gradient of scalar function *f* with respect to some or all variables *x*.
template<typename Function, typename Along, typename At>
auto taylorseries(const Function& f, const Along& along, const At& at)
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

