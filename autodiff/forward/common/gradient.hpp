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

/// Return the gradient of scalar function *f* with respect to some or all variables *x*.
template<typename Function, typename Wrt, typename At, typename U>
auto gradient(const Function& f, const Wrt& wrt, const At& at, U& u) -> Eigen::VectorXd
{
    static_assert(Wrt::numArgs);
    static_assert(At::numArgs);

    const size_t n = _wrt_total_length(wrt.args);

    if(n == 0) return {};

    Eigen::VectorXd g(n);

    size_t offset = 0;

    ForEach(wrt.args, [&](auto& item) constexpr
    {
        if constexpr (hasSize<decltype(item)>)
        {
            const size_t len = item.size();
            for(size_t j = 0; j < len; ++j)
            {
                seednum<1>(item[j], 1.0);
                u = std::apply(f, at.args);
                seednum<1>(item[j], 0.0);
                g[offset + j] = derivative<1>(u);
            }
            offset += len;
        }
        else
        {
            seednum<1>(item, 1.0);
            u = std::apply(f, at.args);
            seednum<1>(item, 0.0);
            g[offset] = derivative<1>(u);
            ++offset;
        }
    });

    return g;
}

/// Return the gradient of scalar function *f* with respect to some or all variables *x*.
template<typename Function, typename Wrt, typename At>
auto gradient(const Function& f, const Wrt& wrt, const At& at) -> Eigen::VectorXd
{
    using U = decltype(std::apply(f, at.args));
    U u;
    return gradient(f, wrt, at, u);
}

/// Return the Jacobian matrix of a function *f* with respect to some or all variables.
template<typename Function, typename Wrt, typename At, typename Result>
auto jacobian(const Function& f, const Wrt& wrt, const At& at, Result& F) -> Eigen::MatrixXd
{
    static_assert(Wrt::numArgs);
    static_assert(At::numArgs);

    Eigen::MatrixXd J;

    const size_t n = _wrt_total_length(wrt.args);
    size_t offset = 0;
    size_t m = 0;

    ForEach(wrt.args, [&](auto& item) constexpr
    {
        if constexpr (hasSize<decltype(item)>)
        {
            const size_t len = item.size();
            for(size_t j = 0; j < len; ++j)
            {
                seednum<1>(item[j], 1.0);
                F = std::apply(f, at.args);
                seednum<1>(item[j], 0.0);
                if(m == 0) { m = F.rows(); J.resize(m, n); };
                for(size_t i = 0; i < m; ++i)
                    J(i, offset + j) = derivative<1>(F[i]);
            }
            offset += len;
        }
        else
        {
            seednum<1>(item, 1.0);
            F = std::apply(f, at.args);
            seednum<1>(item, 0.0);
            if(m == 0) { m = F.rows(); J.resize(m, n); };
            for(size_t i = 0; i < m; ++i)
                J(i, offset) = derivative<1>(F[i]);
            ++offset;
        }
    });

    return J;
}

/// Return the Jacobian matrix of a function *f* with respect to some or all variables.
template<typename Function, typename Wrt, typename At>
auto jacobian(const Function& f, const Wrt& wrt, const At& at) -> Eigen::MatrixXd
{
    using Result = decltype(std::apply(f, at.args));
    Result F;
    return jacobian(f, wrt, at, F);
}

} // namespace detail

using detail::gradient;
using detail::jacobian;

} // namespace autodiff

