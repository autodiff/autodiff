//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2019 Serhii Malyshev
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

#include <vector>

#include <autodiff/mixed/core/node.hpp>
#include <autodiff/mixed/core/variable.hpp>

namespace autodiff::taperep
{
namespace detail
{
/// @brief Compute all first derivatives derivatives using reverse loop.
/// @param y target variable which derivatives we want to compute.
/// @return all derivatives of target variable in <code>std::vector</code>
template<typename Var>
auto all_first_derivatives(const Var& x) -> std::vector<double>
{
    const auto& nodes = x.tape().nodes();
    const auto length = nodes.size();
    
    std::vector<double> derivatives(length, 0.0);

    derivatives[x.index()] = 1.0;

    for (auto i = length; i > 0; i--)
    {
        const auto& node = nodes[i - 1];

        for (std::size_t j = 0; j < node.parents_count(); j++)
            derivatives[node.parent(j)] += node. template partial_derivative<first_order>(j) * derivatives[i - 1];
    }

    return derivatives;
}
}

/// @brief Compute all first derivatives derivatives using reverse loop.
/// @param y Target variable which derivatives we want to compute.
/// @return Function due to we can get derivatives by operator <code>()</code>.
template<typename Var>
auto first_derivatives(const Var& y)
{
    static_assert(traits::is_var_v<Var>, "Please, use 'first_derivatives' function only for var");
    
    return [derivatives = detail::all_first_derivatives(y)](auto x) -> double
    {
        static_assert(traits::is_var_v<decltype(x)>, 
            "Looks like you want to get derivative for non var type, it is prevented");

        return derivatives[x.index()];
    };
}
} // namespace autodiff::taperep
