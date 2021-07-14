#pragma once

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

#include <array>
#include <vector>
#include <memory>

#include <autodiff/mixed/expressions/expression.hpp>

namespace autodiff::taperep
{
/// @brief Struct which represent node of expression graph.
/// @details Give access to topology of node. Makes enable to update children etc.
/// Node holds topology(parent indices) and differential properties (expression) in separate members.
/// @tparam N order of node.
template<std::size_t N>
struct node final
{
    static_assert(N >  0 && "bad idea to create node with zero order derivatives variable, use double or other trivial type");
    static_assert(N <= 2 && "we support only 1st and 2nd order vars");

    /// @brief Construct node from parents indices and underlying expression. 
    /// @param parents parent indices (topological info).
    /// @param expr storage for values and derivatives.
    node(std::array<std::size_t, 2> parents, expression_ptr<N> expr) :
        m_parents { parents }, m_expression { expr } { }

    /// @brief access to class which holds differential props.
    /// @return node differential props.
    auto expression() const -> const expression_ptr<N>&
    { 
        return m_expression; 
    }

    /// @brief Partial derivative at position i.
    /// @details For example if we work with first order derivatives
    /// i = 0 represent dnode / dleft, i = 1 represent dnode / dright.
    /// @tparam Order order of derivatives weights.
    /// @param i index.
    /// @return partial derivative of given order.
    template<std::size_t Order>
    auto partial_derivative(std::size_t i) const -> double
    { 
        return m_expression->template partial_derivatives<Order>()[i]; 
    }

    /// @brief Parent index at position i.
    /// @param i index.
    /// @return Parent node index for given i.
    auto parent(std::size_t i) const -> std::size_t
    { 
        return m_parents[i]; 
    }

    /// @brief Count of parents.
    /// @details For now always two.
    /// @return Number of parent nodes.
    auto parents_count() const -> std::size_t
    { 
        return 2; 
    }
private:
    std::array<std::size_t, 2> m_parents; ///< parents nodes indices
    expression_ptr<N> m_expression;       ///< pointer to expression of node
};
}
