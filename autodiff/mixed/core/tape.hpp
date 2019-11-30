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

#include <autodiff/mixed/core/variable.hpp>
#include <autodiff/mixed/expressions/parameter.hpp>

namespace autodiff::taperep 
{
/// @brief Class which represent tape based storage for expression tree.
/// @tparam N order of tape.
template<std::size_t N>
struct tape_storage final
{
    /// @brief Create tape with preallocated space for nodes.
    /// @param length number of nodes
    explicit tape_storage(std::size_t length)
    {
        m_nodes.reserve(length);
    }

    /// @brief Create tape with preallocated space for nodes = 256.
    tape_storage() : tape_storage(256) { }

    /// @brief Create var from value in current tape.
    /// @param value Numerical value of variable.
    /// @return New variable.
    auto variable(double value) -> var<N>
    {
        return var<N>(add_node(node<N>({ nodes_count() }, std::make_shared<parameter_expression<N>>(value))), this);
    }

    /// @brief Create var from value in current tape.
    /// @param l_index Index of left side of expression.
    /// @param expression Underlying expression.
    /// @return New variable.
    auto variable(std::size_t l_index, expression_ptr<N> expression) -> var<N>
    {
        return var<N>(add_node(node<N>({ l_index }, expression)), this);
    }

    /// @brief Create var from value in current tape.
    /// @param l_index Index of left side of expression.
    /// @param r_index Index of right side of expression.
    /// @param expression Underlying expression.
    /// @return New variable.
    auto variable(std::size_t l_index, std::size_t r_index, expression_ptr<N> expression) -> var<N>
    {
        return var<N>(add_node(node<N>({ l_index, r_index }, expression)), this);
    }

    /// @brief Nodes storage.
    /// @return Mutable reference to nodes
    auto nodes() -> std::vector<node<N>>&
    {
        return m_nodes;
    }
    
    /// @brief Update values and derivatives in tree.
    auto update() -> void 
    {
        for(auto& n : m_nodes)
            n.expression()->update();
    }

    /// @brief Clear noes storage.
    auto clear() -> void
    {
        m_nodes.clear();
    }
private:
    /// @brief Add node to nodes.
    /// @param n Node.
    /// @return Count of nodes.
    auto add_node(node<N> n) -> std::size_t
    {
        m_nodes.push_back(n);
        return nodes_count() - 1;
    }

    /// @brief Count of nodes in tape.
    /// @return Count of nodes.
    auto nodes_count() const -> std::size_t
    {
        return m_nodes.size();
    }

    std::vector<node<N>> m_nodes;   ///< continues data structure to store nodes
};
}
