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

#include <array>
#include <cmath>
#include <utility>

#include <autodiff/mixed/core/meta.hpp>

namespace autodiff::taperep {

constexpr auto first_order = 1;  ///< Constant to get first order partial derivatives.
constexpr auto second_order = 2; ///< Constant to get second order partial derivatives.

/// @brief Base class for expressions.
/// @details We use expression class to re-evaluate expression tree
/// after changing values of some nodes. This class is needed to
/// hold information about type of node expression. We use dynamic
/// polymorphism to satisfy above requirements. Please note that
/// we don't use virtual destructor since we use make_shared function
/// to create node. All expression classes should be inherit from this.
/// TODO: consider using unique pointer to store expression. Then destructor
/// should be virtual.
/// @tparam N Maximal derivative order of expression.
template<std::size_t N>
struct expression {
    /// @brief Construct base expression from value.
    /// @param v Value of this expression.
    explicit expression(double v) : m_value { v } { }

    /// @brief Get partial of given order derivatives for expression.
    /// @tparam Order Maximal derivative order of expression.
    /// @return reference to derivative storage.
    template<std::size_t Order>
    auto& partial_derivatives()
    {
        return std::get<Order - 1>(m_partial_derivatives);
    }

    /// @brief Get reference to numerical value of expression.
    /// @tparam Order Maximal derivative order of expression.
    /// @return reference to derivative storage.
    auto value() -> double&
    {
        return m_value;
    }

    /// @brief Update value and derivatives for expression.
    /// @details This function called from tape to update value
    /// and derivatives. Note: that value of parent expressions
    /// should be updated before updating child(this).
    virtual void update() = 0;
private:
    using weights_container = autodiff::meta::tuple_of_derivatives_weights_t<N>;

    weights_container m_partial_derivatives; ///< partial derivatives storage
    double m_value;                          ///< value of expression
};

template<std::size_t N>
using expression_ptr = std::shared_ptr<expression<N>>; ///< alias for expression pointer
} // namespace autodiff::taperep
