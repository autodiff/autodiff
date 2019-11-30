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

#include <cmath>

#include <autodiff/mixed/expressions/expression.hpp>

namespace autodiff::taperep {
/// @brief Unary expression base class.
/// @tparam N Maximal derivative order of expression.
template<std::size_t N>
struct unary_expression : expression<N>
{
    /// @brief Construct unary expression from value and argument.
    /// @param v Value of unary expression.
    /// @param x Argument constant reference of unary expression.
    explicit unary_expression(double v, const double& x) : 
        expression<N>(v), m_argument { x } { }

    /// @brief Argument of unary expression.
    /// @return Constant reference to argument.
    auto argument() const -> const double&
    {
        return m_argument;
    }
private:
    const double& m_argument;  ///< Value reference to argument of unary node.
};

/// @brief Unary sinus expression class.
/// @tparam N Maximal derivative order of expression.
template<std::size_t N>
struct sin_expression final : unary_expression<N>
{
    using base = unary_expression<N>;

    /// @brief Construct sin expression from argument.
    /// @param x Argument constant reference of unary expression.
    explicit sin_expression(const double& x) : 
        base(std::sin(x), x) 
    {
        eval_derivatives();
    }

    /// @brief Update value and derivatives of sin expression.
    /// @details All derivatives should be updated for sinus expression.
    virtual void update()
    {
        this->value() = std::sin(base::argument());

        eval_derivatives();
    }

private:
    /// @brief eval derivatives of divide expression.
    void eval_derivatives()
    {
        this->template partial_derivatives<first_order>() = { std::cos(base::argument()), 0.0 };

        if constexpr (N > first_order) 
        {
            this->template partial_derivatives<second_order>() = { - std::sin(base::argument()), 0.0, 0.0};
        }
    }
};
} // namespace autodiff::taperep
