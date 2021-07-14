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

#include <autodiff/mixed/expressions/expression.hpp>

namespace autodiff::taperep {
/// @brief Binary expression base class.
/// @tparam N Maximal derivative order of expression.
template<std::size_t N>
struct binary_expression : expression<N>
{
    /// @brief Construct sum expression from left and right expression values.
    /// @param v Value of binary expression.
    /// @param l Value of left expression.
    /// @param r Value of right expression.
    explicit binary_expression(double v, const double& l, const double& r) : 
        expression<N>(v), m_left { l }, m_right { r } { }

    /// @brief Left expression value.
    /// @return Constant reference to left expression value.
    auto left() const -> const double&
    {
        return m_left;
    }

    /// @brief Right expression value.
    /// @return Constant reference to right expression value.
    auto right() const -> const double&
    {
        return m_right;
    }
private:
    const double& m_left;  ///< Value reference to left part of binary node.
    const double& m_right; ///< Value reference to right part of binary node.
};

/// @brief Binary sum expression.
/// @tparam N Maximal derivative order of expression.
template<std::size_t N>
struct sum_expression final : binary_expression<N>
{
    using base = binary_expression<N>;

    /// @brief Construct sum expression from left and right expression values.
    /// @param l Value of left expression.
    /// @param r Value of right expression.
    explicit sum_expression(const double& l, const double& r) : 
        base(l + r, l, r) 
    { 
        this->template partial_derivatives<first_order>() = { 1.0, 1.0 };

        if constexpr (N > first_order)
            this->template partial_derivatives<second_order>() = { };
    }

    /// @brief Update value of sum expression.
    /// @details Values of partial derivatives are always same for sum, 
    /// so we don't need to update them.
    virtual void update()
    {
        this->value() = base::left() + base::right();
    }
};

/// @brief Binary minus expression.
/// @tparam N Maximal derivative order of expression.
template<std::size_t N>
struct minus_expression final : binary_expression<N>
{
    using base = binary_expression<N>;

    /// @brief Construct minus expression from left and right expression values.
    /// @param l Value of left expression.
    /// @param r Value of right expression.
    explicit minus_expression(const double& l, const double& r) :
        base(l - r, l, r)
    {
        this->template partial_derivatives<first_order>() = { 1.0, - 1.0 };

        if constexpr (N > first_order)
            this->template partial_derivatives<second_order>() = { };
    }

    /// @brief Update value of minus expression.
    /// @details Values of partial derivatives are always same for minus, 
    /// so we don't need to update them.
    virtual void update()
    {
        this->value() = base::left() - base::right();
    }
};

/// @brief Binary multiply expression.
/// @tparam N Maximal derivative order of expression.
template<std::size_t N>
struct multiply_expression final : binary_expression<N>
{
    using base = binary_expression<N>;

    /// @brief Construct multiply expression from left and right expression values.
    /// @param l Value of left expression.
    /// @param r Value of right expression.
    explicit multiply_expression(const double& l, const double& r) : 
        base(l * r, l, r)
    { 
        this->template partial_derivatives<first_order>() = { base::right(), base::left() };

        if constexpr (N > first_order)
            this->template partial_derivatives<second_order>() = { };
    }

    /// @brief Update value of multiply expression.
    /// @details Only first order derivatives are need recalculation,
    /// since other are always zero.
    virtual void update()
    {
        this->value() = base::left() * base::right();
        // update derivatives for multiply expression
        this->template partial_derivatives<first_order>() = { base::right(), base::left() };
    }
};

/// @brief Binary divide expression.
/// @tparam N Maximal derivative order of expression.
template<std::size_t N>
struct divide_expression final : binary_expression<N>
{
    using base = binary_expression<N>;

    /// @brief Construct divide expression from left and right expression values.
    /// @param l Value of left expression.
    /// @param r Value of right expression.
    explicit divide_expression(const double& l, const double& r) : 
        base(l / r, l, r)
    { 
        eval_derivatives();
    }

    /// @brief Update value of divide expression.
    /// @details Only first order derivatives are need recalculation,
    /// since other are always zero.
    virtual void update()
    {
        this->value() = base::left() / base::right();

        eval_derivatives();
    }

private:
    /// @brief eval derivatives of divide expression.
    void eval_derivatives()
    {
        const auto aux = 1.0 / base::right();
        this->template partial_derivatives<first_order>() = { aux, - base::left() * aux };

        if constexpr (N > first_order) 
        {
            const auto aux_aux = aux * aux;
            this->template partial_derivatives<second_order>() = { 0.0, - aux_aux, -2 * base::left() * aux_aux};
        }
    }
};
} // namespace autodiff::taperep
