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

#include <cstddef>
#include <functional>
#include <limits>
#include <algorithm>

namespace autodiff::taperep
{
template<std::size_t N>
struct tape_storage;

/// @brief Struct which represent number able to compute derivatives.
/// @details Var can be created only by tape class, which manage allocation of vars.
/// WARNING: If you use uninitialized var you shouldn’t use operator=(double).
/// @tparam N order of var.
template<std::size_t N>
struct var final
{
    using taped_storage_ptr = tape_storage<N>*;
    using taped_storage_ref = tape_storage<N>&;

    constexpr static std::size_t order = N;

    template<std::size_t M>
    friend struct tape_storage;

    /// @brief Default constructor is default.
    var() = default;

    /// @brief Construct variable from arithmetic types. We need it because Eigen requires it.
    /// @details In Eigen code they have check in if statement, where compiler
    /// looking for Scalar(0). Our constructor is empty.
    template<typename U, std::enable_if_t<std::is_arithmetic_v<U>>...>
    var(U) { assert(false && "should be never called"); }

    /// @brief Assignment operator from arithmetic types.
    /// @param value Right part of assignment expression.
    template<typename U, std::enable_if_t<std::is_arithmetic_v<U>>...>
    var& operator= (U value)
    {
        assert(m_tape && "assignment without tape");
        
        if (m_tape->nodes()[m_index].expression()->value() == value)
            return *this;

        m_tape->nodes()[m_index].expression()->value() = value;

        return *this;
    }

    /// @brief Convert var to double.
    explicit operator double() const
    {
        return value();
    }

    /// @brief Get value for var.
    /// @return Numerical value of variable.
    auto value() const -> const double&
    {  
        return m_tape->nodes()[m_index].expression()->value();
    }

    /// @brief Index of this variable.
    /// @return Index in tape.
    auto index() const -> std::size_t
    {
        return m_index;
    }

    /// @brief Get Stack which manage allocation of vars.
    /// @return Pointer to tape.
    auto tape() const -> taped_storage_ref
    {
        return *m_tape;
    }

private:
    var(std::size_t index, taped_storage_ptr tape) : m_index { index }, m_tape { tape } { }

    std::size_t m_index;      ///< Index in tape
    taped_storage_ptr m_tape; ///< Pointer to tape
};

namespace traits 
{
template<typename T>
struct is_var : std::false_type {};

template<std::size_t N>
struct is_var<::autodiff::taperep::var<N>> : std::true_type {};

template<typename T>
inline constexpr bool is_var_v = is_var<T>::value;

template<typename T>
using enable_if_var = std::enable_if_t<is_var_v<T>>;
} // namespace traits
} // namespace autodiff::taperep
