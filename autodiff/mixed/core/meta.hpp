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
#include <tuple>

namespace autodiff::meta
{
namespace detail
{
/// @brief Meta function to generate tuple which contains derivatives array.
/// @return Tuple which contain array of derivatives weights.
constexpr auto generate_tuple_of_derivatives_weights(std::index_sequence<0>)
{
    return std::tuple(std::array<double, 2>{ });
}

/// @brief Meta function to generate tuple which contains derivatives arrays.
/// @return Tuple which contain arrays of derivatives weights.
template<std::size_t I, std::size_t... N>
constexpr auto generate_tuple_of_derivatives_weights(std::index_sequence<I, N...>)
{
    constexpr auto cnt = sizeof...(N);
    return std::tuple_cat(generate_tuple_of_derivatives_weights(std::make_index_sequence<cnt>{}), 
        std::tuple(std::array<double, cnt + 2>{}));
}

/// @brief Meta function to generate type of weights container
/// @return Type of container.
template<std::size_t N>
struct tuple_of_derivatives_weights
{
    using type = decltype(generate_tuple_of_derivatives_weights(std::make_index_sequence<N>{}));
};
}

/// @brief Meta function to generate type of weights container
/// @return Type of container
template<std::size_t N>
using tuple_of_derivatives_weights_t = typename detail::tuple_of_derivatives_weights<N>::type;
}
