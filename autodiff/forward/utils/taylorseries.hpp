//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright © 2018–2024 Allan Leal
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
#include <autodiff/forward/utils/derivative.hpp>
#include <autodiff/common/meta.hpp>
#include <autodiff/common/vectortraits.hpp>

namespace autodiff {
namespace detail {

/// Represents a Taylor series along a direction for either a scalar or vector function.
/// @see taylorseries
template<size_t N, typename V>
class TaylorSeries
{
public:
    /// The numeric floating point type of the derivatives, which can be a vector of values or just one.
    using T = std::conditional_t<isVector<V>, VectorValueType<V>, V>;

    /// Construct a default TaylorSeries object.
    TaylorSeries() = default;

    /// Construct a TaylorSeries object with given directional derivatives.
    explicit TaylorSeries(const std::array<V, N + 1>& derivatives)
    : _derivatives(derivatives)
    {}

    /// Evaluate the Taylor series object with given directional derivatives.
    auto operator()(const T& t)
    {
        auto res = _derivatives[0];
        auto factor = t;
        For<1, N + 1>([&](auto&& i) constexpr {
            res += factor * _derivatives[i];
            factor *= t / static_cast<T>(i + 1);
        });
        return res;
    }

    /// Return the directional derivatives of this TaylorSeries.
    auto derivatives()
    {
        return _derivatives;
    }

private:
    /// The directional derivatives of the function up to Nth order.
    std::array<V, N + 1> _derivatives;
};

/// Return a TaylorSeries of a scalar or vector function *f* along a direction *v* at *x*.
template<typename Fun, typename...Vecs, typename... Args>
auto taylorseries(const Fun& f, const Along<Vecs...>& along, const At<Args...>& at)
{
    auto data = derivatives(f, along, at);
    constexpr auto N = data.size() - 1;
    using V = typename decltype(data)::value_type;
    return TaylorSeries<N, V>(data);
}

} // namespace detail

using detail::taylorseries;

} // namespace autodiff
