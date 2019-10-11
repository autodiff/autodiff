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
#include <autodiff/forward/dual.hpp>
#include <autodiff/utils/eigen.hpp>
#include <autodiff/utils/meta.hpp>

//------------------------------------------------------------------------------
// SUPPORT FOR EIGEN MATRICES AND VECTORS OF DUAL
//------------------------------------------------------------------------------
namespace Eigen {

template<typename T>
struct NumTraits;

template<typename T, typename G>
struct NumTraits<autodiff::Dual<T, G>> : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
    typedef autodiff::Dual<T, G> Real;
    typedef autodiff::Dual<T, G> NonInteger;
    typedef autodiff::Dual<T, G> Nested;
    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};

template<typename T, typename G, typename BinOp>
struct ScalarBinaryOpTraits<autodiff::Dual<T, G>, T, BinOp>
{
    typedef autodiff::Dual<T, G> ReturnType;
};

template<typename T, typename G, typename BinOp>
struct ScalarBinaryOpTraits<T, autodiff::Dual<T, G>, BinOp>
{
    typedef autodiff::Dual<T, G> ReturnType;
};

} // namespace Eigen

namespace autodiff {
namespace detail {

// Create type trait struct `has_member_size`.
CREATE_MEMBER_CHECK(size);

template<typename T>
auto _wrt_item_length(const T& item) -> size_t
{
    if constexpr (has_member_size<T>::value)
        return item.size();
    else return 1;
}

template<typename... Args>
auto _wrt_total_length(const std::tuple<Args...>& args)
{
    return Reduce(args, [&](auto&& item) constexpr {
        return _wrt_item_length(item);
    });
}

/// Return the gradient of scalar function *f* with respect to some or all variables *x*.
template<typename Function, typename Wrt, typename Args, typename U>
auto gradient(const Function& f, Wrt&& wrt, Args&& args, U& u) -> Eigen::VectorXd
{
    const size_t n = _wrt_total_length(wrt);

    if(n == 0) return {};

    Eigen::VectorXd g(n);

    size_t offset = 0;

    ForEach(wrt, [&](auto&& item) constexpr
    {
        if constexpr (isDual<decltype(item)>)
        {
            seed(item);
            u = std::apply(f, std::forward<Args>(args));
            unseed(item);
            g[offset] = derivative<1>(u);
            ++offset;
        }
        else
        {
            const auto len = item.size();
            for(size_t j = 0; j < len; ++j)
            {
                seed(item[j]);
                u = std::apply(f, std::forward<Args>(args));
                unseed(item[j]);
                g[offset + j] = derivative<1>(u);
            }
            offset += len;
        }
    });

    return g;
}

/// Return the gradient of scalar function *f* with respect to some or all variables *x*.
template<typename Function, typename Wrt, typename Args>
auto gradient(const Function& f, Wrt&& wrt, Args&& args)
{
    using U = decltype(std::apply(f, args));
    U u;
    return gradient(f, std::forward<Wrt>(wrt), std::forward<Args>(args), u);
}

/// Return the Jacobian matrix of a function *f* with respect to some or all variables.
template<typename Function, typename Wrt, typename Args, typename Result>
auto jacobian(const Function& f, Wrt&& wrt, Args&& args, Result& F) -> Eigen::MatrixXd
{
    const auto n = std::get<0>(wrt).size();

    if(n == 0) return {};

    std::get<0>(wrt)[0].grad = 1.0;
    F = std::apply(f, args);
    std::get<0>(wrt)[0].grad = 0.0;

    const auto m = F.size();

    Eigen::MatrixXd J(m, n);

    for(auto i = 0; i < m; ++i)
        J(i, 0) = F[i].grad;

    for(auto j = 1; j < n; ++j)
    {
        std::get<0>(wrt)[j].grad = 1.0;
        F = std::apply(f, args);
        std::get<0>(wrt)[j].grad = 0.0;

        for(auto i = 0; i < m; ++i)
            J(i, j) = F[i].grad;
    }

    return J;
}

/// Return the Jacobian matrix of a function *f* with respect to some or all variables.
template<typename Function, typename Wrt, typename Args>
auto jacobian(const Function& f, Wrt&& wrt, Args&& args) -> Eigen::MatrixXd
{
    using Result = decltype(std::apply(f, args));
    Result F;
    return jacobian(f, std::forward<Wrt>(wrt), std::forward<Args>(args), F);
}

} // namespace detail

AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual, dual)

using detail::gradient;
using detail::jacobian;

} // namespace autodiff


