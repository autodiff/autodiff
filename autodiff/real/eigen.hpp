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

// Einge includes
#include <Eigen/Core>

// autodiff includes
#include <autodiff/real/real.hpp>
#include <autodiff/utils/eigen.hpp>
#include <autodiff/utils/meta.hpp>

//------------------------------------------------------------------------------
// SUPPORT FOR EIGEN MATRICES AND VECTORS OF REAL
//------------------------------------------------------------------------------
namespace Eigen {

template<typename T>
struct NumTraits;

template<size_t N, typename T>
struct NumTraits<autodiff::Real<N, T>> : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
    typedef autodiff::Real<N, T> Real;
    typedef autodiff::Real<N, T> NonInteger;
    typedef autodiff::Real<N, T> Nested;
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

template<size_t N, typename T, typename BinOp>
struct ScalarBinaryOpTraits<autodiff::Real<N, T>, T, BinOp>
{
    typedef autodiff::Real<N, T> ReturnType;
};

template<size_t N, typename T, typename BinOp>
struct ScalarBinaryOpTraits<T, autodiff::Real<N, T>, BinOp>
{
    typedef autodiff::Real<N, T> ReturnType;
};

} // namespace Eigen

namespace autodiff {

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
    // Reduce(args, [&](auto&& item) constexpr { return length(item); });
    // const size_t n = Reduce(args, [&](auto&& item) constexpr { return length(item); });

    Eigen::VectorXd g(n);

    size_t offset = 0;

    ForEach(wrt, [&](auto&& item) constexpr
    {
        if constexpr (isReal<decltype(item)>)
        {
            item[1] = 1.0;
            u = std::apply(f, std::forward<Args>(args));
            item[1] = 0.0;
            g[offset] = u[1];
            ++offset;
        }
        else
        {
            const auto len = item.size();
            for(size_t j = 0; j < len; ++j)
            {
                item[j][1] = 1.0;
                u = std::apply(f, std::forward<Args>(args));
                item[j][1] = 0.0;
                g[offset + j] = u[1];
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

// /// Return the Jacobian matrix of a function *f* with respect to some or all variables.
// template<typename Function, typename Wrt, typename Args, typename Result>
// auto jacobian(const Function& f, Wrt&& wrt, Args&& args, Result& F) -> Eigen::MatrixXd
// {
//     const auto n = std::get<0>(wrt).size();

//     if(n == 0) return {};

//     std::get<0>(wrt)[0].grad = 1.0;
//     F = std::apply(f, args);
//     std::get<0>(wrt)[0].grad = 0.0;

//     const auto m = F.size();

//     Eigen::MatrixXd J(m, n);

//     for(auto i = 0; i < m; ++i)
//         J(i, 0) = F[i].grad;

//     for(auto j = 1; j < n; ++j)
//     {
//         std::get<0>(wrt)[j].grad = 1.0;
//         F = std::apply(f, args);
//         std::get<0>(wrt)[j].grad = 0.0;

//         for(auto i = 0; i < m; ++i)
//             J(i, j) = F[i].grad;
//     }

//     return J;
// }

// /// Return the Jacobian matrix of a function *f* with respect to some or all variables.
// template<typename Function, typename Wrt, typename Args>
// auto jacobian(const Function& f, Wrt&& wrt, Args&& args) -> Eigen::MatrixXd
// {
//     using Result = decltype(std::apply(f, args));
//     Result F;
//     return jacobian(f, std::forward<Wrt>(wrt), std::forward<Args>(args), F);
// }

AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(real, real)

} // namespace autodiff

