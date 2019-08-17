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

//------------------------------------------------------------------------------
// SUPPORT FOR EIGEN MATRICES AND VECTORS OF DUAL
//------------------------------------------------------------------------------
namespace Eigen {

template<typename T>
struct NumTraits;

template<> struct NumTraits<autodiff::dual> : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
    typedef autodiff::dual Real;
    typedef autodiff::dual NonInteger;
    typedef autodiff::dual Nested;
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
struct ScalarBinaryOpTraits<autodiff::forward::Dual<T, G>, T, BinOp>
{
    typedef autodiff::dual ReturnType;
};

template<typename T, typename G, typename BinOp>
struct ScalarBinaryOpTraits<T, autodiff::forward::Dual<T, G>, BinOp>
{
    typedef autodiff::dual ReturnType;
};

#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)   \
typedef Matrix<Type, Size, Size, 0, Size, Size> Matrix##SizeSuffix##TypeSuffix;  \
typedef Matrix<Type, Size, 1, 0, Size, 1>       Vector##SizeSuffix##TypeSuffix;  \
typedef Matrix<Type, 1, Size, 1, 1, Size>       RowVector##SizeSuffix##TypeSuffix;

#define EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)         \
typedef Matrix<Type, Size, -1, 0, Size, -1> Matrix##Size##X##TypeSuffix;  \
typedef Matrix<Type, -1, Size, 0, -1, Size> Matrix##X##Size##TypeSuffix;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, -1, X) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

EIGEN_MAKE_TYPEDEFS_ALL_SIZES(autodiff::dual, dual)

#undef EIGEN_MAKE_TYPEDEFS_ALL_SIZES
#undef EIGEN_MAKE_TYPEDEFS
#undef EIGEN_MAKE_FIXED_TYPEDEFS

} // namespace Eigen

namespace autodiff::forward {

namespace detail {
/// Compile time foreach for tuples
template <typename Tuple, typename Callable>
void forEach(Tuple&& tuple, Callable&& callable)
{
    std::apply(
        [&callable](auto&&... args) { (callable(std::forward<decltype(args)>(args)), ...); },
        std::forward<Tuple>(tuple)
    );
}

/// Wrap T to compatible array interface
template<typename T>
struct EigenVectorAdaptor {
    /// implicit construct from value
    constexpr EigenVectorAdaptor(T val) : val(val) { }
    /// operator [] to add array like access
    T operator[](Eigen::Index) const {
        return val;
    }
    /// size for compatibility
    Eigen::Index size() const {
        return 1;
    }
private:
    T val;
};
}

/// Return the gradient vector of scalar function *f* with respect to some or all variables *x*.
template<typename Function, typename Wrt, typename Args, typename Result>
auto gradient(const Function& f, Wrt&& wrt, Args&& args, Result& u) -> Eigen::VectorXd
{
    const std::size_t n = std::get<0>(wrt).size();

    Eigen::VectorXd g(n);

    for(std::size_t j = 0; j < n; ++j)
    {
        std::get<0>(wrt)[j].grad = 1.0;
        u = std::apply(f, args);
        std::get<0>(wrt)[j].grad = 0.0;
        g[j] = u.grad;
    }

    return g;
}

/// Return the gradient vector of scalar function *f* with respect to some or all variables *x*.
template<typename Function, typename Wrt, typename Args>
auto gradient(const Function& f, Wrt&& wrt, Args&& args) -> Eigen::VectorXd
{
    using Result = decltype(std::apply(f, args));
    Result u;
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

} // namespace autodiff::forward


