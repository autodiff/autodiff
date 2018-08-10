//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018 Allan Leal
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

//------------------------------------------------------------------------------
// FORWARD DECLARATIONS
//------------------------------------------------------------------------------
namespace autodiff {

struct var;

namespace forward {

template<typename T>
struct Dual;

} // namespace forward

using dual = forward::Dual<double>;

using Derivatives = std::function<double(const var&)>;
using DerivativesX = std::function<var(const var&)>;

Derivatives derivatives(const var&);
DerivativesX derivativesx(const var&);

} // namespace autodiff

//------------------------------------------------------------------------------
// SUPPORT FOR EIGEN MATRICES AND VECTORS OF VAR AND DUAL
//------------------------------------------------------------------------------
namespace Eigen {

template<typename T>
struct NumTraits;

template<> struct NumTraits<autodiff::var> : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
    typedef autodiff::var Real;
    typedef autodiff::var NonInteger;
    typedef autodiff::var Nested;
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

template<typename T, typename BinOp>
struct ScalarBinaryOpTraits<autodiff::forward::Dual<T>, T, BinOp>
{
    typedef autodiff::dual ReturnType;
};

template<typename T, typename BinOp>
struct ScalarBinaryOpTraits<T, autodiff::forward::Dual<T>, BinOp>
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

EIGEN_MAKE_TYPEDEFS_ALL_SIZES(autodiff::var, var)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(autodiff::dual, dual)

#undef EIGEN_MAKE_TYPEDEFS_ALL_SIZES
#undef EIGEN_MAKE_TYPEDEFS
#undef EIGEN_MAKE_FIXED_TYPEDEFS

} // namespace Eigen

namespace autodiff {

/// Return the gradient vector of variable y with respect to variables x.
template<typename vars>
Eigen::RowVectorXd gradient(const var& y, const vars& x)
{
    const auto n = x.size();
    Eigen::RowVectorXd dydx(n);
    Derivatives dyd = derivatives(y);
    for(auto i = 0; i < n; ++i)
        dydx[i] = dyd(x[i]);
    return dydx;
}

/// Return the Hessian matrix of variable y with respect to variables x.
template<typename vars>
Eigen::MatrixXd hessian(const var& y, const vars& x)
{
    const auto n = x.size();
    Eigen::MatrixXd mat(n, n);
    DerivativesX dyd = derivativesx(y);
    for(auto i = 0; i < n; ++i)
    {
        Derivatives d2yd = derivatives(dyd(x[i]));
        for(auto j = i; j < n; ++j) {
            mat(i, j) = mat(j, i) = d2yd(x(j));
        }
    }
    return mat;
}

/// Return the Jacobian matrix of variables y with respect to variables x.
template<typename Function, typename duals>
Eigen::MatrixXd jacobian(const Function& f, const duals& y, duals& x)
{
    const auto m = y.size();
    const auto n = x.size();
    Eigen::VectorXdual tmp;

    Eigen::MatrixXd mat(m, n);
    for(auto j = 0; j < n; ++j)
    {
        x[j].grad = 1.0;
        auto tmp = f(x);
        x[j].grad = 0.0;
        for(auto i = 0; i < n; ++i)
            mat(i, j) = tmp[i].grad;
    }

    return mat;
}

} // namespace autodiff


