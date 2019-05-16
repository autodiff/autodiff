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

// C++ includes
#include <cmath>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace autodiff {

namespace impl {

template<size_t I, size_t M, typename Function>
auto foreach(Function&& f)
{
    if constexpr (I < M)
    {
        std::forward<Function>(f)(I);
        foreach<I + 1, M>(std::forward<Function>(f));
    }
}

} // namespace impl

template<size_t M, typename Function>
auto foreach(Function&& f)
{
    impl::foreach<0, M>(std::forward<Function>(f));
}

template<size_t M, typename T>
class numarray
{
public:
    constexpr numarray() = delete;
    constexpr numarray(const numarray&) = delete;

    template<typename U>
    constexpr numarray(const numarray<M, U>&) = delete;

    constexpr explicit numarray(T* data)
    : m_data(data)
    {}

    constexpr auto operator[](size_t i) -> T&
    {
        return m_data[i];
    }

    constexpr auto operator[](size_t i) const -> const T&
    {
        return m_data[i];
    }

    constexpr auto operator=(const T& scalar) -> numarray&
    {
        foreach<M>([&](auto i) constexpr { m_data[i] = scalar; });
        return *this;
    }

    constexpr auto operator+=(const T& scalar) -> numarray&
    {
        foreach<M>([&](auto i) constexpr { m_data[i] += scalar; });
        return *this;
    }

    constexpr auto operator-=(const T& scalar) -> numarray&
    {
        foreach<M>([&](auto i) constexpr { m_data[i] -= scalar; });
        return *this;
    }

    constexpr auto operator*=(const T& scalar) -> numarray&
    {
        foreach<M>([&](auto i) constexpr { m_data[i] *= scalar; });
        return *this;
    }

    constexpr auto operator/=(const T& scalar) -> numarray&
    {
        foreach<M>([&](auto i) constexpr { m_data[i] /= scalar; });
        return *this;
    }

    // constexpr auto operator=(const numarray& other) -> numarray&
    // {
    //     foreach<M>([&](auto i) constexpr { m_data[i] = other[i]; });
    //     return *this;
    // }

    // constexpr auto operator+=(const numarray& other) -> numarray&
    // {
    //     foreach<M>([&](auto i) constexpr { m_data[i] += other[i]; });
    //     return *this;
    // }

    // constexpr auto operator-=(const numarray& other) -> numarray&
    // {
    //     foreach<M>([&](auto i) constexpr { m_data[i] -= other[i]; });
    //     return *this;
    // }

    // constexpr auto operator*=(const numarray& other) -> numarray&
    // {
    //     foreach<M>([&](auto i) constexpr { m_data[i] *= other[i]; });
    //     return *this;
    // }

    // constexpr auto operator/=(const numarray& other) -> numarray&
    // {
    //     foreach<M>([&](auto i) constexpr { m_data[i] /= other[i]; });
    //     return *this;
    // }

    template<typename U>
    constexpr auto operator=(const numarray<M, U>& other) -> numarray&
    {
        // foreach<M>([&](auto i) constexpr { m_data[i] = other[i]; });
        foreach<M>([&](auto i) constexpr { m_data[i] = 12345; });
        return *this;
    }

    template<typename U>
    constexpr auto operator+=(const numarray<M, U>& other) -> numarray&
    {
        foreach<M>([&](auto i) constexpr { m_data[i] += other[i]; });
        return *this;
    }

    template<typename U>
    constexpr auto operator-=(const numarray<M, U>& other) -> numarray&
    {
        foreach<M>([&](auto i) constexpr { m_data[i] -= other[i]; });
        return *this;
    }

    template<typename U>
    constexpr auto operator*=(const numarray<M, U>& other) -> numarray&
    {
        foreach<M>([&](auto i) constexpr { m_data[i] *= other[i]; });
        return *this;
    }

    template<typename U>
    constexpr auto operator/=(const numarray<M, U>& other) -> numarray&
    {
        foreach<M>([&](auto i) constexpr { m_data[i] /= other[i]; });
        return *this;
    }

private:
    T* m_data;
};

// template<size_t M, typename T>
// constexpr auto numwrap(std::array<T, M>& a) -> numarray<M, T>
// {
//     return { a.data() };
// }

// template<size_t M, typename T>
// constexpr auto numwrap(const std::array<T, M>& a) -> numarray<M, const T>
// {
//     return { a.data() };
// }

namespace forward2 {

//=====================================================================================================================
//
// STANDARD TEMPLATE LIBRARY MATH FUNCTIONS
//
//=====================================================================================================================

using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::cos;
using std::exp;
using std::log10;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;
using std::tan;

//=====================================================================================================================
//
// OPERATOR TYPES
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// ARITHMETIC OPERATORS
//-----------------------------------------------------------------------------
struct AddOp    {};  // ADDITION OPERATOR
struct SubOp    {};  // SUBTRACTION OPERATOR
struct MulOp    {};  // MULTIPLICATION OPERATOR
struct DivOp    {};  // DIVISION OPERATOR

//-----------------------------------------------------------------------------
// MATHEMATICAL OPERATORS
//-----------------------------------------------------------------------------
struct NegOp    {};  // NEGATIVE OPERATOR
struct InvOp    {};  // INVERSE OPERATOR
struct SinOp    {};  // SINE OPERATOR
struct CosOp    {};  // COSINE OPERATOR
struct TanOp    {};  // TANGENT OPERATOR
struct ArcSinOp {};  // ARC SINE OPERATOR
struct ArcCosOp {};  // ARC COSINE OPERATOR
struct ArcTanOp {};  // ARC TANGENT OPERATOR
struct ExpOp    {};  // EXPONENTIAL OPERATOR
struct LogOp    {};  // NATURAL LOGARITHM OPERATOR
struct Log10Op  {};  // BASE-10 LOGARITHM OPERATOR
struct SqrtOp   {};  // SQUARE ROOT OPERATOR
struct PowOp    {};  // POWER OPERATOR
struct AbsOp    {};  // ABSOLUTE OPERATOR

//-----------------------------------------------------------------------------
// OTHER OPERATORS
//-----------------------------------------------------------------------------
struct NumberDualMulOp     {};  // NUMBER-DUAL MULTIPLICATION OPERATOR
struct NumberDualDualMulOp {};  // NUMBER-DUAL-DUAL MULTIPLICATION OPERATOR

//=====================================================================================================================
//
// BASE EXPRESSION TYPES (DECLARATION)
//
//=====================================================================================================================

template<size_t N, typename T, typename W>
struct BaseDual;

template<size_t N, typename T>
struct Dual;

template<size_t N, typename T>
struct SubDual;

template<typename Op, typename E>
struct UnaryExpr;

template<typename Op, typename L, typename R>
struct BinaryExpr;

template<typename Op, typename L, typename C, typename R>
struct TernaryExpr;

//=====================================================================================================================
//
// DERIVED EXPRESSION TYPES
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// DERIVED MATHEMATICAL EXPRESSIONS
//-----------------------------------------------------------------------------
template<typename R>
using NegExpr = UnaryExpr<NegOp, R>;

template<typename R>
using InvExpr = UnaryExpr<InvOp, R>;

template<typename R>
using SinExpr = UnaryExpr<SinOp, R>;

template<typename R>
using CosExpr = UnaryExpr<CosOp, R>;

template<typename R>
using TanExpr = UnaryExpr<TanOp, R>;

template<typename R>
using ArcSinExpr = UnaryExpr<ArcSinOp, R>;

template<typename R>
using ArcCosExpr = UnaryExpr<ArcCosOp, R>;

template<typename R>
using ArcTanExpr = UnaryExpr<ArcTanOp, R>;

template<typename R>
using ExpExpr = UnaryExpr<ExpOp, R>;

template<typename R>
using LogExpr = UnaryExpr<LogOp, R>;

template<typename R>
using Log10Expr = UnaryExpr<Log10Op, R>;

template<typename R>
using SqrtExpr = UnaryExpr<SqrtOp, R>;

template<typename L, typename R>
using PowExpr = BinaryExpr<PowOp, L, R>;

template<typename R>
using AbsExpr = UnaryExpr<AbsOp, R>;

//-----------------------------------------------------------------------------
// DERIVED ARITHMETIC EXPRESSIONS
//-----------------------------------------------------------------------------
template<typename L, typename R>
using AddExpr = BinaryExpr<AddOp, L, R>;

template<typename L, typename R>
using MulExpr = BinaryExpr<MulOp, L, R>;

//-----------------------------------------------------------------------------
// DERIVED OTHER EXPRESSIONS
//-----------------------------------------------------------------------------
template<typename L, typename R>
using NumberDualMulExpr = BinaryExpr<NumberDualMulOp, L, R>;

template<typename L, typename C, typename R>
using NumberDualDualMulExpr = TernaryExpr<NumberDualDualMulOp, L, C, R>;

//=====================================================================================================================
//
// TYPE TRAITS UTILITIES
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// ENABLE-IF FOR SFINAE USE
//-----------------------------------------------------------------------------
template<bool value>
using enableif = typename std::enable_if<value>::type;

//-----------------------------------------------------------------------------
// CONVENIENT TYPE TRAIT UTILITIES
//-----------------------------------------------------------------------------
template<typename T>
using plain = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template<typename A, typename B>
using common = typename std::common_type<A, B>::type;

namespace traits {

//-----------------------------------------------------------------------------
// IS TYPE T AN EXPRESSION NODE?
//-----------------------------------------------------------------------------
template<typename T>
struct isExpr { constexpr static bool value = false; };

template<size_t N, typename T>
struct isExpr<Dual<N, T>> { constexpr static bool value = true; };

template<size_t N, typename T>
struct isExpr<SubDual<N, T>> { constexpr static bool value = true; };

template<typename Op, typename R>
struct isExpr<UnaryExpr<Op, R>> { constexpr static bool value = true; };

template<typename Op, typename L, typename R>
struct isExpr<BinaryExpr<Op, L, R>> { constexpr static bool value = true; };

template<typename Op, typename L, typename C, typename R>
struct isExpr<TernaryExpr<Op, L, C, R>> { constexpr static bool value = true; };

//-----------------------------------------------------------------------------
// IS TYPE T A DUAL INSTANCE?
//-----------------------------------------------------------------------------
template<typename T>
struct isDual { constexpr static bool value = false; };

template<size_t N, typename T, typename W>
struct isDual<BaseDual<N, T, W>> { constexpr static bool value = true; };

template<size_t N, typename T>
struct isDual<Dual<N, T>> { constexpr static bool value = true; };

template<size_t N, typename T>
struct isDual<SubDual<N, T>> { constexpr static bool value = true; };

//-----------------------------------------------------------------------------
// IS TYPE T A SUBDUAL INSTANCE?
//-----------------------------------------------------------------------------
template<typename T>
struct isSubDual { constexpr static bool value = false; };

template<size_t N, typename T>
struct isSubDual<SubDual<N, T>> { constexpr static bool value = true; };

//-----------------------------------------------------------------------------
// IS TYPE T AN UNARY EXPRESSION?
//-----------------------------------------------------------------------------
template<typename T>
struct isUnaryExpr { constexpr static bool value = false; };

template<typename Op, typename R>
struct isUnaryExpr<UnaryExpr<Op, R>> { constexpr static bool value = true; };

//-----------------------------------------------------------------------------
// IS TYPE T A BINARY EXPRESSION?
//-----------------------------------------------------------------------------
template<typename T>
struct isBinaryExpr { constexpr static bool value = false; };

template<typename Op, typename L, typename R>
struct isBinaryExpr<BinaryExpr<Op, L, R>> { constexpr static bool value = true; };

//-----------------------------------------------------------------------------
// IS TYPE T A TERNARY EXPRESSION?
//-----------------------------------------------------------------------------
template<typename T>
struct isTernaryExpr { constexpr static bool value = false; };

template<typename Op, typename L, typename C, typename R>
struct isTernaryExpr<TernaryExpr<Op, L, C, R>> { constexpr static bool value = true; };

//-----------------------------------------------------------------------------
// IS TYPE T A NEGATIVE EXPRESSION?
//-----------------------------------------------------------------------------
template<typename T>
struct isNegExpr { constexpr static bool value = false; };

template<typename T>
struct isNegExpr<NegExpr<T>> { constexpr static bool value = true; };

//-----------------------------------------------------------------------------
// IS TYPE T AN INVERSE EXPRESSION?
//-----------------------------------------------------------------------------
template<typename T>
struct isInvExpr { constexpr static bool value = false; };

template<typename T>
struct isInvExpr<InvExpr<T>> { constexpr static bool value = true; };

//-----------------------------------------------------------------------------
// IS TYPE T A GENERAL ADDITION EXPRESSION?
//-----------------------------------------------------------------------------
template<typename T>
struct isAddExpr { constexpr static bool value = false; };

template<typename L, typename R>
struct isAddExpr<AddExpr<L, R>> { constexpr static bool value = true; };

//-----------------------------------------------------------------------------
// IS TYPE T A GENERAL MULTIPLICATION EXPRESSION?
//-----------------------------------------------------------------------------
template<typename T>
struct isMulExpr { constexpr static bool value = false; };

template<typename L, typename R>
struct isMulExpr<MulExpr<L, R>> { constexpr static bool value = true; };

//-----------------------------------------------------------------------------
// IS TYPE T A POWER EXPRESSION?
//-----------------------------------------------------------------------------
template<typename T>
struct isPowExpr { constexpr static bool value = false; };

template<typename L, typename R>
struct isPowExpr<PowExpr<L, R>> { constexpr static bool value = true; };

//-----------------------------------------------------------------------------
// IS TYPE T A NUMBER-DUAL MULTIPLICATION EXPRESSION?
//-----------------------------------------------------------------------------
template<typename T>
struct isNumberDualMulExpr { constexpr static bool value = false; };

template<typename L, typename R>
struct isNumberDualMulExpr<NumberDualMulExpr<L, R>> { constexpr static bool value = true; };

//-----------------------------------------------------------------------------
// IS TYPE T A NUMBER-DUAL-DUAL MULTIPLICATION EXPRESSION?
//-----------------------------------------------------------------------------
template<typename T>
struct isNumberDualDualMulExpr { constexpr static bool value = false; };

template<typename L, typename C, typename R>
struct isNumberDualDualMulExpr<NumberDualDualMulExpr<L, C, R>> { constexpr static bool value = true; };

} // namespace traits

template<typename T>
constexpr bool isNumber = std::is_arithmetic<plain<T>>::value;

template<typename T>
constexpr bool isExpr = traits::isExpr<plain<T>>::value;

template<typename T>
constexpr bool isDual = traits::isDual<plain<T>>::value;

template<typename T>
constexpr bool isSubDual = traits::isSubDual<plain<T>>::value;

template<typename T>
constexpr bool isUnaryExpr = traits::isUnaryExpr<plain<T>>::value;

template<typename T>
constexpr bool isBinaryExpr = traits::isBinaryExpr<plain<T>>::value;

template<typename T>
constexpr bool isTernaryExpr = traits::isTernaryExpr<plain<T>>::value;

template<typename T>
constexpr bool isNegExpr = traits::isNegExpr<plain<T>>::value;

template<typename T>
constexpr bool isInvExpr = traits::isInvExpr<plain<T>>::value;

template<typename T>
constexpr bool isAddExpr = traits::isAddExpr<plain<T>>::value;

template<typename T>
constexpr bool isMulExpr = traits::isMulExpr<plain<T>>::value;

template<typename T>
constexpr bool isPowExpr = traits::isPowExpr<plain<T>>::value;

template<typename T>
constexpr bool isNumberDualMulExpr = traits::isNumberDualMulExpr<plain<T>>::value;

template<typename T>
constexpr bool isNumberDualDualMulExpr = traits::isNumberDualDualMulExpr<plain<T>>::value;

//-----------------------------------------------------------------------------
// ARE TYPES L AND R EXPRESSION NODES OR NUMBERS, BUT NOT BOTH NUMBERS?
//-----------------------------------------------------------------------------
template<typename L, typename R>
constexpr bool isOperable = (isExpr<L> && isExpr<R>) || (isNumber<L> && isExpr<R>) || (isExpr<L> && isNumber<R>);

namespace traits {

//-----------------------------------------------------------------------------
// WHAT IS THE VALUE TYPE OF AN EXPRESSION NODE?
//-----------------------------------------------------------------------------

struct ValueTypeInvalid {};

template<typename T>
struct ValueType { using type = std::conditional_t<isNumber<T>, T, ValueTypeInvalid>; };

template<size_t N, typename T, typename W>
struct ValueType<BaseDual<N, T, W>> { using type = typename ValueType<plain<T>>::type; };

template<size_t N, typename T>
struct ValueType<Dual<N, T>> { using type = typename ValueType<plain<T>>::type; };

template<size_t N, typename T>
struct ValueType<SubDual<N, T>> { using type = typename ValueType<plain<T>>::type; };

template<typename Op, typename R>
struct ValueType<UnaryExpr<Op, R>> { using type = typename ValueType<plain<R>>::type; };

template<typename Op, typename L, typename R>
struct ValueType<BinaryExpr<Op, L, R>> { using type = common<typename ValueType<plain<L>>::type, typename ValueType<plain<R>>::type>; };

template<typename Op, typename L, typename C, typename R>
struct ValueType<TernaryExpr<Op, L, C, R>> { using type = common<typename ValueType<plain<L>>::type, common<typename ValueType<plain<C>>::type, typename ValueType<plain<R>>::type>>; };

//-----------------------------------------------------------------------------
// WHAT IS THE GRADIENT TYPE OF AN EXPRESSION NODE?
//-----------------------------------------------------------------------------

struct OrderNumberInvalid {};

template<typename T>
struct OrderNumber { constexpr static size_t value = 0; };

template<size_t N, typename T, typename W>
struct OrderNumber<BaseDual<N, T, W>> { constexpr static size_t value = N; };

template<size_t N, typename T>
struct OrderNumber<Dual<N, T>> { constexpr static size_t value = N; };

template<size_t N, typename T>
struct OrderNumber<SubDual<N, T>> { constexpr static size_t value = N; };

template<typename Op, typename R>
struct OrderNumber<UnaryExpr<Op, R>> { constexpr static size_t value = OrderNumber<plain<R>>::value; };

template<typename Op, typename L, typename R>
struct OrderNumber<BinaryExpr<Op, L, R>> { constexpr static size_t value = std::max(OrderNumber<plain<L>>::value, OrderNumber<plain<R>>::value); };

template<typename Op, typename L, typename C, typename R>
struct OrderNumber<TernaryExpr<Op, L, C, R>> { constexpr static size_t value = std::max(OrderNumber<plain<L>>::value, std::max(OrderNumber<plain<C>>::value, OrderNumber<plain<R>>::value)); };

//-----------------------------------------------------------------------------
// WHAT IS THE OPERATOR TYPE OF AN EXPRESSION NODE?
//-----------------------------------------------------------------------------

struct OperatorTypeInvalid {};

template<typename T>
struct OperatorType { using type = OperatorTypeInvalid; };

template<typename Op, typename R>
struct OperatorType<UnaryExpr<Op, R>> { using type = Op; };

template<typename Op, typename L, typename R>
struct OperatorType<BinaryExpr<Op, L, R>> { using type = Op; };

template<typename Op, typename L, typename C, typename R>
struct OperatorType<TernaryExpr<Op, L, C, R>> { using type = Op; };

} // namespace traits

template<typename T>
using ValueType = typename traits::ValueType<plain<T>>::type;

template<typename T>
constexpr size_t OrderNumber = traits::OrderNumber<plain<T>>::value;

template<typename T>
using OperatorType = typename traits::OperatorType<plain<T>>::type;

//=====================================================================================================================
//
// EXPRESSION TYPES DEFINITION
//
//=====================================================================================================================

namespace detail {

template<size_t M, typename T>
auto begin(const std::array<T, M>& a)
{
    return a.begin();
}

template<size_t M, typename T>
auto begin(std::array<T, M>& a)
{
    return a.begin();
}

template<typename T>
auto begin(T* a) -> T*
{
    return a;
}

} // namespace detail

template<size_t N, typename T, typename W>
class BaseDual
{
public:
    constexpr BaseDual()
    : m_data()
    {}

    constexpr explicit BaseDual(const W& data)
    : m_data(data)
    {}

    constexpr auto data() -> numarray<N + 1, T>
    {
        return numarray<N + 1, T>{ begin() };
    }

    constexpr auto data() const -> numarray<N + 1, const T>
    {
        return numarray<N + 1, const T>{ begin() };
    }

    constexpr auto begin() -> T*
    {
        return detail::begin(m_data);
    }

    constexpr auto begin() const -> const T*
    {
        return detail::begin(m_data);
    }

    template<typename U, enableif<isExpr<U>>...>
    BaseDual(U&& other)
    {
        assign(*this, std::forward<U>(other));
    }

    template<typename U, enableif<isNumber<U> || isExpr<U>>...>
    auto operator=(U&& other) -> BaseDual&
    {
        assign(*this, std::forward<U>(other));
        return *this;
    }

    template<typename U, enableif<isNumber<U> || isExpr<U>>...>
    auto operator+=(U&& other) -> BaseDual&
    {
        assignAdd(*this, std::forward<U>(other));
        return *this;
    }

    template<typename U, enableif<isNumber<U> || isExpr<U>>...>
    auto operator-=(U&& other) -> BaseDual&
    {
        assignSub(*this, std::forward<U>(other));
        return *this;
    }

    template<typename U, enableif<isNumber<U> || isExpr<U>>...>
    auto operator*=(U&& other) -> BaseDual&
    {
        assignMul(*this, std::forward<U>(other));
        return *this;
    }

    template<typename U, enableif<isNumber<U> || isExpr<U>>...>
    auto operator/=(U&& other) -> BaseDual&
    {
        assignDiv(*this, std::forward<U>(other));
        return *this;
    }

private:
    W m_data;
};

template<size_t N, typename T>
class Dual : public BaseDual<N, T, std::array<T, N + 1>>
{
public:
    using Base = BaseDual<N, T, std::array<T, N + 1>>;

    using Base::Base;

    constexpr Dual()
    {}

    constexpr Dual(const T& value)
    : Base({value})
    {
    }

    template<typename U, enableif<isExpr<U>>...>
    Dual(U&& other)
    {
        assign(*this, std::forward<U>(other));
    }

    template<typename U, enableif<isNumber<U> || isExpr<U>>...>
    auto operator=(U&& other) -> Dual&
    {
        assign(*this, std::forward<U>(other));
        return *this;
    }
};

template<size_t N, typename T>
class SubDual : public BaseDual<N, T, T*>
{
public:
    using Base = BaseDual<N, T, T*>;

    SubDual() = delete;

    constexpr explicit SubDual(T* data)
    : Base(data)
    {
    }

    template<typename U, enableif<isExpr<U>>...>
    SubDual(U&& other)
    {
        assign(*this, std::forward<U>(other));
    }

    template<typename U, enableif<isNumber<U> || isExpr<U>>...>
    auto operator=(U&& other) -> SubDual&
    {
        assign(*this, std::forward<U>(other));
        return *this;
    }
};

template<typename Op, typename R>
struct UnaryExpr
{
    R r;
};

template<typename Op, typename L, typename R>
struct BinaryExpr
{
    L l;
    R r;
};

//=====================================================================================================================
//
// UTILITY FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T, typename W>
constexpr auto val(BaseDual<N, T, W>& x) -> T&
{
    return x.begin()[0];
}

template<size_t N, typename T, typename W>
constexpr auto val(const BaseDual<N, T, W>& x) -> const T&
{
    return x.begin()[0];
}


template<typename Dual>
constexpr auto grad(Dual&& x)
{
    constexpr auto N = OrderNumber<Dual>;
    using T = ValueType<Dual>;
    static_assert(N > 0);
    return SubDual<N - 1, T>{ x.begin() + 1 };
}

// template<size_t N, typename T, typename W>
// constexpr auto grad(BaseDual<N, T, W>& x) -> SubDual<N - 1, T>
// {
//     static_assert(N > 0);
//     return SubDual<N - 1, T>{ x.begin() + 1 };
// }

// template<size_t N, typename T, typename W>
// constexpr auto grad(const BaseDual<N, T, W>& x) -> SubDual<N - 1, const T>
// {
//     static_assert(N > 0);
//     return SubDual<N - 1, const T>{ x.begin() + 1 };
// }

template<size_t N, typename T, typename W>
constexpr auto head(BaseDual<N, T, W>& x) -> SubDual<N - 1, T>
{
    static_assert(N > 0);
    return SubDual<N - 1, T>{ x.begin() };
}

template<size_t N, typename T, typename W>
constexpr auto head(const BaseDual<N, T, W>& x) -> SubDual<N - 1, const T>
{
    static_assert(N > 0);
    return SubDual<N - 1, const T>{ x.begin() };
}

template<size_t N, typename T, typename W>
constexpr auto tail(BaseDual<N, T, W>& x) -> SubDual<N - 1, T>
{
    static_assert(N > 0);
    return SubDual<N - 1, T>{ x.begin() + 1 };
}

template<size_t N, typename T, typename W>
constexpr auto tail(const BaseDual<N, T, W>& x) -> SubDual<N - 1, const T>
{
    static_assert(N > 0);
    return SubDual<N - 1, const T>{ x.begin() + 1 };
}

// template<typename T>
// auto eval(T&& expr)
// {
//     if constexpr (isDual<T>)
//         return std::forward<T>(expr);
//     else if constexpr (isExpr<T>)
//         return Dual<OrderNumber<T>, ValueType<T>>(std::forward<T>(expr));
//     else return std::forward<T>(expr);
// }

// template<typename T>
// auto val(T&& expr)
// {
//     if constexpr (isDual<T>)
//         return val(expr.val);
//     else if constexpr (isExpr<T>)
//         return val(eval(std::forward<T>(expr)));
//     else return std::forward<T>(expr);
// }

// // // namespace internal {

// // // template<int num, typename Arg>
// // // auto seed(Arg& dual) -> void
// // // {
// // //     static_assert(isDual<Arg>);
// // //     dual.grad = num;
// // // }

// // // template<int num, typename Arg, typename... Args>
// // // auto seed(Arg& dual, Args&... duals) -> void
// // // {
// // //     static_assert(isDual<Arg>);
// // //     seed<num>(duals.val...);
// // //     dual.grad = num;
// // // }

// // // } // namespace internal

// // template<typename Arg>
// // auto seed(std::tuple<Arg&> dual)
// // {
// //     static_assert(isDual<Arg>);
// //     internal::seed<1>(std::get<0>(dual));
// // }

// // template<typename... Args>
// // auto seed(std::tuple<Args&...> duals)
// // {
// //     std::apply(internal::seed<1, Args&...>, duals);
// // }

// // template<typename Arg>
// // auto unseed(std::tuple<Arg&> dual)
// // {
// //     static_assert(isDual<Arg>);
// //     internal::seed<0>(std::get<0>(dual));
// // }

// // template<typename... Args>
// // auto unseed(std::tuple<Args&...> duals)
// // {
// //     std::apply(internal::seed<0, Args&...>, duals);
// // }

// // template<typename... Args>
// // auto wrt(Args&... duals)
// // {
// //     return std::tuple<Args&...>(duals...);
// // }

// template<size_t order, size_t N, typename T>
// auto derivative(const Dual<N, T>& dual)
// {
//     if constexpr (order == 0)
//         return dual.val;
//     if constexpr (order == 1)
//         return dual.grad;
//     else
//         return derivative<order - 1>(dual.grad);
// }

// template<typename Function, typename... Duals, typename... Args>
// auto derivative(const Function& f, std::tuple<Duals&...> wrt, Args&... args)
// {
//     seed<Duals&...>(wrt);
//     auto res = f(args...);
//     unseed<Duals&...>(wrt);
//     return derivative<std::tuple_size<decltype(wrt)>::value>(res);
// }

// Code below requires template argument deduction, which is not available in clang v4,
// only osx supported compiler in conda-forge at the moment

// namespace internal {

// template<size_t N, typename T, typename... Args>
// auto grad(const std::function<Dual<N, T>(Args...)>& f)
// {
//     auto g = [=](Dual<N, T>& wrt, Args&... args) -> G {
//         return derivative(f, wrt, args...);
//     };
//     return g;
// }

// } // namespace internal

// template<typename Function>
// auto grad(const Function& f)
// {
//     return internal::grad(std::function{f});
// }

//=====================================================================================================================
//
// CONVENIENT FUNCTIONS
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// NEGATIVE EXPRESSION GENERATOR FUNCTION
//-----------------------------------------------------------------------------
template<typename U>
constexpr auto negative(U&& expr)
{
    static_assert(isExpr<U> || isNumber<U>);
    if constexpr (isNegExpr<U>)
        return expr.r;
    else return NegExpr<U>{ std::forward<U>(expr) };
}

//-----------------------------------------------------------------------------
// INVERSE EXPRESSION GENERATOR FUNCTION
//-----------------------------------------------------------------------------
template<typename U>
constexpr size_t inverse(U&& expr)
{
    static_assert(isExpr<U>);
    if constexpr (isInvExpr<U>)
        return expr.r;
    else return InvExpr<U>{ std::forward<U>(expr) };
}

//-----------------------------------------------------------------------------
// AUXILIARY CONSTEXPR CONSTANTS
//-----------------------------------------------------------------------------
template<typename U>
constexpr auto Zero = static_cast<ValueType<U>>(0);

template<typename U>
constexpr auto One = static_cast<ValueType<U>>(1);

//=====================================================================================================================
//
// POSITIVE ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// POSITIVE OPERATOR: +x
//-----------------------------------------------------------------------------
template<typename R, enableif<isExpr<R>>...>
constexpr auto operator+(R&& expr)
{
    return std::forward<R>(expr); // expression optimization: +(expr) => expr
}

//=====================================================================================================================
//
// NEGATIVE ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

template<typename R, enableif<isExpr<R>>...>
constexpr auto operator-(R&& expr)
{
    // NEGATIVE EXPRESSION CASE: -(-x) => x when expr is (-x)
    if constexpr (isNegExpr<R>)
        return expr.r;
    // NEGATIVE EXPRESSION CASE: -(number * dual) => (-number) * dual
    else if constexpr (isNumberDualMulExpr<R>)
        return (-expr.l) * expr.r;
    // default expression
    else return NegExpr<R>{ std::forward<R>(expr) };
}

//=====================================================================================================================
//
// ADDITION ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

template<typename L, typename R, enableif<isOperable<L, R>>...>
constexpr auto operator+(L&& l, R&& r)
{
    // ADDITION EXPRESSION CASE: (-x) + (-y) => -(x + y)
    if constexpr (isNegExpr<L> && isNegExpr<R>)
        return -( l.r + r.r );
    // ADDITION EXPRESSION CASE: expr + number => number + expr (number always on the left)
    else if constexpr (isExpr<L> && isNumber<R>)
        return std::forward<R>(r) + std::forward<L>(l);
    // DEFAULT ADDITION EXPRESSION
    else return AddExpr<L, R>{ std::forward<L>(l), std::forward<R>(r) };
}

//=====================================================================================================================
//
// MULTIPLICATION ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

template<typename L, typename R, enableif<isOperable<L, R>>...>
constexpr auto operator*(L&& l, R&& r)
{
    // MULTIPLICATION EXPRESSION CASE: (-expr) * (-expr) => expr * expr
    if constexpr (isNegExpr<L> && isNegExpr<R>)
        return l.r * r.r;
    // // MULTIPLICATION EXPRESSION CASE: (1 / expr) * (1 / expr) => 1 / (expr * expr)
    else if constexpr (isInvExpr<L> && isInvExpr<R>)
        return inverse(l.r * r.r);
    // // MULTIPLICATION EXPRESSION CASE: expr * number => number * expr
    else if constexpr (isExpr<L> && isNumber<R>)
        return std::forward<R>(r) * std::forward<L>(l);
    // // MULTIPLICATION EXPRESSION CASE: number * (-expr) => (-number) * expr
    else if constexpr (isNumber<L> && isNegExpr<R>)
        return (-l) * r.r;
    // // MULTIPLICATION EXPRESSION CASE: number * (number * expr) => (number * number) * expr
    else if constexpr (isNumber<L> && isNumberDualMulExpr<R>)
        return (l * r.l) * r.r;
    // MULTIPLICATION EXPRESSION CASE: number * dual => NumberDualMulExpr
    else if constexpr (isNumber<L> && isDual<R>)
        return NumberDualMulExpr<L, R>{ std::forward<L>(l), std::forward<R>(r) };
    // DEFAULT MULTIPLICATION EXPRESSION: expr * expr => MulExpr
    else return MulExpr<L, R>{ std::forward<L>(l), std::forward<R>(r) };
}

//=====================================================================================================================
//
// SUBTRACTION ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// SUBTRACTION OPERATOR: expr - expr, scalar - expr, expr - scalar
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isOperable<L, R>>...>
constexpr auto operator-(L&& l, R&& r)
{
    return std::forward<L>(l) + ( -std::forward<R>(r) );
}

//=====================================================================================================================
//
// DIVISION ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// DIVISION OPERATOR: expr / expr
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isOperable<L, R>>...>
constexpr auto operator/(L&& l, R&& r)
{
    if constexpr (isNumber<R>)
        return std::forward<L>(l) * (One<L> / std::forward<R>(r));
    else return std::forward<L>(l) * inverse(std::forward<R>(r));
}

//=====================================================================================================================
//
// TRIGONOMETRIC FUNCTIONS OVERLOADING
//
//=====================================================================================================================

template<typename R, enableif<isExpr<R>>...> constexpr auto sin(R&& r) -> SinExpr<R> { return { std::forward<R>(r) }; }
template<typename R, enableif<isExpr<R>>...> constexpr auto cos(R&& r) -> CosExpr<R> { return { std::forward<R>(r) }; }
template<typename R, enableif<isExpr<R>>...> constexpr auto tan(R&& r) -> TanExpr<R> { return { std::forward<R>(r) }; }
template<typename R, enableif<isExpr<R>>...> constexpr auto asin(R&& r) -> ArcSinExpr<R> { return { std::forward<R>(r) }; }
template<typename R, enableif<isExpr<R>>...> constexpr auto acos(R&& r) -> ArcCosExpr<R> { return { std::forward<R>(r) }; }
template<typename R, enableif<isExpr<R>>...> constexpr auto atan(R&& r) -> ArcTanExpr<R> { return { std::forward<R>(r) }; }

//=====================================================================================================================
//
// HYPERBOLIC FUNCTIONS OVERLOADING
//
//=====================================================================================================================


//=====================================================================================================================
//
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS OVERLOADING
//
//=====================================================================================================================

template<typename R, enableif<isExpr<R>>...> constexpr auto exp(R&& r) -> ExpExpr<R> { return { std::forward<R>(r) }; }
template<typename R, enableif<isExpr<R>>...> constexpr auto log(R&& r) -> LogExpr<R> { return { std::forward<R>(r) }; }
template<typename R, enableif<isExpr<R>>...> constexpr auto log10(R&& r) -> Log10Expr<R> { return { std::forward<R>(r) }; }

//=====================================================================================================================
//
// POWER FUNCTIONS OVERLOADING
//
//=====================================================================================================================

template<typename L, typename R, enableif<isOperable<L, R>>...> constexpr auto pow(L&& l, R&& r) -> PowExpr<L, R> { return { std::forward<L>(l), std::forward<R>(r) }; }
template<typename R, enableif<isExpr<R>>...> constexpr auto sqrt(R&& r) -> SqrtExpr<R> { return { std::forward<R>(r) }; }

//=====================================================================================================================
//
// OTHER FUNCTIONS OVERLOADING
//
//=====================================================================================================================

// template<typename R, enableif<isExpr<R>>...> constexpr auto abs(R&& r) -> AbsExpr<R> { return { std::forward<R>(r) }; }
// template<typename R, enableif<isExpr<R>>...> constexpr auto abs2(R&& r) { return std::forward<R>(r) * std::forward<R>(r); }
// template<typename R, enableif<isExpr<R>>...> constexpr auto conj(R&& r) { return std::forward<R>(r); }
// template<typename R, enableif<isExpr<R>>...> constexpr auto real(R&& r) { return std::forward<R>(r); }
// template<typename R, enableif<isExpr<R>>...> constexpr size_t imag(R&& r) { return 0.0; }

//=====================================================================================================================
//
// COMPARISON OPERATORS OVERLOADING
//
//=====================================================================================================================

// template<typename L, typename R, enableif<isOperable<L, R>>...> bool operator==(L&& l, R&& r) { return val(l) == val(r); }
// template<typename L, typename R, enableif<isOperable<L, R>>...> bool operator!=(L&& l, R&& r) { return val(l) != val(r); }
// template<typename L, typename R, enableif<isOperable<L, R>>...> bool operator<=(L&& l, R&& r) { return val(l) <= val(r); }
// template<typename L, typename R, enableif<isOperable<L, R>>...> bool operator>=(L&& l, R&& r) { return val(l) >= val(r); }
// template<typename L, typename R, enableif<isOperable<L, R>>...> bool operator<(L&& l, R&& r) { return val(l) < val(r); }
// template<typename L, typename R, enableif<isOperable<L, R>>...> bool operator>(L&& l, R&& r) { return val(l) > val(r); }

//=====================================================================================================================
//
// AUXILIARY FUNCTIONS
//
//=====================================================================================================================

// namespace impl {

// template<size_t I, size_t N, typename T, typename U>
// constexpr void assign(T* array, const U& scalar)
// {
//     if constexpr (I < N)
//     {
//         array[I] = scalar;
//         assign<I + 1>(array, scalar);
//     }
// }

// template<size_t I, size_t N, typename T>
// constexpr void negate(T* array)
// {
//     if constexpr (I < N)
//     {
//         array[I] = -array[I];
//         negate<I + 1>(array);
//     }
// }

// template<size_t I, size_t N, typename T, typename U>
// constexpr void scale(T* array, const U& scalar)
// {
//     if constexpr (I < N)
//     {
//         array[I] *= scalar;
//         scale<I + 1>(array, scalar);
//     }
// }

// } // namespace impl

// template<size_t N, typename T, typename U>
// constexpr void assign(T* array, const U& scalar)
// {
//     impl::assign<0, N>(array, scalar);
// }

// template<size_t N, typename T>
// constexpr void negate(T* array)
// {
//     impl::negate<0, N>(array);
// }

// template<size_t N, typename T, typename U>
// constexpr void scale(T* array, const U& scalar)
// {
//     impl::scale<0, N>(array, scalar);
// }

template<size_t N, typename T, typename W>
constexpr void negate(BaseDual<N, T, W>& self)
{
    self.data() *= static_cast<T>(-1);
}

template<size_t N, typename T, typename W, typename U>
constexpr void scale(BaseDual<N, T, W>& self, const U& scalar)
{
    self.data() *= scalar;
}

//=====================================================================================================================
//
// FORWARD DECLARATIONS
//
//=====================================================================================================================

template<typename Op, size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self);

//=====================================================================================================================
//
// ASSIGNMENT FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T, typename W, typename U>
constexpr void assign(BaseDual<N, T, W>& self, U&& other)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN A NUMBER: self = number
    if constexpr (isNumber<U>) {
        self.data() = Zero<T>;
        val(self) = other;
    }
    // ASSIGN A DUAL NUMBER: self = dual
    else if constexpr (isDual<U>) {
        self.data() = other.data();
    }
    // ASSIGN A NUMBER-DUAL MULTIPLICATION EXPRESSION: self = number * dual
    else if constexpr (isNumberDualMulExpr<U>) {
        assign(self, other.r);
        scale(self, other.l);
    }
    // ASSIGN A UNARY EXPRESSION: self = unaryexpr
    else if constexpr (isUnaryExpr<U>) {
        using Op = OperatorType<U>;
        assign(self, other.r);
        apply<Op>(self);
    }
    // ASSIGN AN ADDITION EXPRESSION: self = expr + expr
    else if constexpr (isAddExpr<U>) {
        assign(self, other.l);
        // assign(self, other.r);
        // assignAdd(self, other.l);
    }
    // ASSIGN A MULTIPLICATION EXPRESSION: self = expr * expr
    else if constexpr (isMulExpr<U>) {
        assign(self, other.r);
        assignMul(self, other.l);
    }
    // ASSIGN A POWER EXPRESSION: self = pow(expr)
    else if constexpr (isPowExpr<U>) {
        assign(self, other.l);
        assignPow(self, other.r);
    }
    else {
        int i = self * other;
    }
}

template<size_t N, typename T, typename W, typename U>
constexpr void assign(BaseDual<N, T, W>& self, U&& other, Dual<N, T>& tmp)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN AN UNARY EXPRESSION: self = func(expr)
    if constexpr (isUnaryExpr<U>) {
        using Op = OperatorType<U>;
        assign(self, other.r, tmp);
        apply<Op>(self);
    }
    // ASSIGN AN ADDITION EXPRESSION: self = expr + expr
    else if constexpr (isAddExpr<U>) {
        assign(self, other.r, tmp);
        assignAdd(self, other.l, tmp);
    }
    // ASSIGN A MULTIPLICATION EXPRESSION: self = expr * expr
    else if constexpr (isMulExpr<U>) {
        assign(self, other.r, tmp);
        assignMul(self, other.l, tmp);
    }
    // ASSIGN A POWER EXPRESSION: self = pow(expr, expr)
    else if constexpr (isPowExpr<U>) {
        assign(self, other.l, tmp);
        assignPow(self, other.r, tmp);
    }
    // ALL OTHER EXPRESSIONS
    else {
        assign(tmp, other);
        assign(self, tmp);
    }
}

//=====================================================================================================================
//
// ASSIGNMENT-ADDITION FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T, typename W, typename U>
constexpr void assignAdd(BaseDual<N, T, W>& self, U&& other)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN-ADD A NUMBER: self += number
    if constexpr (isNumber<U>) {
        self.val() += other;
    }
    // ASSIGN-ADD A DUAL NUMBER: self += dual
    else if constexpr (isDual<U>) {
        self.data() += other.data();
    }
    // ASSIGN-ADD A NEGATIVE EXPRESSION: self += -expr => self -= expr
    else if constexpr (isNegExpr<U>) {
        assignSub(self, other.r);
    }
    // ASSIGN-ADD A NUMBER-DUAL MULTIPLICATION EXPRESSION: self += number * dual
    // else if constexpr (isNumberDualMulExpr<U>) {
    //     self.data() += other.l * other.r.data();
    // }
    // ASSIGN-ADD AN ADDITION EXPRESSION: self += expr + expr
    else if constexpr (isAddExpr<U>) {
        assignAdd(self, other.l);
        assignAdd(self, other.r);
    }
    // ASSIGN-ADD ALL OTHER EXPRESSIONS
    else {
        Dual<N, T> tmp;
        assignAdd(self, std::forward<U>(other), tmp);
    }
}

template<size_t N, typename T, typename W, typename U>
constexpr void assignAdd(BaseDual<N, T, W>& self, U&& other, Dual<N, T>& tmp)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN-ADD A NEGATIVE EXPRESSION: self += -expr => self -= expr
    if constexpr (isNegExpr<U>) {
        assignSub(self, other.r, tmp);
    }
    // ASSIGN-ADD AN ADDITION EXPRESSION: self += expr + expr
    else if constexpr (isAddExpr<U>) {
        assignAdd(self, other.l, tmp);
        assignAdd(self, other.r, tmp);
    }
    // ASSIGN-ADD ALL OTHER EXPRESSIONS
    else {
        assign(tmp, other);
        assignAdd(self, tmp);
    }
}

//=====================================================================================================================
//
// ASSIGNMENT-SUBTRACTION FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T, typename U>
constexpr void assignSub(Dual<N, T>& self, U&& other)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN-SUBTRACT A NUMBER: self -= number
    if constexpr (isNumber<U>) {
        self.val() -= other;
    }
    // ASSIGN-SUBTRACT A DUAL NUMBER: self -= dual
    else if constexpr (isDual<U>) {
        self.data() -= other.data();
    }
    // ASSIGN-SUBTRACT A NEGATIVE EXPRESSION: self -= -expr => self += expr
    else if constexpr (isNegExpr<U>) {
        assignAdd(self, other.r);
    }
    // ASSIGN-SUBTRACT A NUMBER-DUAL MULTIPLICATION EXPRESSION: self -= number * dual
    // else if constexpr (isNumberDualMulExpr<U>) {
    //     self.data() -= other.l * other.r.data();
    // }
    // ASSIGN-SUBTRACT AN ADDITION EXPRESSION: self -= expr + expr
    else if constexpr (isAddExpr<U>) {
        assignSub(self, other.l);
        assignSub(self, other.r);
    }
    // ASSIGN-SUBTRACT ALL OTHER EXPRESSIONS
    else {
        Dual<N, T> tmp;
        assignSub(self, std::forward<U>(other), tmp);
    }
}

template<size_t N, typename T, typename U>
constexpr void assignSub(Dual<N, T>& self, U&& other, Dual<N, T>& tmp)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN-SUBTRACT A NEGATIVE EXPRESSION: self -= -expr => self += expr
    if constexpr (isNegExpr<U>) {
        assignAdd(self, other.r, tmp);
    }
    // ASSIGN-SUBTRACT AN ADDITION EXPRESSION: self -= expr + expr
    else if constexpr (isAddExpr<U>) {
        assignSub(self, other.l, tmp);
        assignSub(self, other.r, tmp);
    }
    // ASSIGN-SUBTRACT ALL OTHER EXPRESSIONS
    else {
        assign(tmp, other);
        assignSub(self, tmp);
    }
}

//=====================================================================================================================
//
// ASSIGNMENT-MULTIPLICATION FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T, typename W, typename U>
constexpr void assignMul(BaseDual<N, T, W>& self, U&& other)
{
    static_assert(isExpr<U> || isNumber<U>);

    if constexpr (N == 0) {
        self.data() *= other.data();
    }

    // ASSIGN-MULTIPLY A NUMBER: self *= number
    else if constexpr (isNumber<U>) {
        self.data() *= other;
    }
    // ASSIGN-MULTIPLY A DUAL NUMBER: self *= dual
    else if constexpr (isDual<U>) {
        // grad(self) = val(self) * grad(other) + grad(self) * val(other);
        // Dual<N - 1, T> aux1 = head(self) * grad(other);
        // Dual<N - 1, T> aux2 = grad(self) * head(other);
        // grad(self) = aux1 + aux2;

        const Dual<N - 1, T> aux = grad(other) * head(self);
        grad(self) *= head(other);
        grad(self) += aux;

        // grad(self) = head(self) * grad(other) + grad(self) * head(other);
        // grad(self) = head(self) * grad(other);
        // tail(self) = head(self) * tail(other);
        val(self) *= val(other);
        // const auto aux = self.val() * other.grad(); // to avoid aliasing when self === other
        // self.grad() *= other.val();
        // self.grad() += aux;
        // self.val() *= other.val();
    }
    // ASSIGN-MULTIPLY A NEGATIVE EXPRESSION: self *= (-expr)
    else if constexpr (isNegExpr<U>) {
        assignMul(self, other.r);
        negate(self);
    }
    // ASSIGN-MULTIPLY A NUMBER-DUAL MULTIPLICATION EXPRESSION: self *= number * dual
    else if constexpr (isNumberDualMulExpr<U>) {
        assignMul(self, other.r);
        scale(self, other.l);
    }
    // ASSIGN-MULTIPLY A MULTIPLICATION EXPRESSION: self *= expr * expr
    else if constexpr (isMulExpr<U>) {
        assignMul(self, other.l);
        assignMul(self, other.r);
    }
    // ASSIGN-MULTIPLY ALL OTHER EXPRESSIONS
    else {
        Dual<N, T> tmp;
        assignMul(self, std::forward<U>(other), tmp);
    }
}

template<size_t N, typename T, typename W, typename U>
constexpr void assignMul(BaseDual<N, T, W>& self, U&& other, Dual<N, T>& tmp)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN-MULTIPLY A NEGATIVE EXPRESSION: self *= (-expr)
    if constexpr (isNegExpr<U>) {
        assignMul(self, other.r, tmp);
        negate(self);
    }
    // ASSIGN-MULTIPLY A MULTIPLICATION EXPRESSION: self *= expr * expr
    else if constexpr (isMulExpr<U>) {
        assignMul(self, other.l, tmp);
        assignMul(self, other.r, tmp);
    }
    // ASSIGN-MULTIPLY ALL OTHER EXPRESSIONS
    else {
        assign(tmp, other);
        assignMul(self, tmp);
    }
}

//=====================================================================================================================
//
// ASSIGNMENT-DIVISION FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T, typename U>
constexpr void assignDiv(Dual<N, T>& self, U&& other)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN-DIVIDE A NUMBER: self /= number
    if constexpr (isNumber<U>) {
        self.val /= other;
        self.grad /= other;
    }
    // ASSIGN-DIVIDE A DUAL NUMBER: self /= dual
    else if constexpr (isDual<U>) {
        const auto aux = One<T> / other.val; // to avoid aliasing when self === other
        self.val *= aux;
        self.grad -= self.val * other.grad;
        self.grad *= aux;
    }
    // ASSIGN-DIVIDE A NEGATIVE EXPRESSION: self /= (-expr)
    else if constexpr (isNegExpr<U>) {
        assignDiv(self, other.r);
        negate(self);
    }
    // ASSIGN-DIVIDE AN INVERSE EXPRESSION: self /= 1/expr
    else if constexpr (isInvExpr<U>) {
        assignMul(self, other.r);
    }
    // ASSIGN-DIVIDE A NUMBER-DUAL MULTIPLICATION EXPRESSION: self /= number * dual
    else if constexpr (isNumberDualMulExpr<U>) {
        assignDiv(self, other.r);
        assignDiv(self, other.l);
    }
    // ASSIGN-DIVIDE A MULTIPLICATION EXPRESSION: self /= expr * expr
    else if constexpr (isMulExpr<U>) {
        assignDiv(self, other.l);
        assignDiv(self, other.r);
    }
    // ASSIGN-DIVIDE ALL OTHER EXPRESSIONS
    else {
        Dual<N, T> tmp;
        assignDiv(self, std::forward<U>(other), tmp);
    }
}

template<size_t N, typename T, typename U>
constexpr void assignDiv(Dual<N, T>& self, U&& other, Dual<N, T>& tmp)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN-DIVIDE A NEGATIVE EXPRESSION: self /= (-expr)
    if constexpr (isNegExpr<U>) {
        assignDiv(self, other.r, tmp);
        negate(self);
    }
    // ASSIGN-DIVIDE AN INVERSE EXPRESSION: self /= 1/expr
    else if constexpr (isInvExpr<U>) {
        assignMul(self, other.r, tmp);
    }
    // ASSIGN-DIVIDE A MULTIPLICATION EXPRESSION: self /= expr * expr
    else if constexpr (isMulExpr<U>) {
        assignDiv(self, other.l, tmp);
        assignDiv(self, other.r, tmp);
    }
    // ASSIGN-DIVIDE ALL OTHER EXPRESSIONS
    else {
        assign(tmp, other);
        assignDiv(self, tmp);
    }
}

//=====================================================================================================================
//
// ASSIGNMENT-POWER FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T, typename U>
constexpr void assignPow(Dual<N, T>& self, U&& other)
{
    // ASSIGN-POW A NUMBER: self = pow(self, number)
    if constexpr (isNumber<U>) {
        const auto aux = std::pow(self.val, other);
        self.grad *= other/self.val * aux;
        self.val = aux;
    }
    // ASSIGN-POW A DUAL NUMBER: self = pow(self, dual)
    else if constexpr (isDual<U>) {
        const auto aux1 = std::pow(self.val, other.val);
        const auto aux2 = std::log(self.val);
        self.grad *= other.val/self.val;
        self.grad += aux2 * other.grad;
        self.grad *= aux1;
        self.val = aux1;
    }
    // ASSIGN-POW ALL OTHER EXPRESSIONS: self = pow(self, expr)
    else {
        Dual<N, T> tmp;
        assignPow(self, std::forward<U>(other), tmp);
    }
}

template<size_t N, typename T, typename U>
constexpr void assignPow(Dual<N, T>& self, U&& other, Dual<N, T>& tmp)
{
    assign(tmp, other);
    assignPow(self, tmp);
}

//=====================================================================================================================
//
// APPLY-OPERATOR FUNCTIONS
//
//=====================================================================================================================
template<size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self, NegOp)
{
    self.data() *= static_cast<T>(-1);
}

template<size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self, InvOp)
{
    self.val = One<T> / self.val;
    self.grad *= - self.val * self.val;
}

template<size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self, SinOp)
{
    self.grad *= cos(self.val);
    self.val = sin(self.val);
}

template<size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self, CosOp)
{
    self.grad *= -sin(self.val);
    self.val = cos(self.val);
}

template<size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self, TanOp)
{
    const auto aux = One<T> / std::cos(self.val);
    self.val = tan(self.val);
    self.grad *= aux * aux;
}

template<size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self, ArcSinOp)
{
    const auto aux = One<T> / sqrt(1.0 - self.val * self.val);
    self.val = asin(self.val);
    self.grad *= aux;
}

template<size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self, ArcCosOp)
{
    const auto aux = -One<T> / sqrt(1.0 - self.val * self.val);
    self.val = acos(self.val);
    self.grad *= aux;
}

template<size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self, ArcTanOp)
{
    const auto aux = One<T> / (1.0 + self.val * self.val);
    self.val = atan(self.val);
    self.grad *= aux;
}

template<size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self, ExpOp)
{
    self.val = exp(self.val);
    self.grad *= self.val;
}

template<size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self, LogOp)
{
    const auto aux = One<T> / self.val;
    self.val = log(self.val);
    self.grad *= aux;
}

template<size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self, Log10Op)
{
    constexpr T ln10 = 2.3025850929940456840179914546843;
    const auto aux = One<T> / (ln10 * self.val);
    self.val = log10(self.val);
    self.grad *= aux;
}

template<size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self, SqrtOp)
{
    self.val = sqrt(self.val);
    self.grad *= 0.5 / self.val;
}

template<size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self, AbsOp)
{
    const auto aux = self.val;
    self.val = abs(self.val);
    self.grad *= aux / self.val;
}

template<typename Op, size_t N, typename T, typename W>
constexpr void apply(BaseDual<N, T, W>& self)
{
    apply(self, Op{});
}

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& out, const Dual<N, T>& x)
{
    out << x.data[0];
    return out;
}

using dual = forward2::Dual<2, double>;

} // namespace forward

using forward2::dual;
// using forward2::val;
// using forward2::eval;
// using forward2::derivative;
// using forward2::wrt;

} // namespace autodiff
