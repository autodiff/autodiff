//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2020 Allan Leal
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
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

// autodiff includes
#include <autodiff/common/numbertraits.hpp>
#include <autodiff/common/meta.hpp>

namespace autodiff {
namespace detail {

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
using std::cosh;
using std::sinh;
using std::tanh;

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
struct SinhOp    {};  // HYPERBOLIC SINE OPERATOR
struct CoshOp    {};  // HYPERBOLIC COSINE OPERATOR
struct TanhOp    {};  // HYPERBOLIC TANGENT OPERATOR
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

template<typename T, typename G>
struct Dual;

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
using SinhExpr = UnaryExpr<SinhOp, R>;

template<typename R>
using CoshExpr = UnaryExpr<CoshOp, R>;

template<typename R>
using TanhExpr = UnaryExpr<TanhOp, R>;

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

namespace traits {

//-----------------------------------------------------------------------------
// IS TYPE T AN EXPRESSION NODE?
//-----------------------------------------------------------------------------
template<typename T>
struct isExpr { constexpr static bool value = false; };

template<typename T, typename G>
struct isExpr<Dual<T, G>> { constexpr static bool value = true; };

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

template<typename T, typename G>
struct isDual<Dual<T, G>> { constexpr static bool value = true; };

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
constexpr bool isExpr = traits::isExpr<PlainType<T>>::value;

template<typename T>
constexpr bool isDual = traits::isDual<PlainType<T>>::value;

template<typename T>
constexpr bool isUnaryExpr = traits::isUnaryExpr<PlainType<T>>::value;

template<typename T>
constexpr bool isBinaryExpr = traits::isBinaryExpr<PlainType<T>>::value;

template<typename T>
constexpr bool isTernaryExpr = traits::isTernaryExpr<PlainType<T>>::value;

template<typename T>
constexpr bool isNegExpr = traits::isNegExpr<PlainType<T>>::value;

template<typename T>
constexpr bool isInvExpr = traits::isInvExpr<PlainType<T>>::value;

template<typename T>
constexpr bool isAddExpr = traits::isAddExpr<PlainType<T>>::value;

template<typename T>
constexpr bool isMulExpr = traits::isMulExpr<PlainType<T>>::value;

template<typename T>
constexpr bool isPowExpr = traits::isPowExpr<PlainType<T>>::value;

template<typename T>
constexpr bool isNumberDualMulExpr = traits::isNumberDualMulExpr<PlainType<T>>::value;

template<typename T>
constexpr bool isNumberDualDualMulExpr = traits::isNumberDualDualMulExpr<PlainType<T>>::value;

template<typename... Args>
constexpr bool areDual = (... && isDual<Args>);

//-----------------------------------------------------------------------------
// ARE TYPES L AND R EXPRESSION NODES OR NUMBERS, BUT NOT BOTH NUMBERS?
//-----------------------------------------------------------------------------
template<typename L, typename R>
constexpr bool isOperable = (isExpr<L> && isExpr<R>) || (isNumber<L> && isExpr<R>) || (isExpr<L> && isNumber<R>);

//-----------------------------------------------------------------------------
// VALUE, GRAD, AND OP TYPES IN DUAL EXPRESSIONS
//-----------------------------------------------------------------------------

template<typename T> struct AuxDualType;
template<typename T> struct AuxDualValueType;
template<typename T> struct AuxDualGradType;
template<typename T> struct AuxDualOpType;


template<typename T> struct DualTypeNotDefinedFor {};
template<typename T> struct DualValueTypeNotDefinedFor {};
template<typename T> struct DualGradTypeNotDefinedFor {};
template<typename T> struct DualOpTypeNotDefinedFor {};


template<typename T> using DualType = typename AuxDualType<PlainType<T>>::type;
template<typename T> using DualValueType = typename AuxDualValueType<PlainType<T>>::type;
template<typename T> using DualGradType = typename AuxDualGradType<PlainType<T>>::type;
template<typename T> using DualOpType = typename AuxDualOpType<PlainType<T>>::type;


template<typename T>
struct AuxDualType { using type = DualTypeNotDefinedFor<T>; };

template<typename T, typename G>
struct AuxDualType<Dual<T, G>> { using type = Dual<T, G>; };

template<typename Op, typename R>
struct AuxDualType<UnaryExpr<Op, R>> { using type = DualType<R>; };

template<typename Op, typename L, typename R>
struct AuxDualType<BinaryExpr<Op, L, R>> { using type = std::conditional_t<isDual<DualType<L>>, DualType<L>, DualType<R>>; };

template<typename Op, typename L, typename C, typename R>
struct AuxDualType<TernaryExpr<Op, L, C, R>> { using type = std::conditional_t<isDual<DualType<L>>, DualType<L>, std::conditional_t<isDual<DualType<C>>, DualType<C>, DualType<R>>>; };


template<typename T>
struct AuxDualValueType { using type = std::conditional_t<isNumber<T>, T, DualValueTypeNotDefinedFor<T>>; };

template<typename T, typename G>
struct AuxDualValueType<Dual<T, G>> { using type = DualValueType<T>; };

template<typename Op, typename R>
struct AuxDualValueType<UnaryExpr<Op, R>> { using type = DualValueType<R>; };

template<typename Op, typename L, typename R>
struct AuxDualValueType<BinaryExpr<Op, L, R>> { using type = CommonType<DualValueType<L>, DualValueType<R>>; };

template<typename Op, typename L, typename C, typename R>
struct AuxDualValueType<TernaryExpr<Op, L, C, R>> { using type = CommonType<DualValueType<L>, CommonType<DualValueType<C>, DualValueType<R>>>; };


template<typename T>
struct AuxDualGradType { using type = std::conditional_t<isNumber<T>, T, DualGradTypeNotDefinedFor<T>>; };

template<typename T, typename G>
struct AuxDualGradType<Dual<T, G>> { using type = DualGradType<G>; };

template<typename Op, typename R>
struct AuxDualGradType<UnaryExpr<Op, R>> { using type = DualGradType<R>; };

template<typename Op, typename L, typename R>
struct AuxDualGradType<BinaryExpr<Op, L, R>> { using type = CommonType<DualGradType<L>, DualGradType<R>>; };

template<typename Op, typename L, typename C, typename R>
struct AuxDualGradType<TernaryExpr<Op, L, C, R>> { using type = CommonType<DualGradType<L>, CommonType<DualGradType<C>, DualGradType<R>>>; };


template<typename T>
struct AuxDualOpType { using type = DualOpTypeNotDefinedFor<T>; };

template<typename Op, typename R>
struct AuxDualOpType<UnaryExpr<Op, R>> { using type = Op; };

template<typename Op, typename L, typename R>
struct AuxDualOpType<BinaryExpr<Op, L, R>> { using type = Op; };

template<typename Op, typename L, typename C, typename R>
struct AuxDualOpType<TernaryExpr<Op, L, C, R>> { using type = Op; };

//=====================================================================================================================
//
// EXPRESSION TYPES DEFINITION
//
//=====================================================================================================================

template<typename T, typename G>
struct Dual
{
    T val;

    G grad;

    Dual() : Dual(0.0) {}

    explicit operator T() const { return this->val; }

    Dual(const DualValueType<T>& val)
    : val(val), grad(0)
    {
    }

    template<typename U, EnableIf<isExpr<U> && !isDual<U>>...>
    Dual(U&& other)
    {
        assign(*this, std::forward<U>(other));
    }

    template<typename U, EnableIf<isExpr<U> && !isDual<U>>...>
    Dual& operator=(U&& other)
    {
        Dual tmp;
        assign(tmp, std::forward<U>(other));
        assign(*this, tmp);
        return *this;
    }

    template<typename U, EnableIf<isNumber<U> || isExpr<U>>...>
    Dual& operator+=(U&& other)
    {
        Dual tmp;
        assign(tmp, std::forward<U>(other));
        assignAdd(*this, tmp);
        return *this;
    }

    template<typename U, EnableIf<isNumber<U> || isExpr<U>>...>
    Dual& operator-=(U&& other)
    {
        Dual tmp;
        assign(tmp, std::forward<U>(other));
        assignSub(*this, tmp);
        return *this;
    }

    template<typename U, EnableIf<isNumber<U> || isExpr<U>>...>
    Dual& operator*=(U&& other)
    {
        Dual tmp;
        assign(tmp, std::forward<U>(other));
        assignMul(*this, tmp);
        return *this;
    }

    template<typename U, EnableIf<isNumber<U> || isExpr<U>>...>
    Dual& operator/=(U&& other)
    {
        Dual tmp;
        assign(tmp, std::forward<U>(other));
        assignDiv(*this, tmp);
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

template<typename Op, typename R>
auto inner(const UnaryExpr<Op, R>& expr) -> const R
{
    return expr.r;
}

template<typename Op, typename L, typename R>
auto left(const BinaryExpr<Op, L, R>& expr) -> const L
{
    return expr.l;
}

template<typename Op, typename L, typename R>
auto right(const BinaryExpr<Op, L, R>& expr) -> const R
{
    return expr.r;
}

//=====================================================================================================================
//
// UTILITY FUNCTIONS
//
//=====================================================================================================================

template<typename T>
auto eval(T&& expr)
{
    static_assert(isDual<T> || isExpr<T> || isNumber<T>);
    if constexpr (isDual<T>)
        return std::forward<T>(expr);
    else if constexpr (isExpr<T>)
        return DualType<T>(std::forward<T>(expr));
    else return std::forward<T>(expr);
}

template<typename T>
auto val(T&& expr)
{
    static_assert(isDual<T> || isExpr<T> || isNumber<T>);
    if constexpr (isDual<T>)
        return val(expr.val);
    else if constexpr (isExpr<T>)
        return val(eval(std::forward<T>(expr)));
    else return std::forward<T>(expr);
}

//=====================================================================================================================
//
// DERIVATIVE FUNCTIONS
//
//=====================================================================================================================

template<size_t order, typename T, typename G>
auto derivative(const Dual<T, G>& dual)
{
    if constexpr (order == 0)
        return val(dual.val);
    else if constexpr (order == 1)
        return val(dual.grad);
    else return derivative<order - 1>(dual.grad);
}

//=====================================================================================================================
//
// SEED FUNCTION
//
//=====================================================================================================================

/// Traverse down along the `val` branch until depth `order` is reached, then return the `grad` node.
template<size_t order, typename T, typename G>
auto& gradnode(Dual<T, G>& dual)
{
    constexpr auto N = Order<Dual<T, G>>;
    static_assert(order <= N);
    if constexpr (order == 0) return dual.val;
    else if constexpr (order == 1) return dual.grad;
    else return gradnode<order - 1>(dual.val);
}

/// Set the `grad` node of a dual number along the `val` branch at a depth `order`.
template<size_t order, typename T, typename G, typename U>
auto seed(Dual<T, G>& dual, U&& seedval)
{
    gradnode<order>(dual) = static_cast<NumericType<T>>(seedval);
}

//=====================================================================================================================
//
// CONVENIENT FUNCTIONS
//
//=====================================================================================================================

/// Alias template used to prevent expression nodes to be stored as references.
/// For example, the following should not exist `BinaryExpr<AddOp, const dual&, const UnaryExpr<NegOp, const dual&>&>>`.
/// It should be instead `BinaryExpr<AddOp, const dual&, UnaryExpr<NegOp, const dual&>>`.
/// This alias template allows only dual numbers to have their original type.
/// All other types become plain, without reference and const attributes.
template<typename T>
using PreventExprRef = std::conditional_t<isDual<T>, T, PlainType<T>>;

//-----------------------------------------------------------------------------
// NEGATIVE EXPRESSION GENERATOR FUNCTION
//-----------------------------------------------------------------------------
template<typename U>
constexpr auto negative(U&& expr)
{
    static_assert(isExpr<U> || isNumber<U>);
    if constexpr (isNegExpr<U>)
        return inner(expr);
    else return NegExpr<PreventExprRef<U>>{ expr };
}

//-----------------------------------------------------------------------------
// INVERSE EXPRESSION GENERATOR FUNCTION
//-----------------------------------------------------------------------------
template<typename U>
constexpr auto inverse(U&& expr)
{
    static_assert(isExpr<U>);
    if constexpr (isInvExpr<U>)
        return inner(expr);
    else return InvExpr<PreventExprRef<U>>{ expr };
}

//-----------------------------------------------------------------------------
// AUXILIARY CONSTEXPR CONSTANTS
//-----------------------------------------------------------------------------
template<typename U>
constexpr auto Zero = static_cast<DualValueType<U>>(0);

template<typename U>
constexpr auto One = static_cast<DualValueType<U>>(1);

//=====================================================================================================================
//
// POSITIVE ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// POSITIVE OPERATOR: +x
//-----------------------------------------------------------------------------
template<typename R, EnableIf<isExpr<R>>...>
constexpr auto operator+(R&& expr)
{
    return std::forward<R>(expr); // expression optimization: +(expr) => expr
}

//=====================================================================================================================
//
// NEGATIVE ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

template<typename R, EnableIf<isExpr<R>>...>
constexpr auto operator-(R&& expr)
{
    // NEGATIVE EXPRESSION CASE: -(-x) => x when expr is (-x)
    if constexpr (isNegExpr<R>)
        return inner(expr);
    // NEGATIVE EXPRESSION CASE: -(number * dual) => (-number) * dual
    else if constexpr (isNumberDualMulExpr<R>)
        return (-left(expr)) * right(expr);
    // default expression
    else return NegExpr<PreventExprRef<R>>{ expr };
}

//=====================================================================================================================
//
// ADDITION ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

template<typename L, typename R, EnableIf<isOperable<L, R>>...>
constexpr auto operator+(L&& l, R&& r)
{
    // ADDITION EXPRESSION CASE: (-x) + (-y) => -(x + y)
    if constexpr (isNegExpr<L> && isNegExpr<R>)
        return -( inner(l) + inner(r) );
    // ADDITION EXPRESSION CASE: expr + number => number + expr (number always on the left)
    else if constexpr (isExpr<L> && isNumber<R>)
        return std::forward<R>(r) + std::forward<L>(l);
    // DEFAULT ADDITION EXPRESSION
    else return AddExpr<PreventExprRef<L>, PreventExprRef<R>>{ l, r };
}

//=====================================================================================================================
//
// MULTIPLICATION ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

template<typename L, typename R, EnableIf<isOperable<L, R>>...>
constexpr auto operator*(L&& l, R&& r)
{
    // MULTIPLICATION EXPRESSION CASE: (-expr) * (-expr) => expr * expr
    if constexpr (isNegExpr<L> && isNegExpr<R>)
        return inner(l) * inner(r);
    // // MULTIPLICATION EXPRESSION CASE: (1 / expr) * (1 / expr) => 1 / (expr * expr)
    else if constexpr (isInvExpr<L> && isInvExpr<R>)
        return inverse(inner(l) * inner(r));
    // // MULTIPLICATION EXPRESSION CASE: expr * number => number * expr
    else if constexpr (isExpr<L> && isNumber<R>)
        return std::forward<R>(r) * std::forward<L>(l);
    // // MULTIPLICATION EXPRESSION CASE: number * (-expr) => (-number) * expr
    else if constexpr (isNumber<L> && isNegExpr<R>)
        return (-l) * inner(r);
    // // MULTIPLICATION EXPRESSION CASE: number * (number * expr) => (number * number) * expr
    else if constexpr (isNumber<L> && isNumberDualMulExpr<R>)
        return (l * left(r)) * right(r);
    // MULTIPLICATION EXPRESSION CASE: number * dual => NumberDualMulExpr
    else if constexpr (isNumber<L> && isDual<R>)
        return NumberDualMulExpr<PreventExprRef<L>, PreventExprRef<R>>{ l, r };
    // DEFAULT MULTIPLICATION EXPRESSION: expr * expr => MulExpr
    else return MulExpr<PreventExprRef<L>, PreventExprRef<R>>{ l, r };
}

//=====================================================================================================================
//
// SUBTRACTION ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// SUBTRACTION OPERATOR: expr - expr, scalar - expr, expr - scalar
//-----------------------------------------------------------------------------
template<typename L, typename R, EnableIf<isOperable<L, R>>...>
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
template<typename L, typename R, EnableIf<isOperable<L, R>>...>
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

template<typename R, EnableIf<isExpr<R>>...> constexpr auto sin(R&& r) -> SinExpr<R> { return { r }; }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto cos(R&& r) -> CosExpr<R> { return { r }; }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto tan(R&& r) -> TanExpr<R> { return { r }; }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto asin(R&& r) -> ArcSinExpr<R> { return { r }; }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto acos(R&& r) -> ArcCosExpr<R> { return { r }; }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto atan(R&& r) -> ArcTanExpr<R> { return { r }; }

//=====================================================================================================================
//
// HYPERBOLIC FUNCTIONS OVERLOADING
//
//=====================================================================================================================

template<typename R, EnableIf<isExpr<R>>...> constexpr auto sinh(R&& r) -> SinhExpr<R> { return { r }; }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto cosh(R&& r) -> CoshExpr<R> { return { r }; }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto tanh(R&& r) -> TanhExpr<R> { return { r }; }

//=====================================================================================================================
//
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS OVERLOADING
//
//=====================================================================================================================

template<typename R, EnableIf<isExpr<R>>...> constexpr auto exp(R&& r) -> ExpExpr<R> { return { r }; }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto log(R&& r) -> LogExpr<R> { return { r }; }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto log10(R&& r) -> Log10Expr<R> { return { r }; }

//=====================================================================================================================
//
// POWER FUNCTIONS OVERLOADING
//
//=====================================================================================================================

template<typename L, typename R, EnableIf<isOperable<L, R>>...> constexpr auto pow(L&& l, R&& r) -> PowExpr<L, R> { return { l, r }; }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto sqrt(R&& r) -> SqrtExpr<R> { return { r }; }

//=====================================================================================================================
//
// OTHER FUNCTIONS OVERLOADING
//
//=====================================================================================================================

template<typename R, EnableIf<isExpr<R>>...> constexpr auto abs(R&& r) -> AbsExpr<R> { return { r }; }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto abs2(R&& r) { return std::forward<R>(r) * std::forward<R>(r); }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto conj(R&& r) { return std::forward<R>(r); }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto real(R&& r) { return std::forward<R>(r); }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto imag(R&& r) { return 0.0; }

//=====================================================================================================================
//
// COMPARISON OPERATORS OVERLOADING
//
//=====================================================================================================================

template<typename L, typename R, EnableIf<isOperable<L, R>>...> bool operator==(L&& l, R&& r) { return val(l) == val(r); }
template<typename L, typename R, EnableIf<isOperable<L, R>>...> bool operator!=(L&& l, R&& r) { return val(l) != val(r); }
template<typename L, typename R, EnableIf<isOperable<L, R>>...> bool operator<=(L&& l, R&& r) { return val(l) <= val(r); }
template<typename L, typename R, EnableIf<isOperable<L, R>>...> bool operator>=(L&& l, R&& r) { return val(l) >= val(r); }
template<typename L, typename R, EnableIf<isOperable<L, R>>...> bool operator<(L&& l, R&& r) { return val(l) < val(r); }
template<typename L, typename R, EnableIf<isOperable<L, R>>...> bool operator>(L&& l, R&& r) { return val(l) > val(r); }

//=====================================================================================================================
//
// AUXILIARY FUNCTIONS
//
//=====================================================================================================================
template<typename T, typename G>
constexpr void negate(Dual<T, G>& self)
{
    self.val = -self.val;
    self.grad = -self.grad;
}

template<typename T, typename G, typename U>
constexpr void scale(Dual<T, G>& self, const U& scalar)
{
    self.val *= scalar;
    self.grad *= scalar;
}

//=====================================================================================================================
//
// FORWARD DECLARATIONS
//
//=====================================================================================================================

template<typename Op, typename T, typename G>
constexpr void apply(Dual<T, G>& self);

//=====================================================================================================================
//
// ASSIGNMENT FUNCTIONS
//
//=====================================================================================================================

template<typename T, typename G, typename U>
constexpr void assign(Dual<T, G>& self, U&& other)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN A NUMBER: self = number
    if constexpr (isNumber<U>) {
        self.val = other;
        self.grad = Zero<T>;
    }
    // ASSIGN A DUAL NUMBER: self = dual
    else if constexpr (isDual<U>) {
        self.val = other.val;
        self.grad = other.grad;
    }
    // ASSIGN A NUMBER-DUAL MULTIPLICATION EXPRESSION: self = number * dual
    else if constexpr (isNumberDualMulExpr<U>) {
        assign(self, other.r);
        scale(self, other.l);
    }
    // ASSIGN A UNARY EXPRESSION: self = unaryexpr
    else if constexpr (isUnaryExpr<U>) {
        using Op = DualOpType<U>;
        assign(self, other.r);
        apply<Op>(self);
    }
    // ASSIGN AN ADDITION EXPRESSION: self = expr + expr
    else if constexpr (isAddExpr<U>) {
        assign(self, other.r);
        assignAdd(self, other.l);
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
}

template<typename T, typename G, typename U>
constexpr void assign(Dual<T, G>& self, U&& other, Dual<T, G>& tmp)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN AN UNARY EXPRESSION: self = func(expr)
    if constexpr (isUnaryExpr<U>) {
        using Op = DualOpType<U>;
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

template<typename T, typename G, typename U>
constexpr void assignAdd(Dual<T, G>& self, U&& other)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN-ADD A NUMBER: self += number
    if constexpr (isNumber<U>) {
        self.val += other;
    }
    // ASSIGN-ADD A DUAL NUMBER: self += dual
    else if constexpr (isDual<U>) {
        self.val += other.val;
        self.grad += other.grad;
    }
    // ASSIGN-ADD A NEGATIVE EXPRESSION: self += -expr => self -= expr
    else if constexpr (isNegExpr<U>) {
        assignSub(self, other.r);
    }
    // ASSIGN-ADD A NUMBER-DUAL MULTIPLICATION EXPRESSION: self += number * dual
    else if constexpr (isNumberDualMulExpr<U>) {
        self.val += other.l * other.r.val;
        self.grad += other.l * other.r.grad;
    }
    // ASSIGN-ADD AN ADDITION EXPRESSION: self += expr + expr
    else if constexpr (isAddExpr<U>) {
        assignAdd(self, other.l);
        assignAdd(self, other.r);
    }
    // ASSIGN-ADD ALL OTHER EXPRESSIONS
    else {
        Dual<T, G> tmp;
        assignAdd(self, std::forward<U>(other), tmp);
    }
}

template<typename T, typename G, typename U>
constexpr void assignAdd(Dual<T, G>& self, U&& other, Dual<T, G>& tmp)
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

template<typename T, typename G, typename U>
constexpr void assignSub(Dual<T, G>& self, U&& other)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN-SUBTRACT A NUMBER: self -= number
    if constexpr (isNumber<U>) {
        self.val -= other;
    }
    // ASSIGN-SUBTRACT A DUAL NUMBER: self -= dual
    else if constexpr (isDual<U>) {
        self.val -= other.val;
        self.grad -= other.grad;
    }
    // ASSIGN-SUBTRACT A NEGATIVE EXPRESSION: self -= -expr => self += expr
    else if constexpr (isNegExpr<U>) {
        assignAdd(self, other.r);
    }
    // ASSIGN-SUBTRACT A NUMBER-DUAL MULTIPLICATION EXPRESSION: self -= number * dual
    else if constexpr (isNumberDualMulExpr<U>) {
        self.val -= other.l * other.r.val;
        self.grad -= other.l * other.r.grad;
    }
    // ASSIGN-SUBTRACT AN ADDITION EXPRESSION: self -= expr + expr
    else if constexpr (isAddExpr<U>) {
        assignSub(self, other.l);
        assignSub(self, other.r);
    }
    // ASSIGN-SUBTRACT ALL OTHER EXPRESSIONS
    else {
        Dual<T, G> tmp;
        assignSub(self, std::forward<U>(other), tmp);
    }
}

template<typename T, typename G, typename U>
constexpr void assignSub(Dual<T, G>& self, U&& other, Dual<T, G>& tmp)
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

template<typename T, typename G, typename U>
constexpr void assignMul(Dual<T, G>& self, U&& other)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN-MULTIPLY A NUMBER: self *= number
    if constexpr (isNumber<U>) {
        self.val *= other;
        self.grad *= other;
    }
    // ASSIGN-MULTIPLY A DUAL NUMBER: self *= dual
    else if constexpr (isDual<U>) {
        const G aux = other.grad; // to avoid aliasing when self === other
        self.grad *= other.val;
        self.grad += self.val * aux;
        self.val *= other.val;
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
        Dual<T, G> tmp;
        assignMul(self, std::forward<U>(other), tmp);
    }
}

template<typename T, typename G, typename U>
constexpr void assignMul(Dual<T, G>& self, U&& other, Dual<T, G>& tmp)
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

template<typename T, typename G, typename U>
constexpr void assignDiv(Dual<T, G>& self, U&& other)
{
    static_assert(isExpr<U> || isNumber<U>);

    // ASSIGN-DIVIDE A NUMBER: self /= number
    if constexpr (isNumber<U>) {
        self.val /= other;
        self.grad /= other;
    }
    // ASSIGN-DIVIDE A DUAL NUMBER: self /= dual
    else if constexpr (isDual<U>) {
        const T aux = One<T> / other.val; // to avoid aliasing when self === other
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
        Dual<T, G> tmp;
        assignDiv(self, std::forward<U>(other), tmp);
    }
}

template<typename T, typename G, typename U>
constexpr void assignDiv(Dual<T, G>& self, U&& other, Dual<T, G>& tmp)
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

template<typename T, typename G, typename U>
constexpr void assignPow(Dual<T, G>& self, U&& other)
{
    // ASSIGN-POW A NUMBER: self = pow(self, number)
    if constexpr (isNumber<U>) {
        const T aux = pow(self.val, other - 1);
        self.grad *= other * aux;
        self.val = aux * self.val;
    }
    // ASSIGN-POW A DUAL NUMBER: self = pow(self, dual)
    else if constexpr (isDual<U>) {
        const T aux1 = pow(self.val, other.val);
        const T aux2 = log(self.val);
        self.grad *= other.val/self.val;
        self.grad += aux2 * other.grad;
        self.grad *= aux1;
        self.val = aux1;
    }
    // ASSIGN-POW ALL OTHER EXPRESSIONS: self = pow(self, expr)
    else {
        Dual<T, G> tmp;
        assignPow(self, std::forward<U>(other), tmp);
    }
}

template<typename T, typename G, typename U>
constexpr void assignPow(Dual<T, G>& self, U&& other, Dual<T, G>& tmp)
{
    assign(tmp, other);
    assignPow(self, tmp);
}

//=====================================================================================================================
//
// APPLY-OPERATOR FUNCTIONS
//
//=====================================================================================================================
template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, NegOp)
{
    self.val = -self.val;
    self.grad = -self.grad;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, InvOp)
{
    self.val = One<T> / self.val;
    self.grad *= - self.val * self.val;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, SinOp)
{
    self.grad *= cos(self.val);
    self.val = sin(self.val);
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, CosOp)
{
    self.grad *= -sin(self.val);
    self.val = cos(self.val);
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, TanOp)
{
    const T aux = One<T> / cos(self.val);
    self.val = tan(self.val);
    self.grad *= aux * aux;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, SinhOp)
{
    self.grad *= cosh(self.val);
    self.val = sinh(self.val);
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, CoshOp)
{
    self.grad *= sinh(self.val);
    self.val = cosh(self.val);
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, TanhOp)
{
    const T aux = One<T> / cosh(self.val);
    self.val = tanh(self.val);
    self.grad *= aux * aux;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, ArcSinOp)
{
    const T aux = One<T> / sqrt(1.0 - self.val * self.val);
    self.val = asin(self.val);
    self.grad *= aux;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, ArcCosOp)
{
    const T aux = -One<T> / sqrt(1.0 - self.val * self.val);
    self.val = acos(self.val);
    self.grad *= aux;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, ArcTanOp)
{
    const T aux = One<T> / (1.0 + self.val * self.val);
    self.val = atan(self.val);
    self.grad *= aux;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, ExpOp)
{
    self.val = exp(self.val);
    self.grad *= self.val;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, LogOp)
{
    const T aux = One<T> / self.val;
    self.val = log(self.val);
    self.grad *= aux;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, Log10Op)
{
    constexpr auto ln10 = 2.3025850929940456840179914546843;
    const T aux = One<T> / (ln10 * self.val);
    self.val = log10(self.val);
    self.grad *= aux;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, SqrtOp)
{
    self.val = sqrt(self.val);
    self.grad *= 0.5 / self.val;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, AbsOp)
{
    const T aux = self.val;
    self.val = abs(self.val);
    self.grad *= aux / self.val;
}

template<typename Op, typename T, typename G>
constexpr void apply(Dual<T, G>& self)
{
    apply(self, Op{});
}

template<typename T, typename G>
std::ostream& operator<<(std::ostream& out, const Dual<T, G>& x)
{
    out << x.val;
    return out;
}

//=====================================================================================================================
//
// NUMBER TRAITS DEFINITION
//
//=====================================================================================================================

template<typename T, typename G>
struct NumberTraits<Dual<T, G>>
{
    /// The underlying floating point type of Dual<T, G>.
    using NumericType = DualValueType<T>;

    /// The order of Dual<T, G>.
    static constexpr auto Order = 1 + NumberTraits<PlainType<T>>::Order;
};

//=====================================================================================================================
//
// HIGHER-ORDER DUAL NUMBERS
//
//=====================================================================================================================

template<size_t N, typename T>
struct AuxHigherOrderDual;

template<typename T>
struct AuxHigherOrderDual<0, T>
{
    using type = T;
};

template<size_t N, typename T>
struct AuxHigherOrderDual
{
    using type = Dual<typename AuxHigherOrderDual<N - 1, T>::type, typename AuxHigherOrderDual<N - 1, T>::type>;
};

template<size_t N, typename T>
using HigherOrderDual = typename AuxHigherOrderDual<N, T>::type;

} // namespace detail

using detail::val;
using detail::eval;
using detail::Dual;
using detail::HigherOrderDual;

using dual0th = HigherOrderDual<0, double>;
using dual1st = HigherOrderDual<1, double>;
using dual2nd = HigherOrderDual<2, double>;
using dual3rd = HigherOrderDual<3, double>;
using dual4th = HigherOrderDual<4, double>;

using dual = dual1st;

} // namespace autodiff
