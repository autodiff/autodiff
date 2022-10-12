//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2022 Allan Leal
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
#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>
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
using std::atan2;
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
using std::erf;
using std::hypot;

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
struct NegOp     {};  // NEGATIVE OPERATOR
struct InvOp     {};  // INVERSE OPERATOR
struct SinOp     {};  // SINE OPERATOR
struct CosOp     {};  // COSINE OPERATOR
struct TanOp     {};  // TANGENT OPERATOR
struct SinhOp    {};  // HYPERBOLIC SINE OPERATOR
struct CoshOp    {};  // HYPERBOLIC COSINE OPERATOR
struct TanhOp    {};  // HYPERBOLIC TANGENT OPERATOR
struct ArcSinOp  {};  // ARC SINE OPERATOR
struct ArcCosOp  {};  // ARC COSINE OPERATOR
struct ArcTanOp  {};  // ARC TANGENT OPERATOR
struct ArcTan2Op {};  // 2-ARGUMENT ARC TANGENT OPERATOR
struct ExpOp     {};  // EXPONENTIAL OPERATOR
struct LogOp     {};  // NATURAL LOGARITHM OPERATOR
struct Log10Op   {};  // BASE-10 LOGARITHM OPERATOR
struct SqrtOp    {};  // SQUARE ROOT OPERATOR
struct PowOp     {};  // POWER OPERATOR
struct AbsOp     {};  // ABSOLUTE OPERATOR
struct ErfOp     {};  // ERROR FUNCTION OPERATOR
struct Hypot2Op  {};  // 2D HYPOT OPERATOR
struct Hypot3Op  {};  // 3D HYPOT OPERATOR

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

template<typename L, typename R>
using ArcTan2Expr = BinaryExpr<ArcTan2Op, L, R>;

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

template<typename R>
using ErfExpr = UnaryExpr<ErfOp, R>;

template<typename L, typename R>
using Hypot2Expr = BinaryExpr<Hypot2Op, L, R>;

template<typename L, typename C, typename R>
using Hypot3Expr = TernaryExpr<Hypot3Op, L, C, R>;

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
// IS TYPE T A ARCTAN2 EXPRESSION?
//-----------------------------------------------------------------------------
template<typename T>
struct isArcTan2Expr { constexpr static bool value = false; };

template<typename L, typename R>
struct isArcTan2Expr<ArcTan2Expr<L, R>> { constexpr static bool value = true; };

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

//-----------------------------------------------------------------------------
// IS TYPE T A HYPOT EXPRESSION?
//-----------------------------------------------------------------------------
template<typename T>
struct isHypot2Expr { constexpr static bool value = false; };

template<typename L, typename R>
struct isHypot2Expr<Hypot2Expr<L, R>> { constexpr static bool value = true; };

template<typename T>
struct isHypot3Expr { constexpr static bool value = false; };

template<typename L, typename C, typename R>
struct isHypot3Expr<Hypot3Expr<L, C, R>> { constexpr static bool value = true; };

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
constexpr bool isArcTan2Expr = traits::isArcTan2Expr<PlainType<T>>::value;

template<typename T>
constexpr bool isHypot2Expr = traits::isHypot2Expr<PlainType<T>>::value;

template<typename T>
constexpr bool isHypot3Expr = traits::isHypot3Expr<PlainType<T>>::value;

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
constexpr bool isOperable = (isExpr<L> && isExpr<R>) || (isArithmetic<L> && isExpr<R>) || (isExpr<L> && isArithmetic<R>);

template<typename L, typename C, typename R>
constexpr bool isOperable3 = (isOperable<L,C> && isOperable<L,R>) || (isOperable<C,L> && isOperable<C,R>) || (isOperable<R,L> && isOperable<R,C>);

//-----------------------------------------------------------------------------
// VALUE, GRAD, AND OP TYPES IN DUAL EXPRESSIONS
//-----------------------------------------------------------------------------

template<typename T> struct AuxDualType;
template<typename T> struct AuxDualOpType;
template<typename L, typename R> struct AuxCommonDualType;


template<typename T> struct DualTypeNotDefinedFor {};
template<typename T> struct DualOpTypeNotDefinedFor {};
template<typename L, typename R> struct CommonDualTypeNotDefinedFor {};


template<typename T> using DualType = typename AuxDualType<PlainType<T>>::type;
template<typename T> using DualOpType = typename AuxDualOpType<PlainType<T>>::type;
template<typename L, typename R> using CommonDualType = typename AuxCommonDualType<PlainType<L>, PlainType<R>>::type;


template<typename T>
struct AuxDualType { using type = ConditionalType<isArithmetic<T>, T, DualTypeNotDefinedFor<T>>; };

template<typename T, typename G>
struct AuxDualType<Dual<T, G>> { using type = Dual<T, G>; };

template<typename Op, typename R>
struct AuxDualType<UnaryExpr<Op, R>> { using type = DualType<R>; };

template<typename Op, typename L, typename R>
struct AuxDualType<BinaryExpr<Op, L, R>> { using type = CommonDualType<L, R>; };

template<typename Op, typename L, typename C, typename R>
struct AuxDualType<TernaryExpr<Op, L, C, R>> { using type = CommonDualType<L, CommonDualType<C, R>>; };


template<typename T>
struct AuxDualOpType { using type = DualOpTypeNotDefinedFor<T>; };

template<typename Op, typename R>
struct AuxDualOpType<UnaryExpr<Op, R>> { using type = Op; };

template<typename Op, typename L, typename R>
struct AuxDualOpType<BinaryExpr<Op, L, R>> { using type = Op; };

template<typename Op, typename L, typename C, typename R>
struct AuxDualOpType<TernaryExpr<Op, L, C, R>> { using type = Op; };

template<typename L, typename R>
constexpr auto auxCommonDualType()
{
    if constexpr (isArithmetic<L> && isArithmetic<R>)
        return CommonType<L, R>();
    else if constexpr (isExpr<L> && isArithmetic<R>)
        return DualType<L>();
    else if constexpr (isArithmetic<L> && isExpr<R>)
        return DualType<R>();
    else if constexpr (isExpr<L> && isExpr<R>) {
        using DualTypeL = DualType<L>;
        using DualTypeR = DualType<R>;
        static_assert(isSame<DualTypeL, DualTypeR>);
        return DualTypeL();
    }
    else return CommonDualTypeNotDefinedFor<L, R>();
}

template<typename L, typename R>
struct AuxCommonDualType { using type = decltype(auxCommonDualType<L, R>()); };

//=====================================================================================================================
//
// EXPRESSION TYPES DEFINITION
//
//=====================================================================================================================

template<typename T, typename G>
struct Dual
{
    T val = {};

    G grad = {};

    Dual()
    {}

    template<typename U, EnableIf<isExpr<U> || isArithmetic<U>>...>
    Dual(U&& other)
    {
        assign(*this, std::forward<U>(other));
    }

    template<typename U, EnableIf<isExpr<U> || isArithmetic<U>>...>
    Dual& operator=(U&& other)
    {
        Dual tmp;
        assign(tmp, std::forward<U>(other));
        assign(*this, tmp);
        return *this;
    }

    template<typename U, EnableIf<isExpr<U> || isArithmetic<U>>...>
    Dual& operator+=(U&& other)
    {
        Dual tmp;
        assign(tmp, std::forward<U>(other));
        assignAdd(*this, tmp);
        return *this;
    }

    template<typename U, EnableIf<isExpr<U> || isArithmetic<U>>...>
    Dual& operator-=(U&& other)
    {
        Dual tmp;
        assign(tmp, std::forward<U>(other));
        assignSub(*this, tmp);
        return *this;
    }

    template<typename U, EnableIf<isExpr<U> || isArithmetic<U>>...>
    Dual& operator*=(U&& other)
    {
        Dual tmp;
        assign(tmp, std::forward<U>(other));
        assignMul(*this, tmp);
        return *this;
    }

    template<typename U, EnableIf<isExpr<U> || isArithmetic<U>>...>
    Dual& operator/=(U&& other)
    {
        Dual tmp;
        assign(tmp, std::forward<U>(other));
        assignDiv(*this, tmp);
        return *this;
    }

    /// Convert this Dual number into a value of type @p U.
#if defined(AUTODIFF_ENABLE_IMPLICIT_CONVERSION_DUAL) || defined(AUTODIFF_ENABLE_IMPLICIT_CONVERSION)
    operator T() const { return val; }

    template<typename U>
    operator U() const { return static_cast<U>(val); }
#else
    explicit operator T() const { return val; }

    template<typename U>
    explicit operator U() const { return static_cast<U>(val); }
#endif
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

template<typename Op, typename L, typename C, typename R>
struct TernaryExpr
{
    L l;
    C c;
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
    static_assert(isDual<T> || isExpr<T> || isArithmetic<T>);
    if constexpr (isDual<T>)
        return std::forward<T>(expr);
    else if constexpr (isExpr<T>)
        return DualType<T>(std::forward<T>(expr));
    else return std::forward<T>(expr);
}

template<typename T>
auto val(T&& expr)
{
    static_assert(isDual<T> || isExpr<T> || isArithmetic<T>);
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

template<size_t order = 1, typename T, typename G>
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
using PreventExprRef = ConditionalType<isDual<T>, T, PlainType<T>>;

//-----------------------------------------------------------------------------
// NEGATIVE EXPRESSION GENERATOR FUNCTION
//-----------------------------------------------------------------------------
template<typename U>
constexpr auto negative(U&& expr)
{
    static_assert(isExpr<U> || isArithmetic<U>);
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
constexpr auto Zero() { return static_cast<NumericType<U>>(0); }

template<typename U>
constexpr auto One() { return static_cast<NumericType<U>>(1); }

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
    else if constexpr (isExpr<L> && isArithmetic<R>)
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
    else if constexpr (isExpr<L> && isArithmetic<R>)
        return std::forward<R>(r) * std::forward<L>(l);
    // // MULTIPLICATION EXPRESSION CASE: number * (-expr) => (-number) * expr
    else if constexpr (isArithmetic<L> && isNegExpr<R>)
        return (-l) * inner(r);
    // // MULTIPLICATION EXPRESSION CASE: number * (number * expr) => (number * number) * expr
    else if constexpr (isArithmetic<L> && isNumberDualMulExpr<R>)
        return (l * left(r)) * right(r);
    // MULTIPLICATION EXPRESSION CASE: number * dual => NumberDualMulExpr
    else if constexpr (isArithmetic<L> && isDual<R>)
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
    if constexpr (isArithmetic<R>)
        return std::forward<L>(l) * (One<L>() / std::forward<R>(r));
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
template<typename L, typename R, EnableIf<isOperable<L, R>>...> constexpr auto atan2(L&& l, R&& r) -> ArcTan2Expr<L, R> { return { l, r }; }
template<typename L, typename R, EnableIf<isOperable<L, R>>...> constexpr auto hypot(L&& l, R&& r) -> Hypot2Expr<L, R> { return { l, r }; }
template<typename L, typename C, typename R, EnableIf<isOperable3<L, C, R>>...>
    constexpr auto hypot(L&& l, C&& c, R&& r) -> Hypot3Expr<L, C, R> { return { l, c, r }; }

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
template<typename R, EnableIf<isExpr<R>>...> constexpr auto imag(R&&) { return 0.0; }
template<typename R, EnableIf<isExpr<R>>...> constexpr auto erf(R&& r) -> ErfExpr<R> { return { r }; }

template<typename L, typename R, EnableIf<isOperable<L, R>>...>
constexpr auto min(L&& l, R&& r)
{
    const auto x = eval(l);
    const auto y = eval(r);
    return (x <= y) ? x : y;
}

template<typename L, typename R, EnableIf<isOperable<L, R>>...>
constexpr auto max(L&& l, R&& r)
{
    const auto x = eval(l);
    const auto y = eval(r);
    return (x >= y) ? x : y;
}

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
    static_assert(isExpr<U> || isArithmetic<U>);

    // ASSIGN A NUMBER: self = number
    if constexpr (isArithmetic<U>) {
        self.val = other;
        self.grad = Zero<T>();
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
    // ASSIGN A ATAN2 EXPRESSION: self = atan2(expr, expr)
    else if constexpr (isArcTan2Expr<U>) {
        assignArcTan2(self, other.l, other.r);
    }

    // ASSIGN A HYPOT2 EXPRESSION: self = hypot(expr, expr)
    else if constexpr (isHypot2Expr<U>) {
        assignHypot2(self, other.l, other.r);
    }

    // ASSIGN A HYPOT3 EXPRESSION: self = hypot(expr, expr)
    else if constexpr (isHypot3Expr<U>) {
        assignHypot3(self, other.l, other.c, other.r);
    }
}

template<typename T, typename G, typename U>
constexpr void assign(Dual<T, G>& self, U&& other, Dual<T, G>& tmp)
{
    static_assert(isExpr<U> || isArithmetic<U>);

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
    static_assert(isExpr<U> || isArithmetic<U>);

    // ASSIGN-ADD A NUMBER: self += number
    if constexpr (isArithmetic<U>) {
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
    static_assert(isExpr<U> || isArithmetic<U>);

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
    static_assert(isExpr<U> || isArithmetic<U>);

    // ASSIGN-SUBTRACT A NUMBER: self -= number
    if constexpr (isArithmetic<U>) {
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
    static_assert(isExpr<U> || isArithmetic<U>);

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
    static_assert(isExpr<U> || isArithmetic<U>);

    // ASSIGN-MULTIPLY A NUMBER: self *= number
    if constexpr (isArithmetic<U>) {
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
    static_assert(isExpr<U> || isArithmetic<U>);

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
    static_assert(isExpr<U> || isArithmetic<U>);

    // ASSIGN-DIVIDE A NUMBER: self /= number
    if constexpr (isArithmetic<U>) {
        self.val /= other;
        self.grad /= other;
    }
    // ASSIGN-DIVIDE A DUAL NUMBER: self /= dual
    else if constexpr (isDual<U>) {
        const T aux = One<T>() / other.val; // to avoid aliasing when self === other
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
    static_assert(isExpr<U> || isArithmetic<U>);

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
    if constexpr (isArithmetic<U>) {
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
// ASSIGNMENT-ARCTAN2 FUNCTION
//
//=====================================================================================================================

template<typename T, typename G, typename Y, typename X>
constexpr void assignArcTan2(Dual<T, G>& self, Y&&y, X&&x)
{
    static_assert(isArithmetic<Y> || isExpr<Y>);
    static_assert(isArithmetic<X> || isExpr<X>);

    // self = atan2(number, dual)
    if constexpr (isArithmetic<Y> && isDual<X>) {
        self.val = atan2(y, x.val);
        self.grad = -y/(y*y + x.val * x.val) * x.grad;
    }

    // self = atan2(dual, number)
    else if constexpr (isDual<Y> && isArithmetic<X>) {
        self.val = atan2(y.val, x);
        self.grad = x/(y.val * y.val + x * x) * y.grad;
    }

    // self = atan2(dual, dual)
    else if constexpr (isDual<Y> && isDual<X>) {
        self.val = atan2(y.val, x.val);
        self.grad = (x.val * y.grad - y.val * x.grad)/(y.val * y.val + x.val * x.val);
    }

    // self = atan2(expr, .)
    else if constexpr (!isDual<Y> && !isArithmetic<Y>) {
        Dual<T, G> y_tmp;
        assign(y_tmp, std::forward<Y>(y));
        assignArcTan2(self, std::move(y_tmp), std::forward<X>(x));
    }

    // self = atan2(., expr)
    else {
        Dual<T, G> x_tmp;
        assign(x_tmp, std::forward<X>(x));
        assignArcTan2(self, std::forward<Y>(y), std::move(x_tmp));
    }
}

//=====================================================================================================================
//
// ASSIGNMENT-HYPOT FUNCTION
//
//=====================================================================================================================

template<typename T, typename G, typename X, typename Y>
constexpr void assignHypot2(Dual<T, G>& self, X&& x, Y&& y)
{
    static_assert(isArithmetic<X> || isExpr<X>);
    static_assert(isArithmetic<Y> || isExpr<Y>);

    // self = hypot(number, dual)
    if constexpr (isDual<X> && isArithmetic<Y>) {
        self.val = hypot(x.val, y);
        self.grad = x.val / self.val * x.grad;
    }

    // self = hypot(dual, number)
    else if constexpr (isArithmetic<X> && isDual<Y>) {
        self.val = hypot(x, y.val);
        self.grad = y.val / self.val * y.grad;
    }

    // self = hypot(dual, dual)
    else if constexpr (isDual<X> && isDual<Y>) {
        self.val = hypot(x.val, y.val);
        self.grad = (x.grad * x.val + y.grad * y.val) / self.val;
    }

    // self = hypot(expr, .)
    else if constexpr (!isDual<X> && !isArithmetic<X>) {
        Dual<T, G> x_tmp;
        assign(x_tmp, std::forward<X>(x));
        assignHypot2(self, std::move(x_tmp), std::forward<Y>(y));
    }

    // self = hypot(., expr)
    else {
        Dual<T, G> y_tmp;
        assign(y_tmp, std::forward<Y>(y));
        assignHypot2(self, std::forward<X>(x), std::move(y_tmp));
    }
}

template<typename T, typename G, typename X, typename Y, typename Z>
constexpr void assignHypot3(Dual<T, G>& self, X&& x, Y&& y, Z&& z)
{
    static_assert(isArithmetic<X> || isExpr<X>);
    static_assert(isArithmetic<Y> || isExpr<Y>);
    static_assert(isArithmetic<Z> || isExpr<Z>);

    // self = hypot(dual, number, number)
    if constexpr (isDual<X> && isArithmetic<Y> && isArithmetic<Z>) {
        self.val = hypot(x.val, y, z);
        self.grad = x.val / self.val * x.grad;
    }

    // self = hypot(number, dual, number)
    else if constexpr (isArithmetic<X> && isDual<Y> && isArithmetic<Z>) {
        self.val = hypot(x, y.val, z);
        self.grad = y.val / self.val * y.grad;
    }

    // self = hypot(number, number, dual)
    else if constexpr (isArithmetic<X> && isArithmetic<Y> && isDual<Z>) {
        self.val = hypot(x, y, z.val);
        self.grad = z.val / self.val * z.grad;
    }

    // self = hypot(dual, dual, number)
    else if constexpr (isDual<X> && isDual<Y> && isArithmetic<Z>) {
        self.val = hypot(x.val, y.val, z);
        self.grad = (x.grad * x.val + y.grad * y.val ) / self.val;
    }

    // self = hypot(number, dual, dual)
    else if constexpr (isArithmetic<X> && isDual<Y> && isDual<Z>) {
        self.val = hypot(x, y.val, z.val);
        self.grad = (y.grad * y.val + z.grad * z.val) / self.val;
    }

    // self = hypot(dual, number, dual)
    else if constexpr (isDual<X> && isArithmetic<Y> && isDual<Z>) {
        self.val = hypot(x.val, y, z.val);
        self.grad = (x.grad * x.val + z.grad * z.val) / self.val;
    }

    // self = hypot(dual, dual, dual)
    else if constexpr (isDual<X> && isDual<Y> && isDual<Z>) {
        self.val = hypot(x.val, y.val, z.val);
        self.grad = (x.grad * x.val + y.grad * y.val + z.grad * z.val) / self.val;
    }

    // self = hypot(expr, ., .)
    else if constexpr (!isDual<X> && !isArithmetic<X>) {
        Dual<T, G> tmp;
        assign(tmp, std::forward<X>(x));
        assignHypot3(self, std::move(tmp), std::forward<Y>(y), std::forward<Z>(z));
    }

    // self = hypot(., expr, .)
    else if constexpr (!isDual<Y> && !isArithmetic<Y>) {
        Dual<T, G> tmp;
        assign(tmp, std::forward<Y>(y));
        assignHypot3(self, std::forward<X>(x), std::move(tmp), std::forward<Z>(z));
    }

    // self = hypot(., ., expr)
    else {
        Dual<T, G> tmp;
        assign(tmp, std::forward<Z>(z));
        assignHypot3(self, std::forward<X>(x), std::forward<Y>(y), std::move(tmp));
    }
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
    self.val = One<T>() / self.val;
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
    const T aux = One<T>() / cos(self.val);
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
    const T aux = One<T>() / cosh(self.val);
    self.val = tanh(self.val);
    self.grad *= aux * aux;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, ArcSinOp)
{
    const T aux = One<T>() / sqrt(1.0 - self.val * self.val);
    self.val = asin(self.val);
    self.grad *= aux;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, ArcCosOp)
{
    const T aux = -One<T>() / sqrt(1.0 - self.val * self.val);
    self.val = acos(self.val);
    self.grad *= aux;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, ArcTanOp)
{
    const T aux = One<T>() / (1.0 + self.val * self.val);
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
    const T aux = One<T>() / self.val;
    self.val = log(self.val);
    self.grad *= aux;
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, Log10Op)
{
    constexpr NumericType<T> ln10 = 2.3025850929940456840179914546843;
    const T aux = One<T>() / (ln10 * self.val);
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
    self.grad *= self.val < T(0) ? G(-1) : (self.val > T(0) ? G(1) : G(0));
    self.val = abs(self.val);
}

template<typename T, typename G>
constexpr void apply(Dual<T, G>& self, ErfOp)
{
    constexpr NumericType<T> sqrt_pi = 1.7724538509055160272981674833411451872554456638435;
    const T aux = self.val;
    self.val = erf(aux);
    self.grad *= 2.0 * exp(-aux*aux)/sqrt_pi;
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

template<typename T, EnableIf<isArithmetic<T>>...>
auto reprAux(const T& x)
{
    std::stringstream ss; ss << x;
    return ss.str();
}

template<typename T, typename G>
auto reprAux(const Dual<T, G>& x)
{
    return "(" + reprAux(x.val) + ", " + reprAux(x.grad) + ")";
}

template<typename T, typename G>
auto repr(const Dual<T, G>& x)
{
    return "autodiff.dual" + reprAux(x);
}

//=====================================================================================================================
//
// NUMBER TRAITS DEFINITION
//
//=====================================================================================================================

template<typename T, typename G>
struct NumberTraits<Dual<T, G>>
{
    /// The dual type resulting from the evaluation of the expression (in case T is not double but an expression!).
    using ResultDualType = DualType<T>;

    /// The underlying floating point type of Dual<T, G>.
    using NumericType = typename NumberTraits<ResultDualType>::NumericType;

    /// The order of Dual<T, G>.
    static constexpr auto Order = 1 + NumberTraits<ResultDualType>::Order;
};

template<typename Op, typename R>
struct NumberTraits<UnaryExpr<Op, R>>
{
    /// The dual type resulting from the evaluation of the expression.
    using ResultDualType = DualType<UnaryExpr<Op, R>>;

    /// The underlying floating point type of UnaryExpr<Op, R>.
    using NumericType = typename NumberTraits<ResultDualType>::NumericType;

    /// The order of the expression UnaryExpr<Op, R> as the order of the evaluated dual type.
    static constexpr auto Order = NumberTraits<ResultDualType>::Order;
};

template<typename Op, typename L, typename R>
struct NumberTraits<BinaryExpr<Op, L, R>>
{
    /// The dual type resulting from the evaluation of the expression.
    using ResultDualType = DualType<BinaryExpr<Op, L, R>>;

    /// The underlying floating point type of BinaryExpr<Op, L, R>.
    using NumericType = typename NumberTraits<ResultDualType>::NumericType;

    /// The order of the expression BinaryExpr<Op, L, R> as the order of the evaluated dual type.
    static constexpr auto Order = NumberTraits<ResultDualType>::Order;
};

template<typename Op, typename L, typename C, typename R>
struct NumberTraits<TernaryExpr<Op, L, C, R>>
{
    /// The dual type resulting from the evaluation of the expression.
    using ResultDualType = DualType<TernaryExpr<Op, L, C, R>>;

    /// The underlying floating point type of TernaryExpr<Op, L, C, R>.
    using NumericType = typename NumberTraits<ResultDualType>::NumericType;

    /// The order of the expression TernaryExpr<Op, L, C, R> as the order of the evaluated dual type.
    static constexpr auto Order = NumberTraits<ResultDualType>::Order;
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
using detail::repr;
using detail::Dual;
using detail::HigherOrderDual;

using dual0th = HigherOrderDual<0, double>;
using dual1st = HigherOrderDual<1, double>;
using dual2nd = HigherOrderDual<2, double>;
using dual3rd = HigherOrderDual<3, double>;
using dual4th = HigherOrderDual<4, double>;

using dual = dual1st;

} // namespace autodiff
