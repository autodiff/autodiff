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

// C++ includes
#include <cmath>
#include <type_traits>
#include <utility>

namespace autodiff {
namespace forward {

//=====================================================================================================================
//
// ARITHMETIC AND FUNCTIONAL OPERATORS
//
//=====================================================================================================================

struct NegOp    {};  // NEGATIVE OPERATOR
struct InvOp    {};  // INVERSE OPERATOR
struct AddOp    {};  // ADDITION OPERATOR
struct SubOp    {};  // SUBTRACTION OPERATOR
struct MulOp    {};  // MULTIPLICATION OPERATOR
struct DivOp    {};  // DIVISION OPERATOR
struct SinOp    {};  // SINE OPERATOR
struct CosOp    {};  // COSINE OPERATOR
struct TanOp    {};  // TANGENT OPERATOR
struct ArcSinOp {};  // ARC SINE OPERATOR
struct ArcCosOp {};  // ARC COSINE OPERATOR
struct ArcTanOp {};  // ARC TANGENT OPERATOR
struct ExpOp    {};  // EXPONENTIAL OPERATOR
struct LogOp    {};  // NATURAL LOGARITHM OPERATOR
struct Log10Op  {};  // BASE-10 LOGARITHM OPERATOR
struct PowOp    {};  // POWER OPERATOR
struct SqrtOp   {};  // SQUARE ROOT OPERATOR
struct AbsOp    {};  // ABSOLUTE OPERATOR

//=====================================================================================================================
//
// EXPRESSION TYPES DECLARATION
//
//=====================================================================================================================

template<typename Derived>
struct Expr;

template<typename T>
struct Dual;

template<typename T>
struct ConstantExpr;

template<typename R, typename T>
struct ScaleExpr;

template<typename Op, typename R>
struct UnaryExpr;

template<typename Op, typename L, typename R>
struct BinaryExpr;

template<typename R>
using NegExpr = UnaryExpr<NegOp, R>;

template<typename R>
using InvExpr = UnaryExpr<InvOp, R>;

template<typename L, typename R>
using AddExpr = BinaryExpr<AddOp, L, R>;

template<typename L, typename R>
using MulExpr = BinaryExpr<MulOp, L, R>;

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

template<typename L, typename R>
using PowExpr = BinaryExpr<PowOp, L, R>;

template<typename R>
using SqrtExpr = UnaryExpr<SqrtOp, R>;

template<typename R>
using AbsExpr = UnaryExpr<AbsOp, R>;

//=====================================================================================================================
//
// TYPE TRAITS UTILITIES
//
//=====================================================================================================================

template<bool value>
using enableif = typename std::enable_if<value>::type;

template<bool value>
using disableif = typename std::enable_if<!value>::type;

template<typename T>
using plain = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template<typename A, typename B>
using common = typename std::common_type<A, B>::type;

namespace traits {

template<typename T>
struct isExpr { constexpr static bool value = std::is_base_of<Expr<plain<T>>, plain<T>>::value; };

template<typename Derived>
struct isExpr<Expr<Derived>> { constexpr static bool value = true; };

template<typename T>
struct isDual { constexpr static bool value = false; };

template<typename T>
struct isDual<Dual<T>> { constexpr static bool value = true; };

template<typename T>
struct isConstantExpr { constexpr static bool value = false; };

template<typename T>
struct isConstantExpr<ConstantExpr<T>> { constexpr static bool value = true; };

template<typename T>
struct isNegExpr { constexpr static bool value = false; };

template<typename T>
struct isNegExpr<NegExpr<T>> { constexpr static bool value = true; };

template<typename T>
struct isInvExpr { constexpr static bool value = false; };

template<typename T>
struct isInvExpr<InvExpr<T>> { constexpr static bool value = true; };

template<typename T>
struct isScaleExpr { constexpr static bool value = false; };

template<typename T, typename R>
struct isScaleExpr<ScaleExpr<T, R>> { constexpr static bool value = true; };

template<typename T>
struct isUnaryExpr { constexpr static bool value = false; };

template<typename Op, typename R>
struct isUnaryExpr<UnaryExpr<Op, R>> { constexpr static bool value = true; };

template<typename T>
struct isBinaryExpr { constexpr static bool value = false; };

template<typename Op, typename L, typename R>
struct isBinaryExpr<BinaryExpr<Op, L, R>> { constexpr static bool value = true; };

template<typename T>
struct isAddExpr { constexpr static bool value = false; };

template<typename L, typename R>
struct isAddExpr<AddExpr<L, R>> { constexpr static bool value = true; };

template<typename T>
struct isMulExpr { constexpr static bool value = false; };

template<typename L, typename R>
struct isMulExpr<MulExpr<L, R>> { constexpr static bool value = true; };

} // namespace traits

template<typename A, typename B>
constexpr bool isSame = std::is_same<A, B>::value;

template<typename T>
constexpr bool isNumber = std::is_arithmetic<plain<T>>::value;

template<typename T>
constexpr bool isExpr = traits::isExpr<plain<T>>::value;

template<typename T>
constexpr bool isDual = traits::isDual<plain<T>>::value;

template<typename T>
constexpr bool isConstantExpr = traits::isConstantExpr<plain<T>>::value;

template<typename T>
constexpr bool isNegExpr = traits::isNegExpr<plain<T>>::value;

template<typename T>
constexpr bool isInvExpr = traits::isInvExpr<plain<T>>::value;

template<typename T>
constexpr bool isScaleExpr = traits::isScaleExpr<plain<T>>::value;

template<typename T>
constexpr bool isUnaryExpr = traits::isUnaryExpr<plain<T>>::value;

template<typename T>
constexpr bool isBinaryExpr = traits::isBinaryExpr<plain<T>>::value;

template<typename T>
constexpr bool isAddExpr = traits::isAddExpr<plain<T>>::value;

template<typename T>
constexpr bool isMulExpr = traits::isMulExpr<plain<T>>::value;

//=====================================================================================================================
//
// EXPRESSION TYPES DEFINITION
//
//=====================================================================================================================

template<typename Derived>
struct Expr
{
    auto derived() -> Derived& { return static_cast<Derived&>(*this); }
    auto derived() const -> const Derived& { return static_cast<const Derived&>(*this); }
};

template<typename T>
struct Dual : Expr<Dual<T>>
{
    T val;

    T grad;

    Dual() : Dual(0.0) {}

    explicit Dual(T val) : val(val), grad(0.0)
    {}

    template<typename Other>
    Dual(const Expr<Other>& expr)
    {
        assign(*this, expr.derived());
    }

    template<typename R, enableif<isNumber<R> || isExpr<R>>...>
    Dual& operator+=(R&& other)
    {
        assignAdd(*this, other);
        return *this;
    }

    template<typename R, enableif<isNumber<R> || isExpr<R>>...>
    Dual& operator-=(R&& other)
    {
        assignSub(*this, other);
        return *this;
    }

    template<typename R, enableif<isNumber<R> || isExpr<R>>...>
    Dual& operator*=(R&& other)
    {
        assignMul(*this, other);
        return *this;
    }

    template<typename R, enableif<isNumber<R> || isExpr<R>>...>
    Dual& operator/=(R&& other)
    {
        assignDiv(*this, other);
        return *this;
    }
};

template<typename T>
struct ConstantExpr : Expr<ConstantExpr<T>>
{
    T val;
    constexpr ConstantExpr(T&& val) : val(std::forward<T>(val)) {}
};

template<typename R, typename T>
struct ScaleExpr : Expr<ScaleExpr<T, R>>
{
    T scalar;
    R r;
    constexpr ScaleExpr(T&& scalar, R&& r) : scalar(std::forward<T>(scalar)), r(std::forward<R>(r)) {}
};

template<typename Op, typename R>
struct UnaryExpr : Expr<UnaryExpr<Op, R>>
{
    R r;
    constexpr UnaryExpr(R&& r) : r(std::forward<R>(r)) {}
};

template<typename Op, typename L, typename R>
struct BinaryExpr : Expr<BinaryExpr<Op, L, R>>
{
    L l;
    R r;
    constexpr BinaryExpr(L&& l, R&& r) : l(std::forward<L>(l)), r(std::forward<R>(r)) {}
};

//=====================================================================================================================
//
// CONVENIENT FUNCTIONS
//
//=====================================================================================================================


//-----------------------------------------------------------------------------
// CONSTANT EXPRESSION GENERATOR FUNCTION
//-----------------------------------------------------------------------------
template<typename T>
constexpr auto constant(T&& val) -> ConstantExpr<T>
{
    return { std::forward<T>(val) };
}

//-----------------------------------------------------------------------------
// NEGATIVE EXPRESSION GENERATOR FUNCTION
//-----------------------------------------------------------------------------
template<typename R>
constexpr auto negative(const NegExpr<R>& expr) -> decltype(expr.inner)
{
    return expr.inner;
}

template<typename R, enableif<isExpr<R>>..., disableif<isNegExpr<R>>>
constexpr auto negative(R&& expr) -> NegExpr<R>
{
    return { std::forward<R>(expr) };
}

//-----------------------------------------------------------------------------
// INVERSE EXPRESSION GENERATOR FUNCTION
//-----------------------------------------------------------------------------
template<typename R>
constexpr auto inverse(const InvExpr<R>& expr) -> decltype(expr.inner)
{
    return expr.inner;
}

template<typename R, enableif<isExpr<R>>..., disableif<isInvExpr<R>>>
constexpr auto inverse(R&& expr) -> InvExpr<R>
{
    return { std::forward<R>(expr) };
}

//=====================================================================================================================
//
// POSITIVE ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// operator+: +expr
//-----------------------------------------------------------------------------
template<typename R>
constexpr auto operator+(const Expr<R>& r) -> R
{
    return r.derived();
};

//=====================================================================================================================
//
// NEGATIVE ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// NEGATIVE OPERATOR: -expr
//-----------------------------------------------------------------------------
template<typename R, disableif<isNegExpr<R>>..., disableif<isScaleExpr<R>>...>
constexpr auto operator-(const Expr<R>& expr) -> NegExpr<R>
{
    return { expr.derived() };
}

//-----------------------------------------------------------------------------
// NEGATIVE OPERATOR: -(-expr) => expr
//-----------------------------------------------------------------------------
template<typename R>
constexpr auto operator-(const NegExpr<R>& expr) -> R
{
    return expr.inner;
}

//-----------------------------------------------------------------------------
// NEGATIVE OPERATOR: -(scalar * dual) => -scalar * dual
//-----------------------------------------------------------------------------
template<typename T, typename R>
constexpr auto operator-(const ScaleExpr<T, R>& expr)
{
    return (-expr.scalar) * std::forward<R>(expr.inner);
}

//=====================================================================================================================
//
// ADDITION ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// ADDITION OPERATOR: expr + expr
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isExpr<L> && isExpr<R>>, disableif<isNegExpr<L> && isNegExpr<R>>...>
constexpr auto operator+(L&& l, R&& r) -> AddExpr<L, R>
{
    return { std::forward<L>(l), std::forward<R>(r) };
}

//-----------------------------------------------------------------------------
// ADDITION OPERATOR: scalar + expr
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isNumber<L> && isExpr<R>>...>
constexpr auto operator+(L&& l, R&& r) -> AddExpr<L, R>
{
    return { std::forward<L>(l), std::forward<R>(r) };
}

//-----------------------------------------------------------------------------
// ADDITION OPERATOR: expr + scalar => scalar + expr
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isExpr<L> && isNumber<R>>...>
constexpr auto operator+(L&& l, R&& r) -> AddExpr<R, L>
{
    return { std::forward<R>(r), std::forward<L>(l) };
}

//-----------------------------------------------------------------------------
// ADDITION OPERATOR: (-expr) + (-expr)
//-----------------------------------------------------------------------------
template<typename L, typename R>
constexpr auto operator+(const NegExpr<L>& l, const NegExpr<R>& r)
{
    return -( std::forward<L>(l.inner) + std::forward<R>(r.inner) );
}

//=====================================================================================================================
//
// SUBTRACTION ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

template<typename L, typename R>
constexpr bool isOperable = ( isExpr<L> && isExpr<R> ) || ( isNumber<L> && isExpr<R> ) || ( isExpr<L> && isNumber<R> );

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
// MULTIPLICATION ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: Expr * Expr
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isOperable<L, R>>...,
    disableif<isNegExpr<L> && isNegExpr<R>>...,
    disableif<isInvExpr<L> && isInvExpr<R>>...,
    disableif<isScaleExpr<L> && isScaleExpr<R>>...,
    disableif<isNumber<L> && isScaleExpr<R>>...,
    disableif<isScaleExpr<L> && isNumber<R>>...,
    disableif<isNegExpr<L> && isScaleExpr<R>>...,
    disableif<isScaleExpr<L> && isNegExpr<R>>...>
constexpr auto operator*(L&& l, R&& r) -> MulExpr<L, R>
{
    return { std::forward<L>(l), std::forward<R>(r) };
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: scalar * expr
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isNumber<L>>..., enableif<isExpr<R>>..., disableif<isDual<R>>..., disableif<isScaleExpr<R>>...>
constexpr auto operator*(L&& l, R&& r) -> MulExpr<L, R>
{
    return { std::forward<L>(l), std::forward<R>(r) };
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: Expr * scalar
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isExpr<L>>..., enableif<isNumber<R>>..., disableif<isDual<L>>..., disableif<isScaleExpr<L>>...>
constexpr auto operator*(L&& l, R&& r) -> decltype(std::forward<R>(r) * std::forward<L>(l))
{
    return std::forward<R>(r) * std::forward<L>(l);
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: scalar * Dual
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isNumber<L>>..., enableif<isDual<R>>...>
constexpr auto operator*(L&& l, R&& r) -> ScaleExpr<L, R>
{
    return { std::forward<L>(l), std::forward<R>(r) };
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: Dual * scalar
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isDual<L>>..., enableif<isNumber<R>>...>
constexpr auto operator*(L&& l, R&& r) -> decltype(std::forward<R>(r) * std::forward<L>(l))
{
    return std::forward<R>(r) * std::forward<L>(l);
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: scalar * ScaleExpr
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isNumber<L>>..., enableif<isScaleExpr<R>>...>
constexpr auto operator*(L&& l, R&& r) -> decltype((l * r.scalar) * r.inner)
{
    return (l * r.scalar) * r.inner;
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: ScaleExpr * scalar
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isScaleExpr<L>>..., enableif<isNumber<R>>...>
constexpr auto operator*(L&& l, R&& r) -> decltype((r * l.scalar) * l.inner)
{
    return (r * l.scalar) * l.inner;
}

//=====================================================================================================================
//
// DIVISION ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================



///// >>>>>>>>>>>>>>>>> CONTINUE HERE <<<<<<<<<<<<<<<<<<<<<<




//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: Expr * scalar
//-----------------------------------------------------------------------------
template<typename T, typename L, enableif<isNumber<T>>...>
constexpr auto operator*(const Expr<L>& l, const T& r) -> decltype(constant(r) * l.derived())
{
    return constant(r) * l.derived();
}


//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: ConstantExpr * ConstantExpr
//-----------------------------------------------------------------------------
template<typename L, typename R>
constexpr auto operator*(const ConstantExpr<L>& l, const ConstantExpr<R>& r) -> decltype(constant(l.val * r.val))
{
    return constant(l.val * r.val);
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: NegExpr * NegExpr
//-----------------------------------------------------------------------------
template<typename L, typename R>
constexpr auto operator*(const NegExpr<L>& l, const NegExpr<R>& r) -> decltype(l.inner * r.inner)
{
    return l.inner * r.inner;
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: InvExpr * InvExpr
//-----------------------------------------------------------------------------
template<typename L, typename R>
constexpr auto operator*(const InvExpr<L>& l, const InvExpr<R>& r) -> decltype(inverse(l.inner * r.inner))
{
    return inverse(l.inner * r.inner);
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: ConstantExpr * ScaleExpr
//-----------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr auto operator*(const ConstantExpr<L>& l, const ScaleExpr<T, R>& r) -> decltype((l.val * r.scalar) * r.inner)
{
    return (l.val * r.scalar) * r.inner;
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: ScaleExpr * ConstantExpr
//-----------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr auto operator*(const ScaleExpr<T, L>& l, const ConstantExpr<R>& r) -> decltype((r.val * l.scalar) * l.inner)
{
    return (r.val * l.scalar) * l.inner;
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: NegExpr * ScaleExpr
//-----------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr auto operator*(const NegExpr<L>& l, const ScaleExpr<T, R>& r) -> decltype((-r.scalar) * (l.inner * r.inner))
{
    return (-r.scalar) * (l.inner * r.inner);
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: ScaleExpr * NegExpr
//-----------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr auto operator*(const ScaleExpr<T, L>& l, const NegExpr<R>& r) -> decltype((-l.scalar) * (l.inner * r.inner))
{
    return (-l.scalar) * (l.inner * r.inner);
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: ScaleExpr * ScaleExpr
//-----------------------------------------------------------------------------
template<typename T, typename U, typename L, typename R>
constexpr auto operator*(const ScaleExpr<T, L>& l, const ScaleExpr<U, R>& r) -> decltype((l.scalar * l.scalar) * (l.inner * r.inner))
{
    return (l.scalar * l.scalar) * (l.inner * r.inner);
}

//-----------------------------------------------------------------------------
// DIVISION OPERATOR: GENERAL CASE
//-----------------------------------------------------------------------------
template<typename L, typename R>
constexpr auto operator/(const Expr<L>& l, const Expr<R>& r) -> decltype(l * inverse(r))
{
    return l * inverse(r);
}

template<typename L, typename R> constexpr auto operator+(const Expr<L>& l, const Expr<R>& r) -> AddExpr<L, R> { return { l.derived(), r.derived() }; }
template<typename L, typename R> constexpr auto operator*(const Expr<L>& l, const Expr<R>& r) -> MulExpr<L, R> { return { l.derived(), r.derived() }; }
template<typename L, typename R> constexpr auto operator-(const Expr<L>& l, const Expr<R>& r) -> decltype(l + (-r)) { return l + (-r); }
template<typename L, typename R> constexpr auto operator/(const Expr<L>& l, const Expr<R>& r) -> decltype(l * inverse(r)) { return l * inverse(r); }

template<typename T, typename R, enableif<isNumber<T>>...> constexpr auto operator+(T l, const Expr<R>& r) -> decltype(constant(l) + r.derived()) { return constant(l) + r.derived(); }
template<typename T, typename R, enableif<isNumber<T>>...> constexpr auto operator-(T l, const Expr<R>& r) -> decltype(constant(l) - r.derived()) { return constant(l) - r.derived(); }
template<typename T, typename R, enableif<isNumber<T>>...> constexpr auto operator*(T l, const Expr<R>& r) -> ScaleExpr<T, R> { return { l.derived(), r.derived() }; }
template<typename T, typename R, enableif<isNumber<T>>...> constexpr auto operator/(T l, const Expr<R>& r) -> decltype(constant(l) / r.derived()) { return constant(l) / r.derived(); }

template<typename T, typename L, enableif<isNumber<T>>...> constexpr auto operator+(const Expr<L>& l, T r) -> decltype(l.derived() + constant(r)) { return l.derived() + constant(r); }
template<typename T, typename L, enableif<isNumber<T>>...> constexpr auto operator-(const Expr<L>& l, T r) -> decltype(l.derived() - constant(r)) { return l.derived() - constant(r); }
template<typename T, typename L, enableif<isNumber<T>>...> constexpr auto operator*(const Expr<L>& l, T r) -> ScaleExpr<T, L> { return { r, l.derived() }; }
template<typename T, typename L, enableif<isNumber<T>>...> constexpr auto operator/(const Expr<L>& l, T r) -> decltype((static_cast<T>(1) / r) * l.derived()) { return (static_cast<T>(1) / r) * l.derived(); }


//=====================================================================================================================
//
// TRIGONOMETRIC FUNCTIONS OVERLOADING
//
//=====================================================================================================================

template<typename R> constexpr auto sin(const Expr<R>& r) -> SinExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto cos(const Expr<R>& r) -> CosExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto tan(const Expr<R>& r) -> TanExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto asin(const Expr<R>& r) -> ArcSinExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto acos(const Expr<R>& r) -> ArcCosExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto atan(const Expr<R>& r) -> ArcTanExpr<R> { return { r.derived() }; }

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

template<typename R> constexpr auto exp(const Expr<R>& r) -> ExpExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto log(const Expr<R>& r) -> LogExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto log10(const Expr<R>& r) -> Log10Expr<R> { return { r.derived() }; }

//=====================================================================================================================
//
// POWER FUNCTIONS OVERLOADING
//
//=====================================================================================================================

template<typename L, typename R> constexpr auto pow(const Expr<L>& l, const Expr<R>& r) -> PowExpr<L, R> { return { l.derived(), r.derived() }; }
template<typename R> constexpr auto sqrt(const Expr<R>& r) -> SqrtExpr<R> { return { r.derived() }; }

//=====================================================================================================================
//
// OTHER FUNCTIONS OVERLOADING
//
//=====================================================================================================================

template<typename R> constexpr auto abs(const Expr<R>& r) -> AbsExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto abs2(const Expr<R>& r) -> decltype(r * r) { return r * r; }
template<typename R> constexpr auto conj(const Expr<R>& r) -> decltype(r) { return r; }
template<typename R> constexpr auto real(const Expr<R>& r) -> decltype(r) { return r; }
template<typename R> constexpr auto imag(const Expr<R>& r) -> decltype(constant(0.0)) { return constant(0.0); }

//=====================================================================================================================
//
// COMPARISON OPERATORS OVERLOADING
//
//=====================================================================================================================

// inline bool operator==(const ExprPtr& l, const ExprPtr& r) { return l->val == r->val; }
// inline bool operator!=(const ExprPtr& l, const ExprPtr& r) { return l->val != r->val; }
// inline bool operator<=(const ExprPtr& l, const ExprPtr& r) { return l->val <= r->val; }
// inline bool operator>=(const ExprPtr& l, const ExprPtr& r) { return l->val >= r->val; }
// inline bool operator<(const ExprPtr& l, const ExprPtr& r) { return l->val < r->val; }
// inline bool operator>(const ExprPtr& l, const ExprPtr& r) { return l->val > r->val; }

// inline bool operator==(double l, const ExprPtr& r) { return l == r->val; }
// inline bool operator!=(double l, const ExprPtr& r) { return l != r->val; }
// inline bool operator<=(double l, const ExprPtr& r) { return l <= r->val; }
// inline bool operator>=(double l, const ExprPtr& r) { return l >= r->val; }
// inline bool operator<(double l, const ExprPtr& r) { return l < r->val; }
// inline bool operator>(double l, const ExprPtr& r) { return l > r->val; }

// inline bool operator==(const ExprPtr& l, double r) { return l->val == r; }
// inline bool operator!=(const ExprPtr& l, double r) { return l->val != r; }
// inline bool operator<=(const ExprPtr& l, double r) { return l->val <= r; }
// inline bool operator>=(const ExprPtr& l, double r) { return l->val >= r; }
// inline bool operator<(const ExprPtr& l, double r) { return l->val < r; }
// inline bool operator>(const ExprPtr& l, double r) { return l->val > r; }

//=====================================================================================================================
//
// AUXILIARY FUNCTIONS
//
//=====================================================================================================================
template<typename T>
constexpr void negate(Dual<T>& self)
{
    self.val = -self.val;
    self.grad = -self.grad;
}

template<typename T, typename U>
constexpr void scale(Dual<T>& self, const U& scalar)
{
    self.val *= scalar;
    self.grad *= scalar;
}

//=====================================================================================================================
//
// APPLY-OPERATOR FUNCTIONS (DECLARATIONS)
//
//=====================================================================================================================

template<typename T> constexpr void apply(Dual<T>& self, NegOp);
template<typename T> constexpr void apply(Dual<T>& self, InvOp);
template<typename T> constexpr void apply(Dual<T>& self, SinOp);
template<typename T> constexpr void apply(Dual<T>& self, CosOp);
template<typename T> constexpr void apply(Dual<T>& self, TanOp);
template<typename T> constexpr void apply(Dual<T>& self, ArcSinOp);
template<typename T> constexpr void apply(Dual<T>& self, ArcCosOp);
template<typename T> constexpr void apply(Dual<T>& self, ArcTanOp);
template<typename T> constexpr void apply(Dual<T>& self, ExpOp);
template<typename T> constexpr void apply(Dual<T>& self, LogOp);
template<typename T> constexpr void apply(Dual<T>& self, Log10Op);
template<typename T> constexpr void apply(Dual<T>& self, SqrtOp);
template<typename T> constexpr void apply(Dual<T>& self, AbsOp);

template<typename Op, typename T> constexpr void apply(Dual<T>& self);

//=====================================================================================================================
//
// ASSIGNMENT FUNCTIONS (DECLARATIONS)
//
//=====================================================================================================================

template<typename T, typename U, enableif<isNumber<U>>...> constexpr void assign(Dual<T>& self, const U& other);
template<typename T, typename U, enableif<isNumber<U>>...> constexpr void assign(Dual<T>& self, const U& other, Dual<T>& tmp);

template<typename T> constexpr void assign(Dual<T>& self, const Dual<T>& other);
template<typename T> constexpr void assign(Dual<T>& self, const Dual<T>& other, Dual<T>& tmp);

template<typename T, typename U, typename R> constexpr void assign(Dual<T>& self, const ScaleExpr<U, R>& other);
template<typename T, typename U, typename R> constexpr void assign(Dual<T>& self, const ScaleExpr<U, R>& other, Dual<T>& tmp);

template<typename T, typename Op, typename R> constexpr void assign(Dual<T>& self, const UnaryExpr<Op, R>& other);
template<typename T, typename Op, typename R> constexpr void assign(Dual<T>& self, const UnaryExpr<Op, R>& other, Dual<T>& tmp);

template<typename T, typename L, typename R> constexpr void assign(Dual<T>& self, const AddExpr<L, R>& other);
template<typename T, typename L, typename R> constexpr void assign(Dual<T>& self, const AddExpr<L, R>& other, Dual<T>& tmp);

template<typename T, typename L, typename R> constexpr void assign(Dual<T>& self, const MulExpr<L, R>& other);
template<typename T, typename L, typename R> constexpr void assign(Dual<T>& self, const MulExpr<L, R>& other, Dual<T>& tmp);

//=====================================================================================================================
//
// ASSIGNMENT-ADDITION FUNCTIONS (DECLARATIONS)
//
//=====================================================================================================================

template<typename T, typename U, enableif<isNumber<U>>...> constexpr void assignAdd(Dual<T>& self, const U& other);
template<typename T, typename U, enableif<isNumber<U>>...> constexpr void assignAdd(Dual<T>& self, const U& other, Dual<T>& tmp);

template<typename T> constexpr void assignAdd(Dual<T>& self, const Dual<T>& other);
template<typename T> constexpr void assignAdd(Dual<T>& self, const Dual<T>& other, Dual<T>& tmp);

template<typename T, typename R> constexpr void assignAdd(Dual<T>& self, const NegExpr<R>& other);
template<typename T, typename R> constexpr void assignAdd(Dual<T>& self, const NegExpr<R>& other, Dual<T>& tmp);

template<typename T, typename U, typename R> constexpr void assignAdd(Dual<T>& self, const ScaleExpr<U, R>& other);
template<typename T, typename U, typename R> constexpr void assignAdd(Dual<T>& self, const ScaleExpr<U, R>& other, Dual<T>& tmp);

template<typename T, typename L, typename R> constexpr void assignAdd(Dual<T>& self, const AddExpr<L, R>& other);
template<typename T, typename L, typename R> constexpr void assignAdd(Dual<T>& self, const AddExpr<L, R>& other, Dual<T>& tmp);

template<typename T, typename R, disableif<isDual<R> || isNegExpr<R> || isScaleExpr<R> || isAddExpr<R>>...> constexpr void assignAdd(Dual<T>& self, const Expr<R>& other);
template<typename T, typename R, disableif<isDual<R> || isNegExpr<R> || isScaleExpr<R> || isAddExpr<R>>...> constexpr void assignAdd(Dual<T>& self, const Expr<R>& other, Dual<T>& tmp);

//=====================================================================================================================
//
// ASSIGNMENT-SUBTRACTION FUNCTIONS (DECLARATIONS)
//
//=====================================================================================================================

template<typename T, typename U, enableif<isNumber<U>>...> constexpr void assignSub(Dual<T>& self, const U& other);
template<typename T, typename U, enableif<isNumber<U>>...> constexpr void assignSub(Dual<T>& self, const U& other, Dual<T>& tmp);

template<typename T> constexpr void assignSub(Dual<T>& self, const Dual<T>& other);
template<typename T> constexpr void assignSub(Dual<T>& self, const Dual<T>& other, Dual<T>& tmp);

template<typename T, typename R, disableif<isDual<R>>...> constexpr void assignSub(Dual<T>& self, const Expr<R>& other);
template<typename T, typename R, disableif<isDual<R>>...> constexpr void assignSub(Dual<T>& self, const Expr<R>& other, Dual<T>& tmp);

//=====================================================================================================================
//
// ASSIGNMENT-MULTIPLICATION FUNCTIONS (DECLARATIONS)
//
//=====================================================================================================================

template<typename T, typename U, enableif<isNumber<U>>...> constexpr void assignMul(Dual<T>& self, const U& other);
template<typename T, typename U, enableif<isNumber<U>>...> constexpr void assignMul(Dual<T>& self, const U& other, Dual<T>& tmp);

template<typename T> constexpr void assignMul(Dual<T>& self, const Dual<T>& other);
template<typename T> constexpr void assignMul(Dual<T>& self, const Dual<T>& other, Dual<T>& tmp);

template<typename T, typename R> constexpr void assignMul(Dual<T>& self, const NegExpr<R>& other);
template<typename T, typename R> constexpr void assignMul(Dual<T>& self, const NegExpr<R>& other, Dual<T>& tmp);

template<typename T, typename U, typename R> constexpr void assignMul(Dual<T>& self, const ScaleExpr<U, R>& other);
template<typename T, typename U, typename R> constexpr void assignMul(Dual<T>& self, const ScaleExpr<U, R>& other, Dual<T>& tmp);

template<typename T, typename L, typename R> constexpr void assignMul(Dual<T>& self, const MulExpr<L, R>& other);
template<typename T, typename L, typename R> constexpr void assignMul(Dual<T>& self, const MulExpr<L, R>& other, Dual<T>& tmp);

template<typename T, typename R, disableif<isDual<R> || isNegExpr<R> || isScaleExpr<R> || isMulExpr<R>>...> constexpr void assignMul(Dual<T>& self, const Expr<R>& other);
template<typename T, typename R, disableif<isDual<R> || isNegExpr<R> || isScaleExpr<R> || isMulExpr<R>>...> constexpr void assignMul(Dual<T>& self, const Expr<R>& other, Dual<T>& tmp);

//=====================================================================================================================
//
// ASSIGNMENT-DIVISION FUNCTIONS (DECLARATIONS)
//
//=====================================================================================================================

template<typename T, typename U, enableif<isNumber<U>>...> constexpr void assignDiv(Dual<T>& self, const U& other);
template<typename T, typename U, enableif<isNumber<U>>...> constexpr void assignDiv(Dual<T>& self, const U& other, Dual<T>& tmp);

template<typename T> constexpr void assignDiv(Dual<T>& self, const Dual<T>& other);
template<typename T> constexpr void assignDiv(Dual<T>& self, const Dual<T>& other, Dual<T>& tmp);

template<typename T, typename R, disableif<isDual<R>>...> constexpr void assignDiv(Dual<T>& self, const Expr<R>& other);
template<typename T, typename R, disableif<isDual<R>>...> constexpr void assignDiv(Dual<T>& self, const Expr<R>& other, Dual<T>& tmp);

//=====================================================================================================================
//
// ASSIGNMENT FUNCTIONS
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// assign: self = scalar
//-----------------------------------------------------------------------------
template<typename T, typename U, enableif<isNumber<U>>...>
constexpr void assign(Dual<T>& self, const U& other)
{
    self.val = other;
    self.grad = 0.0;
}

template<typename T, typename U, enableif<isNumber<U>>...>
constexpr void assign(Dual<T>& self, const U& other, Dual<T>& tmp)
{
    assign(self, other);
}

//-----------------------------------------------------------------------------
// assign: self = dual
//-----------------------------------------------------------------------------
template<typename T>
constexpr void assign(Dual<T>& self, const Dual<T>& other)
{
    self.val = other.val;
    self.grad = other.grad;
}

template<typename T>
constexpr void assign(Dual<T>& self, const Dual<T>& other, Dual<T>& tmp)
{
    assign(self, other);
}

//-----------------------------------------------------------------------------
// assign: self = scalar * dual
//-----------------------------------------------------------------------------
template<typename T, typename U, typename R>
constexpr void assign(Dual<T>& self, const ScaleExpr<U, R>& other)
{
    assign(self, other.inner);
    scale(self, other.scalar);
}

template<typename T, typename U, typename R>
constexpr void assign(Dual<T>& self, const ScaleExpr<U, R>& other, Dual<T>& tmp)
{
    assign(self, other);
}

//-----------------------------------------------------------------------------
// assign: self = function(expr)
//-----------------------------------------------------------------------------
template<typename T, typename Op, typename R>
constexpr void assign(Dual<T>& self, const UnaryExpr<Op, R>& other)
{
    assign(self, other.inner);
    apply<Op>(self);
}

template<typename T, typename Op, typename R>
constexpr void assign(Dual<T>& self, const UnaryExpr<Op, R>& other, Dual<T>& tmp)
{
    assign(self, other.inner, tmp);
    apply<Op>(self);
}

//-----------------------------------------------------------------------------
// assign: self = expr + expr
//-----------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr void assign(Dual<T>& self, const AddExpr<L, R>& other)
{
    assign(self, other.inner);
    assignAdd(self, other.l);
}

template<typename T, typename L, typename R>
constexpr void assign(Dual<T>& self, const AddExpr<L, R>& other, Dual<T>& tmp)
{
    assign(self, other.inner, tmp);
    assignAdd(self, other.l, tmp);
}

//-----------------------------------------------------------------------------
// assign: self = expr * expr
//-----------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr void assign(Dual<T>& self, const MulExpr<L, R>& other)
{
    assign(self, other.inner);
    assignMul(self, other.l);
}

template<typename T, typename L, typename R>
constexpr void assign(Dual<T>& self, const MulExpr<L, R>& other, Dual<T>& tmp)
{
    assign(self, other.inner, tmp);
    assignMul(self, other.l, tmp);
}

//=====================================================================================================================
//
// ASSIGNMENT-ADDITION FUNCTIONS
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// assignAdd: self += scalar
//-----------------------------------------------------------------------------
template<typename T, typename U, enableif<isNumber<U>>...>
constexpr void assignAdd(Dual<T>& self, const U& other)
{
    self.val += other;
}

template<typename T, typename U, enableif<isNumber<U>>...>
constexpr void assignAdd(Dual<T>& self, const U& other, Dual<T>& tmp)
{
    self.val += other;
}

//-----------------------------------------------------------------------------
// assignAdd: self += dual
//-----------------------------------------------------------------------------
template<typename T>
constexpr void assignAdd(Dual<T>& self, const Dual<T>& other)
{
    self.val += other.val;
    self.grad += other.grad;
}

template<typename T>
constexpr void assignAdd(Dual<T>& self, const Dual<T>& other, Dual<T>& tmp)
{
    assignAdd(self, other);
}

//-----------------------------------------------------------------------------
// assignAdd: self += -(expr)
//-----------------------------------------------------------------------------
template<typename T, typename R>
constexpr void assignAdd(Dual<T>& self, const NegExpr<R>& other)
{
    assignSub(self, other.inner);
}

template<typename T, typename R>
constexpr void assignAdd(Dual<T>& self, const NegExpr<R>& other, Dual<T>& tmp)
{
    assignSub(self, other.inner, tmp);
}

//-----------------------------------------------------------------------------
// assignAdd: self += scalar * dual
//-----------------------------------------------------------------------------
template<typename T, typename U, typename R>
constexpr void assignAdd(Dual<T>& self, const ScaleExpr<U, R>& other)
{
    self.val += other.scalar * other.val;
    self.grad += other.scalar * other.grad;
}

template<typename T, typename U, typename R>
constexpr void assignAdd(Dual<T>& self, const ScaleExpr<U, R>& other, Dual<T>& tmp)
{
    assignAdd(self, other);
}

//-----------------------------------------------------------------------------
// assignAdd: self += expr + expr
//-----------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr void assignAdd(Dual<T>& self, const AddExpr<L, R>& other)
{
    assignAdd(self, other.l);
    assignAdd(self, other.inner);
}

template<typename T, typename L, typename R>
constexpr void assignAdd(Dual<T>& self, const AddExpr<L, R>& other, Dual<T>& tmp)
{
    assignAdd(self, other.l, tmp);
    assignAdd(self, other.inner, tmp);
}

//-----------------------------------------------------------------------------
// assignAdd: self += expr
//-----------------------------------------------------------------------------
template<typename T, typename R, disableif<isDual<R> || isNegExpr<R> || isScaleExpr<R> || isAddExpr<R>>...>
constexpr void assignAdd(Dual<T>& self, const Expr<R>& other)
{
    Dual<T> tmp;
    assignAdd(self, tmp, other.derived());
}

template<typename T, typename R, disableif<isDual<R> || isNegExpr<R> || isScaleExpr<R> || isAddExpr<R>>...>
constexpr void assignAdd(Dual<T>& self, const Expr<R>& other, Dual<T>& tmp)
{
    assign(tmp, other.derived());
    assignAdd(self, tmp);
}

//=====================================================================================================================
//
// ASSIGNMENT-SUBTRACTION FUNCTIONS
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// assignSub: self -= scalar
//-----------------------------------------------------------------------------
template<typename T, typename U, enableif<isNumber<U>>...>
constexpr void assignSub(Dual<T>& self, const U& other)
{
    self.val -= other;
}

template<typename T, typename U, enableif<isNumber<U>>...>
constexpr void assignSub(Dual<T>& self, const U& other, Dual<T>& tmp)
{
    self.val -= other;
}

//-----------------------------------------------------------------------------
// assignSub: self -= dual
//-----------------------------------------------------------------------------
template<typename T>
constexpr void assignSub(Dual<T>& self, const Dual<T>& other)
{
    self.val -= other.val;
    self.grad -= other.grad;
}

template<typename T>
constexpr void assignSub(Dual<T>& self, const Dual<T>& other, Dual<T>& tmp)
{
    assignSub(self, other);
}

//-----------------------------------------------------------------------------
// assignSub: self -= expr
//-----------------------------------------------------------------------------
template<typename T, typename R, disableif<isDual<R>>...>
constexpr void assignSub(Dual<T>& self, const Expr<R>& other)
{
    assignAdd(self, negative(other.derived()));
}

template<typename T, typename R, disableif<isDual<R>>...>
constexpr void assignSub(Dual<T>& self, const Expr<R>& other, Dual<T>& tmp)
{
    assignAdd(self, negative(other.derived()), tmp);
}

//=====================================================================================================================
//
// ASSIGNMENT-MULTIPLICATION FUNCTIONS
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// assignMul: self *= scalar
//-----------------------------------------------------------------------------
template<typename T, typename U, enableif<isNumber<U>>...>
constexpr void assignMul(Dual<T>& self, const U& other)
{
    self.val *= other;
    self.grad *= other;
}

template<typename T, typename U, enableif<isNumber<U>>...>
constexpr void assignMul(Dual<T>& self, const U& other, Dual<T>& tmp)
{
    assignMul(self, other);
}

//-----------------------------------------------------------------------------
// assignMul: self *= dual
//-----------------------------------------------------------------------------
template<typename T>
constexpr void assignMul(Dual<T>& self, const Dual<T>& other)
{
    self.grad *= other.val;
    self.grad += self.val * other.grad;
    self.val *= other.val;
}

template<typename T>
constexpr void assignMul(Dual<T>& self, const Dual<T>& other, Dual<T>& tmp)
{
    assignMul(self, other);
}

//-----------------------------------------------------------------------------
// assignMul: self *= -(expr)
//-----------------------------------------------------------------------------
template<typename T, typename R>
constexpr void assignMul(Dual<T>& self, const NegExpr<R>& other)
{
    assignMul(self, other.inner);
    negate(self);
}

template<typename T, typename R>
constexpr void assignMul(Dual<T>& self, const NegExpr<R>& other, Dual<T>& tmp)
{
    assignMul(self, other.inner, tmp);
    negate(self);
}

//-----------------------------------------------------------------------------
// assignMul: self *= scalar * dual
//-----------------------------------------------------------------------------
template<typename T, typename U, typename R>
constexpr void assignMul(Dual<T>& self, const ScaleExpr<U, R>& other)
{
    self.val *= other.scalar * other.val;
    self.grad *= other.scalar * other.grad;
}

template<typename T, typename U, typename R>
constexpr void assignMul(Dual<T>& self, const ScaleExpr<U, R>& other, Dual<T>& tmp)
{
    assignMul(self, other);
}

//-----------------------------------------------------------------------------
// assignMul: self *= expr * expr
//-----------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr void assignMul(Dual<T>& self, const MulExpr<L, R>& other)
{
    assignMul(self, other.l);
    assignMul(self, other.inner);
}

template<typename T, typename L, typename R>
constexpr void assignMul(Dual<T>& self, const MulExpr<L, R>& other, Dual<T>& tmp)
{
    assignMul(self, other.l, tmp);
    assignMul(self, other.inner, tmp);
}

//-----------------------------------------------------------------------------
// assignMul: self *= expr
//-----------------------------------------------------------------------------
template<typename T, typename R, disableif<isDual<R> || isNegExpr<R> || isScaleExpr<R> || isMulExpr<R>>...>
constexpr void assignMul(Dual<T>& self, const Expr<R>& other)
{
    Dual<T> tmp;
    assignMul(self, tmp, other.derived());
}

template<typename T, typename R, disableif<isDual<R> || isNegExpr<R> || isScaleExpr<R> || isMulExpr<R>>...>
constexpr void assignMul(Dual<T>& self, const Expr<R>& other, Dual<T>& tmp)
{
    assign(tmp, other.derived());
    assignMul(self, tmp);
}

//=====================================================================================================================
//
// ASSIGNMENT-DIVISION FUNCTIONS
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// assignDiv: self /= scalar
//-----------------------------------------------------------------------------
template<typename T, typename U, enableif<isNumber<U>>...>
constexpr void assignDiv(Dual<T>& self, const U& other)
{
    const auto aux = static_cast<T>(1) / other;
    assignMul(self, aux);
}

template<typename T, typename U, enableif<isNumber<U>>...>
constexpr void assignDiv(Dual<T>& self, const U& other, Dual<T>& tmp)
{
    assignDiv(self, other);
}

//-----------------------------------------------------------------------------
// assignDiv: self /= dual
//-----------------------------------------------------------------------------
template<typename T>
constexpr void assignDiv(Dual<T>& self, const Dual<T>& other)
{
    const auto aux = static_cast<T>(1) / other.val;
    self.val *= aux;
    self.grad -= self.val * other.grad;
    self.grad *= aux;
}

template<typename T>
constexpr void assignDiv(Dual<T>& self, const Dual<T>& other, Dual<T>& tmp)
{
    assignDiv(self, other);
}

//-----------------------------------------------------------------------------
// assignDiv: self /= expr
//-----------------------------------------------------------------------------
template<typename T, typename R, disableif<isDual<R>>...>
constexpr void assignDiv(Dual<T>& self, const Expr<R>& other)
{
    assignMul(self, inverse(other.derived()));
}

template<typename T, typename R, disableif<isDual<R>>...>
constexpr void assignDiv(Dual<T>& self, const Expr<R>& other, Dual<T>& tmp)
{
    assignMul(self, inverse(other.derived()), tmp);
}

//=====================================================================================================================
//
// APPLY-OPERATOR FUNCTIONS
//
//=====================================================================================================================
template<typename T>
constexpr void apply(Dual<T>& self, NegOp)
{
    self.val = -self.val;
    self.grad = -self.grad;
}

template<typename T>
constexpr void apply(Dual<T>& self, InvOp)
{
    self.val = static_cast<T>(1) / self.val;
    self.grad *= - self.val * self.val;
}

template<typename T>
constexpr void apply(Dual<T>& self, SinOp)
{
    self.grad *= std::cos(self.val);
    self.val = std::sin(self.val);
}

template<typename T>
constexpr void apply(Dual<T>& self, CosOp)
{
    self.grad *= -std::sin(self.val);
    self.val = std::cos(self.val);
}

template<typename T>
constexpr void apply(Dual<T>& self, TanOp)
{
    const auto aux = static_cast<T>(1) / std::cos(self.val);
    self.val = std::tan(self.val);
    self.grad *= aux * aux;
}

template<typename T>
constexpr void apply(Dual<T>& self, ArcSinOp)
{
    const auto aux = static_cast<T>(1) / std::sqrt(1.0 - self.val * self.val);
    self.val = std::asin(self.val);
    self.grad *= aux;
}

template<typename T>
constexpr void apply(Dual<T>& self, ArcCosOp)
{
    const auto aux = -1.0 / std::sqrt(1.0 - self.val * self.val);
    self.val = std::acos(self.val);
    self.grad *= aux;
}

template<typename T>
constexpr void apply(Dual<T>& self, ArcTanOp)
{
    const auto aux = static_cast<T>(1) / (1.0 + self.val * self.val);
    self.val = std::atan(self.val);
    self.grad *= aux;
}

template<typename T>
constexpr void apply(Dual<T>& self, ExpOp)
{
    self.val = std::exp(self.val);
    self.grad *= self.val;
}

template<typename T>
constexpr void apply(Dual<T>& self, LogOp)
{
    const auto aux = static_cast<T>(1) / self.val;
    self.val = std::log(self.val);
    self.grad *= aux;
}

template<typename T>
constexpr void apply(Dual<T>& self, Log10Op)
{
    constexpr T ln10 = 2.3025850929940456840179914546843;
    const auto aux = static_cast<T>(1) / (ln10 * self.val);
    self.val = std::log10(self.val);
    self.grad *= aux;
}

template<typename T>
constexpr void apply(Dual<T>& self, SqrtOp)
{
    self.val = std::sqrt(self.val);
    self.grad *= 0.5 / self.val;
}

template<typename T>
constexpr void apply(Dual<T>& self, AbsOp)
{
    const auto aux = self.val;
    self.val = std::abs(self.val);
    self.grad *= aux / self.val;
}

template<typename Op, typename T>
constexpr void apply(Dual<T>& self)
{
    apply(self, Op{});
}

} // namespace forward

using dual = forward::Dual<double>;

} // namespace autodiff
