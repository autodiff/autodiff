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

struct AddOp    {};  // ADDITION OPERATOR
struct SubOp    {};  // SUBTRACTION OPERATOR
struct MulOp    {};  // MULTIPLICATION OPERATOR
struct DivOp    {};  // DIVISION OPERATOR
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

template<typename U, typename R>
struct ScalarDualMulExpr;

template<typename Op, typename R>
struct UnaryExpr;

template<typename Op, typename L, typename R>
struct BinaryExpr;

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

template<typename R>
using AbsExpr = UnaryExpr<AbsOp, R>;

template<typename L, typename R>
using AddExpr = BinaryExpr<AddOp, L, R>;

template<typename L, typename R>
using MulExpr = BinaryExpr<MulOp, L, R>;

template<typename L, typename R>
using PowExpr = BinaryExpr<PowOp, L, R>;

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

template<typename A, typename B>
constexpr bool isSame = std::is_same<A, B>::value;

template<typename T>
constexpr bool isNumber = std::is_arithmetic<plain<T>>::value;

namespace traits {

template<typename T>
struct isExpr { constexpr static bool value = std::is_base_of<Expr<plain<T>>, plain<T>>::value; };

template<typename T>
struct isDual { constexpr static bool value = false; };

template<typename T>
struct isDual<Dual<T>> { constexpr static bool value = true; };

template<typename T>
struct isScalarDualMulExpr { constexpr static bool value = false; };

template<typename U, typename R>
struct isScalarDualMulExpr<ScalarDualMulExpr<U, R>> { constexpr static bool value = true; };

template<typename T>
struct isNegExpr { constexpr static bool value = false; };

template<typename T>
struct isNegExpr<NegExpr<T>> { constexpr static bool value = true; };

template<typename T>
struct isInvExpr { constexpr static bool value = false; };

template<typename T>
struct isInvExpr<InvExpr<T>> { constexpr static bool value = true; };

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

template<typename T>
struct ValueType { using type = T; };

template<typename Derived>
struct ValueType<Expr<Derived>> { using type = typename ValueType<plain<Derived>>::type; };

template<typename T>
struct ValueType<Dual<T>> { using type = T; };

template<typename U, typename R>
struct ValueType<ScalarDualMulExpr<U, R>> { using type = common<plain<U>, typename ValueType<plain<R>>::type>; };

template<typename Op, typename R>
struct ValueType<UnaryExpr<Op, R>> { using type = typename ValueType<plain<R>>::type; };

template<typename Op, typename L, typename R>
struct ValueType<BinaryExpr<Op, L, R>> { using type = common<typename ValueType<plain<L>>::type, typename ValueType<plain<R>>::type>; };

} // namespace traits

template<typename T>
constexpr bool isExpr = traits::isExpr<plain<T>>::value;

template<typename T>
constexpr bool isDual = traits::isDual<plain<T>>::value;

template<typename T>
constexpr bool isScalarDualMulExpr = traits::isScalarDualMulExpr<plain<T>>::value;

template<typename T>
constexpr bool isNegExpr = traits::isNegExpr<plain<T>>::value;

template<typename T>
constexpr bool isInvExpr = traits::isInvExpr<plain<T>>::value;

template<typename T>
constexpr bool isUnaryExpr = traits::isUnaryExpr<plain<T>>::value;

template<typename T>
constexpr bool isBinaryExpr = traits::isBinaryExpr<plain<T>>::value;

template<typename T>
constexpr bool isAddExpr = traits::isAddExpr<plain<T>>::value;

template<typename T>
constexpr bool isMulExpr = traits::isMulExpr<plain<T>>::value;

template<typename L, typename R>
constexpr bool isOperable = (isExpr<L> && isExpr<R>) || (isNumber<L> && isExpr<R>) || (isExpr<L> && isNumber<R>);

template<typename T>
using ValueType = typename traits::ValueType<plain<T>>::type;

//=====================================================================================================================
//
// EXPRESSION TYPES DEFINITION
//
//=====================================================================================================================

template<typename Derived>
struct Expr
{
    // TODO both derived methods below are not actually used. In fact, this curiously recursive template pattern is also not used. Implement autodiff::dual without this by relying more on type traits.
    auto derived() -> Derived& { return static_cast<Derived&>(*this); }
    auto derived() const -> const Derived& { return static_cast<const Derived&>(*this); }
};

template<typename T>
struct Dual : Expr<Dual<T>>
{
    T val;

    T grad;

    Dual() : Dual(0.0) {}

    template<typename U, enableif<isNumber<U>>...>
    Dual(const U& val) : val(val), grad(0.0)
    {}

    template<typename R, enableif<isExpr<R>>...>
    Dual(const R& other)
    {
        assign(*this, other);
    }

    template<typename R, enableif<isNumber<R> || isExpr<R>>...>
    Dual& operator=(const R& other)
    {
        assign(*this, other);
        return *this;
    }

    template<typename R, enableif<isNumber<R> || isExpr<R>>...>
    Dual& operator+=(const R& other)
    {
        assignAdd(*this, other);
        return *this;
    }

    template<typename R, enableif<isNumber<R> || isExpr<R>>...>
    Dual& operator-=(const R& other)
    {
        assignSub(*this, other);
        return *this;
    }

    template<typename R, enableif<isNumber<R> || isExpr<R>>...>
    Dual& operator*=(const R& other)
    {
        assignMul(*this, other);
        return *this;
    }

    template<typename R, enableif<isNumber<R> || isExpr<R>>...>
    Dual& operator/=(const R& other)
    {
        assignDiv(*this, other);
        return *this;
    }
};

template<typename T, typename R>
struct ScalarDualMulExpr : Expr<ScalarDualMulExpr<T, R>>
{
    const T scalar;
    const R dual;
    constexpr ScalarDualMulExpr(T&& scalar, R&& r) : scalar(std::forward<T>(scalar)), dual(std::forward<R>(r)) {}
};

template<typename T, typename L, typename R>
struct ScalarDualDualMulExpr : Expr<ScalarDualDualMulExpr<T, L, R>>
{
    const T scalar;
    const L ldual;
    const R rdual;
    constexpr ScalarDualDualMulExpr(T&& scalar, L&& l, R&& r) : scalar(std::forward<T>(scalar)), ldual(std::forward<L>(l)), rdual(std::forward<R>(r)) {}
};

template<typename Op, typename R>
struct UnaryExpr : Expr<UnaryExpr<Op, R>>
{
    const R r;
    constexpr UnaryExpr(R&& r) : r(std::forward<R>(r)) {}
};

template<typename Op, typename L, typename R>
struct BinaryExpr : Expr<BinaryExpr<Op, L, R>>
{
    const L l;
    const R r;
    constexpr BinaryExpr(L&& l, R&& r) : l(std::forward<L>(l)), r(std::forward<R>(r)) {}
};

//=====================================================================================================================
//
// UTILITY FUNCTIONS
//
//=====================================================================================================================

template<typename T>
auto eval(const Dual<T>& dual)
{
    return dual;
}

template<typename R, enableif<isExpr<R>>..., disableif<isDual<R>>...>
auto eval(R&& expr)
{
    return Dual<ValueType<R>>(expr);
}

template<typename T>
auto val(const Dual<T>& dual)
{
    return dual.val;
}

template<typename T, enableif<isNumber<T>>...>
auto val(const T& scalar)
{
    return scalar;
}

template<typename R, enableif<isExpr<R>>..., disableif<isDual<R>>...>
auto val(R&& expr)
{
    return eval(expr).val;
}

template<typename Function, typename T, typename... Args>
double grad(const Function& f, Dual<T>& wrt, const Args&... args)
{
    wrt.grad = 1.0;
    Dual<T> res = f(args...);
    wrt.grad = 0.0;
    return res.grad;
}

template<typename T>
Dual<T>& wrt(Dual<T>& x)
{
    return x;
}

//=====================================================================================================================
//
// CONVENIENT FUNCTIONS
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// NEGATIVE EXPRESSION GENERATOR FUNCTION
//-----------------------------------------------------------------------------
template<typename R, enableif<isExpr<R>>..., disableif<isNegExpr<R>>>
constexpr auto negative(R&& expr) -> NegExpr<R>
{
    return { std::forward<R>(expr) };
}

template<typename R, enableif<isNegExpr<R>>...>
constexpr auto negative(R&& expr)
{
    return expr.r;
}

//-----------------------------------------------------------------------------
// INVERSE EXPRESSION GENERATOR FUNCTION
//-----------------------------------------------------------------------------
template<typename R, enableif<isExpr<R>>..., disableif<isInvExpr<R>>...>
constexpr auto inverse(R&& r) -> InvExpr<R>
{
    return { std::forward<R>(r) };
}

template<typename R>
constexpr auto inverse(const InvExpr<R>& r)
{
    return std::forward<R>(r.r);
}

template<typename T, typename R, enableif<isNumber<R>>...>
constexpr auto inverse(R&& r)
{
    return static_cast<T>(1) / r;
}

//=====================================================================================================================
//
// POSITIVE ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// operator+: +expr => expr
//-----------------------------------------------------------------------------
template<typename R, enableif<isExpr<R>>...>
constexpr auto operator+(R&& r)
{
    return std::forward<R>(r);
}

//=====================================================================================================================
//
// NEGATIVE ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// NEGATIVE OPERATOR: -expr
//-----------------------------------------------------------------------------
template<typename R, enableif<isExpr<R>>..., disableif<isNegExpr<R> || isScalarDualMulExpr<R>>...>
constexpr auto operator-(R&& r) -> NegExpr<R>
{
    return { std::forward<R>(r) };
}

//-----------------------------------------------------------------------------
// NEGATIVE OPERATOR: -(-expr) => expr
//-----------------------------------------------------------------------------
template<typename R>
constexpr auto operator-(const NegExpr<R>& expr)
{
    return std::forward<R>(expr.r);
}

//-----------------------------------------------------------------------------
// NEGATIVE OPERATOR: -(scalar * dual) => -scalar * dual
//-----------------------------------------------------------------------------
template<typename U, typename R>
constexpr auto operator-(const ScalarDualMulExpr<U, R>& expr) -> ScalarDualMulExpr<U, R>
{
    return { (-expr.scalar), std::forward<R>(expr.dual) };
}

//=====================================================================================================================
//
// ADDITION ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// ADDITION OPERATOR: expr + expr
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isExpr<L> && isExpr<R>>..., disableif<isNegExpr<L> && isNegExpr<R>>...>
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
    return -( std::forward<L>(l.r) + std::forward<R>(r.r) );
}

//=====================================================================================================================
//
// MULTIPLICATION ARITHMETIC OPERATOR OVERLOADING
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: scalar * expr
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isNumber<L> && isExpr<R>>..., disableif<isDual<R> || isScalarDualMulExpr<R> || isNegExpr<R>>...>
constexpr auto operator*(L&& l, R&& r) -> MulExpr<L, R>
{
    return { std::forward<L>(l), std::forward<R>(r) };
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: scalar * dual
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isNumber<L> && isDual<R>>...>
constexpr auto operator*(L&& l, R&& r) -> ScalarDualMulExpr<L, R>
{
    return { std::forward<L>(l), std::forward<R>(r) };
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: scalar * (scalar * expr) => (scalar * scalar) * expr
//-----------------------------------------------------------------------------
template<typename L, typename U, typename R, enableif<isNumber<L>>...>
constexpr auto operator*(L&& l, const ScalarDualMulExpr<U, R>& r)
{
    return (l * r.scalar) * std::forward<R>(r.dual);
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: scalar * (-expr) => (-scalar) * expr
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isNumber<L>>...>
constexpr auto operator*(L&& l, const NegExpr<R>& r)
{
    return (-l) * std::forward<R>(r.r);
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: expr * scalar => scalar * expr
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isExpr<L> && isNumber<R>>...>
constexpr auto operator*(L&& l, R&& r) ->  MulExpr<R, L>
{
    return { std::forward<R>(r), std::forward<L>(l) };
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: expr * expr
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isExpr<L> && isExpr<R>>..., disableif<isNegExpr<L> && isNegExpr<R>>..., disableif<isInvExpr<L> && isInvExpr<R>>...>
constexpr auto operator*(L&& l, R&& r) -> MulExpr<L, R>
{
    return { std::forward<L>(l), std::forward<R>(r) };
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: (-expr) * (-expr) => expr * expr
//-----------------------------------------------------------------------------
template<typename L, typename R>
constexpr auto operator*(const NegExpr<L>& l, const NegExpr<R>& r)
{
    return std::forward<L>(l.r) * std::forward<L>(r.r);
}

//-----------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: (1 / expr) * (1 / expr) => 1 / (expr * expr)
//-----------------------------------------------------------------------------
template<typename L, typename R>
constexpr auto operator*(const InvExpr<L>& l, const InvExpr<R>& r)
{
    return inverse(std::forward<L>(l.r) * std::forward<R>(r.r));
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
template<typename L, typename R, enableif<isExpr<L> && isExpr<R>>...>
constexpr auto operator/(L&& l, R&& r)
{
    return std::forward<L>(l) * inverse(std::forward<R>(r));
}

//-----------------------------------------------------------------------------
// DIVISION OPERATOR: scalar / expr
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isNumber<L> && isExpr<R>>...>
constexpr auto operator/(L&& l, R&& r)
{
    return std::forward<L>(l) * inverse(std::forward<R>(r));
}

//-----------------------------------------------------------------------------
// DIVISION OPERATOR: expr / scalar
//-----------------------------------------------------------------------------
template<typename L, typename R, enableif<isExpr<L> && isNumber<R>>...>
constexpr auto operator/(L&& l, R&& r)
{
    using T = ValueType<L>;
    return std::forward<L>(l) * inverse<T>(std::forward<R>(r));
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

template<typename L, typename R, enableif<isExpr<R>>...> constexpr auto pow(L&& l, R&& r) -> PowExpr<L, R> { return { std::forward<L>(l), std::forward<R>(r) }; }
template<typename R, enableif<isExpr<R>>...> constexpr auto sqrt(R&& r) -> SqrtExpr<R> { return { std::forward<R>(r) }; }

//=====================================================================================================================
//
// OTHER FUNCTIONS OVERLOADING
//
//=====================================================================================================================

template<typename R, enableif<isExpr<R>>...> constexpr auto abs(R&& r) -> AbsExpr<R> { return { std::forward<R>(r) }; }
template<typename R, enableif<isExpr<R>>...> constexpr auto abs2(R&& r) { return std::forward<R>(r) * std::forward<R>(r); }
template<typename R, enableif<isExpr<R>>...> constexpr auto conj(R&& r) { return std::forward<R>(r); }
template<typename R, enableif<isExpr<R>>...> constexpr auto real(R&& r) { return std::forward<R>(r); }
template<typename R, enableif<isExpr<R>>...> constexpr auto imag(R&& r) { return 0.0; }

//=====================================================================================================================
//
// COMPARISON OPERATORS OVERLOADING
//
//=====================================================================================================================

template<typename L, typename R, enableif<isOperable<L, R>>...> bool operator==(L&& l, R&& r) { return val(l) == val(r); }
template<typename L, typename R, enableif<isOperable<L, R>>...> bool operator!=(L&& l, R&& r) { return val(l) != val(r); }
template<typename L, typename R, enableif<isOperable<L, R>>...> bool operator<=(L&& l, R&& r) { return val(l) <= val(r); }
template<typename L, typename R, enableif<isOperable<L, R>>...> bool operator>=(L&& l, R&& r) { return val(l) >= val(r); }
template<typename L, typename R, enableif<isOperable<L, R>>...> bool operator<(L&& l, R&& r) { return val(l) < val(r); }
template<typename L, typename R, enableif<isOperable<L, R>>...> bool operator>(L&& l, R&& r) { return val(l) > val(r); }

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
// FORWARD DECLARATIONS
//
//=====================================================================================================================

template<typename Op, typename T> constexpr void apply(Dual<T>& self);

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
constexpr void assign(Dual<T>& self, const ScalarDualMulExpr<U, R>& other)
{
    assign(self, other.dual);
    scale(self, other.scalar);
}

template<typename T, typename U, typename R>
constexpr void assign(Dual<T>& self, const ScalarDualMulExpr<U, R>& other, Dual<T>& tmp)
{
    assign(self, other);
}

//-----------------------------------------------------------------------------
// assign: self = function(expr)
//-----------------------------------------------------------------------------
template<typename T, typename Op, typename R>
constexpr void assign(Dual<T>& self, const UnaryExpr<Op, R>& other)
{
    assign(self, other.r);
    apply<Op>(self);
}

template<typename T, typename Op, typename R>
constexpr void assign(Dual<T>& self, const UnaryExpr<Op, R>& other, Dual<T>& tmp)
{
    assign(self, other.r, tmp);
    apply<Op>(self);
}

//-----------------------------------------------------------------------------
// assign: self = expr + expr
//-----------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr void assign(Dual<T>& self, const AddExpr<L, R>& other)
{
    assign(self, other.r);
    assignAdd(self, other.l);
}

template<typename T, typename L, typename R>
constexpr void assign(Dual<T>& self, const AddExpr<L, R>& other, Dual<T>& tmp)
{
    assign(self, other.r, tmp);
    assignAdd(self, other.l, tmp);
}

//-----------------------------------------------------------------------------
// assign: self = expr * expr
//-----------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr void assign(Dual<T>& self, const MulExpr<L, R>& other)
{
    assign(self, other.r);
    assignMul(self, other.l);
}

template<typename T, typename L, typename R>
constexpr void assign(Dual<T>& self, const MulExpr<L, R>& other, Dual<T>& tmp)
{
    assign(self, other.r, tmp);
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
// assignAdd: self += scalar * dual
//-----------------------------------------------------------------------------
template<typename T, typename U, typename R>
constexpr void assignAdd(Dual<T>& self, const ScalarDualMulExpr<U, R>& other)
{
    self.val += other.scalar * other.dual.val;
    self.grad += other.scalar * other.dual.grad;
}

template<typename T, typename U, typename R>
constexpr void assignAdd(Dual<T>& self, const ScalarDualMulExpr<U, R>& other, Dual<T>& tmp)
{
    assignAdd(self, other);
}

//-----------------------------------------------------------------------------
// assignAdd: self += -expr
//-----------------------------------------------------------------------------
template<typename T>
constexpr void assignAddNegativeDual(Dual<T>& self, const Dual<T>& tmp)
{
    self.val -= tmp.val;
    self.grad -= tmp.grad;
}

template<typename T, typename R>
constexpr void assignAdd(Dual<T>& self, const NegExpr<R>& other)
{
    assignAddNegativeDual(self, eval(other.r));
}

template<typename T, typename R>
constexpr void assignAdd(Dual<T>& self, const NegExpr<R>& other, Dual<T>& tmp)
{
    tmp = eval(other.r);
    assignAddNegativeDual(self, tmp);
}

//-----------------------------------------------------------------------------
// assignAdd: self += 1 / expr
//-----------------------------------------------------------------------------
template<typename T>
constexpr void assignAddInverseDual(Dual<T>& self, const Dual<T>& tmp)
{
    const auto aux = static_cast<T>(1) / tmp.val;
    self.val += aux;
    self.grad -= aux * aux * tmp.grad;
}

template<typename T, typename R>
constexpr void assignAdd(Dual<T>& self, const InvExpr<R>& other)
{
    assignAddInverseDual(self, eval(other.r));
}

template<typename T, typename R>
constexpr void assignAdd(Dual<T>& self, const InvExpr<R>& other, Dual<T>& tmp)
{
    tmp = eval(other.r);
    assignAddInverseDual(self, tmp);
}

//-----------------------------------------------------------------------------
// assignAdd: self += expr + expr
//-----------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr void assignAdd(Dual<T>& self, const AddExpr<L, R>& other)
{
    assignAdd(self, other.l);
    assignAdd(self, other.r);
}

template<typename T, typename L, typename R>
constexpr void assignAdd(Dual<T>& self, const AddExpr<L, R>& other, Dual<T>& tmp)
{
    assignAdd(self, other.l, tmp);
    assignAdd(self, other.r, tmp);
}

//-----------------------------------------------------------------------------
// assignAdd: self += expr
//-----------------------------------------------------------------------------
template<typename T, typename R, enableif<isExpr<R>>..., disableif<isDual<R> || isScalarDualMulExpr<R> || isNegExpr<R> || isInvExpr<R> || isAddExpr<R>>...>
constexpr void assignAdd(Dual<T>& self, const R& other)
{
    Dual<T> tmp;
    assignAdd(self, other, tmp);
}

template<typename T, typename R, enableif<isExpr<R>>..., disableif<isDual<R> || isScalarDualMulExpr<R> || isNegExpr<R> || isInvExpr<R> || isAddExpr<R>>...>
constexpr void assignAdd(Dual<T>& self, const R& other, Dual<T>& tmp)
{
    assign(tmp, other);
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
template<typename T, typename R, enableif<isExpr<R>>..., disableif<isDual<R>>...>
constexpr void assignSub(Dual<T>& self, const R& other)
{
    assignAdd(self, negative(other));
}

template<typename T, typename R, enableif<isExpr<R>>..., disableif<isDual<R>>...>
constexpr void assignSub(Dual<T>& self, const R& other, Dual<T>& tmp)
{
    assignAdd(self, negative(other), tmp);
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
    assignMul(self, other.r);
    negate(self);
}

template<typename T, typename R>
constexpr void assignMul(Dual<T>& self, const NegExpr<R>& other, Dual<T>& tmp)
{
    assignMul(self, other.r, tmp);
    negate(self);
}

//-----------------------------------------------------------------------------
// assignMul: self *= scalar * dual
//-----------------------------------------------------------------------------
template<typename T, typename U, typename R>
constexpr void assignMul(Dual<T>& self, const ScalarDualMulExpr<U, R>& other)
{
    self.val *= other.scalar;
    self.grad *= other.scalar;
    self.grad *= other.dual.val;
    self.grad += self.val * other.dual.grad;
    self.val *= other.dual.val;
}

template<typename T, typename U, typename R>
constexpr void assignMul(Dual<T>& self, const ScalarDualMulExpr<U, R>& other, Dual<T>& tmp)
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
    assignMul(self, other.r);
}

template<typename T, typename L, typename R>
constexpr void assignMul(Dual<T>& self, const MulExpr<L, R>& other, Dual<T>& tmp)
{
    assignMul(self, other.l, tmp);
    assignMul(self, other.r, tmp);
}

//-----------------------------------------------------------------------------
// assignMul: self *= expr
//-----------------------------------------------------------------------------
template<typename T, typename R, enableif<isExpr<R>>..., disableif<isDual<R> || isNegExpr<R> || isScalarDualMulExpr<R> || isMulExpr<R>>...>
constexpr void assignMul(Dual<T>& self, const R& other)
{
    Dual<T> tmp;
    assignMul(self, other, tmp);
}

template<typename T, typename R, enableif<isExpr<R>>..., disableif<isDual<R> || isNegExpr<R> || isScalarDualMulExpr<R> || isMulExpr<R>>...>
constexpr void assignMul(Dual<T>& self, const R& other, Dual<T>& tmp)
{
    assign(tmp, other);
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
template<typename T, typename R, enableif<isExpr<R>>..., disableif<isDual<R>>...>
constexpr void assignDiv(Dual<T>& self, const R& other)
{
    assignMul(self, inverse(other));
}

template<typename T, typename R, enableif<isExpr<R>>..., disableif<isDual<R>>...>
constexpr void assignDiv(Dual<T>& self, const R& other, Dual<T>& tmp)
{
    assignMul(self, inverse(other), tmp);
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
    const auto aux = -static_cast<T>(1) / std::sqrt(1.0 - self.val * self.val);
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
    constexpr auto ln10 = 2.3025850929940456840179914546843;
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

template<typename T>
std::ostream& operator<<(std::ostream& out, const Dual<T>& x)
{
    out << x.val;
    return out;
}

} // namespace forward

using forward::val;
using forward::eval;
using forward::grad;
using forward::wrt;

using dual = forward::Dual<double>;

} // namespace autodiff
