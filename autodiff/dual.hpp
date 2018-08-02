// autodiff - automatic differentiation made easier in C++
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

namespace autodiff {
namespace forward {

template<typename Derived>
struct Expr;

template<typename T>
struct Dual;

template<typename Derived>
struct Expr
{
    auto derived() -> Derived& { return static_cast<Derived&>(*this); }
    auto derived() const -> const Derived& { return static_cast<const Derived&>(*this); }
};

template<typename T = double>
struct Dual : Expr<Dual<T>>
{
    T val;

    T grad;

    Dual() : Dual(0.0) {}

    Dual(T val) : val(val), grad(0.0) {}

    template<typename Other>
    Dual(const Expr<Other>& expr) : Dual(expr.eval()) {}

    // Arithmetic-assignment operators
    // Dual& operator+=(const ExprPtr& other) { expr = expr + other; return *this; }
    // Dual& operator-=(const ExprPtr& other) { expr = expr - other; return *this; }
    // Dual& operator*=(const ExprPtr& other) { expr = expr * other; return *this; }
    // Dual& operator/=(const ExprPtr& other) { expr = expr / other; return *this; }
    // Dual& operator+=(double other) { expr = expr + constant(other); return *this; }
    // Dual& operator-=(double other) { expr = expr - constant(other); return *this; }
    // Dual& operator*=(double other) { expr = expr * constant(other); return *this; }
    // Dual& operator/=(double other) { expr = expr / constant(other); return *this; }
};


template<typename T = double>
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

//------------------------------------------------------------------------------
// LIST OF OPERATORS AND EXPRESSION ALIAS
//------------------------------------------------------------------------------
// NEGATIVE OPERATOR
struct NegOp {};

// INVERSE OPERATOR
struct InvOp {};

// BINARY ARITHMETIC OPERATORS
struct AddOp {};
struct SubOp {};
struct MulOp {};
struct DivOp {};

// TRIGONOMETRIC OPERATORS
struct SinOp {};
struct CosOp {};
struct TanOp {};
struct ArcSinOp {};
struct ArcCosOp {};
struct ArcTanOp {};

// EXPONENTIAL AND LOGARITHMIC OPERATORS
struct ExpOp {};
struct LogOp {};
struct Log10Op {};

// POWER OPERATORS
struct PowOp {};
struct SqrtOp {};

// OTHER OPERATORS
struct AbsOp {};

//------------------------------------------------------------------------------
// LIST OF EXPRESSION ALIASES
//------------------------------------------------------------------------------
// INVERSE EXPRESSION
template<typename R> using NegExpr = UnaryExpr<NegOp, R>;

// INVERSE EXPRESSION
template<typename R> using InvExpr = UnaryExpr<InvOp, R>;

// BINARY ARITHMETIC EXPRESSION ALIASES
template<typename L, typename R> using AddExpr = UnaryExpr<AddOp, R>;
template<typename L, typename R> using MulExpr = UnaryExpr<MulOp, R>;

// TRIGONOMETRIC EXPRESSION ALIASES
template<typename R> using SinExpr    = UnaryExpr<SinOp, R>;
template<typename R> using CosExpr    = UnaryExpr<CosOp, R>;
template<typename R> using TanExpr    = UnaryExpr<TanOp, R>;
template<typename R> using ArcSinExpr = UnaryExpr<ArcSinOp, R>;
template<typename R> using ArcCosExpr = UnaryExpr<ArcCosOp, R>;
template<typename R> using ArcTanExpr = UnaryExpr<ArcTanOp, R>;

// EXPONENTIAL AND LOGARITHMIC EXPRESSION ALIASES
template<typename R> using ExpExpr   = UnaryExpr<ExpOp, R>;
template<typename R> using LogExpr   = UnaryExpr<LogOp, R>;
template<typename R> using Log10Expr = UnaryExpr<Log10Op, R>;

// POWER EXPRESSION ALIASES
template<typename L, typename R> using PowExpr  = BinaryExpr<PowOp, L, R>;
template<typename R>             using SqrtExpr = UnaryExpr<SqrtOp, R>;

// OTHER EXPRESSION ALIASES
template<typename R> using AbsExpr = UnaryExpr<AbsOp, R>;


//------------------------------------------------------------------------------
// TYPE TRAITS UTILITIES
//------------------------------------------------------------------------------
namespace traits {

template<typename Expr>
struct isDualExpr { constexpr static bool value = false; };

template<typename T>
struct isDualExpr<Dual<T>> { constexpr static bool value = true; };

template<typename Expr>
struct isConstantExpr { constexpr static bool value = false; };

template<typename T>
struct isConstantExpr<ConstantExpr<T>> { constexpr static bool value = true; };

template<typename Expr>
struct isNegExpr { constexpr static bool value = false; };

template<typename T>
struct isNegExpr<NegExpr<T>> { constexpr static bool value = true; };

template<typename Expr>
struct isInvExpr { constexpr static bool value = false; };

template<typename T>
struct isInvExpr<InvExpr<T>> { constexpr static bool value = true; };

template<typename Expr>
struct isScaleExpr { constexpr static bool value = false; };

template<typename T, typename R>
struct isScaleExpr<ScaleExpr<T, R>> { constexpr static bool value = true; };

template<typename Expr>
struct isUnaryExpr { constexpr static bool value = false; };

template<typename Op, typename R>
struct isUnaryExpr<UnaryExpr<Op, R>> { constexpr static bool value = true; };

template<typename Expr>
struct isBinaryExpr { constexpr static bool value = false; };

template<typename Op, typename L, typename R>
struct isBinaryExpr<BinaryExpr<Op, L, R>> { constexpr static bool value = true; };

} // namespace traits


template<bool value>
using enableif = typename std::enable_if<value>::type;

template<bool value>
using disableif = typename std::enable_if<!value>::type;

template<typename T>
using plain = typename std::remove_cv<std::remove_reference<T>::type>::type;

template<typename A, typename B>
using common = typename std::common_type<A, B>::type;

template<typename A, typename B>
constexpr bool isSame() { return std::is_same<A, B>::value; }

template<typename T>
constexpr bool isNumber() { return std::is_arithmetic<T>::value; }

template<typename Expr>
constexpr bool isDualExpr() { return traits::isDualExpr<plain<Expr>>::value; }

template<typename Expr>
constexpr bool isConstantExpr() { return traits::isConstantExpr<plain<Expr>>::value; }

template<typename Expr>
constexpr bool isNegExpr() { return traits::isNegExpr<plain<Expr>>::value; }

template<typename Expr>
constexpr bool isInvExpr() { return traits::isInvExpr<plain<Expr>>::value; }

template<typename Expr>
constexpr bool isScaleExpr() { return traits::isScaleExpr<plain<Expr>>::value; }

template<typename Expr>
constexpr bool isUnaryExpr() { return traits::isUnaryExpr<plain<Expr>>::value; }

template<typename Expr>
constexpr bool isBinaryExpr() { return traits::isBinaryExpr<plain<Expr>>::value; }


//------------------------------------------------------------------------------
// CONVENIENT FUNCTIONS
//------------------------------------------------------------------------------
template<typename T>
constexpr auto constant(T&& val) -> ConstantExpr<T>
{
    return { std::forward<T>(val) };
}

//------------------------------------------------------------------------------
// INVERSE FUNCTION: GENERAL CASE
//------------------------------------------------------------------------------
template<typename R, disableif<isInvExpr<R>>...>
constexpr auto inverse(const Expr<R>& expr) -> InvExpr<R>
{
    return { expr.derived() };
}

//------------------------------------------------------------------------------
// INVERSE FUNCTION: inverse(InvExpr)
//------------------------------------------------------------------------------
template<typename R>
constexpr auto inverse(const InvExpr<R>& expr) -> const R&
{
    return expr.r;
}

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// POSITIVE OPERATOR: GENERAL CASE
//------------------------------------------------------------------------------
template<typename R>
constexpr auto operator+(const Expr<R>& r) -> const R&
{
    return r.derived();
};

//------------------------------------------------------------------------------
// NEGATIVE OPERATOR: GENERAL CASE
//------------------------------------------------------------------------------
template<typename R,
    disableif<isNegExpr<R>>...,
    disableif<isScaleExpr<R>>...>
constexpr auto operator-(const Expr<R>& expr) -> NegExpr<R>
{
    return { expr.derived() };
}

//------------------------------------------------------------------------------
// NEGATIVE OPERATOR: -NegExpr
//------------------------------------------------------------------------------
template<typename R>
constexpr auto operator-(const NegExpr<R>& expr) -> decltype(expr.r)
{
    return expr.r;
}

//------------------------------------------------------------------------------
// NEGATIVE OPERATOR: -ScaleExpr
//------------------------------------------------------------------------------
template<typename T, typename R>
constexpr auto operator-(const ScaleExpr<T, R>& expr) -> decltype((-expr.scalar) * expr.r)
{
    return (-expr.scalar) * expr.r;
}

//------------------------------------------------------------------------------
// ADDITION OPERATOR: GENERAL CASE
//------------------------------------------------------------------------------
template<typename L, typename R,
    disableif<isConstantExpr<L> && isConstantExpr<R>>...,
    disableif<isNegExpr<L> && isNegExpr<R>>...>
constexpr auto operator+(const Expr<L>& l, const Expr<R>& r) -> AddExpr<L, R>
{
    return { l.derived(), r.derived() };
}

//------------------------------------------------------------------------------
// ADDITION OPERATOR: ConstantExpr + ConstantExpr
//------------------------------------------------------------------------------
template<typename L, typename R>
constexpr auto operator+(const ConstantExpr<L>& l, const ConstantExpr<R>& r) -> decltype(constant(l.val + r.val))
{
    return constant(l.val + r.val);
}

//------------------------------------------------------------------------------
// ADDITION OPERATOR: NegExpr + NegExpr
//------------------------------------------------------------------------------
template<typename L, typename R>
constexpr auto operator+(const NegExpr<L>& l, const NegExpr<R>& r) -> decltype(-(l.r + r.r))
{
    return -(l.r + r.r);
}

//------------------------------------------------------------------------------
// SUBTRACTION OPERATOR: GENERAL CASE
//------------------------------------------------------------------------------
template<typename L, typename R,
    disableif<isConstantExpr<L> && isConstantExpr<R>>...>
constexpr auto operator-(const Expr<L>& l, const Expr<R>& r) -> decltype(l + (-r))
{
    return l + (-r);
}

//------------------------------------------------------------------------------
// SUBTRACTION OPERATOR: ConstantExpr - ConstantExpr
//------------------------------------------------------------------------------
template<typename L, typename R>
constexpr auto operator-(const ConstantExpr<L>& l, const ConstantExpr<R>& r) -> decltype(constant(l.val - r.val))
{
    return constant(l.val - r.val);
}

//------------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: GENERAL CASE
//------------------------------------------------------------------------------
template<typename L, typename R,
    disableif<isConstantExpr<L> && isConstantExpr<R>>...,
    disableif<isNegExpr<L> && isNegExpr<R>>...,
    disableif<isInvExpr<L> && isInvExpr<R>>...,
    disableif<isConstantExpr<L> && isScaleExpr<R>>...,
    disableif<isScaleExpr<L> && isConstantExpr<R>>...,
    disableif<isNegExpr<L> && isScaleExpr<R>>...,
    disableif<isScaleExpr<L> && isNegExpr<R>>...,
    disableif<isScaleExpr<L> && isScaleExpr<R>>...>
constexpr auto operator*(const Expr<L>& l, const Expr<R>& r) -> MulExpr<L, R>
{
    return { l.derived(), r.derived() };
}

//------------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: ConstantExpr * ConstantExpr
//------------------------------------------------------------------------------
template<typename L, typename R>
constexpr auto operator*(const ConstantExpr<L>& l, const ConstantExpr<R>& r) -> decltype(constant(l.val * r.val))
{
    return constant(l.val * r.val);
}

//------------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: NegExpr * NegExpr
//------------------------------------------------------------------------------
template<typename L, typename R>
constexpr auto operator*(const NegExpr<L>& l, const NegExpr<R>& r) -> decltype(l.r * r.r)
{
    return l.r * r.r;
}

//------------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: InvExpr * InvExpr
//------------------------------------------------------------------------------
template<typename L, typename R>
constexpr auto operator*(const InvExpr<L>& l, const InvExpr<R>& r) -> decltype(inverse(l.r * r.r))
{
    return inverse(l.r * r.r);
}

//------------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: ConstantExpr * ScaleExpr
//------------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr auto operator*(const ConstantExpr<L>& l, const ScaleExpr<T, R>& r) -> decltype((l.val * r.scalar) * r.r)
{
    return (l.val * r.scalar) * r.r;
}

//------------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: ScaleExpr * ConstantExpr
//------------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr auto operator*(const ScaleExpr<T, L>& l, const ConstantExpr<R>& r) -> decltype((r.val * l.scalar) * l.r)
{
    return (r.val * l.scalar) * l.r;
}

//------------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: NegExpr * ScaleExpr
//------------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr auto operator*(const NegExpr<L>& l, const ScaleExpr<T, R>& r) -> decltype((-r.scalar) * (l.r * r.r))
{
    return (-r.scalar) * (l.r * r.r);
}

//------------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: ScaleExpr * NegExpr
//------------------------------------------------------------------------------
template<typename T, typename L, typename R>
constexpr auto operator*(const ScaleExpr<T, L>& l, const NegExpr<R>& r) -> decltype((-l.scalar) * (l.r * r.r))
{
    return (-l.scalar) * (l.r * r.r);
}

//------------------------------------------------------------------------------
// MULTIPLICATION OPERATOR: ScaleExpr * ScaleExpr
//------------------------------------------------------------------------------
template<typename T, typename U, typename L, typename R>
constexpr auto operator*(const ScaleExpr<T, L>& l, const ScaleExpr<U, R>& r) -> decltype((l.scalar * l.scalar) * (l.r * r.r))
{
    return (l.scalar * l.scalar) * (l.r * r.r);
}

//------------------------------------------------------------------------------
// DIVISION OPERATOR: GENERAL CASE
//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS
//------------------------------------------------------------------------------
template<typename R> constexpr auto sin(const Expr<R>& r) -> SinExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto cos(const Expr<R>& r) -> CosExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto tan(const Expr<R>& r) -> TanExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto asin(const Expr<R>& r) -> ArcSinExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto acos(const Expr<R>& r) -> ArcCosExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto atan(const Expr<R>& r) -> ArcTanExpr<R> { return { r.derived() }; }

//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS
//------------------------------------------------------------------------------
template<typename R> constexpr auto exp(const Expr<R>& r) -> ExpExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto log(const Expr<R>& r) -> LogExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto log10(const Expr<R>& r) -> Log10Expr<R> { return { r.derived() }; }

//------------------------------------------------------------------------------
// POWER FUNCTIONS
//------------------------------------------------------------------------------
template<typename L, typename R> constexpr auto pow(const Expr<L>& l, const Expr<R>& r) -> PowExpr<L, R> { return { l.derived(), r.derived() }; }
template<typename R> constexpr auto sqrt(const Expr<R>& r) -> SqrtExpr<R> { return { r.derived() }; }

//------------------------------------------------------------------------------
// OTHER FUNCTIONS
//------------------------------------------------------------------------------
template<typename R> constexpr auto abs(const Expr<R>& r) -> AbsExpr<R> { return { r.derived() }; }
template<typename R> constexpr auto abs2(const Expr<R>& r) -> decltype(r * r) { return r * r; }
template<typename R> constexpr auto conj(const Expr<R>& r) -> decltype(r) { return r; }
template<typename R> constexpr auto real(const Expr<R>& r) -> decltype(r) { return r; }
template<typename R> constexpr auto imag(const Expr<R>& r) -> decltype(constant(0.0)) { return constant(0.0); }


//------------------------------------------------------------------------------
// COMPARISON OPERATORS
//------------------------------------------------------------------------------
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


//------------------------------------------------------------------------------
// FORWARD DECLARATIONS
//------------------------------------------------------------------------------
template<typename Op, typename T>
constexpr void apply(Dual<T>& self);

template<typename IncrementOp, typename T, typename Derived>
constexpr void increment(Dual<T>& self, const Expr<Derived>& other);


//------------------------------------------------------------------------------
// AUXILIARY FUNCTIONS
//------------------------------------------------------------------------------
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


//------------------------------------------------------------------------------
// EXPRESSION EVALUATION FUNCTIONS
//------------------------------------------------------------------------------
template<typename T, typename R, disableif<isDualExpr<R>>..., disableif<isConstantExpr<R>>...>
constexpr auto eval(const Expr<R>& expr) -> Dual<T>
{
    Dual<T> tmp;
    assign(tmp, expr);
    return tmp;
}

template<typename T>
constexpr auto eval(const Dual<T>& expr) -> const Dual<T>&
{
    return expr;
}

template<typename T>
constexpr auto eval(const ConstantExpr<T>& expr) -> const ConstantExpr<T>&
{
    return expr;
}


//------------------------------------------------------------------------------
// ASSIGNMENT FUNCTIONS
//------------------------------------------------------------------------------
template<typename T>
constexpr void assign(Dual<T>& self, const Dual<T>& other)
{
    self.val = other.val;
    self.grad = other.grad;
}

template<typename T, typename U>
constexpr void assign(Dual<T>& self, const ConstantExpr<U>& other)
{
    self.val = other.val;
    self.grad = 0.0;
}

template<typename T, typename U, typename R>
constexpr void assign(Dual<T>& self, const ScaleExpr<U, R>& other)
{
    assign(self, other.r);
    scale(self, other.scalar);
}

template<typename T, typename Op, typename R>
constexpr void assign(Dual<T>& self, const UnaryExpr<Op, R>& other)
{
    assign(self, other.r);
    apply<Op>(self);
}

template<typename T, typename Op, typename L, typename R>
constexpr void assign(Dual<T>& self, const BinaryExpr<Op, L, R>& other)
{
    assign(self, other.l);
    increment<Op>(self, other.r);
}


//------------------------------------------------------------------------------
// INCREMENT FUNCTIONS FOR DUAL EXPRESSIONS
//------------------------------------------------------------------------------
template<typename T>
constexpr void increment(Dual<T>& self, const Dual<T>& other, AddOp)
{
    self.val += other.val;
    self.grad += other.grad;
}

template<typename T>
constexpr void increment(Dual<T>& self, const Dual<T>& other, SubOp)
{
    self.val -= other.val;
    self.grad -= other.grad;
}

template<typename T>
constexpr void increment(Dual<T>& self, const Dual<T>& other, MulOp)
{
    self.grad *= other.val;
    self.grad += self.val * other.grad;
    self.val  *= other.val;
}

template<typename T>
constexpr void increment(Dual<T>& self, const Dual<T>& other, DivOp)
{
    const auto aux = static_cast<T>(1) / other.val;
    self.val  *= aux;
    self.grad *= aux;
    self.grad -= self.val * aux;
}

//------------------------------------------------------------------------------
// INCREMENT FUNCTIONS FOR CONSTANT EXPRESSIONS
//------------------------------------------------------------------------------
template<typename T, typename U>
constexpr void increment(Dual<T>& self, const ConstantExpr<U>& other, AddOp)
{
    self.val += other.val;
}

template<typename T, typename U>
constexpr void increment(Dual<T>& self, const ConstantExpr<U>& other, SubOp)
{
    self.val -= other.val;
}

template<typename T, typename U>
constexpr void increment(Dual<T>& self, const ConstantExpr<U>& other, MulOp)
{
    self.val *= other.val;
    self.grad *= other.val;
}

template<typename T, typename U>
constexpr void increment(Dual<T>& self, const ConstantExpr<U>& other, DivOp)
{
    const auto aux = static_cast<T>(1) / other.val;
    self.val *= aux;
    self.grad *= aux;
}

//------------------------------------------------------------------------------
// INCREMENT FUNCTIONS FOR NEGATIVE EXPRESSIONS
//------------------------------------------------------------------------------
template<typename T, typename R>
constexpr void increment(Dual<T>& self, const NegExpr<R>& other, AddOp)
{
    increment<SubOp>(self, other);
}

template<typename T, typename R>
constexpr void increment(Dual<T>& self, const NegExpr<R>& other, SubOp)
{
    increment<AddOp>(self, other);
}

template<typename T, typename R>
constexpr void increment(Dual<T>& self, const NegExpr<R>& other, MulOp)
{
    increment<MulOp>(self, other.r);
    negate(self);
}

template<typename T, typename R>
constexpr void increment(Dual<T>& self, const NegExpr<R>& other, DivOp)
{
    increment<DivOp>(self, other.r);
    negate(self);
}

//------------------------------------------------------------------------------
// INCREMENT FUNCTIONS FOR SCALE EXPRESSIONS
//------------------------------------------------------------------------------
template<typename T, typename U, typename R>
constexpr void increment(Dual<T>& self, const ScaleExpr<U, R>& other, AddOp)
{
    const auto tmp = eval(other.r);
    self.val += other.scalar * tmp.val;
    self.grad += other.scalar * tmp.grad;
}

template<typename T, typename U, typename R>
constexpr void increment(Dual<T>& self, const ScaleExpr<U, R>& other, SubOp)
{
    increment<AddOp>(self, other);
}

template<typename T, typename U, typename R>
constexpr void increment(Dual<T>& self, const ScaleExpr<U, R>& other, MulOp)
{
    increment<MulOp>(self, other.r);
    negate(self);
}

template<typename T, typename U, typename R>
constexpr void increment(Dual<T>& self, const ScaleExpr<U, R>& other, DivOp)
{
    increment<DivOp>(self, other.r);
    negate(self);
}

//------------------------------------------------------------------------------
// INCREMENT FUNCTIONS FOR UNARY AND BINARY EXPRESSIONS
//------------------------------------------------------------------------------
template<typename T, typename Op, typename R, typename IncrementOp>
constexpr void increment(Dual<T>& self, const UnaryExpr<Op, R>& other, IncrementOp)
{
    Dual<T> tmp;
    assign(tmp, other);
    increment<IncrementOp>(self, tmp);
}

template<typename T, typename Op, typename L, typename R, typename IncrementOp>
constexpr void increment(Dual<T>& self, const BinaryExpr<Op, L, R>& other, IncrementOp)
{
    Dual<T> tmp;
    assign(tmp, other);
    increment<IncrementOp>(self, tmp);
}

template<typename IncrementOp, typename T, typename Derived>
constexpr void increment(Dual<T>& self, const Expr<Derived>& other)
{
    increment(self, other.derived(), IncrementOp{});
}

//------------------------------------------------------------------------------
// APPLY FUNCTIONS
//------------------------------------------------------------------------------
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
    self.val = std::sin(self.val);
    self.grad *= std::cos(self.val);
}

template<typename T>
constexpr void apply(Dual<T>& self, CosOp)
{
    self.val = std::cos(self.val);
    self.grad *= -std::sin(self.val);
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
