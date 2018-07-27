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

template<typename Op, typename R>
struct UnaryExpr : Expr<UnaryExpr<Op, R>>
{
    R r;
    constexpr UnaryExpr(R&& r) : r(std::forward<Right>(r)) {}
};

template<typename Op, typename L, typename R>
struct BinaryExpr : Expr<BinaryExpr<Op, L, R>>
{
    L l;
    R r;
    constexpr BinaryExpr(L&& l, R&& r) : l(std::forward<Left>(l)), r(std::forward<Right>(r)) {}
};

//------------------------------------------------------------------------------
// LIST OF OPERATORS AND EXPRESSION ALIAS
//------------------------------------------------------------------------------
// UNARY ARITHMETIC OPERATORS
//struct PosOp {}; //  todo remove as only sub is transformed into add and div in mul
struct NegOp {};

// INVERSE OPERATOR
struct InvOp {};

// BINARY ARITHMETIC OPERATORS
struct AddOp {};
struct MulOp {};
//struct SubOp {}; //  todo remove as only sub is transformed into add and div in mul
//struct DivOp {}; //  todo remove as only sub is transformed into add and div in mul

template<double scalar>
struct ScaOp;

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
// UNARY ARITHMETIC EXPRESSION ALIASES
template<typename R> using NegExpr = UnaryExpr<NegOp, R>;
//template<typename R> using PosExpr = UnaryExpr<PosOp, R>; //  todo remove as only sub is transformed into add and div in mul

// INVERSE OPERATOR
template<typename R> using InvExpr = UnaryExpr<InvOp, R>;

// BINARY ARITHMETIC EXPRESSION ALIASES
template<typename L, typename R> using AddExpr = UnaryExpr<AddOp, R>;
template<typename L, typename R> using MulExpr = UnaryExpr<MulOp, R>;
//template<typename L, typename R> using SubExpr = UnaryExpr<SubOp, R>; //  todo remove as only sub is transformed into add and div in mul
//template<typename L, typename R> using DivExpr = UnaryExpr<DivOp, R>; //  todo remove as only sub is transformed into add and div in mul

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
// CONVENIENT FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> constexpr auto constant(T&& val) -> ConstantExpr<T> { return { std::forward<T>(val) }; }
template<typename R> constexpr auto inverse(const Expr<R>& expr) -> InvExpr<R> { return { expr.derived() }; }

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS
//------------------------------------------------------------------------------
//template<typename R> constexpr auto operator+(const Expr<R>& r) -> PosExpr<R> { return { r }; } //  todo remove as only sub is transformed into add and div in mul
template<typename R> constexpr auto operator+(const Expr<R>& r) -> const R& { return r.derived(); };
template<typename R> constexpr auto operator-(const Expr<R>& r) -> NegExpr<R> { return { r }; }

template<typename L, typename R> constexpr auto operator+(const Expr<L>& l, const Expr<R>& r) -> AddExpr<L, R> { return { l, r }; }
template<typename L, typename R> constexpr auto operator*(const Expr<L>& l, const Expr<R>& r) -> MulExpr<L, R> { return { l, r }; }
template<typename L, typename R> constexpr auto operator-(const Expr<L>& l, const Expr<R>& r) -> decltype(l + (-r)) { return l + (-r); }
template<typename L, typename R> constexpr auto operator/(const Expr<L>& l, const Expr<R>& r) -> decltype(l * inverse(r)) { return l * inverse(r); }
//template<typename L, typename R> constexpr auto operator-(const Expr<L>& l, const Expr<R>& r) -> SubExpr<L, R> { return { l, r }; } //  todo remove as only sub is transformed into add and div in mul
//template<typename L, typename R> constexpr auto operator/(const Expr<L>& l, const Expr<R>& r) -> DivExpr<L, R> { return { l, r }; } //  todo remove as only sub is transformed into add and div in mul

template<typename L, typename R> constexpr auto operator+(double l, const Expr<R>& r) -> decltype(constant(l) + r) { return constant(l) + r; }
template<typename L, typename R> constexpr auto operator-(double l, const Expr<R>& r) -> decltype(constant(l) - r) { return constant(l) - r; }
template<typename L, typename R> constexpr auto operator*(double l, const Expr<R>& r) -> decltype(constant(l) * r) { return constant(l) * r; }
template<typename L, typename R> constexpr auto operator/(double l, const Expr<R>& r) -> decltype(constant(l) / r) { return constant(l) / r; }

template<typename L, typename R> constexpr auto operator+(const Expr<L>& l, double r) -> decltype(l + constant(r)) { return l + constant(r); }
template<typename L, typename R> constexpr auto operator-(const Expr<L>& l, double r) -> decltype(l - constant(r)) { return l - constant(r); }
template<typename L, typename R> constexpr auto operator*(const Expr<L>& l, double r) -> decltype(l * constant(r)) { return l * constant(r); }
template<typename L, typename R> constexpr auto operator/(const Expr<L>& l, double r) -> decltype(l / constant(r)) { return l / constant(r); }

//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS
//------------------------------------------------------------------------------
template<typename R> constexpr auto sin(const Expr<R>& r) -> SinExpr<R> { return { r }; }
template<typename R> constexpr auto cos(const Expr<R>& r) -> CosExpr<R> { return { r }; }
template<typename R> constexpr auto tan(const Expr<R>& r) -> TanExpr<R> { return { r }; }
template<typename R> constexpr auto asin(const Expr<R>& r) -> ArcSinExpr<R> { return { r }; }
template<typename R> constexpr auto acos(const Expr<R>& r) -> ArcCosExpr<R> { return { r }; }
template<typename R> constexpr auto atan(const Expr<R>& r) -> ArcTanExpr<R> { return { r }; }

//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS
//------------------------------------------------------------------------------
template<typename R> constexpr auto exp(const Expr<R>& r) -> ExpExpr<R> { return { r }; }
template<typename R> constexpr auto log(const Expr<R>& r) -> LogExpr<R> { return { r }; }
template<typename R> constexpr auto log10(const Expr<R>& r) -> Log10Expr<R> { return { r }; }

//------------------------------------------------------------------------------
// POWER FUNCTIONS
//------------------------------------------------------------------------------
template<typename L, typename R> constexpr auto pow(const Expr<L>& l, const Expr<R>& r) -> PowExpr<L, R> { return { l, r }; }
template<typename R> constexpr auto sqrt(const Expr<R>& r) -> SqrtExpr<R> { return { r }; }

//------------------------------------------------------------------------------
// OTHER FUNCTIONS
//------------------------------------------------------------------------------
template<typename R> constexpr auto abs(const Expr<R>& r) -> AbsExpr<R> { return { r }; }
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
// TYPE TRAITS UTILITIES
//------------------------------------------------------------------------------
template<bool value>
using enableif = typename std::enable_if<value>::type;

template<bool value>
using disableif = typename std::enable_if<!value>::type;

template<typename A, typename B>
constexpr bool same() { return std::is_same<A, B>::value; }

//------------------------------------------------------------------------------
// EXPRESSION EVALUATION FUNCTIONS
//------------------------------------------------------------------------------
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

template<typename T, typename Op, typename R>
constexpr auto eval(const UnaryExpr<Op, R>& expr) -> Dual<T>
{
    Dual<T> res;
    assign(res, expr);
    return res;
}

template<typename T, typename Op, typename L, typename R>
constexpr auto eval(const BinaryExpr<Op, L, R>& expr) -> Dual<T>
{
    Dual<T> res;
    assign(res, expr);
    return res;
}

//------------------------------------------------------------------------------
// INCREMENT FUNCTIONS FOR DUAL AND CONSTANT EXPRESSIONS
//------------------------------------------------------------------------------
template<typename T> constexpr void increment(Dual<T>& self, const Dual<T>& other, AddOp) { self += other; }
template<typename T> constexpr void increment(Dual<T>& self, const Dual<T>& other, MulOp) { self *= other; }

template<typename T> constexpr void increment(Dual<T>& self, const ConstantExpr<T>& other, AddOp) { self.val += other.val; }
template<typename T> constexpr void increment(Dual<T>& self, const ConstantExpr<T>& other, MulOp) { self.val *= other.val; }

template<typename T, typename R> constexpr void increment(Dual<T>& self, const NegExpr<R>& other, AddOp) { self -= eval(other.r); }
template<typename T, typename R> constexpr void increment(Dual<T>& self, const NegExpr<R>& other, MulOp) { self *= eval(other.r); self *= -1.0; }

//------------------------------------------------------------------------------
// INCREMENT FUNCTIONS FOR UNARY EXPRESSIONS
//------------------------------------------------------------------------------
template<typename T, typename IncrementOp, typename Op, typename R, disableif<same<Op, NegOp>()>...>
constexpr void increment(Dual<T>& self, const UnaryExpr<Op, R>& expr, IncrementOp)
{
    Dual<T> tmp = eval(expr);
    increment(self, eval(expr), IncrementOp{});
}

//------------------------------------------------------------------------------
// INCREMENT FUNCTIONS FOR BINARY EXPRESSIONS
//------------------------------------------------------------------------------
template<typename T, typename IncrementOp, typename Op, typename L, typename R, enableif<same<IncrementOp, Op>()>...>
constexpr void increment(Dual<T>& self, const BinaryExpr<Op, L, R>& expr, IncrementOp)
{
    increment(self, expr.l, IncrementOp{});
    increment(self, expr.r, IncrementOp{});
}

template<typename T, typename IncrementOp, typename Op, typename L, typename R, disableif<same<IncrementOp, Op>()>...>
constexpr void increment(Dual<T>& self, const BinaryExpr<Op, L, R>& expr, IncrementOp)
{
    increment(self, eval(expr), IncrementOp{});
}

template<typename Op, typename T, typename Derived>
constexpr void increment(Dual<T>& self, const Expr<Derived>& expr) 
{
    increment(self, expr.derived(), Op{}); 
}

//------------------------------------------------------------------------------
// APPLY FUNCTIONS
//------------------------------------------------------------------------------
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
    const double aux = 1.0 / std::cos(self.val);
    self.val = std::tan(self.val);
    self.grad *= aux * aux;
}

template<typename T> 
constexpr void apply(Dual<T>& self, ArcSinOp)
{
    const double aux = 1.0 / std::sqrt(1.0 - self.val * self.val);
    self.val = std::asin(self.val);
    self.grad *= aux;
}

template<typename T> 
constexpr void apply(Dual<T>& self, ArcCosOp)
{
    const double aux = -1.0 / std::sqrt(1.0 - self.val * self.val);
    self.val = std::acos(self.val);
    self.grad *= aux;
}

template<typename T> 
constexpr void apply(Dual<T>& self, ArcTanOp)
{
    const double aux = 1.0 / (1.0 + self.val * self.val);
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
    const double aux = 1.0 / self.val;
    self.val = std::log(self.val);
    self.grad *= aux;
}

template<typename T> 
constexpr void apply(Dual<T>& self, Log10Op)
{
    constexpr double ln10 = 2.3025850929940456840179914546843;
    const double aux = 1.0 / (ln10 * self.val);
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
    const double aux = self.val;
    self.val = std::abs(self.val);
    self.grad *= aux / self.val;
}

template<typename Op, typename T>
constexpr void apply(Dual<T>& self)
{
    apply(self, Op{});
}

//------------------------------------------------------------------------------
// ASSIGN FUNCTION FOR BASE EXPR
//------------------------------------------------------------------------------

template<typename T, typename Derived>
void assign(Dual<T>& self, const Expr<Derived>& expr)
{
    assign(self, expr.derived()); 
}

template<typename T> constexpr void assign(Dual<T>& self, const Dual<T>& other) { self = other; }
template<typename T> constexpr void assign(Dual<T>& self, const ConstantExpr<T>& other) { self.val = other.val; self.grad = 0.0; }


template<typename T, typename Op, typename R>
void assign(Dual<T>& self, const UnaryExpr<Op, R>& expr) 
{
    assign(self, expr.r);
    apply<Op>(self);
}

template<typename T, typename Op, typename L, typename R>
void assign(Dual<T>& self, const BinaryExpr<Op, L, R>& expr)
{
    assign(self, expr.l);
    increment<Op>(self, expr.r);
}


//------------------------------------------------------------------------------
// ARITHMETIC-ASSIGN FUNCTIONS FOR BASE EXPR
//------------------------------------------------------------------------------
//template<typename Derived> void assignAdd(Dual<T>& self, const Expr<Derived>& expr) { assignAdd(self, expr.derived()); }
//template<typename Derived> void assignSub(Dual<T>& self, const Expr<Derived>& expr) { assignSub(self, expr.derived()); }
//template<typename Derived> void assignMul(Dual<T>& self, const Expr<Derived>& expr) { assignMul(self, expr.derived()); }
//template<typename Derived> void assignDiv(Dual<T>& self, const Expr<Derived>& expr) { assignDiv(self, expr.derived()); }
//
////------------------------------------------------------------------------------
//// ASSIGN FUNCTION FOR CONSTANT EXPR
////------------------------------------------------------------------------------
//template<typename T> void assign(Dual<T>& self, const ConstantExpr<T>& expr) { self.val = expr.val; self.grad = 0.0; }
//
////------------------------------------------------------------------------------
//// ARITHMETIC-ASSIGN FUNCTIONS FOR CONSTANT EXPR
////------------------------------------------------------------------------------
//template<typename T> void assignAdd(Dual<T>& self, const ConstantExpr<T>& expr) { self.val += expr.val; }
//template<typename T> void assignSub(Dual<T>& self, const ConstantExpr<T>& expr) { self.val -= expr.val; }
//template<typename T> void assignMul(Dual<T>& self, const ConstantExpr<T>& expr) { self.val *= expr.val; }
//template<typename T> void assignDiv(Dual<T>& self, const ConstantExpr<T>& expr) { self.val /= expr.val; }
//
////------------------------------------------------------------------------------
//// ASSIGN FUNCTIONS FOR POS AND NEG UNARY OPERATORS
////------------------------------------------------------------------------------
//template<typename R> void assign(Dual<T>& self, const PosExpr<R>& expr) { assign(self, expr.r); }
//template<typename R> void assign(Dual<T>& self, const NegExpr<R>& expr) { assign(self, expr.r); applyNeg(self); }
//
////------------------------------------------------------------------------------
//// ARITHMETIC-ASSIGN FUNCTIONS FOR POS AND NEG UNARY OPERATORS
////------------------------------------------------------------------------------
//template<typename R> void assignAdd(Dual<T>& self, const PosExpr<R>& expr) { assignAdd(self, expr.r); }
//template<typename R> void assignSub(Dual<T>& self, const PosExpr<R>& expr) { assignSub(self, expr.r); }
//template<typename R> void assignMul(Dual<T>& self, const PosExpr<R>& expr) { assignMul(self, expr.r); }
//template<typename R> void assignDiv(Dual<T>& self, const PosExpr<R>& expr) { assignDiv(self, expr.r); }
//
//template<typename R> void assignAdd(Dual<T>& self, const NegExpr<R>& expr) { assignAdd(self, expr.r); }
//template<typename R> void assignSub(Dual<T>& self, const NegExpr<R>& expr) { assignSub(self, expr.r); }
//template<typename R> void assignMul(Dual<T>& self, const NegExpr<R>& expr) { assignMul(self, expr.r); }
//template<typename R> void assignDiv(Dual<T>& self, const NegExpr<R>& expr) { assignDiv(self, expr.r); }
//
////------------------------------------------------------------------------------
//// ASSIGN FUNCTIONS FOR ADD, SUB, MUL, AND DIV BINARY OPERATORS
////------------------------------------------------------------------------------
//template<typename L, typename R> void assign(Dual<T>& self, const AddExpr<L, R>& expr) { assign(self, expr.l); assignAdd(self, expr.r); }
//template<typename L, typename R> void assign(Dual<T>& self, const SubExpr<L, R>& expr) { assign(self, expr.l); assignSub(self, expr.r); }
//template<typename L, typename R> void assign(Dual<T>& self, const MulExpr<L, R>& expr) { assign(self, expr.l); assignMul(self, expr.r); }
//template<typename L, typename R> void assign(Dual<T>& self, const DivExpr<L, R>& expr) { assign(self, expr.l); assignDiv(self, expr.r); }
//
////------------------------------------------------------------------------------
//// ASSIGN FUNCTIONS FOR TRIGONOMETRIC FUNCTIONS
////------------------------------------------------------------------------------
//template<typename R> void assign(Dual<T>& self, const SinExpr<R>& expr)    { assign(self, expr.r); applySin(self); }
//template<typename R> void assign(Dual<T>& self, const CosExpr<R>& expr)    { assign(self, expr.r); applyCos(self); }
//template<typename R> void assign(Dual<T>& self, const TanExpr<R>& expr)    { assign(self, expr.r); applyTan(self); }
//template<typename R> void assign(Dual<T>& self, const ArcSinExpr<R>& expr) { assign(self, expr.r); applyArcSin(self); }
//template<typename R> void assign(Dual<T>& self, const ArcCosExpr<R>& expr) { assign(self, expr.r); applyArcCos(self); }
//template<typename R> void assign(Dual<T>& self, const ArcTanExpr<R>& expr) { assign(self, expr.r); applyArcTan(self); }
//
////------------------------------------------------------------------------------
//// ASSIGN FUNCTIONS FOR EXPONENTIAL AND LOGARITHMIC FUNCTIONS
////------------------------------------------------------------------------------
//template<typename R> void assign(Dual<T>& self, const ExpExpr<R>& expr)    { assign(self, expr.r); applyExp(self); }
//template<typename R> void assign(Dual<T>& self, const LogExpr<R>& expr)    { assign(self, expr.r); applyLog(self); }
//template<typename R> void assign(Dual<T>& self, const Log10Expr<R>& expr)  { assign(self, expr.r); applyLog10(self); }
//
////------------------------------------------------------------------------------
//// ASSIGN FUNCTIONS FOR POWER FUNCTIONS
////------------------------------------------------------------------------------
//template<typename L, typename R> void assign(Dual<T>& self, const PowExpr<L, R>& expr) { assign(self, expr.r); applyPow(self, expr.r); }
//template<typename R>             void assign(Dual<T>& self, const SqrtExpr<R>& expr)   { assign(self, expr.r); applySqrt(self); }
//
////------------------------------------------------------------------------------
//// ASSIGN FUNCTIONS FOR OTHER FUNCTIONS
////------------------------------------------------------------------------------
//template<typename R> void assign(Dual<T>& self, const AbsExpr<R>& expr) { assign(self, expr.r); applyAbs(self); }
//
//
//template<typename Derived> void assignAdd(Dual<T>& self, const Expr<Derived>& expr) { assignAdd(self, expr.derived()); }
//template<typename Derived> void assignSub(Dual<T>& self, const Expr<Derived>& expr) { assignSub(self, expr.derived()); }
//template<typename Derived> void assignMul(Dual<T>& self, const Expr<Derived>& expr) { assignMul(self, expr.derived()); }
//template<typename Derived> void assignDiv(Dual<T>& self, const Expr<Derived>& expr) { assignDiv(self, expr.derived()); }
//
//template<typename Derived>
//void assignDiv(Dual<T>& self, const Expr<Derived>& expr) { assignDiv(self, expr.derived()); }
//
//template<typename L, typename R>
//void assign(Dual<T>& self, const AddExpr<L, R>& expr)
//{
//    assign(self, expr.l);
//    assignAdd(self, expr.l);
//}


} // namespace forward

using dual = forward::Dual<double>;

} // namespace autodiff
