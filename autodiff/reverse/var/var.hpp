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
#include <memory>
#include <unordered_map>

/// autodiff namespace where @ref var and @ref grad are defined.
namespace autodiff {}

namespace autodiff {
namespace reverse {

struct Expr;
struct ParameterExpr;
struct VariableExpr;
struct ConstantExpr;
struct UnaryExpr;
struct NegativeExpr;
struct BinaryExpr;
struct AddExpr;
struct SubExpr;
struct MulExpr;
struct DivExpr;
struct SinExpr;
struct CosExpr;
struct TanExpr;
struct SinhExpr;
struct CoshExpr;
struct TanhExpr;
struct ArcSinExpr;
struct ArcCosExpr;
struct ArcTanExpr;
struct ExpExpr;
struct LogExpr;
struct Log10Expr;
struct PowExpr;
struct SqrtExpr;
struct AbsExpr;
struct ErfExpr;

using ExprPtr = std::shared_ptr<const Expr>;

using DerivativesMap = std::unordered_map<const Expr*, double>;
using DerivativesMapX = std::unordered_map<const Expr*, ExprPtr>;

//------------------------------------------------------------------------------
// CONVENIENT FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
ExprPtr constant(double val);

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS (DECLARATION ONLY)
//------------------------------------------------------------------------------
ExprPtr operator+(const ExprPtr& r);
ExprPtr operator-(const ExprPtr& r);

ExprPtr operator+(const ExprPtr& l, const ExprPtr& r);
ExprPtr operator-(const ExprPtr& l, const ExprPtr& r);
ExprPtr operator*(const ExprPtr& l, const ExprPtr& r);
ExprPtr operator/(const ExprPtr& l, const ExprPtr& r);

ExprPtr operator+(double l, const ExprPtr& r);
ExprPtr operator-(double l, const ExprPtr& r);
ExprPtr operator*(double l, const ExprPtr& r);
ExprPtr operator/(double l, const ExprPtr& r);

ExprPtr operator+(const ExprPtr& l, double r);
ExprPtr operator-(const ExprPtr& l, double r);
ExprPtr operator*(const ExprPtr& l, double r);
ExprPtr operator/(const ExprPtr& l, double r);

//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
ExprPtr sin(const ExprPtr& x);
ExprPtr cos(const ExprPtr& x);
ExprPtr tan(const ExprPtr& x);
ExprPtr asin(const ExprPtr& x);
ExprPtr acos(const ExprPtr& x);
ExprPtr atan(const ExprPtr& x);

//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
ExprPtr sinh(const ExprPtr& x);
ExprPtr cosh(const ExprPtr& x);
ExprPtr tanh(const ExprPtr& x);

//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
ExprPtr exp(const ExprPtr& x);
ExprPtr log(const ExprPtr& x);
ExprPtr log10(const ExprPtr& x);

//------------------------------------------------------------------------------
// POWER FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
ExprPtr pow(const ExprPtr& l, const ExprPtr& r);
ExprPtr pow(double l, const ExprPtr& r);
ExprPtr pow(const ExprPtr& l, double r);
ExprPtr sqrt(const ExprPtr& x);

//------------------------------------------------------------------------------
// OTHER FUNCTIONS (DECLARATION ONLY)
//------------------------------------------------------------------------------
ExprPtr abs(const ExprPtr& x);
ExprPtr abs2(const ExprPtr& x);
ExprPtr conj(const ExprPtr& x);
ExprPtr real(const ExprPtr& x);
ExprPtr imag(const ExprPtr& x);
ExprPtr erf(const ExprPtr& x);

//------------------------------------------------------------------------------
// COMPARISON OPERATORS (DECLARATION ONLY)
//------------------------------------------------------------------------------
bool operator==(const ExprPtr& l, const ExprPtr& r);
bool operator!=(const ExprPtr& l, const ExprPtr& r);
bool operator<=(const ExprPtr& l, const ExprPtr& r);
bool operator>=(const ExprPtr& l, const ExprPtr& r);
bool operator<(const ExprPtr& l, const ExprPtr& r);
bool operator>(const ExprPtr& l, const ExprPtr& r);

bool operator==(double l, const ExprPtr& r);
bool operator!=(double l, const ExprPtr& r);
bool operator<=(double l, const ExprPtr& r);
bool operator>=(double l, const ExprPtr& r);
bool operator<(double l, const ExprPtr& r);
bool operator>(double l, const ExprPtr& r);

bool operator==(const ExprPtr& l, double r);
bool operator!=(const ExprPtr& l, double r);
bool operator<=(const ExprPtr& l, double r);
bool operator>=(const ExprPtr& l, double r);
bool operator<(const ExprPtr& l, double r);
bool operator>(const ExprPtr& l, double r);

struct Expr
{
    /// The numerical value of this expression.
    double val;

    /// Construct an Expr object with given numerical value.
    explicit Expr(double val) : val(val) {}

    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param derivatives The container where the derivatives of the root variable w.r.t. to leaf variables are stored.
    /// @param wprime The derivative of the root variable w.r.t. a child expression of this expression.
    virtual void propagate(DerivativesMap& derivatives, double wprime) const = 0;

    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param derivatives The container where the derivatives of the root variable w.r.t. to leaf variables are stored.
    /// @param wprime The derivative of the root variable w.r.t. a child expression of this expression (as an expression).
    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const = 0;
};

struct ParameterExpr : Expr
{
    using Expr::Expr;

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        const auto it = derivatives.find(this);
        if(it != derivatives.end()) it->second += wprime;
        else derivatives.insert({ this, wprime });
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        const auto it = derivatives.find(this);
        if(it != derivatives.end()) it->second = it->second + wprime;
        else derivatives.insert({ this, wprime });
    }
};

struct VariableExpr : Expr
{
    ExprPtr expr;

    VariableExpr(const ExprPtr& expr) : Expr(expr->val), expr(expr) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        const auto it = derivatives.find(this);
        if(it != derivatives.end()) it->second += wprime;
        else derivatives.insert({ this, wprime });
        expr->propagate(derivatives, wprime);
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        const auto it = derivatives.find(this);
        if(it != derivatives.end()) it->second = it->second + wprime;
        else derivatives.insert({ this, wprime });
        expr->propagate(derivatives, wprime);
    }
};

struct ConstantExpr : Expr
{
    using Expr::Expr;

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {}

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {}
};

struct UnaryExpr : Expr
{
    ExprPtr x;

    UnaryExpr(double val, const ExprPtr& x) : Expr(val), x(x) {}
};

struct NegativeExpr : UnaryExpr
{
    using UnaryExpr::UnaryExpr;

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        x->propagate(derivatives, -wprime);
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        x->propagate(derivatives, -wprime);
    }
};

struct BinaryExpr : Expr
{
    ExprPtr l, r;

    BinaryExpr(double val, const ExprPtr& l, const ExprPtr& r) : Expr(val), l(l), r(r) {}
};

struct AddExpr : BinaryExpr
{
    using BinaryExpr::BinaryExpr;

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        l->propagate(derivatives, wprime);
        r->propagate(derivatives, wprime);
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        l->propagate(derivatives, wprime);
        r->propagate(derivatives, wprime);
    }
};

struct SubExpr : BinaryExpr
{
    using BinaryExpr::BinaryExpr;

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        l->propagate(derivatives,  wprime);
        r->propagate(derivatives, -wprime);
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        l->propagate(derivatives,  wprime);
        r->propagate(derivatives, -wprime);
    }
};

struct MulExpr : BinaryExpr
{
    using BinaryExpr::BinaryExpr;

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        l->propagate(derivatives, wprime * r->val);
        r->propagate(derivatives, wprime * l->val);
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        l->propagate(derivatives, wprime * r);
        r->propagate(derivatives, wprime * l);
    }
};

struct DivExpr : BinaryExpr
{
    using BinaryExpr::BinaryExpr;

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        const auto aux1 = 1.0 / r->val;
        const auto aux2 = -l->val * aux1 * aux1;
        l->propagate(derivatives, wprime * aux1);
        r->propagate(derivatives, wprime * aux2);
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        const auto aux1 = 1.0 / r;
        const auto aux2 = -l * aux1 * aux1;
        l->propagate(derivatives, wprime * aux1);
        r->propagate(derivatives, wprime * aux2);
    }
};

struct SinExpr : UnaryExpr
{
    SinExpr(double val, const ExprPtr& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        x->propagate(derivatives, wprime * std::cos(x->val));
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        x->propagate(derivatives, wprime * cos(x));
    }
};

struct CosExpr : UnaryExpr
{
    CosExpr(double val, const ExprPtr& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        x->propagate(derivatives, -wprime * std::sin(x->val));
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        x->propagate(derivatives, -wprime * sin(x));
    }
};

struct TanExpr : UnaryExpr
{
    TanExpr(double val, const ExprPtr& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        const auto aux = 1.0 / std::cos(x->val);
        x->propagate(derivatives, wprime * aux * aux);
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        const auto aux = 1.0 / cos(x);
        x->propagate(derivatives, wprime * aux * aux);
    }
};

struct SinhExpr : UnaryExpr
{
    SinhExpr(double val, const ExprPtr& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        x->propagate(derivatives, wprime * std::cosh(x->val));
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        x->propagate(derivatives, wprime * cosh(x));
    }
};

struct CoshExpr : UnaryExpr
{
    CoshExpr(double val, const ExprPtr& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        x->propagate(derivatives, wprime * std::sinh(x->val));
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        x->propagate(derivatives, wprime * sinh(x));
    }
};

struct TanhExpr : UnaryExpr
{
    TanhExpr(double val, const ExprPtr& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        const auto aux = 1.0 / std::cosh(x->val);
        x->propagate(derivatives, wprime * aux * aux);
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        const auto aux = 1.0 / cosh(x);
        x->propagate(derivatives, wprime * aux * aux);
    }
};

struct ArcSinExpr : UnaryExpr
{
    ArcSinExpr(double val, const ExprPtr& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        x->propagate(derivatives, wprime / std::sqrt(1.0 - x->val * x->val));
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        x->propagate(derivatives, wprime / sqrt(1.0 - x * x));
    }
};

struct ArcCosExpr : UnaryExpr
{
    ArcCosExpr(double val, const ExprPtr& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        x->propagate(derivatives, -wprime / std::sqrt(1.0 - x->val * x->val));
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        x->propagate(derivatives, -wprime / sqrt(1.0 - x * x));
    }
};

struct ArcTanExpr : UnaryExpr
{
    ArcTanExpr(double val, const ExprPtr& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        x->propagate(derivatives, wprime / (1.0 + x->val * x->val));
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        x->propagate(derivatives, wprime / (1.0 + x * x));
    }
};

struct ExpExpr : UnaryExpr
{
    using UnaryExpr::UnaryExpr;

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        x->propagate(derivatives, wprime * val);
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        x->propagate(derivatives, wprime * exp(x));
    }
};

struct LogExpr : UnaryExpr
{
    using UnaryExpr::UnaryExpr;

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        x->propagate(derivatives, wprime / x->val);
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        x->propagate(derivatives, wprime / x);
    }
};

struct Log10Expr : UnaryExpr
{
    constexpr static double ln10 = 2.3025850929940456840179914546843;

    Log10Expr(double val, const ExprPtr& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        x->propagate(derivatives, wprime / (ln10 * x->val));
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        x->propagate(derivatives, wprime / (ln10 * x));
    }
};

struct PowExpr : BinaryExpr
{
    double log_l;

    PowExpr(double val, const ExprPtr& l, const ExprPtr& r) : BinaryExpr(val, l, r), log_l(std::log(l->val)) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        const auto lval = l->val;
        const auto rval = r->val;
        const auto aux = wprime * val;
        l->propagate(derivatives, aux * rval / lval);
        r->propagate(derivatives, aux * std::log(lval));
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        const auto aux = wprime * pow(l, r - 1);
        l->propagate(derivatives, aux * r);
        r->propagate(derivatives, aux *l * log(l));
    }
};

struct PowConstantLeftExpr : BinaryExpr
{
    PowConstantLeftExpr(double val, const ExprPtr& l, const ExprPtr& r) : BinaryExpr(val, l, r) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        r->propagate(derivatives, wprime * val * std::log(l->val));
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        r->propagate(derivatives, wprime * pow(l, r) * log(l));
    }
};

struct PowConstantRightExpr : BinaryExpr
{
    PowConstantRightExpr(double val, const ExprPtr& l, const ExprPtr& r) : BinaryExpr(val, l, r) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        l->propagate(derivatives, wprime * val * r->val / l->val);
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        l->propagate(derivatives, wprime * pow(l, r - 1) * r);
    }
};

struct SqrtExpr : UnaryExpr
{
    SqrtExpr(double val, const ExprPtr& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        x->propagate(derivatives, wprime / (2.0 * std::sqrt(x->val)));
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        x->propagate(derivatives, wprime / (2.0 * sqrt(x)));
    }
};

struct AbsExpr : UnaryExpr
{
    AbsExpr(double val, const ExprPtr& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        x->propagate(derivatives, wprime * std::copysign(1.0, x->val));
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        x->propagate(derivatives, wprime * std::copysign(1.0, x->val));
    }
};

struct ErfExpr : UnaryExpr
{
    constexpr static auto sqrt_pi = 1.7724538509055160272981674833411451872554456638435;

    ErfExpr(double val, const ExprPtr& x) : UnaryExpr(val, x) {}

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {
        const auto aux = 2.0/sqrt_pi * std::exp(-(x->val)*(x->val));
        x->propagate(derivatives, wprime * aux);
    }

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {
        const auto aux = 2.0/sqrt_pi * exp(-x*x);
        x->propagate(derivatives, wprime * aux);
    }
};

//------------------------------------------------------------------------------
// CONVENIENT FUNCTIONS
//------------------------------------------------------------------------------
inline ExprPtr constant(double val) { return std::make_shared<ConstantExpr>(val); }

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS
//------------------------------------------------------------------------------
inline ExprPtr operator+(const ExprPtr& r) { return r; }
inline ExprPtr operator-(const ExprPtr& r) { return std::make_shared<NegativeExpr>(-r->val, r); }

inline ExprPtr operator+(const ExprPtr& l, const ExprPtr& r) { return std::make_shared<AddExpr>(l->val + r->val, l, r); }
inline ExprPtr operator-(const ExprPtr& l, const ExprPtr& r) { return std::make_shared<SubExpr>(l->val - r->val, l, r); }
inline ExprPtr operator*(const ExprPtr& l, const ExprPtr& r) { return std::make_shared<MulExpr>(l->val * r->val, l, r); }
inline ExprPtr operator/(const ExprPtr& l, const ExprPtr& r) { return std::make_shared<DivExpr>(l->val / r->val, l, r); }

inline ExprPtr operator+(double l, const ExprPtr& r) { return constant(l) + r; }
inline ExprPtr operator-(double l, const ExprPtr& r) { return constant(l) - r; }
inline ExprPtr operator*(double l, const ExprPtr& r) { return constant(l) * r; }
inline ExprPtr operator/(double l, const ExprPtr& r) { return constant(l) / r; }

inline ExprPtr operator+(const ExprPtr& l, double r) { return l + constant(r); }
inline ExprPtr operator-(const ExprPtr& l, double r) { return l - constant(r); }
inline ExprPtr operator*(const ExprPtr& l, double r) { return l * constant(r); }
inline ExprPtr operator/(const ExprPtr& l, double r) { return l / constant(r); }

//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS
//------------------------------------------------------------------------------
inline ExprPtr sin(const ExprPtr& x) { return std::make_shared<SinExpr>(std::sin(x->val), x); }
inline ExprPtr cos(const ExprPtr& x) { return std::make_shared<CosExpr>(std::cos(x->val), x); }
inline ExprPtr tan(const ExprPtr& x) { return std::make_shared<TanExpr>(std::tan(x->val), x); }
inline ExprPtr asin(const ExprPtr& x) { return std::make_shared<ArcSinExpr>(std::asin(x->val), x); }
inline ExprPtr acos(const ExprPtr& x) { return std::make_shared<ArcCosExpr>(std::acos(x->val), x); }
inline ExprPtr atan(const ExprPtr& x) { return std::make_shared<ArcTanExpr>(std::atan(x->val), x); }

//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS
//------------------------------------------------------------------------------
inline ExprPtr sinh(const ExprPtr& x) { return std::make_shared<SinhExpr>(std::sinh(x->val), x); }
inline ExprPtr cosh(const ExprPtr& x) { return std::make_shared<CoshExpr>(std::cosh(x->val), x); }
inline ExprPtr tanh(const ExprPtr& x) { return std::make_shared<TanhExpr>(std::tanh(x->val), x); }

//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS
//------------------------------------------------------------------------------
inline ExprPtr exp(const ExprPtr& x) { return std::make_shared<ExpExpr>(std::exp(x->val), x); }
inline ExprPtr log(const ExprPtr& x) { return std::make_shared<LogExpr>(std::log(x->val), x); }
inline ExprPtr log10(const ExprPtr& x) { return std::make_shared<Log10Expr>(std::log10(x->val), x); }

//------------------------------------------------------------------------------
// POWER FUNCTIONS
//------------------------------------------------------------------------------
inline ExprPtr pow(const ExprPtr& l, const ExprPtr& r) { return std::make_shared<PowExpr>(std::pow(l->val, r->val), l, r); }
inline ExprPtr pow(double l, const ExprPtr& r) { return std::make_shared<PowConstantLeftExpr>(std::pow(l, r->val), constant(l), r); }
inline ExprPtr pow(const ExprPtr& l, double r) { return std::make_shared<PowConstantRightExpr>(std::pow(l->val, r), l, constant(r)); }
inline ExprPtr sqrt(const ExprPtr& x) { return std::make_shared<SqrtExpr>(std::sqrt(x->val), x); }

//------------------------------------------------------------------------------
// OTHER FUNCTIONS
//------------------------------------------------------------------------------
inline ExprPtr abs(const ExprPtr& x) { return std::make_shared<AbsExpr>(std::abs(x->val), x); }
inline ExprPtr abs2(const ExprPtr& x) { return x * x; }
inline ExprPtr conj(const ExprPtr& x) { return x; }
inline ExprPtr real(const ExprPtr& x) { return x; }
inline ExprPtr imag(const ExprPtr& x) { return constant(0.0); }
inline ExprPtr erf(const ExprPtr& x) { return std::make_shared<ErfExpr>(std::erf(x->val), x); }

//------------------------------------------------------------------------------
// COMPARISON OPERATORS
//------------------------------------------------------------------------------
inline bool operator==(const ExprPtr& l, const ExprPtr& r) { return l->val == r->val; }
inline bool operator!=(const ExprPtr& l, const ExprPtr& r) { return l->val != r->val; }
inline bool operator<=(const ExprPtr& l, const ExprPtr& r) { return l->val <= r->val; }
inline bool operator>=(const ExprPtr& l, const ExprPtr& r) { return l->val >= r->val; }
inline bool operator<(const ExprPtr& l, const ExprPtr& r) { return l->val < r->val; }
inline bool operator>(const ExprPtr& l, const ExprPtr& r) { return l->val > r->val; }

inline bool operator==(double l, const ExprPtr& r) { return l == r->val; }
inline bool operator!=(double l, const ExprPtr& r) { return l != r->val; }
inline bool operator<=(double l, const ExprPtr& r) { return l <= r->val; }
inline bool operator>=(double l, const ExprPtr& r) { return l >= r->val; }
inline bool operator<(double l, const ExprPtr& r) { return l < r->val; }
inline bool operator>(double l, const ExprPtr& r) { return l > r->val; }

inline bool operator==(const ExprPtr& l, double r) { return l->val == r; }
inline bool operator!=(const ExprPtr& l, double r) { return l->val != r; }
inline bool operator<=(const ExprPtr& l, double r) { return l->val <= r; }
inline bool operator>=(const ExprPtr& l, double r) { return l->val >= r; }
inline bool operator<(const ExprPtr& l, double r) { return l->val < r; }
inline bool operator>(const ExprPtr& l, double r) { return l->val > r; }

} // namespace reverse

using namespace reverse;

/// The autodiff variable type used for automatic differentiation.
struct var
{
    /// The pointer to the expression tree of variable operations
    ExprPtr expr;

    /// Construct a default var object variable
    var() : var(0.0) {}

    /// Construct a var object variable with given value
    var(double val) : expr(std::make_shared<ParameterExpr>(val)) {}

    /// Construct a var object variable with given expression
    var(const ExprPtr& expr) : expr(std::make_shared<VariableExpr>(expr)) {}

    /// Implicitly convert this var object variable into an expression pointer
    operator ExprPtr() const { return expr; }

    /// Explicitly convert this var object variable into a double value
    explicit operator double() const { return expr->val; }

	// Arithmetic-assignment operators
    var& operator+=(const ExprPtr& other) { expr = expr + other; return *this; }
    var& operator-=(const ExprPtr& other) { expr = expr - other; return *this; }
    var& operator*=(const ExprPtr& other) { expr = expr * other; return *this; }
    var& operator/=(const ExprPtr& other) { expr = expr / other; return *this; }
    var& operator+=(double other) { expr = expr + constant(other); return *this; }
    var& operator-=(double other) { expr = expr - constant(other); return *this; }
    var& operator*=(double other) { expr = expr * constant(other); return *this; }
    var& operator/=(double other) { expr = expr / constant(other); return *this; }
};

//------------------------------------------------------------------------------
// COMPARISON OPERATORS (DEFINED FOR ARGUMENTS OF TYPE var)
//------------------------------------------------------------------------------
inline bool operator==(const var& l, const var& r) { return l.expr == r.expr; }
inline bool operator!=(const var& l, const var& r) { return l.expr != r.expr; }
inline bool operator<=(const var& l, const var& r) { return l.expr <= r.expr; }
inline bool operator>=(const var& l, const var& r) { return l.expr >= r.expr; }
inline bool operator<(const var& l, const var& r) { return l.expr < r.expr; }
inline bool operator>(const var& l, const var& r) { return l.expr > r.expr; }

inline bool operator==(double l, const var& r) { return l == r.expr; }
inline bool operator!=(double l, const var& r) { return l != r.expr; }
inline bool operator<=(double l, const var& r) { return l <= r.expr; }
inline bool operator>=(double l, const var& r) { return l >= r.expr; }
inline bool operator<(double l, const var& r) { return l < r.expr; }
inline bool operator>(double l, const var& r) { return l > r.expr; }

inline bool operator==(const var& l, double r) { return l.expr == r; }
inline bool operator!=(const var& l, double r) { return l.expr != r; }
inline bool operator<=(const var& l, double r) { return l.expr <= r; }
inline bool operator>=(const var& l, double r) { return l.expr >= r; }
inline bool operator<(const var& l, double r) { return l.expr < r; }
inline bool operator>(const var& l, double r) { return l.expr > r; }

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS (DEFINED FOR ARGUMENTS OF TYPE var)
//------------------------------------------------------------------------------
inline const ExprPtr& operator+(const var& r) { return r.expr; }
inline ExprPtr operator-(const var& r) { return -r.expr; }

inline ExprPtr operator+(const var& l, const var& r) { return l.expr + r.expr; }
inline ExprPtr operator-(const var& l, const var& r) { return l.expr - r.expr; }
inline ExprPtr operator*(const var& l, const var& r) { return l.expr * r.expr; }
inline ExprPtr operator/(const var& l, const var& r) { return l.expr / r.expr; }

inline ExprPtr operator+(const ExprPtr& l, const var& r) { return l + r.expr; }
inline ExprPtr operator-(const ExprPtr& l, const var& r) { return l - r.expr; }
inline ExprPtr operator*(const ExprPtr& l, const var& r) { return l * r.expr; }
inline ExprPtr operator/(const ExprPtr& l, const var& r) { return l / r.expr; }

inline ExprPtr operator+(const var& l, const ExprPtr& r) { return l.expr + r; }
inline ExprPtr operator-(const var& l, const ExprPtr& r) { return l.expr - r; }
inline ExprPtr operator*(const var& l, const ExprPtr& r) { return l.expr * r; }
inline ExprPtr operator/(const var& l, const ExprPtr& r) { return l.expr / r; }

inline ExprPtr operator+(double l, const var& r) { return l + r.expr; }
inline ExprPtr operator-(double l, const var& r) { return l - r.expr; }
inline ExprPtr operator*(double l, const var& r) { return l * r.expr; }
inline ExprPtr operator/(double l, const var& r) { return l / r.expr; }

inline ExprPtr operator+(const var& l, double r) { return l.expr + r; }
inline ExprPtr operator-(const var& l, double r) { return l.expr - r; }
inline ExprPtr operator*(const var& l, double r) { return l.expr * r; }
inline ExprPtr operator/(const var& l, double r) { return l.expr / r; }

//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE var)
//------------------------------------------------------------------------------
inline ExprPtr sin(const var& x) { return sin(x.expr); }
inline ExprPtr cos(const var& x) { return cos(x.expr); }
inline ExprPtr tan(const var& x) { return tan(x.expr); }
inline ExprPtr asin(const var& x) { return asin(x.expr); }
inline ExprPtr acos(const var& x) { return acos(x.expr); }
inline ExprPtr atan(const var& x) { return atan(x.expr); }

//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE var)
//------------------------------------------------------------------------------
inline ExprPtr sinh(const var& x) { return sinh(x.expr); }
inline ExprPtr cosh(const var& x) { return cosh(x.expr); }
inline ExprPtr tanh(const var& x) { return tanh(x.expr); }

//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE var)
//------------------------------------------------------------------------------
inline ExprPtr exp(const var& x) { return exp(x.expr); }
inline ExprPtr log(const var& x) { return log(x.expr); }
inline ExprPtr log10(const var& x) { return log10(x.expr); }

//------------------------------------------------------------------------------
// POWER FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE var)
//------------------------------------------------------------------------------
inline ExprPtr pow(const var& l, const var& r) { return pow(l.expr, r.expr); }
inline ExprPtr pow(double l, const var& r) { return pow(l, r.expr); }
inline ExprPtr pow(const var& l, double r) { return pow(l.expr, r); }
inline ExprPtr sqrt(const var& x) { return sqrt(x.expr); }

//------------------------------------------------------------------------------
// OTHER FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE var)
//------------------------------------------------------------------------------
inline ExprPtr abs(const var& x) { return abs(x.expr); }
inline ExprPtr abs2(const var& x) { return abs2(x.expr); }
inline ExprPtr conj(const var& x) { return conj(x.expr); }
inline ExprPtr real(const var& x) { return real(x.expr); }
inline ExprPtr imag(const var& x) { return imag(x.expr); }
inline ExprPtr erf(const var& x) { return erf(x.expr); }

/// Return the value of a variable x.
inline double val(const var& x)
{
    return x.expr->val;
}

using Derivatives = std::function<double(const var&)>;
using DerivativesX = std::function<var(const var&)>;

/// Return the derivatives of a variable y with respect to all independent variables.
inline Derivatives derivatives(const var& y)
{
    DerivativesMap map;

    y.expr->propagate(map, 1.0);

    auto fn = [=](const var& x)
    {
        const auto it = map.find(x.expr.get());
        return it != map.end() ? it->second : 0.0;
    };

    return fn;
}

/// Return the derivatives of a variable y with respect to all independent variables.
inline DerivativesX derivativesx(const var& y)
{
    DerivativesMapX map;

    y.expr->propagate(map, constant(1.0));

    auto fn = [=](const var& x)
    {
        const auto it = map.find(x.expr.get());
        return it != map.end() ? it->second : constant(0.0);
    };

    return fn;
}

/// Output a var object variable to the output stream.
inline std::ostream& operator<<(std::ostream& out, const var& x)
{
    out << autodiff::val(x);
    return out;
}

} // namespace autodiff
