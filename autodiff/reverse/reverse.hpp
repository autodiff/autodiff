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
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>

// autodiff includes
#include <autodiff/common/meta.hpp>

/// autodiff namespace where @ref var and @ref grad are defined.
namespace autodiff {}

namespace autodiff {
namespace reverse {

struct Expr;
struct VariableExpr;
struct IndependentVariableExpr;
struct DependentVariableExpr;
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

using ExprPtr = std::shared_ptr<Expr>;

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

/// The abstract type of any node type in the expression tree.
struct Expr
{
    /// The value of this expression node.
    double val = {};

    /// Construct an Expr object with given value.
    explicit Expr(double val) : val(val) {}

    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param derivatives The container where the derivatives of the root variable w.r.t. to leaf variables are stored.
    /// @param wprime The derivative of the root variable w.r.t. a child expression of this expression.
    virtual void propagate(DerivativesMap& derivatives, double wprime) const = 0;

    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param derivatives The container where the derivatives of the root variable w.r.t. to leaf variables are stored.
    /// @param wprime The derivative of the root variable w.r.t. a child expression of this expression (as an expression).
    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const = 0;


    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param wprime The derivative of the root expression node w.r.t. the child expression of this expression node.
    virtual void propagate(double wprime) = 0;

    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param wprime The derivative of the root expression node w.r.t. the child expression of this expression node (as an expression).
    virtual void propagate(const ExprPtr& wprime) = 0;
};

/// The node in the expression tree representing either an independent or dependent variable.
struct VariableExpr : Expr
{
    /// The derivative of the root expression node with respect to this variable.
    double grad = {};

    /// The derivative of the root expression node with respect to this variable (as an expression for higher-order derivatives).
    ExprPtr gradx = {};

    /// Construct a VariableExpr object with given value.
    VariableExpr(double val) : Expr(val) {}
};

/// The node in the expression tree representing an independent variable.
struct IndependentVariableExpr : VariableExpr
{
    /// Construct an IndependentVariableExpr object with given value.
    IndependentVariableExpr(double val) : VariableExpr(val)
    {
        gradx = constant(0.0);
    }

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

    virtual void propagate(double wprime)
    {
        grad += wprime;
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        gradx = gradx + wprime;
    }
};

/// The node in the expression tree representing a dependent variable.
struct DependentVariableExpr : VariableExpr
{
    /// The expression tree that defines how the dependent variable is calculated.
    ExprPtr expr;

    /// Construct an DependentVariableExpr object with given value.
    DependentVariableExpr(const ExprPtr& expr) : VariableExpr(expr->val), expr(expr)
    {
        gradx = constant(0.0);
    }

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

    virtual void propagate(double wprime)
    {
        grad += wprime;
        expr->propagate(wprime);
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        gradx = gradx + wprime;
        expr->propagate(wprime);
    }
};

struct ConstantExpr : Expr
{
    using Expr::Expr;

    virtual void propagate(DerivativesMap& derivatives, double wprime) const
    {}

    virtual void propagate(DerivativesMapX& derivatives, const ExprPtr& wprime) const
    {}

    virtual void propagate(double wprime)
    {}

    virtual void propagate(const ExprPtr& wprime)
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

    virtual void propagate(double wprime)
    {
        x->propagate(-wprime);
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        x->propagate(-wprime);
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

    virtual void propagate(double wprime)
    {
        l->propagate(wprime);
        r->propagate(wprime);
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        l->propagate(wprime);
        r->propagate(wprime);
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

    virtual void propagate(double wprime)
    {
        l->propagate( wprime);
        r->propagate(-wprime);
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        l->propagate( wprime);
        r->propagate(-wprime);
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

    virtual void propagate(double wprime)
    {
        l->propagate(wprime * r->val);
        r->propagate(wprime * l->val);
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        l->propagate(wprime * r);
        r->propagate(wprime * l);
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

    virtual void propagate(double wprime)
    {
        const auto aux1 = 1.0 / r->val;
        const auto aux2 = -l->val * aux1 * aux1;
        l->propagate(wprime * aux1);
        r->propagate(wprime * aux2);
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        const auto aux1 = 1.0 / r;
        const auto aux2 = -l * aux1 * aux1;
        l->propagate(wprime * aux1);
        r->propagate(wprime * aux2);
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

    virtual void propagate(double wprime)
    {
        x->propagate(wprime * std::cos(x->val));
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        x->propagate(wprime * cos(x));
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

    virtual void propagate(double wprime)
    {
        x->propagate(-wprime * std::sin(x->val));
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        x->propagate(-wprime * sin(x));
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

    virtual void propagate(double wprime)
    {
        const auto aux = 1.0 / std::cos(x->val);
        x->propagate(wprime * aux * aux);
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        const auto aux = 1.0 / cos(x);
        x->propagate(wprime * aux * aux);
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

    virtual void propagate(double wprime)
    {
        x->propagate(wprime * std::cosh(x->val));
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        x->propagate(wprime * cosh(x));
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

    virtual void propagate(double wprime)
    {
        x->propagate(wprime * std::sinh(x->val));
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        x->propagate(wprime * sinh(x));
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

    virtual void propagate(double wprime)
    {
        const auto aux = 1.0 / std::cosh(x->val);
        x->propagate(wprime * aux * aux);
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        const auto aux = 1.0 / cosh(x);
        x->propagate(wprime * aux * aux);
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

    virtual void propagate(double wprime)
    {
        x->propagate(wprime / std::sqrt(1.0 - x->val * x->val));
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        x->propagate(wprime / sqrt(1.0 - x * x));
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

    virtual void propagate(double wprime)
    {
        x->propagate(-wprime / std::sqrt(1.0 - x->val * x->val));
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        x->propagate(-wprime / sqrt(1.0 - x * x));
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

    virtual void propagate(double wprime)
    {
        x->propagate(wprime / (1.0 + x->val * x->val));
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        x->propagate(wprime / (1.0 + x * x));
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

    virtual void propagate(double wprime)
    {
        x->propagate(wprime * val);
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        x->propagate(wprime * exp(x));
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

    virtual void propagate(double wprime)
    {
        x->propagate(wprime / x->val);
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        x->propagate(wprime / x);
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

    virtual void propagate(double wprime)
    {
        x->propagate(wprime / (ln10 * x->val));
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        x->propagate(wprime / (ln10 * x));
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

    virtual void propagate(double wprime)
    {
        const auto lval = l->val;
        const auto rval = r->val;
        const auto aux = wprime * val;
        l->propagate(aux * rval / lval);
        r->propagate(aux * std::log(lval));
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        const auto aux = wprime * pow(l, r - 1);
        l->propagate(aux * r);
        r->propagate(aux *l * log(l));
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

    virtual void propagate(double wprime)
    {
        r->propagate(wprime * val * std::log(l->val));
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        r->propagate(wprime * pow(l, r) * log(l));
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

    virtual void propagate(double wprime)
    {
        l->propagate(wprime * val * r->val / l->val);
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        l->propagate(wprime * pow(l, r - 1) * r);
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

    virtual void propagate(double wprime)
    {
        x->propagate(wprime / (2.0 * std::sqrt(x->val)));
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        x->propagate(wprime / (2.0 * sqrt(x)));
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

    virtual void propagate(double wprime)
    {
        x->propagate(wprime * std::copysign(1.0, x->val));
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        x->propagate(wprime * std::copysign(1.0, x->val));
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

    virtual void propagate(double wprime)
    {
        const auto aux = 2.0/sqrt_pi * std::exp(-(x->val)*(x->val));
        x->propagate(wprime * aux);
    }

    virtual void propagate(const ExprPtr& wprime)
    {
        const auto aux = 2.0/sqrt_pi * exp(-x*x);
        x->propagate(wprime * aux);
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

    /// Construct a default var object variable
    var(const var& other) : var(other.expr) {}

    /// Construct a var object variable with given value
    var(double val) : expr(std::make_shared<IndependentVariableExpr>(val)) {}

    /// Construct a var object variable with given expression
    var(const ExprPtr& expr) : expr(std::make_shared<DependentVariableExpr>(expr)) {}

    // auto grad() const { return expr->grad; }
    auto grad() const { return static_cast<DependentVariableExpr*>(expr.get())->grad; }

    auto gradx() const { return static_cast<DependentVariableExpr*>(expr.get())->gradx; }

    auto seed() { static_cast<VariableExpr*>(expr.get())->grad = 0.0; }

    // auto seedx() { expr->gradx = constant(0.0); }
    auto seedx() { static_cast<VariableExpr*>(expr.get())->gradx = constant(0.0); }

    /// Implicitly convert this var object variable into an expression pointer
    operator ExprPtr() const { return expr; }

    /// Explicitly convert this var object variable into a double value
    explicit operator double() const { return expr->val; }

    auto operator=(int val) -> var& { expr->val = val; return *this; }
    auto operator=(double val) -> var& { expr->val = val; return *this; }

    // auto operator=(const var& other) -> var& { expr->val = other.expr->val; return *this; }

    // auto operator=(const ExprPtr& e) -> var& { expr = e; return *this; }

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
inline double val(double x)
{
    return x;
}

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

namespace detail {

template<typename... Vars>
struct Wrt
{
    std::tuple<Vars...> args;
};

/// The keyword used to denote the variables *with respect to* the derivative is calculated.
template<typename... Args>
auto wrt(Args&&... args)
{
    return detail::Wrt<Args&&...>{ std::forward_as_tuple(std::forward<Args>(args)...) };
}

/// Seed each var number in the **wrt** list.
template<typename... Vars>
auto seed(const Wrt<Vars...>& wrt)
{
    constexpr static auto N = sizeof...(Vars);
    For<N>([&](auto i) constexpr {
        std::get<i>(wrt.args).seed();
    });
}

/// Seed each var number in the **wrt** list.
template<typename... Vars>
auto seedx(const Wrt<Vars...>& wrt)
{
    constexpr static auto N = sizeof...(Vars);
    For<N>([&](auto i) constexpr {
        std::get<i>(wrt.args).seedx();
    });
}

/// Return the derivatives of a variable y with respect to all independent variables.
template<typename... Vars>
auto gradient(const var& y, const Wrt<Vars...>& wrt)
{
    seed(wrt);
    y.expr->propagate(1.0);

    constexpr static auto N = sizeof...(Vars);
    std::array<double, N> values;
    For<N>([&](auto i) constexpr {
        values[i.index] = std::get<i>(wrt.args).grad();
    });

    return values;
}

/// Return the derivatives of a variable y with respect to all independent variables.
template<typename... Vars>
auto gradientx(const var& y, const Wrt<Vars...>& wrt)
{
    seedx(wrt);
    y.expr->propagate(constant(1.0));

    constexpr static auto N = sizeof...(Vars);
    std::array<var, N> values;
    For<N>([&](auto i) constexpr {
        values[i.index] = std::get<i>(wrt.args).gradx();
    });

    return values;
}

// /// Return the gradient of scalar function *f* with respect to some or all variables *x*.
// template<typename Fun, typename... Vars, typename... Args>
// auto gradient(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, ReturnType<Fun, Args...>& u)
// {
//     static_assert(sizeof...(Vars) >= 1);
//     static_assert(sizeof...(Args) >= 1);

//     using T = NumericType<decltype(u)>; // the underlying numeric floating point type in the autodiff number u
//     using Vec = VectorX<T>; // the gradient vector type with floating point values (not autodiff numbers!)

//     const size_t n = wrt_total_length(wrt);

//     if(n == 0) return Vec{};

//     Vec g(n);

//     ForEachWrtVar(wrt, [&](auto&& i, auto&& xi) constexpr
//     {
//         u = eval(f, at, detail::wrt(xi)); // evaluate u with xi seeded so that du/dxi is also computed
//         g[i] = derivative<1>(u);
//     });

//     return g;
// }

} // namespace detail

using detail::wrt;
using detail::gradient;

/// Output a var object variable to the output stream.
inline std::ostream& operator<<(std::ostream& out, const var& x)
{
    out << autodiff::val(x);
    return out;
}

} // namespace autodiff
