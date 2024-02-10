//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright © 2018–2024 Allan Leal
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
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>

// autodiff includes
#include <autodiff/common/meta.hpp>
#include <autodiff/common/numbertraits.hpp>

/// autodiff namespace where @ref Variable and @ref grad are defined.
namespace autodiff {}

namespace autodiff {
// avoid clash with autodiff::detail in autodiff/forward/dual/dual.hpp
namespace reverse {
using detail::Requires;
using detail::For;
using detail::isArithmetic;
namespace detail {


using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cos;
using std::cosh;
using std::erf;
using std::exp;
using std::hypot;
using std::log;
using std::log10;
using std::pow;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;

template<typename T> struct Expr;
template<typename T> struct VariableExpr;
template<typename T> struct IndependentVariableExpr;
template<typename T> struct DependentVariableExpr;
template<typename T> struct ConstantExpr;
template<typename T> struct UnaryExpr;
template<typename T> struct NegativeExpr;
template<typename T> struct BinaryExpr;
template<typename T> struct TernaryExpr;
template<typename T> struct AddExpr;
template<typename T> struct SubExpr;
template<typename T> struct MulExpr;
template<typename T> struct DivExpr;
template<typename T> struct SinExpr;
template<typename T> struct CosExpr;
template<typename T> struct TanExpr;
template<typename T> struct SinhExpr;
template<typename T> struct CoshExpr;
template<typename T> struct TanhExpr;
template<typename T> struct ArcSinExpr;
template<typename T> struct ArcCosExpr;
template<typename T> struct ArcTanExpr;
template<typename T> struct ArcTan2Expr;
template<typename T> struct ExpExpr;
template<typename T> struct LogExpr;
template<typename T> struct Log10Expr;
template<typename T> struct PowExpr;
template<typename T> struct SqrtExpr;
template<typename T> struct AbsExpr;
template<typename T> struct ErfExpr;
template<typename T> struct Hypot2Expr;
template<typename T> struct Hypot3Expr;
template<typename T> struct Variable;

template<typename T> using ExprPtr = std::shared_ptr<Expr<T>>;

namespace traits {

template<typename T>
struct VariableValueTypeNotDefinedFor {};

template<typename T>
struct VariableValueType;

template<typename T>
struct VariableValueType { using type = std::conditional_t<isArithmetic<T>, T, VariableValueTypeNotDefinedFor<T>>; };

template<typename T>
struct VariableValueType<Variable<T>> { using type = typename VariableValueType<T>::type; };

template<typename T>
struct VariableValueType<ExprPtr<T>> { using type = typename VariableValueType<T>::type; };

template<typename T>
struct VariableOrder { constexpr static auto value = 0; };

template<typename T>
struct VariableOrder<Variable<T>> { constexpr static auto value = 1 + VariableOrder<T>::value; };

template<typename T>
struct isVariable { constexpr static bool value = false; };

template<typename T>
struct isVariable<Variable<T>> { constexpr static bool value = true; };

} // namespace traits

template<typename T>
using VariableValueType = typename traits::VariableValueType<T>::type;

template<typename T>
constexpr auto VariableOrder = traits::VariableOrder<T>::value;

template<typename T>
constexpr auto isVariable = traits::isVariable<T>::value;

/// The abstract type of any node type in the expression tree.
template<typename T>
struct Expr
{
    /// The value of this expression node.
    T val = {};

    /// Construct an Expr object with given value.
    explicit Expr(const T& v) : val(v) {}

    /// Destructor (to avoid warning)
    virtual ~Expr() {}

    /// Bind a value pointer for writing the derivative during propagation
    virtual void bind_value(T* /* grad */) {}

    /// Bind an expression pointer for writing the derivative expression during propagation
    virtual void bind_expr(ExprPtr<T>* /* gradx */) {}

    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param wprime The derivative of the root expression node w.r.t. the child expression of this expression node.
    virtual void propagate(const T& wprime) = 0;

    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param wprime The derivative of the root expression node w.r.t. the child expression of this expression node (as an expression).
    virtual void propagatex(const ExprPtr<T>& wprime) = 0;

    /// Update the value of this expression
    virtual void update() = 0;
};

/// The node in the expression tree representing either an independent or dependent variable.
template<typename T>
struct VariableExpr : Expr<T>
{
    /// The derivative value of the root expression node w.r.t. this variable.
    T* gradPtr = {};

    /// The derivative expression of the root expression node w.r.t. this variable (reusable for higher-order derivatives).
    ExprPtr<T>* gradxPtr = {};

    /// Construct a VariableExpr object with given value.
    VariableExpr(const T& v) : Expr<T>(v) {}

    virtual void bind_value(T* grad) { gradPtr = grad; }
    virtual void bind_expr(ExprPtr<T>* gradx) { gradxPtr = gradx; }
};

/// The node in the expression tree representing an independent variable.
template<typename T>
struct IndependentVariableExpr : VariableExpr<T>
{
    using VariableExpr<T>::gradPtr;
    using VariableExpr<T>::gradxPtr;

    /// Construct an IndependentVariableExpr object with given value.
    IndependentVariableExpr(const T& v) : VariableExpr<T>(v) {}

    void propagate(const T& wprime) override {
        if(gradPtr) { *gradPtr += wprime; }
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        if(gradxPtr) { *gradxPtr = *gradxPtr + wprime; }
    }

    void update() override {}
};

/// The node in the expression tree representing a dependent variable.
template<typename T>
struct DependentVariableExpr : VariableExpr<T>
{
    using VariableExpr<T>::gradPtr;
    using VariableExpr<T>::gradxPtr;

    /// The expression tree that defines how the dependent variable is calculated.
    ExprPtr<T> expr;

    /// Construct an DependentVariableExpr object with given value.
    DependentVariableExpr(const ExprPtr<T>& e) : VariableExpr<T>(e->val), expr(e) {}

    void propagate(const T& wprime) override
    {
        if(gradPtr) { *gradPtr += wprime; }
        expr->propagate(wprime);
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        if(gradxPtr) { *gradxPtr = *gradxPtr + wprime; }
        expr->propagatex(wprime);
    }

    void update() override
    {
        expr->update();
        this->val = expr->val;
    }
};

template<typename T>
struct ConstantExpr : Expr<T>
{
    using Expr<T>::Expr;

    void propagate([[maybe_unused]] const T& wprime) override
    {}

    void propagatex([[maybe_unused]] const ExprPtr<T>& wprime) override
    {}

    void update() override {}
};

template<typename T> ExprPtr<T> constant(const T& val) { return std::make_shared<ConstantExpr<T>>(val); }

template<typename T>
struct UnaryExpr : Expr<T>
{
    ExprPtr<T> x;

    UnaryExpr(const T& v, const ExprPtr<T>& e) : Expr<T>(v), x(e) {}
};

template<typename T>
struct NegativeExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    using UnaryExpr<T>::UnaryExpr;

    void propagate(const T& wprime) override
    {
        x->propagate(-wprime);
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(-wprime);
    }

    void update() override
    {
        x->update();
        this->val = -x->val;
    }
};

template<typename T>
struct BinaryExpr : Expr<T>
{
    ExprPtr<T> l, r;

    BinaryExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : Expr<T>(v), l(ll), r(rr) {}
};

template<typename T>
struct TernaryExpr : Expr<T>
{
    ExprPtr<T> l, c, r;

    TernaryExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& cc, const ExprPtr<T>& rr) : Expr<T>(v), l(ll), c(cc), r(rr) {}
};

template<typename T>
struct AddExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    using BinaryExpr<T>::BinaryExpr;

    void propagate(const T& wprime) override
    {
        l->propagate(wprime);
        r->propagate(wprime);
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(wprime);
        r->propagatex(wprime);
    }

    void update() override
    {
        l->update();
        r->update();
        this->val = l->val + r->val;
    }
};

template<typename T>
struct SubExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;
    using BinaryExpr<T>::BinaryExpr;

    void propagate(const T& wprime) override
    {
        l->propagate(wprime);
        r->propagate(-wprime);
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(wprime);  // (l - r)'l =  l'
        r->propagatex(-wprime); // (l - r)'r = -r'
    }

    void update() override
    {
        l->update();
        r->update();
        this->val = l->val - r->val;
    }
};

template<typename T>
struct MulExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;
    using BinaryExpr<T>::BinaryExpr;

    void propagate(const T& wprime) override
    {
        l->propagate(wprime * r->val); // (l * r)'l = w' * r
        r->propagate(wprime * l->val); // (l * r)'r = l * w'
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(wprime * r);
        r->propagatex(wprime * l);
    }

    void update() override
    {
        l->update();
        r->update();
        this->val = l->val * r->val;
    }
};

template<typename T>
struct DivExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;
    using BinaryExpr<T>::BinaryExpr;

    void propagate(const T& wprime) override
    {
        const auto aux1 = 1.0 / r->val;
        const auto aux2 = -l->val * aux1 * aux1;
        l->propagate(wprime * aux1);
        r->propagate(wprime * aux2);
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        const auto aux1 = 1.0 / r;
        const auto aux2 = -l * aux1 * aux1;
        l->propagatex(wprime * aux1);
        r->propagatex(wprime * aux2);
    }

    void update() override
    {
        l->update();
        r->update();
        this->val = l->val / r->val;
    }
};

template<typename T>
struct SinExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    SinExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override
    {
        x->propagate(wprime * cos(x->val));
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime * cos(x));
    }

    void update() override
    {
        x->update();
        this->val = sin(x->val);
    }
};

template<typename T>
struct CosExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    CosExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override
    {
        x->propagate(-wprime * sin(x->val));
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(-wprime * sin(x));
    }

    void update() override
    {
        x->update();
        this->val = cos(x->val);
    }
};

template<typename T>
struct TanExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    TanExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override
    {
        const auto aux = 1.0 / cos(x->val);
        x->propagate(wprime * aux * aux);
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        const auto aux = 1.0 / cos(x);
        x->propagatex(wprime * aux * aux);
    }

    void update() override
    {
        x->update();
        this->val = tan(x->val);
    }
};

template<typename T>
struct SinhExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    SinhExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override
    {
        x->propagate(wprime * cosh(x->val));
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime * cosh(x));
    }

    void update() override
    {
        x->update();
        this->val = sinh(x->val);
    }
};

template<typename T>
struct CoshExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    CoshExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override
    {
        x->propagate(wprime * sinh(x->val));
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime * sinh(x));
    }

    void update() override
    {
        x->update();
        this->val = cosh(x->val);
    }
};

template<typename T>
struct TanhExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    TanhExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override
    {
        const auto aux = 1.0 / cosh(x->val);
        x->propagate(wprime * aux * aux);
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        const auto aux = 1.0 / cosh(x);
        x->propagatex(wprime * aux * aux);
    }

    void update() override
    {
        x->update();
        this->val = tanh(x->val);
    }
};

template<typename T>
struct ArcSinExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    ArcSinExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override
    {
        x->propagate(wprime / sqrt(1.0 - x->val * x->val));
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / sqrt(1.0 - x * x));
    }

    void update() override
    {
        x->update();
        this->val = asin(x->val);
    }
};

template<typename T>
struct ArcCosExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    ArcCosExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override
    {
        x->propagate(-wprime / sqrt(1.0 - x->val * x->val));
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(-wprime / sqrt(1.0 - x * x));
    }

    void update() override
    {
        x->update();
        this->val = acos(x->val);
    }
};

template<typename T>
struct ArcTanExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    ArcTanExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override
    {
        x->propagate(wprime / (1.0 + x->val * x->val));
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / (1.0 + x * x));
    }

    void update() override
    {
        x->update();
        this->val = atan(x->val);
    }
};

template<typename T>
struct ArcTan2Expr : BinaryExpr<T>
{
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    ArcTan2Expr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : BinaryExpr<T>(v, ll, rr) {}

    void propagate(const T& wprime) override
    {
        const auto aux = wprime / (l->val * l->val + r->val * r->val);
        l->propagate(r->val * aux);
        r->propagate(-l->val * aux);
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        const auto aux = wprime / (l * l + r * r);
        l->propagatex(r * aux);
        r->propagatex(-l * aux);
    }

    void update() override
    {
        l->update();
        r->update();
        this->val = atan2(l->val, r->val);
    }
};

template<typename T>
struct ExpExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::UnaryExpr;
    using UnaryExpr<T>::val;
    using UnaryExpr<T>::x;

    void propagate(const T& wprime) override
    {
        x->propagate(wprime * val); // exp(x)' = exp(x) * x'
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime * exp(x));
    }

    void update() override
    {
        x->update();
        this->val = exp(x->val);
    }
};

template<typename T>
struct LogExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;
    using UnaryExpr<T>::UnaryExpr;

    void propagate(const T& wprime) override
    {
        x->propagate(wprime / x->val); // log(x)' = x'/x
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / x);
    }

    void update() override
    {
        x->update();
        this->val = log(x->val);
    }
};

template<typename T>
struct Log10Expr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    constexpr static auto ln10 = static_cast<VariableValueType<T>>(2.3025850929940456840179914546843);

    Log10Expr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override
    {
        x->propagate(wprime / (ln10 * x->val));
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / (ln10 * x));
    }

    void update() override
    {
        x->update();
        this->val = log10(x->val);
    }
};

template<typename T>
struct PowExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    T log_l;

    PowExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : BinaryExpr<T>(v, ll, rr), log_l(log(ll->val)) {}

    void propagate(const T& wprime) override
    {
        using U = VariableValueType<T>;
        constexpr auto zero = U(0.0);
        const auto lval = l->val;
        const auto rval = r->val;
        const auto aux = wprime * pow(lval, rval - 1);
        l->propagate(aux * rval);
        const auto auxr = lval == zero ? 0.0 : lval * log(lval); // since x*log(x) -> 0 as x -> 0
        r->propagate(aux * auxr);
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        using U = VariableValueType<T>;
        constexpr auto zero = U(0.0);
        const auto aux = wprime * pow(l, r - 1);
        l->propagatex(aux * r);
        const auto auxr = l == zero ? 0.0*l : l * log(l); // since x*log(x) -> 0 as x -> 0
        r->propagatex(aux * auxr);
    }

    void update() override
    {
        l->update();
        r->update();
        this->val = pow(l->val, r->val);
    }
};

template<typename T>
struct PowConstantLeftExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    PowConstantLeftExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : BinaryExpr<T>(v, ll, rr) {}

    void propagate(const T& wprime) override
    {
        const auto lval = l->val;
        const auto rval = r->val;
        const auto aux = wprime * pow(lval, rval - 1);
        const auto auxr = lval == 0.0 ? 0.0 : lval * log(lval); // since x*log(x) -> 0 as x -> 0
        r->propagate(aux * auxr);
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        const auto aux = wprime * pow(l, r - 1);
        const auto auxr = l == 0.0 ? 0.0*l : l * log(l); // since x*log(x) -> 0 as x -> 0
        r->propagatex(aux * auxr);
    }

    void update() override
    {
        r->update();
        this->val = pow(l->val, r->val);
    }
};

template<typename T>
struct PowConstantRightExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    PowConstantRightExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : BinaryExpr<T>(v, ll, rr) {}

    void propagate(const T& wprime) override
    {
        l->propagate(wprime * pow(l->val, r->val - 1) * r->val); // pow(l, r)'l = r * pow(l, r - 1) * l'
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(wprime * pow(l, r - 1) * r);
    }

    void update() override
    {
        l->update();
        this->val = pow(l->val, r->val);
    }
};

template<typename T>
struct SqrtExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    SqrtExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override
    {
        x->propagate(wprime / (2.0 * sqrt(x->val))); // sqrt(x)' = 1/2 * 1/sqrt(x) * x'
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / (2.0 * sqrt(x)));
    }

    void update() override
    {
        x->update();
        this->val = sqrt(x->val);
    }
};

template<typename T>
struct AbsExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;
    using U = VariableValueType<T>;

    AbsExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override
    {
        if(x->val < 0.0) x->propagate(-wprime);
        else if(x->val > 0.0) x->propagate(wprime);
        else x->propagate(T(0));
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        if(x->val < 0.0) x->propagatex(-wprime);
        else if(x->val > 0.0) x->propagatex(wprime);
        else x->propagate(T(0));
    }

    void update() override
    {
        x->update();
        this->val = abs(x->val);
    }
};

template<typename T>
struct ErfExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    constexpr static auto sqrt_pi = static_cast<VariableValueType<T>>(1.7724538509055160272981674833411451872554456638435);

    ErfExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    void propagate(const T& wprime) override
    {
        const auto aux = 2.0 / sqrt_pi * exp(-(x->val) * (x->val)); // erf(x)' = 2/sqrt(pi) * exp(-x * x) * x'
        x->propagate(wprime * aux);
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        const auto aux = 2.0 / sqrt_pi * exp(-x * x);
        x->propagatex(wprime * aux);
    }

    void update() override
    {
        x->update();
        this->val = erf(x->val);
    }
};

template<typename T>
struct Hypot2Expr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    Hypot2Expr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : BinaryExpr<T>(v, ll, rr) {}

    void propagate(const T& wprime) override
    {
        l->propagate(wprime * l->val / val); // sqrt(l*l + r*r)'l = 1/2 * 1/sqrt(l*l + r*r) * (2*l*l') = (l*l')/sqrt(l*l + r*r)
        r->propagate(wprime * r->val / val); // sqrt(l*l + r*r)'r = 1/2 * 1/sqrt(l*l + r*r) * (2*r*r') = (r*r')/sqrt(l*l + r*r)
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(wprime * l / hypot(l, r));
        r->propagatex(wprime * r / hypot(l, r));
    }

    void update() override
    {
        l->update();
        r->update();
        this->val = hypot(l->val, r->val);
    }
};

template<typename T>
struct Hypot3Expr : TernaryExpr<T>
{
    // Using declarations for data members of base class
    using TernaryExpr<T>::val;
    using TernaryExpr<T>::l;
    using TernaryExpr<T>::c;
    using TernaryExpr<T>::r;

    Hypot3Expr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& cc, const ExprPtr<T>& rr) : TernaryExpr<T>(v, ll, cc, rr) {}

    void propagate(const T& wprime) override
    {
        l->propagate(wprime * l->val / val);
        c->propagate(wprime * c->val / val);
        r->propagate(wprime * r->val / val);
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(wprime * l / hypot(l, c, r));
        c->propagatex(wprime * c / hypot(l, c, r));
        r->propagatex(wprime * r / hypot(l, c, r));
    }

    void update() override
    {
        l->update();
        c->update();
        r->update();
        this->val = hypot(l->val, c->val, r->val);
    }
};

// Any expression yielding a boolean depending on arithmetic subexpressions
struct BooleanExpr
{
    std::function<bool()> expr;
    bool val = {};

    explicit BooleanExpr(std::function<bool()> expression) : expr(std::move(expression)) { update(); }
    operator bool() const { return val; }

    void update() { val = expr(); }

    auto operator! () const { return BooleanExpr([=]() { return !(expr()); }); }
};

/// Capture numeric comparison between two expression trees
template<typename T, typename Comparator>
auto expr_comparison(const ExprPtr<T>& l, const ExprPtr<T>& r, Comparator&& compare) {
    return BooleanExpr([=]() mutable -> bool {
        l->update();
        r->update();
        return compare(l->val, r->val);
    });
}

template<typename Op> auto bool_expr_op(BooleanExpr& l, BooleanExpr& r, Op op) {
    return BooleanExpr([=]() mutable -> bool {
        l.update();
        r.update();
        return op(l, r);
    });
}

inline auto operator && (BooleanExpr&& l, BooleanExpr&& r) { return bool_expr_op(l, r, std::logical_and<> {}); }
inline auto operator || (BooleanExpr&& l, BooleanExpr&& r) { return bool_expr_op(l, r, std::logical_or<> {}); }

/// Select between expression branches depending on a boolean expression
template<typename T>
struct ConditionalExpr : Expr<T>
{
    // Using declarations for data members of base class
    BooleanExpr predicate;
    using Expr<T>::val;
    ExprPtr<T> l, r;

    ConditionalExpr(const BooleanExpr& wrappedPred, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : Expr<T>(wrappedPred ? ll->val : rr->val), predicate(wrappedPred), l(ll), r(rr) {}

    void propagate(const T& wprime) override
    {
        if(predicate.val) l->propagate(wprime);
        else r->propagate(wprime);
    }

    void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(derive(wprime, constant<T>(0.0)));
        r->propagatex(derive(constant<T>(0.0), wprime));
    }

    void update() override
    {
        predicate.update();
        if(predicate.val) {
            l->update();
            this->val = l->val;
        } else {
            r->update();
            this->val = r->val;
        }
    }

    ExprPtr<T> derive(const ExprPtr<T>& left, const ExprPtr<T>& right) const {
      return std::make_shared<ConditionalExpr>(predicate, left, right);
    }
};

//------------------------------------------------------------------------------
// CONVENIENT FUNCTIONS
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> operator+(const ExprPtr<T>& r) { return r; }
template<typename T> ExprPtr<T> operator-(const ExprPtr<T>& r) { return std::make_shared<NegativeExpr<T>>(-r->val, r); }

template<typename T> ExprPtr<T> operator+(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<AddExpr<T>>(l->val + r->val, l, r); }
template<typename T> ExprPtr<T> operator-(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<SubExpr<T>>(l->val - r->val, l, r); }
template<typename T> ExprPtr<T> operator*(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<MulExpr<T>>(l->val * r->val, l, r); }
template<typename T> ExprPtr<T> operator/(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<DivExpr<T>>(l->val / r->val, l, r); }

template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator+(const U& l, const ExprPtr<T>& r) { return constant<T>(l) + r; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator-(const U& l, const ExprPtr<T>& r) { return constant<T>(l) - r; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator*(const U& l, const ExprPtr<T>& r) { return constant<T>(l) * r; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator/(const U& l, const ExprPtr<T>& r) { return constant<T>(l) / r; }

template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator+(const ExprPtr<T>& l, const U& r) { return l + constant<T>(r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator-(const ExprPtr<T>& l, const U& r) { return l - constant<T>(r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator*(const ExprPtr<T>& l, const U& r) { return l * constant<T>(r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator/(const ExprPtr<T>& l, const U& r) { return l / constant<T>(r); }

//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sin(const ExprPtr<T>& x) { return std::make_shared<SinExpr<T>>(sin(x->val), x); }
template<typename T> ExprPtr<T> cos(const ExprPtr<T>& x) { return std::make_shared<CosExpr<T>>(cos(x->val), x); }
template<typename T> ExprPtr<T> tan(const ExprPtr<T>& x) { return std::make_shared<TanExpr<T>>(tan(x->val), x); }
template<typename T> ExprPtr<T> asin(const ExprPtr<T>& x) { return std::make_shared<ArcSinExpr<T>>(asin(x->val), x); }
template<typename T> ExprPtr<T> acos(const ExprPtr<T>& x) { return std::make_shared<ArcCosExpr<T>>(acos(x->val), x); }
template<typename T> ExprPtr<T> atan(const ExprPtr<T>& x) { return std::make_shared<ArcTanExpr<T>>(atan(x->val), x); }
template<typename T> ExprPtr<T> atan2(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<ArcTan2Expr<T>>(atan2(l->val, r->val), l, r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> atan2(const U& l, const ExprPtr<T>& r) { return std::make_shared<ArcTan2Expr<T>>(atan2(l, r->val), constant<T>(l), r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> atan2(const ExprPtr<T>& l, const U& r) { return std::make_shared<ArcTan2Expr<T>>(atan2(l->val, r), l, constant<T>(r)); }


//------------------------------------------------------------------------------
// HYPOT2 FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> hypot(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<Hypot2Expr<T>>(hypot(l->val, r->val), l, r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> hypot(const U& l, const ExprPtr<T>& r) { return std::make_shared<Hypot2Expr<T>>(hypot(l, r->val), constant<T>(l), r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> hypot(const ExprPtr<T>& l, const U& r) { return std::make_shared<Hypot2Expr<T>>(hypot(l->val, r), l, constant<T>(r)); }

//------------------------------------------------------------------------------
// HYPOT3 FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> hypot(const ExprPtr<T>& l, const ExprPtr<T>& c, const ExprPtr<T>& r) { return std::make_shared<Hypot3Expr<T>>(hypot(l->val,c->val, r->val), l, c, r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> hypot(const ExprPtr<T>& l, const ExprPtr<T>& c, const U& r) { return std::make_shared<Hypot3Expr<T>>(hypot(l->val, c->val, r), l, c, constant<T>(r)); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> hypot(const U& l, const ExprPtr<T>& c, const ExprPtr<T>& r) { return std::make_shared<Hypot3Expr<T>>(hypot(l, c->val, r->val), constant<T>(l), c, r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> hypot(const ExprPtr<T>& l,const U& c, const ExprPtr<T>& r) { return std::make_shared<Hypot3Expr<T>>(hypot(l->val, c, r->val), l, constant<T>(c), r); }
template<typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true> ExprPtr<T> hypot(const ExprPtr<T>& l, const U& c, const V& r) { return std::make_shared<Hypot3Expr<T>>(hypot(l->val, c, r), l, constant<T>(c), constant<T>(r)); }
template<typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true> ExprPtr<T> hypot(const U& l, const ExprPtr<T>& c, const V& r) { return std::make_shared<Hypot3Expr<T>>(hypot(l, c->val, r), constant<T>(l), c, constant<T>(r)); }
template<typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true> ExprPtr<T> hypot(const V& l, const U& c, const ExprPtr<T>& r) { return std::make_shared<Hypot3Expr<T>>(hypot(l, c, r->val), constant<T>(l), constant<T>(c), r); }

//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sinh(const ExprPtr<T>& x) { return std::make_shared<SinhExpr<T>>(sinh(x->val), x); }
template<typename T> ExprPtr<T> cosh(const ExprPtr<T>& x) { return std::make_shared<CoshExpr<T>>(cosh(x->val), x); }
template<typename T> ExprPtr<T> tanh(const ExprPtr<T>& x) { return std::make_shared<TanhExpr<T>>(tanh(x->val), x); }

//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> exp(const ExprPtr<T>& x) { return std::make_shared<ExpExpr<T>>(exp(x->val), x); }
template<typename T> ExprPtr<T> log(const ExprPtr<T>& x) { return std::make_shared<LogExpr<T>>(log(x->val), x); }
template<typename T> ExprPtr<T> log10(const ExprPtr<T>& x) { return std::make_shared<Log10Expr<T>>(log10(x->val), x); }

//------------------------------------------------------------------------------
// POWER FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sqrt(const ExprPtr<T>& x) { return std::make_shared<SqrtExpr<T>>(sqrt(x->val), x); }
template<typename T> ExprPtr<T> pow(const ExprPtr<T>& l, const ExprPtr<T>& r) { return std::make_shared<PowExpr<T>>(pow(l->val, r->val), l, r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> pow(const U& l, const ExprPtr<T>& r) { return std::make_shared<PowConstantLeftExpr<T>>(pow(l, r->val), constant<T>(l), r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> pow(const ExprPtr<T>& l, const U& r) { return std::make_shared<PowConstantRightExpr<T>>(pow(l->val, r), l, constant<T>(r)); }

//------------------------------------------------------------------------------
// OTHER FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> abs(const ExprPtr<T>& x) { return std::make_shared<AbsExpr<T>>(abs(x->val), x); }
template<typename T> ExprPtr<T> abs2(const ExprPtr<T>& x) { return x * x; }
template<typename T> ExprPtr<T> conj(const ExprPtr<T>& x) { return x; }
template<typename T> ExprPtr<T> real(const ExprPtr<T>& x) { return x; }
template<typename T> ExprPtr<T> imag(const ExprPtr<T>&) { return constant<T>(0.0); }
template<typename T> ExprPtr<T> erf(const ExprPtr<T>& x) { return std::make_shared<ErfExpr<T>>(erf(x->val), x); }

/// The autodiff variable type used for detail mode automatic differentiation.
template<typename T>
struct Variable
{
    /// The pointer to the expression tree of variable operations
    ExprPtr<T> expr;

    /// Construct a default Variable object
    Variable() : Variable(0.0) {}

    /// Construct a copy of a Variable object
    Variable(const Variable& other) : Variable(other.expr) {}

    /// Construct a Variable object with given arithmetic value
    template<typename U, Requires<isArithmetic<U>> = true>
    Variable(const U& val) : expr(std::make_shared<IndependentVariableExpr<T>>(val)) {}

    /// Construct a Variable object with given expression
    Variable(const ExprPtr<T>& e) : expr(std::make_shared<DependentVariableExpr<T>>(e)) {}

    /// Default copy assignment
    Variable& operator=(const Variable&) = default;

    /// Update the value of this variable with changes in its expression tree
    void update() { expr->update(); }

    void update(T value) {
      if(auto independentExpr = std::dynamic_pointer_cast<IndependentVariableExpr<T>>(expr)) {
        independentExpr->val = value;
        independentExpr->update();
      } else {
        throw std::logic_error("Cannot update the value of a dependent expression stored in a variable");
      }
    }

    /// Implicitly convert this Variable object into an expression pointer.
    operator const ExprPtr<T>&() const { return expr; }

    /// Assign an arithmetic value to this variable.
    template<typename U, Requires<isArithmetic<U>> = true>
    auto operator=(const U& val) -> Variable& { *this = Variable(val); return *this; }

    /// Assign an expression to this variable.
    auto operator=(const ExprPtr<T>& x) -> Variable& { *this = Variable(x); return *this; }

    // Assignment operators
    Variable& operator+=(const ExprPtr<T>& x) { *this = Variable(expr + x); return *this; }
    Variable& operator-=(const ExprPtr<T>& x) { *this = Variable(expr - x); return *this; }
    Variable& operator*=(const ExprPtr<T>& x) { *this = Variable(expr * x); return *this; }
    Variable& operator/=(const ExprPtr<T>& x) { *this = Variable(expr / x); return *this; }

	// Assignment operators with arithmetic values
    template<typename U, Requires<isArithmetic<U>> = true> Variable& operator+=(const U& x) { *this = Variable(expr + x); return *this; }
    template<typename U, Requires<isArithmetic<U>> = true> Variable& operator-=(const U& x) { *this = Variable(expr - x); return *this; }
    template<typename U, Requires<isArithmetic<U>> = true> Variable& operator*=(const U& x) { *this = Variable(expr * x); return *this; }
    template<typename U, Requires<isArithmetic<U>> = true> Variable& operator/=(const U& x) { *this = Variable(expr / x); return *this; }

#if defined(AUTODIFF_ENABLE_IMPLICIT_CONVERSION_VAR) || defined(AUTODIFF_ENABLE_IMPLICIT_CONVERSION)
    operator T() const { return expr->val; }

    template<typename U>
    operator U() const { return static_cast<U>(expr->val); }
#else
    explicit operator T() const { return expr->val; }

    template<typename U>
    explicit operator U() const { return static_cast<U>(expr->val); }
#endif
};

//------------------------------------------------------------------------------
// EXPRESSION TRAITS
//------------------------------------------------------------------------------

template<typename T, Requires<isArithmetic<T>> = true> T expr_value(const T& t) { return t; }
template<typename T> T expr_value(const ExprPtr<T>& t) { return t->val; }
template<typename T> T expr_value(const Variable<T>& t) { return t.expr->val; }

template<typename T, typename U>
using expr_common_t = std::common_type_t<decltype(expr_value(std::declval<T>())), decltype(expr_value(std::declval<U>()))>;

template<class> struct sfinae_true : std::true_type {};
template<typename T> static auto is_expr_test(int) -> sfinae_true<decltype(expr_value(std::declval<T>()))>;
template<typename T> static auto is_expr_test(long) -> std::false_type;
template<typename T> struct is_expr : decltype(is_expr_test<T>(0)) {};
template<typename T> constexpr bool is_expr_v = is_expr<T>::value;

template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> coerce_expr(const U& u) { return constant<T>(u); }
template<typename T> ExprPtr<T> coerce_expr(const ExprPtr<T>& t) { return t; }
template<typename T> ExprPtr<T> coerce_expr(const Variable<T>& t) { return t.expr; }

template<typename T, typename U> struct is_binary_expr : std::conditional_t<!(isArithmetic<T> && isArithmetic<U>) && is_expr_v<T> && is_expr_v<U>, std::true_type, std::false_type> {};
template<typename T, typename U> constexpr bool is_binary_expr_v = is_binary_expr<T, U>::value;


//------------------------------------------------------------------------------
// COMPARISON OPERATORS
//------------------------------------------------------------------------------

template<typename Comparator, typename T, typename U>
auto comparison_operator(const T& t, const U& u) {
    using C = expr_common_t<T, U>;
    return expr_comparison(coerce_expr<C>(t), coerce_expr<C>(u), Comparator {});
}

template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
auto operator == (const T& t, const U& u) { return comparison_operator<std::equal_to<>>(t, u); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
auto operator != (const T& t, const U& u) { return comparison_operator<std::not_equal_to<>>(t, u); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
auto operator <= (const T& t, const U& u) { return comparison_operator<std::less_equal<>>(t, u); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
auto operator >= (const T& t, const U& u) { return comparison_operator<std::greater_equal<>>(t, u); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
auto operator < (const T& t, const U& u) { return comparison_operator<std::less<>>(t, u); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
auto operator > (const T& t, const U& u) { return comparison_operator<std::greater<>>(t, u); }

//------------------------------------------------------------------------------
// CONDITION AND RELATED FUNCTIONS
//------------------------------------------------------------------------------

template<typename T, typename U, Requires<is_expr_v<T> && is_expr_v<U>> = true>
auto condition(BooleanExpr&& p, const T& t, const U& u) {
  using C = expr_common_t<T, U>;
  ExprPtr<C> expr = std::make_shared<ConditionalExpr<C>>(std::forward<BooleanExpr>(p), coerce_expr<C>(t), coerce_expr<C>(u));
  return expr;
}

template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true> auto min(const T& x, const U& y) { return condition(x < y, x, y); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true> auto max(const T& x, const U& y) { return condition(x > y, x, y); }
template<typename T> ExprPtr<T> sgn(const ExprPtr<T>& x) { return condition(x < 0, -1.0, condition(x > 0, 1.0, 0.0)); }
template<typename T> ExprPtr<T> sgn(const Variable<T>& x) { return condition(x.expr < 0, -1.0, condition(x.expr > 0, 1.0, 0.0)); }

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> const ExprPtr<T>& operator+(const Variable<T>& r) { return r.expr; }
template<typename T> ExprPtr<T> operator-(const Variable<T>& r) { return -r.expr; }

template<typename T> ExprPtr<T> operator+(const Variable<T>& l, const Variable<T>& r) { return l.expr + r.expr; }
template<typename T> ExprPtr<T> operator-(const Variable<T>& l, const Variable<T>& r) { return l.expr - r.expr; }
template<typename T> ExprPtr<T> operator*(const Variable<T>& l, const Variable<T>& r) { return l.expr * r.expr; }
template<typename T> ExprPtr<T> operator/(const Variable<T>& l, const Variable<T>& r) { return l.expr / r.expr; }

template<typename T> ExprPtr<T> operator+(const ExprPtr<T>& l, const Variable<T>& r) { return l + r.expr; }
template<typename T> ExprPtr<T> operator-(const ExprPtr<T>& l, const Variable<T>& r) { return l - r.expr; }
template<typename T> ExprPtr<T> operator*(const ExprPtr<T>& l, const Variable<T>& r) { return l * r.expr; }
template<typename T> ExprPtr<T> operator/(const ExprPtr<T>& l, const Variable<T>& r) { return l / r.expr; }

template<typename T> ExprPtr<T> operator+(const Variable<T>& l, const ExprPtr<T>& r) { return l.expr + r; }
template<typename T> ExprPtr<T> operator-(const Variable<T>& l, const ExprPtr<T>& r) { return l.expr - r; }
template<typename T> ExprPtr<T> operator*(const Variable<T>& l, const ExprPtr<T>& r) { return l.expr * r; }
template<typename T> ExprPtr<T> operator/(const Variable<T>& l, const ExprPtr<T>& r) { return l.expr / r; }

template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator+(const U& l, const Variable<T>& r) { return l + r.expr; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator-(const U& l, const Variable<T>& r) { return l - r.expr; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator*(const U& l, const Variable<T>& r) { return l * r.expr; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator/(const U& l, const Variable<T>& r) { return l / r.expr; }

template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator+(const Variable<T>& l, const U& r) { return l.expr + r; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator-(const Variable<T>& l, const U& r) { return l.expr - r; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator*(const Variable<T>& l, const U& r) { return l.expr * r; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> operator/(const Variable<T>& l, const U& r) { return l.expr / r; }

//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sin(const Variable<T>& x) { return sin(x.expr); }
template<typename T> ExprPtr<T> cos(const Variable<T>& x) { return cos(x.expr); }
template<typename T> ExprPtr<T> tan(const Variable<T>& x) { return tan(x.expr); }
template<typename T> ExprPtr<T> asin(const Variable<T>& x) { return asin(x.expr); }
template<typename T> ExprPtr<T> acos(const Variable<T>& x) { return acos(x.expr); }
template<typename T> ExprPtr<T> atan(const Variable<T>& x) { return atan(x.expr); }
template<typename T> ExprPtr<T> atan2(const Variable<T> & l, const Variable<T> & r) { return atan2(l.expr, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> atan2(const U& l, const Variable<T>& r) { return atan2(l, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> atan2(const Variable<T>& l, const U& r) { return atan2(l.expr, r); }

//------------------------------------------------------------------------------
// HYPOT2 FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> hypot(const Variable<T>& l, const Variable<T>& r) { return hypot(l.expr, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> hypot(const U& l, const Variable<T>& r) { return hypot(l, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> hypot(const Variable<T>& l, const U& r) { return hypot(l.expr, r); }

//------------------------------------------------------------------------------
// HYPOT3 FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> hypot(const Variable<T> &l, const Variable<T> &c, const Variable<T> &r) { return hypot(l.expr, c.expr, r.expr); }
template<typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true> ExprPtr<T> hypot(const Variable<T>& l, const U& c, const V& r) { return hypot(l.expr, c, r); }
template<typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true> ExprPtr<T> hypot(const U& l, const Variable<T>& c, const V& r) { return hypot(l, c.expr, r); }
template<typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true> ExprPtr<T> hypot(const U& l, const V& c, const Variable<T>& r) { return hypot(l, c, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> hypot(const Variable<T> &l, const Variable<T> &c, const U& r) { return hypot(l.expr, c.expr, r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> hypot(const U &l, const Variable<T> &c, const Variable<T>& r) { return hypot(l, c.expr, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> hypot(const Variable<T> &l, const U &c, const Variable<T>& r) { return hypot(l.expr, c, r.expr); }

//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sinh(const Variable<T>& x) { return sinh(x.expr); }
template<typename T> ExprPtr<T> cosh(const Variable<T>& x) { return cosh(x.expr); }
template<typename T> ExprPtr<T> tanh(const Variable<T>& x) { return tanh(x.expr); }

//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> exp(const Variable<T>& x) { return exp(x.expr); }
template<typename T> ExprPtr<T> log(const Variable<T>& x) { return log(x.expr); }
template<typename T> ExprPtr<T> log10(const Variable<T>& x) { return log10(x.expr); }

//------------------------------------------------------------------------------
// POWER FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> sqrt(const Variable<T>& x) { return sqrt(x.expr); }
template<typename T> ExprPtr<T> pow(const Variable<T>& l, const Variable<T>& r) { return pow(l.expr, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> pow(const U& l, const Variable<T>& r) { return pow(l, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> ExprPtr<T> pow(const Variable<T>& l, const U& r) { return pow(l.expr, r); }

//------------------------------------------------------------------------------
// OTHER FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> ExprPtr<T> abs(const Variable<T>& x) { return abs(x.expr); }
template<typename T> ExprPtr<T> abs2(const Variable<T>& x) { return abs2(x.expr); }
template<typename T> ExprPtr<T> conj(const Variable<T>& x) { return conj(x.expr); }
template<typename T> ExprPtr<T> real(const Variable<T>& x) { return real(x.expr); }
template<typename T> ExprPtr<T> imag(const Variable<T>& x) { return imag(x.expr); }
template<typename T> ExprPtr<T> erf(const Variable<T>& x) { return erf(x.expr); }

template<typename T, Requires<is_expr_v<T>> = true>
auto val(const T& t) { return expr_value(t); }

/// Return the derivatives of a variable y with respect to all independent variables.
template<typename T>
[[deprecated("Use method `derivatives(y, wrt(a, b, c,...)` instead.")]]
auto derivatives(const T&)
{
    static_assert(!std::is_same_v<T,T>, "Method derivatives(const var&) has been deprecated. Use method derivatives(y, wrt(a, b, c,...) instead.");
}

/// Return the derivatives of a variable y with respect to all independent variables.
template<typename T>
[[deprecated("Use method derivativesx(y, wrt(a, b, c,...) instead.")]]
auto derivativesx(const T&)
{
    static_assert(!std::is_same_v<T,T>, "Method derivativesx(const var&) has been deprecated. Use method derivativesx(y, wrt(a, b, c,...) instead.");
}

template<typename... Vars>
struct Wrt
{
    std::tuple<Vars...> args;
};

/// The keyword used to denote the variables *with respect to* the derivative is calculated.
template<typename... Args>
auto wrt(Args&&... args)
{
    return Wrt<Args&&...>{ std::forward_as_tuple(std::forward<Args>(args)...) };
}

/// Return the derivatives of a dependent variable y with respect given independent variables.
template<typename T, typename... Vars>
auto derivatives(const Variable<T>& y, const Wrt<Vars...>& wrt)
{
    constexpr auto N = sizeof...(Vars);
    std::array<T, N> values;
    values.fill(0.0);

    For<N>([&](auto i) constexpr {
        std::get<i>(wrt.args).expr->bind_value(&values.at(i));
    });

    y.expr->propagate(1.0);

    For<N>([&](auto i) constexpr {
        std::get<i>(wrt.args).expr->bind_value(nullptr);
    });

    return values;
}

/// Return the derivatives of a dependent variable y with respect given independent variables.
template<typename T, typename... Vars>
auto derivativesx(const Variable<T>& y, const Wrt<Vars...>& wrt)
{
    constexpr auto N = sizeof...(Vars);
    std::array<Variable<T>, N> values;

    For<N>([&](auto i) constexpr {
        std::get<i>(wrt.args).expr->bind_expr(&values.at(i).expr);
    });

    y.expr->propagatex(constant<T>(1.0));

    For<N>([&](auto i) constexpr {
        std::get<i>(wrt.args).expr->bind_expr(nullptr);
    });

    return values;
}

/// Output a Variable object to the output stream.
template<typename T>
std::ostream& operator<<(std::ostream& out, const Variable<T>& x)
{
    out << val(x);
    return out;
}

/// Output an ExprPrt object to the output stream.
template<typename T>
std::ostream& operator<<(std::ostream& out, const ExprPtr<T>& x)
{
    out << val(x);
    return out;
}

//=====================================================================================================================
//
// HIGHER-ORDER VAR NUMBERS
//
//=====================================================================================================================

template<size_t N, typename T>
struct AuxHigherOrderVariable;

template<typename T>
struct AuxHigherOrderVariable<0, T>
{
    using type = T;
};

template<size_t N, typename T>
struct AuxHigherOrderVariable
{
    using type = Variable<typename AuxHigherOrderVariable<N - 1, T>::type>;
};

template<size_t N, typename T>
using HigherOrderVariable = typename AuxHigherOrderVariable<N, T>::type;

} // namespace detail

} // namespace reverse

using reverse::detail::wrt;
using reverse::detail::derivatives;
using reverse::detail::Variable;
using reverse::detail::val;

using var = Variable<double>;

inline reverse::detail::BooleanExpr boolref(const bool& v) { return reverse::detail::BooleanExpr([&]() { return v; }); }

} // namespace autodiff
