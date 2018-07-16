// C++ includes
#include <cmath>
#include <iostream>
#include <memory>

namespace autodiff {

namespace internal {

struct Expr;

using ExprPtr = std::shared_ptr<Expr>;

struct Expr
{
    double val;

    explicit Expr(double val) : val(val) {}

    virtual double grad(const ExprPtr& param) const = 0;
};

struct ParameterExpr : Expr
{
    using Expr::Expr;

    virtual double grad(const ExprPtr& param) const
    {
        return this == param.get();
    }
};

struct ConstantExpr : Expr
{
    using Expr::Expr;

    virtual double grad(const ExprPtr& param) const
    {
        return 0.0;
    }
};

struct UnaryExpr : Expr
{
    ExprPtr x;

    UnaryExpr(double val, const ExprPtr& x) : Expr(val), x(x) {}
};

struct NegativeExpr : UnaryExpr
{
    using UnaryExpr::UnaryExpr;

    virtual double grad(const ExprPtr& param) const
    {
        return -x->grad(param);
    }
};

struct BinaryExpr : Expr
{
    ExprPtr l;
    ExprPtr r;

    BinaryExpr(double val, const ExprPtr& l, const ExprPtr& r) : Expr(val), l(l), r(r) {}
};

struct AddExpr : BinaryExpr
{
    using BinaryExpr::BinaryExpr;

    virtual double grad(const ExprPtr& param) const
    {
        return l->grad(param) + r->grad(param);
    }
};

struct SubExpr : BinaryExpr
{
    using BinaryExpr::BinaryExpr;

    virtual double grad(const ExprPtr& param) const
    {
        return l->grad(param) - r->grad(param);
    }
};

struct MulExpr : BinaryExpr
{
    using BinaryExpr::BinaryExpr;

    virtual double grad(const ExprPtr& param) const
    {
        return l->grad(param) * r->val + l->val * r->grad(param);
    }
};

struct DivExpr : BinaryExpr
{
    using BinaryExpr::BinaryExpr;

    virtual double grad(const ExprPtr& param) const
    {
        const double rval = r->val;
        return ( l->grad(param) * rval - l->val * r->grad(param) ) / (rval * rval);
    }
};

struct SinExpr : UnaryExpr
{
    double cos_x;

    SinExpr(double val, const ExprPtr& x) : UnaryExpr(val, x), cos_x(std::cos(x->val)) {}

    virtual double grad(const ExprPtr& param) const
    {
        return cos_x * x->grad(param);
    }
};

struct CosExpr : UnaryExpr
{
    double sin_x;

    CosExpr(double val, const ExprPtr& x) : UnaryExpr(val, x), sin_x(std::sin(x->val)) {}

    virtual double grad(const ExprPtr& param) const
    {
        return -sin_x * x->grad(param);
    }
};

struct ExpExpr : UnaryExpr
{
    using UnaryExpr::UnaryExpr;

    virtual double grad(const ExprPtr& param) const
    {
        return val * x->grad(param);
    }
};

struct LogExpr : UnaryExpr
{
    using UnaryExpr::UnaryExpr;

    virtual double grad(const ExprPtr& param) const
    {
        return 1.0 / x->val * x->grad(param);
    }
};

//struct PowerExpr : Expr
//{
//    ExprPtr x;
//
//    long p;
//
//    PowerExpr(const ExprPtr& x, long p) : x(x), p(p) {}
//};

auto operator+(const ExprPtr& r) -> ExprPtr
{
    return r;
}

auto operator-(const ExprPtr& r) -> ExprPtr
{
    return std::make_shared<NegativeExpr>(-r->val, r);
}

auto operator+(const ExprPtr& l, const ExprPtr& r) -> ExprPtr
{
    return std::make_shared<AddExpr>(l->val + r->val, l, r);
}

auto operator-(const ExprPtr& l, const ExprPtr& r) -> ExprPtr
{
    return std::make_shared<SubExpr>(l->val - r->val, l, r);
}

auto operator*(const ExprPtr& l, const ExprPtr& r) -> ExprPtr
{
    return std::make_shared<MulExpr>(l->val * r->val, l, r);
}

auto operator/(const ExprPtr& l, const ExprPtr& r) -> ExprPtr
{
    return std::make_shared<DivExpr>(l->val / r->val, l, r);
}

auto constant(double val) -> ExprPtr
{
    return std::make_shared<ConstantExpr>(val);
}

} // namespace internal

using namespace internal;

struct var
{
    ExprPtr expr;

    var() : var(0.0) {}

    var(double val) : expr(std::make_shared<ParameterExpr>(val)) {}

    var(const ExprPtr& expr) : expr(expr) {}
};

auto operator+(const var& r) -> ExprPtr { return  r.expr; }
auto operator-(const var& r) -> ExprPtr { return -r.expr; }

auto operator+(const var& l, const var& r) -> ExprPtr { return l.expr + r.expr; }
auto operator-(const var& l, const var& r) -> ExprPtr { return l.expr - r.expr; }
auto operator*(const var& l, const var& r) -> ExprPtr { return l.expr * r.expr; }
auto operator/(const var& l, const var& r) -> ExprPtr { return l.expr / r.expr; }

auto operator+(double l, const var& r) -> ExprPtr { return constant(l) + r.expr; }
auto operator-(double l, const var& r) -> ExprPtr { return constant(l) - r.expr; }
auto operator*(double l, const var& r) -> ExprPtr { return constant(l) * r.expr; }
auto operator/(double l, const var& r) -> ExprPtr { return constant(l) / r.expr; }

auto operator+(const var& l, double r) -> ExprPtr { return l.expr + constant(r); }
auto operator-(const var& l, double r) -> ExprPtr { return l.expr - constant(r); }
auto operator*(const var& l, double r) -> ExprPtr { return l.expr * constant(r); }
auto operator/(const var& l, double r) -> ExprPtr { return l.expr / constant(r); }

auto grad(const var& y, const var& x) -> double
{
    return y.expr->grad(x.expr);
}

auto operator<<(std::ostream& out, const var& x) -> std::ostream&
{
    out << x.expr->val;
    return out;
}

} // namespace autodiff
