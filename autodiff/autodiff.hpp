// C++ includes
#include <cmath>
#include <iostream>
#include <memory>

namespace autodiff {

namespace internal {

struct Expr;
struct ParameterExpr;

using ExprPtr = std::shared_ptr<Expr>;
using ValuePtr = std::shared_ptr<double>;

struct Expr
{
    virtual double eval() const = 0;
    virtual double grad(const ExprPtr& param) const = 0;
};

struct UnaryExpr : Expr
{
    UnaryExpr(const ExprPtr& x) : x(x) {}
    ExprPtr x;
};

struct BinaryExpr : Expr
{
    BinaryExpr(const ExprPtr& l, const ExprPtr& r) : l(l), r(r) {}
    ExprPtr l;
    ExprPtr r;
};

struct ZeroExpr : Expr
{
    virtual double eval() const
    {
        return 0.0;
    }

    virtual double grad(const ExprPtr& param) const
    {
        return 0.0;
    }
};

struct ConstantExpr : Expr
{
    double val;

    explicit ConstantExpr(double val) : val(val) {}

    virtual double eval() const
    {
        return val;
    }

    virtual double grad(const ExprPtr& param) const
    {
        return 0.0;
    }
};

struct AddExpr : BinaryExpr 
{
    using BinaryExpr::BinaryExpr;

    virtual double eval() const
    {
        return l->eval() + r->eval();
    }

    virtual double grad(const ExprPtr& param) const
    {
        return l->grad(param) + r->grad(param);
    }
};

struct SubExpr : BinaryExpr
{
    using BinaryExpr::BinaryExpr;
    
    virtual double eval() const
    {
        return l->eval() - r->eval();
    }

    virtual double grad(const ExprPtr& param) const
    {
        return l->grad(param) - r->grad(param);
    }
};

struct MulExpr : BinaryExpr
{
    using BinaryExpr::BinaryExpr;
    
    virtual double eval() const
    {
        return l->eval() * r->eval();
    }

    virtual double grad(const ExprPtr& param) const
    {
        return l->grad(param) * r->eval() + l->eval() * r->grad(param);
    }
};

struct DivExpr : BinaryExpr
{
    using BinaryExpr::BinaryExpr;
    
    virtual double eval() const
    {
        return l->eval() / r->eval();
    }

    virtual double grad(const ExprPtr& param) const
    {
        return ( l->grad(param) * r->eval() - l->eval() * r->grad(param) ) / (r->eval() * r->eval());
    }
};

struct SinExpr : UnaryExpr
{
    using UnaryExpr::UnaryExpr;
    
    virtual double eval() const
    {
        return std::sin(x->eval());
    }

    virtual double grad(const ExprPtr& param) const
    {
        return std::cos(x->eval()) * x->grad(param);
    }
};

struct CosExpr : UnaryExpr
{
    using UnaryExpr::UnaryExpr;

    virtual double eval() const
    {
        return std::cos(x->eval());
    }

    virtual double grad(const ExprPtr& param) const
    {
        return -std::sin(x->eval()) * x->grad(param);
    }
};

struct ExpExpr : UnaryExpr
{
    using UnaryExpr::UnaryExpr;
    
    virtual double eval() const
    {
        return std::exp(x->eval());
    }

    virtual double grad(const ExprPtr& param) const
    {
        return std::exp(x->eval()) * x->grad(param);
    }
};

struct LogExpr : UnaryExpr
{
    using UnaryExpr::UnaryExpr;
    
    virtual double eval() const
    {
        return std::log(x->eval());
    }

    virtual double grad(const ExprPtr& param) const
    {
        return 1.0 / x->eval() * x->grad(param);
    }
};

struct PowerExpr : Expr
{
    ExprPtr x;
    
    long p;

    PowerExpr(const ExprPtr& x, long p) : x(x), p(p) {}
};

struct ParameterExpr : Expr
{
    ValuePtr val;

    ParameterExpr(const ValuePtr& val) : val(val) {}

    virtual double eval() const
    {
        return *val;
    }

    virtual double grad(const ExprPtr& param) const
    {
        return this == param.get();
    }
};

auto operator+(const ExprPtr& l, const ExprPtr& r) -> ExprPtr
{
    return std::make_shared<AddExpr>(l, r);
}

auto operator-(const ExprPtr& l, const ExprPtr& r) -> ExprPtr
{
    return std::make_shared<SubExpr>(l, r);
}

auto operator*(const ExprPtr& l, const ExprPtr& r) -> ExprPtr
{
    return std::make_shared<MulExpr>(l, r);
}

auto operator/(const ExprPtr& l, const ExprPtr& r) -> ExprPtr
{
    return std::make_shared<DivExpr>(l, r);
}

auto constant(double val) -> ExprPtr
{
    return std::make_shared<ConstantExpr>(val);
}

} // namespace internal

using namespace internal;

struct var
{
    ValuePtr value;

    ExprPtr expr;

    var() : var(0.0) {}
    
    var(double val) : value(std::make_shared<double>(val)), expr(std::make_shared<ParameterExpr>(value)) {}

    var(const ExprPtr& expr) : value(std::make_shared<double>(expr->eval())), expr(expr) {}
};

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
    out << *x.value;
    return out;
}

} // namespace autodiff
