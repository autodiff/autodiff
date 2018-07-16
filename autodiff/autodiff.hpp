// C++ includes
#include <cmath>
#include <iostream>
#include <memory>

/// autodiff namespace where @ref var and @ref grad are defined.
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

struct VariableExpr : Expr
{
    ExprPtr expr;

    VariableExpr(const ExprPtr& expr) : Expr(expr->val), expr(expr) {}

    virtual double grad(const ExprPtr& param) const
    {
        return this == param.get() ? 1.0 : expr->grad(param);
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
    ExprPtr l, r;

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

struct TanExpr : UnaryExpr
{
    double rcos_x, sec_x;

    TanExpr(double val, const ExprPtr& x) : UnaryExpr(val, x), rcos_x(1.0 / std::cos(x->val)), sec_x(rcos_x * rcos_x) {}

    virtual double grad(const ExprPtr& param) const
    {
        return sec_x * x->grad(param);
    }
};

struct ArcSinExpr : UnaryExpr
{
    double ddx_arcsin_x;

    ArcSinExpr(double val, const ExprPtr& x) : UnaryExpr(val, x), ddx_arcsin_x(1.0 / std::sqrt(1 - x->val * x->val)) {}

    virtual double grad(const ExprPtr& param) const
    {
        return ddx_arcsin_x * x->grad(param);
    }
};

struct ArcCosExpr : UnaryExpr
{
    double ddx_arccos_x;

    ArcCosExpr(double val, const ExprPtr& x) : UnaryExpr(val, x), ddx_arccos_x(-1.0 / std::sqrt(1 - x->val * x->val)) {}

    virtual double grad(const ExprPtr& param) const
    {
        return ddx_arccos_x * x->grad(param);
    }
};

struct ArcTanExpr : UnaryExpr
{
    double ddx_arctan_x;

    ArcTanExpr(double val, const ExprPtr& x) : UnaryExpr(val, x), ddx_arctan_x(1.0 / (1 + x->val * x->val)) {}

    virtual double grad(const ExprPtr& param) const
    {
        return ddx_arctan_x * x->grad(param);
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

struct Log10Expr : UnaryExpr
{
    using UnaryExpr::UnaryExpr;

    const double ln10 = std::log(10.0);

    virtual double grad(const ExprPtr& param) const
    {
        return 1.0 / (ln10 * x->val) * x->grad(param);
    }
};

struct PowExpr : BinaryExpr
{
    double log_l;

    PowExpr(double val, const ExprPtr& l, const ExprPtr& r) : BinaryExpr(val, l, r), log_l(std::log(l->val)) {}

    virtual double grad(const ExprPtr& param) const
    {
        return ( log_l * r->grad(param) + r->val / l->val * l->grad(param) ) * val;
    }
};

struct SqrtExpr : UnaryExpr
{
    double aux;

    SqrtExpr(double val, const ExprPtr& x) : UnaryExpr(val, x), aux(-0.5 / val) {}

    virtual double grad(const ExprPtr& param) const
    {
        return aux * x->grad(param);
    }
};

struct AbsExpr : UnaryExpr
{
    double aux;

    AbsExpr(double val, const ExprPtr& x) : UnaryExpr(val, x), aux(x->val / val) {}

    virtual double grad(const ExprPtr& param) const
    {
        return aux * x->grad(param);
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
inline ExprPtr pow(double l, const ExprPtr& r) { return pow(constant(l), r); }
inline ExprPtr pow(const ExprPtr& l, double r) { return pow(l, constant(r)); }
inline ExprPtr sqrt(const ExprPtr& x) { return std::make_shared<SqrtExpr>(std::sqrt(x->val), x); }

//------------------------------------------------------------------------------
// OTHER FUNCTIONS
//------------------------------------------------------------------------------
inline ExprPtr abs(const ExprPtr& x) { return std::make_shared<AbsExpr>(std::abs(x->val), x); }

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

} // namespace internal

using namespace internal;

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
};

/// Return the value of a variable x.
inline double val(const var& x)
{
    return x.expr->val;
}

/// Return the derivative of a variable y with respect to variable x.
inline double grad(const var& y, const var& x)
{
    return y.expr->grad(x.expr);
}

/// Output a var object variable to the output stream.
inline std::ostream& operator<<(std::ostream& out, const var& x)
{
    out << val(x);
    return out;
}

} // namespace autodiff
