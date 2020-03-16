// Catch includes
#include "catch.hpp"

// C++ includes
#include <iostream>

// autodiff includes
#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>

using autodiff::derivatives;
using autodiff::gradient;
using autodiff::hessian;
using autodiff::val;
using autodiff::var;
using autodiff::wrt;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXvar;

/// Convenient function used in the tests to calculate the derivative of a variable y with respect to a variable x.
inline auto grad(const var& y, var& x)
{
    auto g = derivatives(y, wrt(x));
    return val(g[0]);
}

inline auto gradx(const var& y, var& x)
{
    auto g = derivativesx(y, wrt(x));
    return g[0];
}

/// A helper class to deal with catch::Approx combined with autodiff::Variable number type.
template<typename Var>
auto approx(const Var& x) -> Approx
{
    return Approx(val(x));
}

namespace autodiff {
namespace reverse {

/// A helper equal operator to deal with catch::Approx combined with autodiff::Variable number type.
template<typename T>
bool operator==(const Variable<T>& l, const Approx& r) { return val(l) == r; }

} // namespace reverse
} // namespace autodiff

TEST_CASE("autodiff::var tests", "[var]")
{
    var a = 10;
    var b = 20;
    var c = a;
    var x, y;
    var r;

    //------------------------------------------------------------------------------
    // TEST TRIVIAL DERIVATIVE CALCULATIONS
    //------------------------------------------------------------------------------
    REQUIRE( grad(a, a) == 1 );
    REQUIRE( grad(a, b) == 0 );
    REQUIRE( grad(c, c) == 1 );
    REQUIRE( grad(c, a) == 1 );
    REQUIRE( grad(c, b) == 0 );

    //------------------------------------------------------------------------------
    // TEST POSITIVE OPERATOR
    //------------------------------------------------------------------------------
    c = +a;

    REQUIRE( val(c) == val(a) );
    REQUIRE( grad(c, a) == 1 );

    //------------------------------------------------------------------------------
    // TEST NEGATIVE OPERATOR
    //------------------------------------------------------------------------------
    c = -a;

    REQUIRE( val(c) == -val(a) );
    REQUIRE( grad(c, a) == -1 );

    //------------------------------------------------------------------------------
    // TEST WHEN IDENTICAL/EQUIVALENT VARIABLES ARE PRESENT
    //------------------------------------------------------------------------------
    x = a;
    c = a*a + x;

    REQUIRE( val(c) == val(a)*val(a) + val(x) );
    REQUIRE( grad(c, a) == 2*val(a) + grad(x, a) );
    REQUIRE( grad(c, x) == 2*val(a) * grad(a, x) + 1 );

    //------------------------------------------------------------------------------
    // TEST DERIVATIVES COMPUTATION AFTER CHANGING VAR VALUE
    //------------------------------------------------------------------------------
    a = 20.0; // a is now a new independent variable

    REQUIRE( grad(c, a) == approx(0.0) );
    REQUIRE( grad(c, x) == 2*val(x) + 1 );

    //------------------------------------------------------------------------------
    // TEST MULTIPLICATION OPERATOR (USING CONSTANT FACTOR)
    //------------------------------------------------------------------------------
    c = -2*a;

    REQUIRE( grad(c, a) == -2 );

    //------------------------------------------------------------------------------
    // TEST DIVISION OPERATOR (USING CONSTANT FACTOR)
    //------------------------------------------------------------------------------
    c = a / 3.0;

    REQUIRE( grad(c, a) == approx(1.0/3.0) );

    //------------------------------------------------------------------------------
    // TEST DERIVATIVES WITH RESPECT TO DEPENDENT VARIABLES USING += -= *= /=
    //------------------------------------------------------------------------------

    a += 2.0;
    c = a * b;

    REQUIRE( grad(c, a) == approx(b) );

    a -= 3.0;
    c = a * b;

    REQUIRE( grad(c, a) == approx(b) );

    a *= 2.0;
    c = a * b;

    REQUIRE( grad(c, a) == approx(b) );

    a /= 3.0;
    c = a * b;

    REQUIRE( grad(c, a) == approx(b) );

    a += 2*b;
    c = a * b;

    REQUIRE( grad(c, a) == approx(b + a * grad(b, a)) );

    a -= 3*b;
    c = a * b;

    REQUIRE( grad(c, a) == approx(b) );

    a *= b;
    c = a * b;

    REQUIRE( grad(c, a) == approx(b) );

    a /= b;
    c = a * b;

    REQUIRE( grad(c, a) == approx(b) );

    //------------------------------------------------------------------------------
    // TEST BINARY ARITHMETIC OPERATORS
    //------------------------------------------------------------------------------
    a = 100.0;
    b = 200.0;

    c = a + b;

    REQUIRE( grad(c, a) == 1.0 );
    REQUIRE( grad(c, b) == 1.0 );

    c = a - b;

    REQUIRE( grad(c, a) ==  1.0 );
    REQUIRE( grad(c, b) == -1.0 );

    c = -a + b;

    REQUIRE( grad(c, a) == -1.0 );
    REQUIRE( grad(c, b) ==  1.0 );

    c = a + 1;

    REQUIRE( grad(c, a) == 1.0 );

    //------------------------------------------------------------------------------
    // TEST DERIVATIVES WITH RESPECT TO SUB-EXPRESSIONS
    //------------------------------------------------------------------------------
    x = 2 * a + b;
    r = x * x - a + b;

    REQUIRE( grad(r, x) == approx(2 * x) );
    REQUIRE( grad(r, a) == approx(2 * x * grad(x, a) - 1.0) );
    REQUIRE( grad(r, b) == approx(2 * x * grad(x, b) + 1.0) );

    //------------------------------------------------------------------------------
    // TEST COMPARISON OPERATORS
    //------------------------------------------------------------------------------
    a = 10;
    x = 10;

    REQUIRE( a == a );
    REQUIRE( a == x );
    REQUIRE( a == 10 );
    REQUIRE( 10 == a );

    REQUIRE( a != b );
    REQUIRE( a != 20 );
    REQUIRE( 20 != a );

    REQUIRE( a < b);
    REQUIRE( a < 20);

    REQUIRE( b > a );
    REQUIRE( 20 > a );

    REQUIRE( a <= a);
    REQUIRE( a <= x);
    REQUIRE( a <= b);
    REQUIRE( a <= 10);
    REQUIRE( a <= 20);

    REQUIRE( a >= a );
    REQUIRE( x >= a );
    REQUIRE( b >= a );
    REQUIRE( 10 >= a );
    REQUIRE( 20 >= a );

    //--------------------------------------------------------------------------
    // TEST TRIGONOMETRIC FUNCTIONS
    //--------------------------------------------------------------------------
    x = 0.5;

    REQUIRE( val(sin(x)) == approx(std::sin(val(x))) );
    REQUIRE( grad(sin(x), x) == approx(std::cos(val(x))) );

    REQUIRE( val(cos(x)) == approx(std::cos(val(x))) );
    REQUIRE( grad(cos(x), x) == approx(-std::sin(val(x))) );

    REQUIRE( val(tan(x)) == approx(std::tan(val(x))) );
    REQUIRE( grad(tan(x), x) == approx(1.0 / (std::cos(val(x)) * std::cos(val(x)))) );

    REQUIRE( val(asin(x)) == approx(std::asin(val(x))) );
    REQUIRE( grad(asin(x), x) == approx(1.0 / std::sqrt(1 - val(x) * val(x))) );

    REQUIRE( val(acos(x)) == approx(std::acos(val(x))) );
    REQUIRE( grad(acos(x), x) == approx(-1.0 / std::sqrt(1 - val(x) * val(x))) );

    REQUIRE( val(atan(x)) == approx(std::atan(val(x))) );
    REQUIRE( grad(atan(x), x) == approx(1.0 / (1 + val(x) * val(x))) );

    //--------------------------------------------------------------------------
    // TEST HYPERBOLIC FUNCTIONS
    //--------------------------------------------------------------------------
    REQUIRE( val(sinh(x)) == approx(std::sinh(val(x))) );
    REQUIRE( grad(sinh(x), x) == approx(std::cosh(val(x))) );

    REQUIRE( val(cosh(x)) == approx(std::cosh(val(x))) );
    REQUIRE( grad(cosh(x), x) == approx(std::sinh(val(x))) );

    REQUIRE( val(tanh(x)) == approx(std::tanh(val(x))) );
    REQUIRE( grad(tanh(x), x) == approx(1.0 / (std::cosh(val(x)) * std::cosh(val(x)))) );

    //--------------------------------------------------------------------------
    // TEST EXPONENTIAL AND LOGARITHMIC FUNCTIONS
    //--------------------------------------------------------------------------
    REQUIRE( val(log(x)) == approx(std::log(val(x))) );
    REQUIRE( grad(log(x), x) == approx(1.0 / val(x)) );

    REQUIRE( val(log10(x)) == approx(std::log10(val(x))) );
    REQUIRE( grad(log10(x), x) == approx(1.0 / (std::log(10) * val(x))) );

    REQUIRE( val(exp(x)) == approx(std::exp(val(x))) );
    REQUIRE( grad(exp(x), x) == approx(std::exp(val(x))) );

    //--------------------------------------------------------------------------
    // TEST POWER FUNCTIONS
    //--------------------------------------------------------------------------
    REQUIRE( val(sqrt(x)) == Approx(std::sqrt(val(x))) );
    REQUIRE( grad(sqrt(x), x) == Approx(0.5 / std::sqrt(val(x))) );

    REQUIRE( val(pow(x, 2.0)) == Approx(std::pow(val(x), 2.0)) );
    REQUIRE( grad(pow(x, 2.0), x) == Approx(2.0 * val(x)) );

    REQUIRE( val(pow(2.0, x)) == Approx(std::pow(2.0, val(x))) );
    REQUIRE( grad(pow(2.0, x), x) == Approx(std::log(2.0) * std::pow(2.0, val(x))) );

    REQUIRE( val(pow(x, x)) == Approx(std::pow(val(x), val(x))) );
    REQUIRE( grad(pow(x, x), x) == Approx(val((log(x) + 1) * pow(x, x))) );

    y = 2 * a;

    REQUIRE( y == Approx( 2 * val(a) ) );
    REQUIRE( grad(y, a) == Approx( 2.0 ) );

    REQUIRE( val(pow(x, y)) == Approx(std::pow(val(x), val(y))) );
    REQUIRE( grad(pow(x, y), x) == Approx( val(y)/val(x) * std::pow(val(x), val(y)) ) );
    REQUIRE( grad(pow(x, y), a) == Approx( std::pow(val(x), val(y)) * (val(y)/val(x) * grad(x, a) + std::log(val(x)) * grad(y, a)) ) );
    REQUIRE( grad(pow(x, y), y) == Approx( std::log(val(x)) * std::pow(val(x), val(y)) ) );

    //--------------------------------------------------------------------------
    // TEST OTHER FUNCTIONS
    //--------------------------------------------------------------------------
    x = 3.0;
    y = x;

    REQUIRE( val(abs(x)) == Approx(std::abs(val(x))) );
    REQUIRE( grad(x, x) == Approx(1.0) );

    x = 0.5;
    constexpr double pi = 3.141592653589793238462643383279502884197169399375105820974;
    REQUIRE( val(erf(x)) == approx(std::erf(val(x))) );
    REQUIRE( grad(erf(x), x) == approx(2/sqrt(pi) * std::exp(-val(x)*val(x))) );

    //--------------------------------------------------------------------------
    // TEST HIGHER ORDER DERIVATIVES (2nd order)
    //--------------------------------------------------------------------------
    REQUIRE( val(gradx(gradx(x * x, x), x)) == Approx(2.0) );
    REQUIRE( val(gradx(gradx(1.0/x, x), x)) == Approx(val(2.0/(x * x * x))) );
    REQUIRE( val(gradx(gradx(sin(x), x), x)) == Approx(val(-sin(x))) );
    REQUIRE( val(gradx(gradx(cos(x), x), x)) == Approx(val(-cos(x))) );
    REQUIRE( val(gradx(gradx(log(x), x), x)) == Approx(val(-1.0/(x * x))) );
    REQUIRE( val(gradx(gradx(exp(x), x), x)) == Approx(val(exp(x))) );
    REQUIRE( val(gradx(gradx(pow(x, 2.0), x), x)) == Approx(2.0) );
    REQUIRE( val(gradx(gradx(pow(2.0, x), x), x)) == Approx(val(std::log(2.0) * std::log(2.0) * pow(2.0, x))) );
    REQUIRE( val(gradx(gradx(pow(x, x), x), x)) == Approx(val(((log(x) + 1) * (log(x) + 1) + 1.0/x) * pow(x, x))) );
    REQUIRE( val(gradx(gradx(sqrt(x), x), x)) == Approx(val(-0.25 / (x * sqrt(x)))) );

    //--------------------------------------------------------------------------
    // TEST HIGHER ORDER DERIVATIVES (3rd order)
    //--------------------------------------------------------------------------
    REQUIRE( val(gradx(gradx(gradx(log(x), x), x), x)) == approx(val(2.0/(x * x * x))) );
    REQUIRE( val(gradx(gradx(gradx(exp(x), x), x), x)) == approx(val(exp(x))) );
}

TEST_CASE("autodiff::VectorXvar tests", "[VectorXvar]")
{
    SECTION("Testing VectorXvar")
    {
        var y;
        VectorXd g;
        MatrixXd H;
        VectorXvar x(5);
        x.setConstant(3.0);

        //--------------------------------------------------------------------------
        // TESTING GRADIENT AND HESSIAN WHEN y = sum(x)
        //--------------------------------------------------------------------------
        y = x.sum();
        g = gradient(y, x);

        CHECK( val(y) == approx(15.0) );
        for(auto i = 0; i < x.size(); ++i)
            CHECK( g[i] == approx(1.0) );

        H = hessian(y, x, g);
        for(auto i = 0; i < x.size(); ++i) {
            CHECK( val(g[i]) == approx(1.0) );
            for(auto j = 0; j < x.size(); ++j)
                CHECK( H(i, j) == approx(0.0) );
        }

        //--------------------------------------------------------------------------
        // TESTING GRADIENT AND HESSIAN WHEN y = ||x||^2
        //--------------------------------------------------------------------------
        x << 1, 2, 3, 4, 5;
        y = x.cwiseProduct(x).sum();
        g = gradient(y, x);

        CHECK( val(y) == approx(1 + 2*2 + 3*3 + 4*4 + 5*5) );
        for(auto i = 0; i < x.size(); ++i)
            CHECK( val(g[i]) == approx(2 * x[i]) );

        H = hessian(y, x, g);
        for(auto i = 0; i < x.size(); ++i) {
            CHECK( val(g[i]) == approx(2 * x[i]) );
            for(auto j = 0; j < x.size(); ++j)
                CHECK( H(i, j) == approx(i == j ? 2.0 : 0.0) );
        }

        //--------------------------------------------------------------------------
        // TESTING GRADIENT AND HESSIAN WHEN y = prod(sin(x))
        //--------------------------------------------------------------------------
        y = x.array().sin().prod();
        g = gradient(y, x);

        CHECK( val(y) == approx(sin(1) * sin(2) * sin(3) * sin(4) * sin(5)) );
        for(auto i = 0; i < x.size(); ++i)
            CHECK( val(g[i]) == approx(y / tan(x[i])) );

        H = hessian(y, x, g);
        for(auto i = 0; i < x.size(); ++i) {
            CHECK( val(g[i]) == approx(y / tan(x[i])) );
            for(auto j = 0; j < x.size(); ++j)
                if(i == j)
                    CHECK( H(i, j) == Approx(val(g[i] / tan(x[i]) * (1.0 - 1.0/(cos(x[i]) * cos(x[i]))))) );
                else
                    CHECK( H(i, j) == Approx(val(g[j] / tan(x[i]))) );
        }
    }
}