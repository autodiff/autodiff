// C++ includes
#include <iostream>

// Catch includes
#include "catch.hpp"

// autodiff includes
#include <dual.hpp>
using namespace autodiff;
using namespace autodiff::forward;

auto approx(double value)
{
    const double zero = std::numeric_limits<double>::epsilon();
    return Approx(value).margin(zero);
}

template<typename R, enableif<isExpr<R>>...>
auto approx(R&& expr)
{
    return approx(val(expr));
}

bool operator==(const dual& l, double r) { return val(l) == approx(r); }
bool operator==(double l, const dual& r) { return approx(l) == val(r); }

TEST_CASE("autodiff::dual tests", "[dual]")
{
    dual x = 100;
    dual y = 10;

    SECTION("trivial tests")
    {
        dual x = 100;

        REQUIRE( x == 100 );
        x += 1;
        REQUIRE( x == 101 );
        x -= 1;
        REQUIRE( x == 100 );
        x *= 2;
        REQUIRE( x == 200 );
        x /= 20;
        REQUIRE( x == 10 );
    }

    SECTION("testing unary operators")
    {
        std::function<dual(dual)> f;

        // Testing positive operator
        f = [](dual x) -> dual { return +x; };
        REQUIRE( f(x) == x );
        REQUIRE( grad(f, wrt(x), x) == 1.0 );

        // Testing negative operator
        f = [](dual x) -> dual { return -x; };
        REQUIRE( f(x) == -x );
        REQUIRE( grad(f, wrt(x), x) == -1.0 );

        // Testing negative operator on a negative expression
        f = [](dual x) -> dual { return -(-x); };
        REQUIRE( f(x) == x );
        REQUIRE( grad(f, wrt(x), x) == 1.0 );

        // Testing negative operator on a scaling expression expression
        f = [](dual x) -> dual { return -(2 * x); };
        REQUIRE( f(x) == -2 * val(x) );
        REQUIRE( grad(f, wrt(x), x) == -2.0 );
    }

    SECTION("testing binary addition operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing addition operator on a `scalar + dual` expression
        f = [](dual x, dual y) -> dual { return 1 + x; };
        REQUIRE( f(x, y) == 1 + val(x) );
        REQUIRE( grad(f, wrt(x), x, y) == 1.0 );
        REQUIRE( grad(f, wrt(y), x, y) == 0.0 );

        // Testing addition operator on a `dual + scalar` expression
        f = [](dual x, dual y) -> dual { return x + 1; };
        REQUIRE( f(x, y) == val(x) + 1 );
        REQUIRE( grad(f, wrt(x), x, y) == 1.0 );
        REQUIRE( grad(f, wrt(y), x, y) == 0.0 );

        // Testing addition operator on a `(-dual) + (-dual)` expression
        f = [](dual x, dual y) -> dual { return (-x) + (-y); };
        REQUIRE( f(x, y) == -(val(x) + val(y)) );
        REQUIRE( grad(f, wrt(x), x, y) == -1.0 );
        REQUIRE( grad(f, wrt(y), x, y) == -1.0 );
    }

    SECTION("testing binary subtraction operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing subtraction operator on a `scalar - dual` expression
        f = [](dual x, dual y) -> dual { return 1 - x; };
        REQUIRE( f(x, y) == 1 - val(x) );
        REQUIRE( grad(f, wrt(x), x, y) == -1.0 );
        REQUIRE( grad(f, wrt(y), x, y) ==  0.0 );

        // Testing subtraction operator on a `dual - scalar` expression
        f = [](dual x, dual y) -> dual { return x - 1; };
        REQUIRE( f(x, y) == val(x) - 1 );
        REQUIRE( grad(f, wrt(x), x, y) == 1.0 );
        REQUIRE( grad(f, wrt(y), x, y) == 0.0 );

        // Testing subtraction operator on a `(-dual) - (-dual)` expression
        f = [](dual x, dual y) -> dual { return (-x) - (-y); };
        REQUIRE( f(x, y) == -(val(x) - val(y)) );
        REQUIRE( grad(f, wrt(x), x, y) == -1.0 );
        REQUIRE( grad(f, wrt(y), x, y) ==  1.0 );
    }

    SECTION("testing binary multiplication operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing multiplication operator on a `scalar * dual` expression
        f = [](dual x, dual y) -> dual { return 2 * x; };
        REQUIRE( f(x, y) == 2 * val(x) );
        REQUIRE( grad(f, wrt(x), x, y) == 2.0 );
        REQUIRE( grad(f, wrt(y), x, y) == 0.0 );

        // Testing multiplication operator on a `dual * scalar` expression
        f = [](dual x, dual y) -> dual { return x * 2; };
        REQUIRE( f(x, y) == 2 * val(x) );
        REQUIRE( grad(f, wrt(x), x, y) == 2.0 );
        REQUIRE( grad(f, wrt(y), x, y) == 0.0 );

        // Testing multiplication operator on a `dual * dual` expression
        f = [](dual x, dual y) -> dual { return x * y; };
        REQUIRE( f(x, y) == val(x) * val(y) );
        REQUIRE( grad(f, wrt(x), x, y) == y );
        REQUIRE( grad(f, wrt(y), x, y) == x );

        // Testing multiplication operator on a `scalar * (scalar * dual)` expression
        f = [](dual x, dual y) -> dual { return 5 * (2 * x); };
        REQUIRE( f(x, y) == 10 * val(x) );
        REQUIRE( grad(f, wrt(x), x, y) == 10.0 );
        REQUIRE( grad(f, wrt(y), x, y) ==  0.0 );

        // Testing multiplication operator on a `(dual * scalar) * scalar` expression
        f = [](dual x, dual y) -> dual { return (x * 2) * 5; };
        REQUIRE( f(x, y) == 10 * val(x) );
        REQUIRE( grad(f, wrt(x), x, y) == 10.0 );
        REQUIRE( grad(f, wrt(y), x, y) ==  0.0 );

        // Testing multiplication operator on a `scalar * (-dual)` expression
        f = [](dual x, dual y) -> dual { return 2 * (-x); };
        REQUIRE( f(x, y) == -2 * val(x) );
        REQUIRE( grad(f, wrt(x), x, y) == -2.0 );
        REQUIRE( grad(f, wrt(y), x, y) ==  0.0 );

        // Testing multiplication operator on a `(-dual) * scalar` expression
        f = [](dual x, dual y) -> dual { return (-x) * 2; };
        REQUIRE( f(x, y) == -2 * val(x) );
        REQUIRE( grad(f, wrt(x), x, y) == -2.0 );
        REQUIRE( grad(f, wrt(y), x, y) ==  0.0 );

        // Testing multiplication operator on a `(-dual) * (-dual)` expression
        f = [](dual x, dual y) -> dual { return (-x) * (-y); };
        REQUIRE( f(x, y) == val(x) * val(y) );
        REQUIRE( grad(f, wrt(x), x, y) == y );
        REQUIRE( grad(f, wrt(y), x, y) == x );

        // Testing multiplication operator on a `(1/dual) * (1/dual)` expression
        f = [](dual x, dual y) -> dual { return (1 / x) * (1 / y); };
        REQUIRE( f(x, y) == 1 / (val(x) * val(y)) );
        REQUIRE( grad(f, wrt(x), x, y) == approx(-1/(x * x * y)) );
        REQUIRE( grad(f, wrt(y), x, y) == approx(-1/(x * y * y)) );
    }

    SECTION("testing binary division operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing division operator on a `scalar / dual` expression
        f = [](dual x, dual y) -> dual { return 1 / x; };
        REQUIRE( f(x, y) == 1 / val(x) );
        REQUIRE( grad(f, wrt(x), x, y) == approx(-1.0 / (x * x)) );
        REQUIRE( grad(f, wrt(y), x, y) == approx(0.0) );

        // Testing division operator on a `dual / scalar` expression
        f = [](dual x, dual y) -> dual { return x / 2; };
        REQUIRE( f(x, y) == val(x) / 2 );
        REQUIRE( grad(f, wrt(x), x, y) == approx(0.5) );
        REQUIRE( grad(f, wrt(y), x, y) == approx(0.0) );

        // Testing division operator on a `dual / dual` expression
        f = [](dual x, dual y) -> dual { return x / y; };
        REQUIRE( f(x, y) == val(x) / val(y) );
        REQUIRE( grad(f, wrt(x), x, y) == approx(1 / y) );
        REQUIRE( grad(f, wrt(y), x, y) == approx(-x / (y * y)) );

        // Testing division operator on a `1 / (1 / dual)` expression
        f = [](dual x, dual y) -> dual { return 1 / (1 / x); };
        REQUIRE( f(x, y) == val(x) );
        REQUIRE( grad(f, wrt(x), x, y) == 1.0 );
        REQUIRE( grad(f, wrt(y), x, y) == 0.0 );
    }

    SECTION("testing combination of operations")
    {
        std::function<dual(dual, dual)> f;

        // Testing multiplication with addition
        f = [](dual x, dual y) -> dual { return 2 * x + y; };
        REQUIRE( f(x, y) == 2 * val(x) + val(y) );
        REQUIRE( grad(f, wrt(x), x, y) == approx(2.0) );
        REQUIRE( grad(f, wrt(y), x, y) == approx(1.0) );

        // Testing a complex expression that is actually equivalent to one
        f = [](dual x, dual y) -> dual { return (2 * x * x - x * y + x/y) / (x * (2 * x - y + 1/y)); };
        REQUIRE( f(x, y) == 1.0 );
        REQUIRE( grad(f, wrt(x), x, y) == approx(0.0) );
        REQUIRE( grad(f, wrt(y), x, y) == approx(0.0) );
    }

    SECTION("testing mathematical functions")
    {
        std::function<dual(dual)> f;

        dual x = 0.5;

        // Testing sin function
        f = [](dual x) -> dual { return sin(x); };
        REQUIRE( f(x) == std::sin(val(x)) );
        REQUIRE( grad(f, wrt(x), x) == approx(cos(x)) );

        // Testing cos function
        f = [](dual x) -> dual { return cos(x); };
        REQUIRE( f(x) == std::cos(val(x)) );
        REQUIRE( grad(f, wrt(x), x) == approx(-sin(x)) );

        // Testing tan function
        f = [](dual x) -> dual { return tan(x); };
        REQUIRE( f(x) == std::tan(val(x)) );
        REQUIRE( grad(f, wrt(x), x) == approx(1 / (cos(x) * cos(x))) );

        // Testing asin function
        f = [](dual x) -> dual { return asin(x); };
        REQUIRE( f(x) == std::asin(val(x)) );
        REQUIRE( grad(f, wrt(x), x) == approx(1 / sqrt(1 - x * x)) );

        // Testing acos function
        f = [](dual x) -> dual { return acos(x); };
        REQUIRE( f(x) == std::acos(val(x)) );
        REQUIRE( grad(f, wrt(x), x) == approx(-1 / sqrt(1 - x * x)) );

        // Testing atan function
        f = [](dual x) -> dual { return atan(x); };
        REQUIRE( f(x) == std::atan(val(x)) );
        REQUIRE( grad(f, wrt(x), x) == approx(1 / (1 + x * x)) );

        // Testing exp function
        f = [](dual x) -> dual { return exp(x); };
        REQUIRE( f(x) == std::exp(val(x)) );
        REQUIRE( grad(f, wrt(x), x) == approx(exp(x)) );

        // Testing log function
        f = [](dual x) -> dual { return log(x); };
        REQUIRE( f(x) == std::log(val(x)) );
        REQUIRE( grad(f, wrt(x), x) == approx(1 / x) );

        // Testing log function
        f = [](dual x) -> dual { return log10(x); };
        REQUIRE( f(x) == std::log10(val(x)) );
        REQUIRE( grad(f, wrt(x), x) == approx(1 / (log(10) * x)) );

        // Testing sqrt function
        f = [](dual x) -> dual { return sqrt(x); };
        REQUIRE( f(x) == std::sqrt(val(x)) );
        REQUIRE( grad(f, wrt(x), x) == approx(0.5 / sqrt(x)) );

        // Testing abs function (when x > 0 and when x < 0)
        f = [](dual x) -> dual { return abs(x); };
        x = 1.0;
        REQUIRE( f(x) == std::abs(val(x)) );
        REQUIRE( grad(f, wrt(x), x) == approx(1.0) );
        x = -1.0;
        REQUIRE( f(x) == std::abs(val(x)) );
        REQUIRE( grad(f, wrt(x), x) == approx(-1.0) );
    }

    //------------------------------------------------------------------------------
    // TEST TRIVIAL DERIVATIVE CALCULATIONS
    //------------------------------------------------------------------------------
//    REQUIRE( grad(a, a) == 1 );
//    REQUIRE( grad(a, b) == 0 );
//    REQUIRE( grad(c, c) == 1 );
//    REQUIRE( grad(c, a) == 1 );
//    REQUIRE( grad(c, b) == 0 );

//    //------------------------------------------------------------------------------
//    // TEST POSITIVE OPERATOR
//    //------------------------------------------------------------------------------
//    c = +a;
//
//    REQUIRE( grad(c, a) == 1 );
//
//    //------------------------------------------------------------------------------
//    // TEST NEGATIVE OPERATOR
//    //------------------------------------------------------------------------------
//    c = -a;
//
//    REQUIRE( grad(c, a) == -1 );
//
//    //------------------------------------------------------------------------------
//    // TEST WHEN IDENTICAL/EQUIVALENT VARIABLES ARE PRESENT
//    //------------------------------------------------------------------------------
//    x = a;
//    c = a + x;
//
//    REQUIRE( grad(c, a) == 2 );
//    REQUIRE( grad(c, x) == 2 );
//
//    //------------------------------------------------------------------------------
//    // TEST MULTIPLICATION OPERATOR (USING CONSTANT FACTOR)
//    //------------------------------------------------------------------------------
//    c = -2*a;
//
//    REQUIRE( grad(c, a) == -2 );
//
//    //------------------------------------------------------------------------------
//    // TEST DIVISION OPERATOR (USING CONSTANT FACTOR)
//    //------------------------------------------------------------------------------
//    c = a / 3;
//
//    REQUIRE( grad(c, a) == 1.0/3.0 );
//
//    //------------------------------------------------------------------------------
//    // TEST BINARY ARITHMETIC OPERATORS
//    //------------------------------------------------------------------------------
//    c = a + b;
//
//    REQUIRE( grad(c, a) == 1.0 );
//    REQUIRE( grad(c, b) == 1.0 );
//
//    c = a - b;
//
//    REQUIRE( grad(c, a) ==  1.0 );
//    REQUIRE( grad(c, b) == -1.0 );
//
//    c = -a + b;
//
//    REQUIRE( grad(c, a) == -1.0 );
//    REQUIRE( grad(c, b) ==  1.0 );
//
//    c = a + 1;
//
//    REQUIRE( grad(c, a) == 1.0 );
//
//    //------------------------------------------------------------------------------
//    // TEST DERIVATIVES WITH RESPECT TO SUB-EXPRESSIONS
//    //------------------------------------------------------------------------------
//    x = 2 * a + b;
//    r = x * x - a + b;
//
//    REQUIRE( grad(r, x) == approx(2 * x) );
//    REQUIRE( grad(r, a) == approx(2 * x * grad(x, a) - 1.0) );
//    REQUIRE( grad(r, b) == approx(2 * x * grad(x, b) + 1.0) );
//
//    //------------------------------------------------------------------------------
//    // TEST COMPARISON OPERATORS
//    //------------------------------------------------------------------------------
//    x = 10;
//
//    REQUIRE( a == a );
//    REQUIRE( a == x );
//    REQUIRE( a == 10 );
//    REQUIRE( 10 == a );
//
//    REQUIRE( a != b );
//    REQUIRE( a != 20 );
//    REQUIRE( 20 != a );
//
//    REQUIRE( a < b);
//    REQUIRE( a < 20);
//
//    REQUIRE( b > a );
//    REQUIRE( 20 > a );
//
//    REQUIRE( a <= a);
//    REQUIRE( a <= x);
//    REQUIRE( a <= b);
//    REQUIRE( a <= 10);
//    REQUIRE( a <= 20);
//
//    REQUIRE( a >= a );
//    REQUIRE( x >= a );
//    REQUIRE( b >= a );
//    REQUIRE( 10 >= a );
//    REQUIRE( 20 >= a );
//
//    //--------------------------------------------------------------------------
//    // TEST TRIGONOMETRIC FUNCTIONS
//    //--------------------------------------------------------------------------
//    x = 0.5;
//
//    REQUIRE( val(sin(x)) == approx(std::sin(val(x))) );
//    REQUIRE( grad(sin(x), x) == approx(std::cos(val(x))) );
//
//    REQUIRE( val(cos(x)) == approx(std::cos(val(x))) );
//    REQUIRE( grad(cos(x), x) == approx(-std::sin(val(x))) );
//
//    REQUIRE( val(tan(x)) == approx(std::tan(val(x))) );
//    REQUIRE( grad(tan(x), x) == approx(1.0 / (std::cos(val(x)) * std::cos(val(x)))) );
//
//    REQUIRE( val(asin(x)) == approx(std::asin(val(x))) );
//    REQUIRE( grad(asin(x), x) == approx(1.0 / std::sqrt(1 - val(x) * val(x))) );
//
//    REQUIRE( val(acos(x)) == approx(std::acos(val(x))) );
//    REQUIRE( grad(acos(x), x) == approx(-1.0 / std::sqrt(1 - val(x) * val(x))) );
//
//    REQUIRE( val(atan(x)) == approx(std::atan(val(x))) );
//    REQUIRE( grad(atan(x), x) == approx(1.0 / (1 + val(x) * val(x))) );
//
//    //--------------------------------------------------------------------------
//    // TEST HYPERBOLIC FUNCTIONS
//    //--------------------------------------------------------------------------
//    //--------------------------------------------------------------------------
//    // TEST EXPONENTIAL AND LOGARITHMIC FUNCTIONS
//    //--------------------------------------------------------------------------
//    REQUIRE( val(log(x)) == approx(std::log(val(x))) );
//    REQUIRE( grad(log(x), x) == approx(1.0 / val(x)) );
//
//    REQUIRE( val(log10(x)) == approx(std::log10(val(x))) );
//    REQUIRE( grad(log10(x), x) == approx(1.0 / (std::log(10) * val(x))) );
//
//    REQUIRE( val(exp(x)) == approx(std::exp(val(x))) );
//    REQUIRE( grad(exp(x), x) == approx(std::exp(val(x))) );
//
//    //--------------------------------------------------------------------------
//    // TEST POWER FUNCTIONS
//    //--------------------------------------------------------------------------
//    REQUIRE( val(sqrt(x)) == Approx(std::sqrt(val(x))) );
//    REQUIRE( grad(sqrt(x), x) == Approx(0.5 / std::sqrt(val(x))) );
//
//    REQUIRE( val(pow(x, 2.0)) == Approx(std::pow(val(x), 2.0)) );
//    REQUIRE( grad(pow(x, 2.0), x) == Approx(2.0 * val(x)) );
//
//    REQUIRE( val(pow(2.0, x)) == Approx(std::pow(2.0, val(x))) );
//    REQUIRE( grad(pow(2.0, x), x) == Approx(std::log(2.0) * std::pow(2.0, val(x))) );
//
//    REQUIRE( val(pow(x, x)) == Approx(std::pow(val(x), val(x))) );
//    REQUIRE( grad(pow(x, x), x) == Approx(val((log(x) + 1) * pow(x, x))) );
//
//    y = 2 * a;
//
//    REQUIRE( val(pow(x, y)) == Approx(std::pow(val(x), val(y))) );
//    REQUIRE( grad(pow(x, y), x) == Approx( val(y)/val(x) * std::pow(val(x), val(y)) ) );
//    REQUIRE( grad(pow(x, y), a) == Approx( std::log(val(x)) * grad(y, a) * std::pow(val(x), val(y)) ) );
//    REQUIRE( grad(pow(x, y), y) == Approx( std::log(val(x)) * std::pow(val(x), val(y)) ) );
//
//    //--------------------------------------------------------------------------
//    // TEST OTHER FUNCTIONS
//    //--------------------------------------------------------------------------
//    x = 3.0;
//    y = x;
//
//    REQUIRE( val(abs(x)) == Approx(std::abs(val(x))) );
//    REQUIRE( val(grad(x, x)) == Approx(1.0) );
//
//    //--------------------------------------------------------------------------
//    // TEST HIGHER ORDER DERIVATIVES (2nd order)
//    //--------------------------------------------------------------------------
//    REQUIRE( val(gradx(gradx(x * x, x), x)) == Approx(2.0) );
//    REQUIRE( val(gradx(gradx(1.0/x, x), x)) == Approx(val(2.0/(x * x * x))) );
//    REQUIRE( val(gradx(gradx(sin(x), x), x)) == Approx(val(-sin(x))) );
//    REQUIRE( val(gradx(gradx(cos(x), x), x)) == Approx(val(-cos(x))) );
//    REQUIRE( val(gradx(gradx(log(x), x), x)) == Approx(val(-1.0/(x * x))) );
//    REQUIRE( val(gradx(gradx(exp(x), x), x)) == Approx(val(exp(x))) );
//    REQUIRE( val(gradx(gradx(pow(x, 2.0), x), x)) == Approx(2.0) );
//    REQUIRE( val(gradx(gradx(pow(2.0, x), x), x)) == Approx(val(std::log(2.0) * std::log(2.0) * pow(2.0, x))) );
//    REQUIRE( val(gradx(gradx(pow(x, x), x), x)) == Approx(val(((log(x) + 1) * (log(x) + 1) + 1.0/x) * pow(x, x))) );
//    REQUIRE( val(gradx(gradx(sqrt(x), x), x)) == Approx(val(-0.25 / (x * sqrt(x)))) );
//
//    //--------------------------------------------------------------------------
//    // TEST HIGHER ORDER DERIVATIVES (3rd order)
//    //--------------------------------------------------------------------------
//    REQUIRE( val(gradx(gradx(gradx(log(x), x), x), x)) == approx(val(2.0/(x * x * x))) );
//    REQUIRE( val(gradx(gradx(gradx(exp(x), x), x), x)) == approx(val(exp(x))) );
}
