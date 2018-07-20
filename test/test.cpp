// Catch includes
#include "catch.hpp"

// autodiff includes
#include <autodiff.hpp>
using namespace autodiff;

TEST_CASE("autodiff tests", "[autodiff]")
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

    REQUIRE( grad(c, a) == 1 );

    //------------------------------------------------------------------------------
    // TEST NEGATIVE OPERATOR
    //------------------------------------------------------------------------------
    c = -a;

    REQUIRE( grad(c, a) == -1 );

    //------------------------------------------------------------------------------
    // TEST WHEN IDENTICAL/EQUIVALENT VARIABLES ARE PRESENT
    //------------------------------------------------------------------------------
    x = a;
    c = a + x;

    REQUIRE( grad(c, a) == 2 );
    REQUIRE( grad(c, x) == 2 );

    //------------------------------------------------------------------------------
    // TEST MULTIPLICATION OPERATOR (USING CONSTANT FACTOR)
    //------------------------------------------------------------------------------
    c = -2*a;

    REQUIRE( grad(c, a) == -2 );

    //------------------------------------------------------------------------------
    // TEST DIVISION OPERATOR (USING CONSTANT FACTOR)
    //------------------------------------------------------------------------------
    c = a / 3;

    REQUIRE( grad(c, a) == 1.0/3.0 );

    //------------------------------------------------------------------------------
    // TEST BINARY ARITHMETIC OPERATORS
    //------------------------------------------------------------------------------
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

    REQUIRE( grad(r, x) == Approx(2 * val(x)) );
    REQUIRE( grad(r, a) == Approx(2 * val(x) * grad(x, a) - 1.0) );
    REQUIRE( grad(r, b) == Approx(2 * val(x) * grad(x, b) + 1.0) );

    //------------------------------------------------------------------------------
    // TEST COMPARISON OPERATORS
    //------------------------------------------------------------------------------
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

    REQUIRE( val(sin(x)) == Approx(std::sin(val(x))) );
    REQUIRE( grad(sin(x), x) == Approx(std::cos(val(x))) );

    REQUIRE( val(cos(x)) == Approx(std::cos(val(x))) );
    REQUIRE( grad(cos(x), x) == Approx(-std::sin(val(x))) );

    REQUIRE( val(tan(x)) == Approx(std::tan(val(x))) );
    REQUIRE( grad(tan(x), x) == Approx(1.0 / (std::cos(val(x)) * std::cos(val(x)))) );

    REQUIRE( val(asin(x)) == Approx(std::asin(val(x))) );
    REQUIRE( grad(asin(x), x) == Approx(1.0 / std::sqrt(1 - val(x) * val(x))) );

    REQUIRE( val(acos(x)) == Approx(std::acos(val(x))) );
    REQUIRE( grad(acos(x), x) == Approx(-1.0 / std::sqrt(1 - val(x) * val(x))) );

    REQUIRE( val(atan(x)) == Approx(std::atan(val(x))) );
    REQUIRE( grad(atan(x), x) == Approx(1.0 / (1 + val(x) * val(x))) );

    //--------------------------------------------------------------------------
    // TEST HYPERBOLIC FUNCTIONS
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    // TEST EXPONENTIAL AND LOGARITHMIC FUNCTIONS
    //--------------------------------------------------------------------------
    REQUIRE( val(log(x)) == Approx(std::log(val(x))) );
    REQUIRE( grad(log(x), x) == Approx(1.0 / val(x)) );

    REQUIRE( val(log10(x)) == Approx(std::log10(val(x))) );
    REQUIRE( grad(log10(x), x) == Approx(1.0 / (std::log(10) * val(x))) );

    REQUIRE( val(exp(x)) == Approx(std::exp(val(x))) );
    REQUIRE( grad(exp(x), x) == Approx(std::exp(val(x))) );

    //--------------------------------------------------------------------------
    // TEST POWER FUNCTIONS
    //--------------------------------------------------------------------------
    REQUIRE( val(sqrt(x)) == Approx(std::sqrt(val(x))) );
    REQUIRE( grad(sqrt(x), x) == Approx(-0.5 / std::sqrt(val(x))) );

    REQUIRE( val(pow(x, 2.0)) == Approx(std::pow(val(x), 2.0)) );
    REQUIRE( grad(pow(x, 2.0), x) == Approx(2.0 * val(x)) );

    REQUIRE( val(pow(x, -2.0)) == Approx(std::pow(val(x), -2.0)) );
    REQUIRE( grad(pow(x, -2.0), x) == Approx(-2.0 * std::pow(val(x), -3.0)) );

    y = 2 * a;

    REQUIRE( val(pow(x, y)) == Approx(std::pow(val(x), val(y))) );
    REQUIRE( grad(pow(x, y), x) == Approx( val(y)/val(x) * std::pow(val(x), val(y)) ) );
    REQUIRE( grad(pow(x, y), a) == Approx( std::log(val(x)) * grad(y, a) * std::pow(val(x), val(y)) ) );
    REQUIRE( grad(pow(x, y), y) == Approx( std::log(val(x)) * std::pow(val(x), val(y)) ) );

    //--------------------------------------------------------------------------
    // TEST OTHER FUNCTIONS
    //--------------------------------------------------------------------------
    x = 1.0;
    y = x;

    REQUIRE( val(abs(x)) == Approx(std::abs(val(x))) );
    REQUIRE( val(grad(x, x)) == Approx(1.0) );

    REQUIRE( val(abs(y)) == Approx(std::abs(val(y))) );
    REQUIRE( val(grad(y, x)) == Approx(val(y) / std::abs(val(y)) * grad(y, x)) );
}
