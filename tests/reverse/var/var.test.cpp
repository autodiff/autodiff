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

// Catch includes
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// autodiff includes
#include <autodiff/reverse/var.hpp>

using autodiff::derivatives;
using autodiff::val;
using autodiff::var;
using autodiff::wrt;

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

template<typename Var>
auto approx(const Var& x) -> Catch::Approx
{
    return Catch::Approx(val(x));
}

TEST_CASE("testing autodiff::var", "[reverse][var]")
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

    //------------------------------------------------------------------------------
    // TEST COMPARISON OPERATORS BETWEEN VARIABLE AND EXPRPTR
    //------------------------------------------------------------------------------
    REQUIRE( a == a/a * a );
    REQUIRE( a/a * a == a );

    REQUIRE( a != a - a );
    REQUIRE( a - a != a );

    REQUIRE( a - a < a );
    REQUIRE( a < a + a );

    REQUIRE( a + a > a );
    REQUIRE( a > a - a );

    REQUIRE( a <= a - a + a );
    REQUIRE( a - a + a <= a );

    REQUIRE( a <= a + a );
    REQUIRE( a - a <= a );

    REQUIRE( a >= a - a + a );
    REQUIRE( a - a + a >= a );

    REQUIRE( a + a >= a );
    REQUIRE( a >= a - a );

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
    REQUIRE( val(sqrt(x)) == Catch::Approx(std::sqrt(val(x))) );
    REQUIRE( grad(sqrt(x), x) == Catch::Approx(0.5 / std::sqrt(val(x))) );

    REQUIRE( val(pow(x, 2.0)) == Catch::Approx(std::pow(val(x), 2.0)) );
    REQUIRE( grad(pow(x, 2.0), x) == Catch::Approx(2.0 * val(x)) );

    REQUIRE( val(pow(2.0, x)) == Catch::Approx(std::pow(2.0, val(x))) );
    REQUIRE( grad(pow(2.0, x), x) == Catch::Approx(std::log(2.0) * std::pow(2.0, val(x))) );

    REQUIRE( val(pow(x, x)) == Catch::Approx(std::pow(val(x), val(x))) );
    REQUIRE( grad(pow(x, x), x) == Catch::Approx(val((log(x) + 1) * pow(x, x))) );

    y = 2 * a;

    REQUIRE( val(y) == Catch::Approx( 2 * val(a) ) );
    REQUIRE( grad(y, a) == Catch::Approx( 2.0 ) );

    REQUIRE( val(pow(x, y)) == Catch::Approx(std::pow(val(x), val(y))) );
    REQUIRE( grad(pow(x, y), x) == Catch::Approx( val(y)/val(x) * std::pow(val(x), val(y)) ) );
    REQUIRE( grad(pow(x, y), a) == Catch::Approx( std::pow(val(x), val(y)) * (val(y)/val(x) * grad(x, a) + std::log(val(x)) * grad(y, a)) ) );
    REQUIRE( grad(pow(x, y), y) == Catch::Approx( std::log(val(x)) * std::pow(val(x), val(y)) ) );


    //--------------------------------------------------------------------------
    // TEST ABS FUNCTION
    //--------------------------------------------------------------------------

    x = 1.0;
    REQUIRE( val(abs(x)) == std::abs(val(x)) );
    REQUIRE( grad(abs(x), x) == approx(1.0) );
    x = -1.0;
    REQUIRE( val(abs(x)) == std::abs(val(x)) );
    REQUIRE( grad(abs(x), x) == approx(-1.0) );
    x = 0.0;
    REQUIRE( val(abs(x)) == std::abs(val(x)) );
    REQUIRE( grad(abs(x), x) == approx(0.0) );


    //--------------------------------------------------------------------------
    // TEST ATAN2 FUNCTION
    //--------------------------------------------------------------------------

    // Testing atan2 function on (double, var)
    x = 1.0;
    REQUIRE( atan2(2.0, x) == std::atan2(2.0, val(x)) );
    REQUIRE( grad(atan2(2.0, x), x) == approx(-2.0 / (2*2 + x*x)) );

    // Testing atan2 function on (var, double)
    x = 1.0;
    REQUIRE( atan2(x, 2.0) == std::atan2(val(x), 2.0) );
    REQUIRE( grad(atan2(x, 2.0), x) == approx(2.0 / (2*2 + x*x)) );

    // Testing atan2 function on (var, var)
    x = 1.1;
    y = 0.9;
    REQUIRE( atan2(y, x) == std::atan2(val(y), val(x)) );
    REQUIRE( grad(atan2(y, x), y) == approx(x / (x*x + y*y)) );
    REQUIRE( grad(atan2(y, x), x) == approx(-y / (x*x + y*y)) );

    // Testing atan2 function on (expr, expr)
    REQUIRE( 3*atan2(sin(y), 2 * x + 1) == 3 * std::atan2(sin(val(y)), 2*val(x)+1) );
    REQUIRE( grad(3*atan2(sin(y), 2 * x + 1), y) == approx(3*(2*x+1)*cos(y) / ((2*x+1)*(2*x+1) + sin(y)*sin(y))) );
    REQUIRE( grad(3*atan2(sin(y), 2 * x + 1), x) == approx(3*-2*sin(y) / ((2*x+1)*(2*x+1) + sin(y)*sin(y))) );


    //--------------------------------------------------------------------------
    // TEST HYPOT2 FUNCTIONS
    //--------------------------------------------------------------------------

    // Testing hypot function on (var, double)
    x = 1.8;
    REQUIRE( hypot(x, 2.0) == std::hypot(val(x), 2.0) );
    REQUIRE( grad(hypot(x, 2.0), x) == approx(x / std::hypot(val(x), 2.0)) );

    // Testing hypot function on (double, var)
    y = 1.5;
    REQUIRE( hypot(2.0, y) == std::hypot(2.0, val(y)) );
    REQUIRE( grad(hypot(2.0, y), y) == approx(y / std::hypot(2.0, val(y))) );

    // Testing hypot function on (var, var)
    x = 1.3;
    y = 2.3;
    REQUIRE( hypot(x, y) == std::hypot(val(x), val(y)) );
    REQUIRE( grad(hypot(x, y), x) == approx(x / std::hypot(val(x), val(y))) );
    REQUIRE( grad(hypot(x, y), y) == approx(y / std::hypot(val(x), val(y))) );

    // Testing hypot function on (expr, expr)
    x = 1.3;
    y = 2.3;
    REQUIRE( hypot(2.0*x, 3.0*y) == std::hypot(2.0*val(x), 3.0*val(y)) );
    REQUIRE( grad(hypot(2.0*x, 3.0*y), x) == approx(4.0*x / std::hypot(2.0*val(x), 3.0*val(y))) );
    REQUIRE( grad(hypot(2.0*x, 3.0*y), y) == approx(9.0*y / std::hypot(2.0*val(x), 3.0*val(y))) );


    //--------------------------------------------------------------------------
    // TEST HYPOT3 FUNCTIONS
    //--------------------------------------------------------------------------

    // Testing hypot function on (var, double, double)
    x = 1.5;
    REQUIRE( hypot(x, 2.0, 3.0) == std::hypot(val(x), 2.0, 3.0) );
    REQUIRE( grad(hypot(x, 2.0, 3.0), x) == approx(x / std::hypot(val(x), 2.0, 3.0)) );

    // Testing hypot function on (double, var, double)
    y = 1.8;
    REQUIRE( hypot(2.0, y, 3.0) == std::hypot(2.0, val(y), 3.0) );
    REQUIRE( grad(hypot(2.0, y, 3.0), y) == approx(y / std::hypot(2.0, val(y), 3.0)) );

    // Testing hypot function on (double, var, double)
    var z = 1.9;
    REQUIRE( hypot(2.0, 3.0, z) == std::hypot(2.0, 3.0, val(z)) );
    REQUIRE( grad(hypot(2.0, 3.0, z), z) == approx(z / std::hypot(2.0, 3.0, val(z))) );

    // Testing hypot function on (var, var, double)
    x = 1.3;
    y = 2.3;
    REQUIRE( hypot(x, y, 2.0) == std::hypot(val(x), val(y), 2.0) );
    REQUIRE( grad(hypot(x, y, 2.0), x) == approx(x / std::hypot(val(x), val(y), 2.0)) );
    REQUIRE( grad(hypot(x, y, 2.0), y) == approx(y / std::hypot(val(x), val(y), 2.0)) );

    // Testing hypot function on (double, var, var)
    y = 2.3;
    z = 3.3;
    REQUIRE( hypot(2.0, y, z) == std::hypot(2.0, val(y), val(z)) );
    REQUIRE( grad(hypot(2.0, y, z), y) == approx(y / std::hypot(2.0, val(y), val(z))) );
    REQUIRE( grad(hypot(2.0, y, z), z) == approx(z / std::hypot(2.0, val(y), val(z))) );

    // Testing hypot function on (double, var, var)
    x = 3.3;
    z = 4.3;
    REQUIRE( hypot(x, 2.0, z) == std::hypot(val(x), 2.0, val(z)) );
    REQUIRE( grad(hypot(x, 2.0, z), x) == approx(x / std::hypot(val(x), 2.0, val(z))) );
    REQUIRE( grad(hypot(x, 2.0, z), z) == approx(z / std::hypot(val(x), 2.0, val(z))) );

    // Testing hypot function on (var, var, var)
    x = 4.3;
    y = 5.3;
    z = 6.3;
    REQUIRE( hypot(x, y, z) == std::hypot(val(x), val(y), val(z)) );
    REQUIRE( grad(hypot(x, y, z), x) == approx(x / std::hypot(val(x), val(y), val(z))) );
    REQUIRE( grad(hypot(x, y, z), y) == approx(y / std::hypot(val(x), val(y), val(z))) );
    REQUIRE( grad(hypot(x, y, z), z) == approx(z / std::hypot(val(x), val(y), val(z))) );

    // Testing hypot function on (expr, expr, expr)
    REQUIRE( hypot(2.0*x, 3.0*y, 4.0*z) == std::hypot(2.0*val(x), 3.0*val(y), 4.0*val(z)) );
    REQUIRE( grad(hypot(2.0*x, 3.0*y, 4.0*z), x) == approx(4.0*x / std::hypot(2.0*val(x), 3.0*val(y), 4.0*val(z))) );
    REQUIRE( grad(hypot(2.0*x, 3.0*y, 4.0*z), y) == approx(9.0*y / std::hypot(2.0*val(x), 3.0*val(y), 4.0*val(z))) );
    REQUIRE( grad(hypot(2.0*x, 3.0*y, 4.0*z), z) == approx(16.*z / std::hypot(2.0*val(x), 3.0*val(y), 4.0*val(z))) );


    //--------------------------------------------------------------------------
    // TEST CONDITION FUNCTIONS
    //--------------------------------------------------------------------------

    // Single condition
    x = 2.0;
    REQUIRE( condition(x > 0, x * x, x * x * x) == 4 );
    REQUIRE( grad(condition(x > 0, x * x, x * x * x), x) == approx(2 * val(x)) );

    x.update(-2.0);
    var conditional = condition(x > 0, x * x, x * x * x);
    REQUIRE( conditional == -8 );
    REQUIRE( grad(conditional, x) == approx(3 * val(x) * val(x)) );

    x.update(3.0);
    conditional.update();
    REQUIRE( x == 3.0 );
    REQUIRE( conditional == 9 );
    REQUIRE( grad(conditional, x) == approx(2 * val(x)) );

    // Conjunction of conditions
    var square = condition(0 <= x && x <= 1, 1.0, 0.0);
    REQUIRE( square == 0.0 );
    x.update(0.5);
    square.update();
    REQUIRE( square == 1.0 );
    x.update(-1.0);
    square.update();
    REQUIRE( square == 0.0 );

    // boolref
    bool arbitraryCondition = true;
    conditional = condition(autodiff::boolref(arbitraryCondition), 1.0, 0.0);
    REQUIRE( conditional == 1.0 );
    arbitraryCondition = false;
    conditional.update();
    REQUIRE( conditional == 0.0 );

    // min/max/sgn
    x = 1.0;
    y = 2.0;
    conditional = min(x, y);
    REQUIRE(conditional == 1.0);
    conditional = max(x, y);
    REQUIRE(conditional == 2.0);
    conditional = sgn(x);
    REQUIRE(conditional == 1.0);

    x.update(-1);
    conditional.update();
    REQUIRE(conditional == -1.0);

    x.update(0);
    conditional.update();
    REQUIRE(conditional == 0.0);

    // Gradients of conditional functions with updating
    x = 1;
    y = 2;
    conditional = condition(x < y, x * y, x * x);
    REQUIRE(grad(conditional, x) == approx(val(y)));
    REQUIRE(grad(conditional, y) == approx(val(x)));
    x.update(3.0);
    conditional.update();
    REQUIRE(grad(conditional, x) == approx(2 * val(x)));
    REQUIRE(grad(conditional, y) == approx(0.0));

    //--------------------------------------------------------------------------
    // TEST OTHER FUNCTIONS
    //--------------------------------------------------------------------------
    x = 3.0;
    y = x;

    REQUIRE( val(abs(x)) == Catch::Approx(std::abs(val(x))) );
    REQUIRE( grad(x, x) == Catch::Approx(1.0) );

    x = 0.5;
    constexpr double pi = 3.141592653589793238462643383279502884197169399375105820974;
    REQUIRE( val(erf(x)) == approx(std::erf(val(x))) );
    REQUIRE( grad(erf(x), x) == approx(2/sqrt(pi) * std::exp(-val(x)*val(x))) );

    //--------------------------------------------------------------------------
    // TEST HIGHER ORDER DERIVATIVES (2nd order)
    //--------------------------------------------------------------------------
    x = 0.5;
    y = 0.7;
    z = 0.3;

    REQUIRE( val(gradx(gradx(x * x, x), x)) == Catch::Approx(2.0) );
    REQUIRE( val(gradx(gradx(1.0/x, x), x)) == Catch::Approx(val(2.0/(x * x * x))) );
    REQUIRE( val(gradx(gradx(sin(x), x), x)) == Catch::Approx(val(-sin(x))) );
    REQUIRE( val(gradx(gradx(cos(x), x), x)) == Catch::Approx(val(-cos(x))) );
    REQUIRE( val(gradx(gradx(log(x), x), x)) == Catch::Approx(val(-1.0/(x * x))) );
    REQUIRE( val(gradx(gradx(exp(x), x), x)) == Catch::Approx(val(exp(x))) );
    REQUIRE( val(gradx(gradx(pow(x, 2.0), x), x)) == Catch::Approx(2.0) );
    REQUIRE( val(gradx(gradx(pow(2.0, x), x), x)) == Catch::Approx(val(std::log(2.0) * std::log(2.0) * pow(2.0, x))) );
    REQUIRE( val(gradx(gradx(pow(x, x), x), x)) == Catch::Approx(val(((log(x) + 1) * (log(x) + 1) + 1.0/x) * pow(x, x))) );
    REQUIRE( val(gradx(gradx(sqrt(x), x), x)) == Catch::Approx(val(-0.25 / (x * sqrt(x)))) );

    REQUIRE( val(gradx(gradx(hypot(x, y), x), x)) == Catch::Approx(val(grad(x / hypot(x, y), x))) ); // hypot(x, y)'x = 1/2 * 1/hypot(x, y) * (2*x) = x/hypot(x, y)
    REQUIRE( val(gradx(gradx(hypot(x, y), x), y)) == Catch::Approx(val(grad(x / hypot(x, y), y))) ); // hypot(x, y)'y = 1/2 * 1/hypot(x, y) * (2*y) = y/hypot(x, y)
    REQUIRE( val(gradx(gradx(hypot(x, y), y), x)) == Catch::Approx(val(grad(y / hypot(x, y), x))) );
    REQUIRE( val(gradx(gradx(hypot(x, y), y), y)) == Catch::Approx(val(grad(y / hypot(x, y), y))) );

    REQUIRE( val(gradx(gradx(hypot(x, y, z), x), x)) == Catch::Approx(val(grad(x / hypot(x, y, z), x))) ); // hypot(x, y, z)'x = 1/2 * 1/hypot(x, y, z) * (2*x) = x/hypot(x, y, z)
    REQUIRE( val(gradx(gradx(hypot(x, y, z), x), y)) == Catch::Approx(val(grad(x / hypot(x, y, z), y))) ); // hypot(x, y, z)'y = 1/2 * 1/hypot(x, y, z) * (2*y) = y/hypot(x, y, z)
    REQUIRE( val(gradx(gradx(hypot(x, y, z), x), z)) == Catch::Approx(val(grad(x / hypot(x, y, z), z))) ); // hypot(x, y, z)'z = 1/2 * 1/hypot(x, y, z) * (2*z) = z/hypot(x, y, z)
    REQUIRE( val(gradx(gradx(hypot(x, y, z), y), x)) == Catch::Approx(val(grad(y / hypot(x, y, z), x))) );
    REQUIRE( val(gradx(gradx(hypot(x, y, z), y), y)) == Catch::Approx(val(grad(y / hypot(x, y, z), y))) );
    REQUIRE( val(gradx(gradx(hypot(x, y, z), y), z)) == Catch::Approx(val(grad(y / hypot(x, y, z), z))) );
    REQUIRE( val(gradx(gradx(hypot(x, y, z), z), x)) == Catch::Approx(val(grad(z / hypot(x, y, z), x))) );
    REQUIRE( val(gradx(gradx(hypot(x, y, z), z), y)) == Catch::Approx(val(grad(z / hypot(x, y, z), y))) );
    REQUIRE( val(gradx(gradx(hypot(x, y, z), z), z)) == Catch::Approx(val(grad(z / hypot(x, y, z), z))) );

    //--------------------------------------------------------------------------
    // TEST HIGHER ORDER DERIVATIVES (3rd order)
    //--------------------------------------------------------------------------
    REQUIRE( val(gradx(gradx(gradx(log(x), x), x), x)) == approx(val(2.0/(x * x * x))) );
    REQUIRE( val(gradx(gradx(gradx(exp(x), x), x), x)) == approx(val(exp(x))) );
}
