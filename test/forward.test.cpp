// Catch includes
#include "catch.hpp"

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

// autodiff includes
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
using namespace autodiff;
using namespace autodiff::forward;

#include <iostream>

template<typename T>
auto approx(T&& expr) -> Approx
{
    const double zero = std::numeric_limits<double>::epsilon();
    return Approx(val(std::forward<T>(expr))).margin(zero);
}

TEST_CASE("autodiff::dual tests", "[dual]")
{
    dual x = 100;
    dual y = 10;

    SECTION("trivial tests")
    {
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

    SECTION("aliasing tests")
    {
        x = 1; x = x + 3*x - 2*x + x;
        REQUIRE( x == 3 );

        x = 1; x += x + 3*x - 2*x + x;
        REQUIRE( x == 4 );

        x = 1; x -= x + 3*x - 2*x + x;
        REQUIRE( x == -2 );

        x = 1; x *= x + 3*x - 2*x + x;
        REQUIRE( x == 3 );

        x = 1; x /= x + x;
        REQUIRE( x == 0.5 );
    }

    SECTION("testing comparison operators")
    {
        x = 6;
        y = 5;

        REQUIRE( x == 6 );
        REQUIRE( 6 == x );
        REQUIRE( x == x );

        REQUIRE( x != 5 );
        REQUIRE( 5 != x );
        REQUIRE( x != y );

        REQUIRE( x > 5 );
        REQUIRE( x > y );

        REQUIRE( x >= 6 );
        REQUIRE( x >= x );
        REQUIRE( x >= y );

        REQUIRE( 5 < x );
        REQUIRE( y < x );

        REQUIRE( 6 <= x );
        REQUIRE( x <= x );
        REQUIRE( y <= x );
    }

    SECTION("testing unary negative operator")
    {
        std::function<dual(dual)> f;

        // Testing positive operator on a dual
        f = [](dual x) -> dual { return +x; };
        REQUIRE( f(x) == x );
        REQUIRE( derivative(f, wrt(x), at(x)) == 1.0 );

        // Testing positive operator on an expression
        f = [](dual x) -> dual { return +(x * x); };
        REQUIRE( f(x) == x * x );
        REQUIRE( derivative(f, wrt(x), at(x)) == 2.0 * x );

        // Testing negative operator on a dual
        f = [](dual x) -> dual { return -x; };
        REQUIRE( f(x) == -x );
        REQUIRE( derivative(f, wrt(x), at(x)) == -1.0 );

        // Testing negative operator on a negative expression
        f = [](dual x) -> dual { return -(-x); };
        REQUIRE( f(x) == x );
        REQUIRE( derivative(f, wrt(x), at(x)) == 1.0 );

        // Testing negative operator on a scaling expression expression
        f = [](dual x) -> dual { return -(2.0 * x); };
        REQUIRE( f(x) == -2.0 * x );
        REQUIRE( derivative(f, wrt(x), at(x)) == -2.0 );

        // Testing negative operator on a more complex expression
        f = [](dual x) -> dual { return -x - (2*x); };
        REQUIRE( f(x) == -val(x) - 2.0 * val(x) );
        REQUIRE( derivative(f, wrt(x), at(x)) == -3.0 );

        // Testing negative operator on a more complex expression
        f = [](dual x) -> dual { return -x - 2.0 * log(1.0 + exp(-x)); };
        REQUIRE( f(x) == -val(x) - 2.0 * std::log(1.0 + std::exp(-val(x))) );
        REQUIRE( derivative(f, wrt(x), at(x)) == -1.0 - 2.0/(1.0 + std::exp(-val(x))) * (-std::exp(-val(x))) );

        // Testing negative operator on a more complex expression
        f = [](dual x) -> dual { return -x - log(1.0 + exp(-x)); };
        REQUIRE( f(x) == -val(x) - std::log(1.0 + std::exp(-val(x))) );
        REQUIRE( derivative(f, wrt(x), at(x)) == -1.0 - 1.0/(1.0 + std::exp(-val(x))) * (-std::exp(-val(x))) );
    }

    SECTION("testing unary inverse operator")
    {
        std::function<dual(dual)> f;

        // Testing inverse operator on a dual
        f = [](dual x) -> dual { return 1.0 / x; };
        REQUIRE( f(x) == 1/x );
        REQUIRE( derivative(f, wrt(x), at(x)) == -1.0 / (x * x) );

        // Testing inverse operator on a trivial expression
        f = [](dual x) -> dual { return x / x; };
        REQUIRE( f(x) == 1.0 );
        REQUIRE( derivative(f, wrt(x), at(x)) == 0.0 );

        // Testing inverse operator on an inverse expression
        f = [](dual x) -> dual { return 1.0 / (1.0 / x); };
        REQUIRE( f(x) == x );
        REQUIRE( derivative(f, wrt(x), at(x)) == 1.0 );

        // Testing inverse operator on a scaling expression
        f = [](dual x) -> dual { return 1.0 / (2.0 * x); };
        REQUIRE( f(x) == 0.5 / x );
        REQUIRE( derivative(f, wrt(x), at(x)) == -0.5 / (x * x) );
    }

    SECTION("testing binary addition operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing addition operator on a `number + dual` expression
        f = [](dual x, dual y) -> dual { return 1 + x; };
        REQUIRE( f(x, y) == 1 + val(x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 0.0 );

        // Testing addition operator on a `dual + number` expression
        f = [](dual x, dual y) -> dual { return x + 1; };
        REQUIRE( f(x, y) == val(x) + 1 );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 0.0 );

        // Testing addition operator on a `dual + dual` expression
        f = [](dual x, dual y) -> dual { return (-x) + (-y); };
        REQUIRE( f(x, y) == -(val(x) + val(y)) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == -1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == -1.0 );

        // Testing addition operator on a `(-dual) + (-dual)` expression
        f = [](dual x, dual y) -> dual { return (-x) + (-y); };
        REQUIRE( f(x, y) == -(val(x) + val(y)) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == -1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == -1.0 );

        // Testing addition operator on a `dual * dual + dual * dual` expression
        f = [](dual x, dual y) -> dual { return x * y + x * y; };
        REQUIRE( f(x, y) == 2.0 * (val(x) * val(y)) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx( 2.0 * y ) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx( 2.0 * x ) );

        // Testing addition operator on a `1/dual * 1/dual` expression
        f = [](dual x, dual y) -> dual { return 1.0 / x + 1.0 / y; };
        REQUIRE( f(x, y) == approx( 1.0 / val(x) + 1.0 / val(y)) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx( -1.0 / (x * x) ) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx( -1.0 / (y * y) ) );
    }

    SECTION("testing binary subtraction operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing subtraction operator on a `number - dual` expression
        f = [](dual x, dual y) -> dual { return 1 - x; };
        REQUIRE( f(x, y) == 1 - val(x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == -1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) ==  0.0 );

        // Testing subtraction operator on a `dual - number` expression
        f = [](dual x, dual y) -> dual { return x - 1; };
        REQUIRE( f(x, y) == val(x) - 1 );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 0.0 );

        // Testing subtraction operator on a `(-dual) - (-dual)` expression
        f = [](dual x, dual y) -> dual { return (-x) - (-y); };
        REQUIRE( f(x, y) == -(val(x) - val(y)) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == -1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) ==  1.0 );

        // Testing subtraction operator on a `dual * dual - dual * dual` expression
        f = [](dual x, dual y) -> dual { return x * y - x * y; };
        REQUIRE( f(x, y) == 0.0 );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx( 0.0 ) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx( 0.0 ) );

        // Testing subtraction operator on a `1/dual * 1/dual` expression
        f = [](dual x, dual y) -> dual { return 1.0 / x - 1.0 / y; };
        REQUIRE( f(x, y) == approx( 1.0 / val(x) - 1.0 / val(y)) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx( -1.0 / (x * x) ) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(  1.0 / (y * y) ) );
    }

    SECTION("testing binary multiplication operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing multiplication operator on a `number * dual` expression
        f = [](dual x, dual y) -> dual { return 2.0 * x; };
        REQUIRE( f(x, y) == 2.0 * val(x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 2.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 0.0 );

        // Testing multiplication operator on a `dual * number` expression
        f = [](dual x, dual y) -> dual { return x * 2; };
        REQUIRE( f(x, y) == 2.0 * val(x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 2.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 0.0 );

        // Testing multiplication operator on a `dual * dual` expression
        f = [](dual x, dual y) -> dual { return x * y; };
        REQUIRE( f(x, y) == val(x) * val(y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == y );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == x );

        // Testing multiplication operator on a `number * (number * dual)` expression
        f = [](dual x, dual y) -> dual { return 5 * (2.0 * x); };
        REQUIRE( f(x, y) == 10 * val(x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 10.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) ==  0.0 );

        // Testing multiplication operator on a `(dual * number) * number` expression
        f = [](dual x, dual y) -> dual { return (x * 2) * 5; };
        REQUIRE( f(x, y) == 10 * val(x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 10.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) ==  0.0 );

        // Testing multiplication operator on a `number * (-dual)` expression
        f = [](dual x, dual y) -> dual { return 2.0 * (-x); };
        REQUIRE( f(x, y) == -2.0 * val(x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == -2.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) ==  0.0 );

        // Testing multiplication operator on a `(-dual) * number` expression
        f = [](dual x, dual y) -> dual { return (-x) * 2; };
        REQUIRE( f(x, y) == -2.0 * val(x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == -2.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) ==  0.0 );

        // Testing multiplication operator on a `(-dual) * (-dual)` expression
        f = [](dual x, dual y) -> dual { return (-x) * (-y); };
        REQUIRE( f(x, y) == val(x) * val(y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == y );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == x );

        // Testing multiplication operator on a `(1/dual) * (1/dual)` expression
        f = [](dual x, dual y) -> dual { return (1 / x) * (1 / y); };
        REQUIRE( f(x, y) == 1 / (val(x) * val(y)) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(-1/(x * x * y)) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(-1/(x * y * y)) );
    }

    SECTION("testing binary division operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing division operator on a `number / dual` expression
        f = [](dual x, dual y) -> dual { return 1 / x; };
        REQUIRE( f(x, y) == 1 / val(x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(-1.0 / (x * x)) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        // Testing division operator on a `dual / number` expression
        f = [](dual x, dual y) -> dual { return x / 2; };
        REQUIRE( f(x, y) == val(x) / 2 );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(0.5) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        // Testing division operator on a `dual / dual` expression
        f = [](dual x, dual y) -> dual { return x / y; };
        REQUIRE( f(x, y) == val(x) / val(y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(1 / y) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(-x / (y * y)) );

        // Testing division operator on a `1 / (number * dual)` expression
        f = [](dual x, dual y) -> dual { return 1 / (2.0 * x); };
        REQUIRE( f(x, y) == approx(0.5 / x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(-0.5 / (x * x)) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        // Testing division operator on a `1 / (1 / dual)` expression
        f = [](dual x, dual y) -> dual { return 1 / (1 / x); };
        REQUIRE( f(x, y) == val(x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 0.0 );
    }

    SECTION("testing operator+=")
    {
        std::function<dual(dual, dual)> f;

        f = [](dual x, dual y) -> dual { return x += 1; };
        REQUIRE( f(x, y) == approx(x + 1) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 0.0 );

        f = [](dual x, dual y) -> dual { return x += y; };
        REQUIRE( f(x, y) == approx(x + y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 1.0 );

        f = [](dual x, dual y) -> dual { return x += -y; };
        REQUIRE( f(x, y) == approx(x - y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) ==  1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == -1.0 );

        f = [](dual x, dual y) -> dual { return x += -(x - y); };
        REQUIRE( f(x, y) == approx(y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 0.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 1.0 );

        f = [](dual x, dual y) -> dual { return x += 1.0 / y; };
        REQUIRE( f(x, y) == approx(x + 1.0 / y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) ==  1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(-1.0 / (y * y)) );

        f = [](dual x, dual y) -> dual { return x += 2.0 * y; };
        REQUIRE( f(x, y) == approx(x + 2.0 * y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 2.0 );

        f = [](dual x, dual y) -> dual { return x += x * y; };
        REQUIRE( f(x, y) == approx(x + x * y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 1.0 + y );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == x );

        f = [](dual x, dual y) -> dual { return x += x + y; };
        REQUIRE( f(x, y) == approx(x + x + y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 2.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 1.0 );
    }

    SECTION("testing operator-=")
    {
        std::function<dual(dual, dual)> f;

        f = [](dual x, dual y) -> dual { return x -= 1; };
        REQUIRE( f(x, y) == approx(x - 1) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 0.0 );

        f = [](dual x, dual y) -> dual { return x -= y; };
        REQUIRE( f(x, y) == approx(x - y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) ==  1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == -1.0 );

        f = [](dual x, dual y) -> dual { return x -= -y; };
        REQUIRE( f(x, y) == approx(x + y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 1.0 );

        f = [](dual x, dual y) -> dual { return x -= -(x - y); };
        REQUIRE( f(x, y) == approx(2.0 * x - y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) ==  2.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == -1.0 );

        f = [](dual x, dual y) -> dual { return x -= 1.0 / y; };
        REQUIRE( f(x, y) == approx(x - 1.0 / y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) ==  1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(1.0 / (y * y)) );

        f = [](dual x, dual y) -> dual { return x -= 2.0 * y; };
        REQUIRE( f(x, y) == approx(x - 2.0 * y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) ==  1.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == -2.0 );

        f = [](dual x, dual y) -> dual { return x -= x * y; };
        REQUIRE( f(x, y) == approx(x - x * y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 1.0 - y );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == -x );

        f = [](dual x, dual y) -> dual { return x -= x - y; };
        REQUIRE( f(x, y) == approx(y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == 0.0 );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == 1.0 );
    }

    SECTION("testing operator*=")
    {
        std::function<dual(dual, dual)> f;

        f = [](dual x, dual y) -> dual { return x *= 2; };
        REQUIRE( f(x, y) == approx(2.0 * x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(2.0) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        f = [](dual x, dual y) -> dual { return x *= y; };
        REQUIRE( f(x, y) == approx(x * y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(y) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(x) );

        f = [](dual x, dual y) -> dual { return x *= -x; };
        REQUIRE( f(x, y) == approx(-x * x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(-2.0 * x) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        f = [](dual x, dual y) -> dual { return x *= (2.0 / y); };
        REQUIRE( f(x, y) == approx(2.0 * x / y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(2.0 / y) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(-2.0 * x / (y * y)) );

        f = [](dual x, dual y) -> dual { return x *= (2.0 * x); };
        REQUIRE( f(x, y) == approx(2.0 * x * x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(4.0 * x) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        f = [](dual x, dual y) -> dual { return x *= (2.0 * y); };
        REQUIRE( f(x, y) == approx(2.0 * x * y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(2.0 * y) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(2.0 * x) );

        f = [](dual x, dual y) -> dual { return x *= x + y; };
        REQUIRE( f(x, y) == approx(x * (x + y)) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(2.0 * x + y) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(x) );

        f = [](dual x, dual y) -> dual { return x *= x * y; };
        REQUIRE( f(x, y) == approx(x * (x * y)) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(2.0 * x * y) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(x * x) );
    }

    SECTION("testing operator/=")
    {
        std::function<dual(dual, dual)> f;

        f = [](dual x, dual y) -> dual { return x /= 2; };
        REQUIRE( f(x, y) == approx(0.5 * x) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(0.5) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        f = [](dual x, dual y) -> dual { return x /= y; };
        REQUIRE( f(x, y) == approx(x / y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(1.0 / y) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(-x / (y * y)) );

        f = [](dual x, dual y) -> dual { return x /= -x; };
        REQUIRE( f(x, y) == approx(-1.0) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(0.0) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        f = [](dual x, dual y) -> dual { return x /= (2.0 / y); };
        REQUIRE( f(x, y) == approx(0.5 * x * y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(0.5 * y) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(0.5 * x) );

        f = [](dual x, dual y) -> dual { return x /= (2.0 * y); };
        REQUIRE( f(x, y) == approx(0.5 * x / y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx( 0.5 / y) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(-0.5 * x / (y * y)) );

        f = [](dual x, dual y) -> dual { return x /= x + y; };
        REQUIRE( f(x, y) == approx(x / (x + y)) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(1.0 / (x + y) - x / (x + y) / (x + y)) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(-x / (x + y) / (x + y)) );

        f = [](dual x, dual y) -> dual { return x /= x * y; };
        REQUIRE( f(x, y) == approx(1.0 / y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(0.0) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(-1.0 / (y * y)) );
    }

    SECTION("testing combination of operations")
    {
        std::function<dual(dual, dual)> f;

        // Testing multiplication with addition
        f = [](dual x, dual y) -> dual { return 2.0 * x + y; };
        REQUIRE( f(x, y) == 2.0 * val(x) + val(y) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(2.0) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(1.0) );

        // Testing a complex expression that is actually equivalent to one
        f = [](dual x, dual y) -> dual { return (2.0 * x * x - x * y + x / y + x / (2.0 * y)) / (x * (2.0 * x - y + 1 / y + 1 / (2.0 * y))); };
        REQUIRE( f(x, y) == approx(1.0) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(0.0) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(0.0) );
    }

    SECTION("testing mathematical functions")
    {
        std::function<dual(dual)> f;

        x = 0.5;

        // Testing sin function
        f = [](dual x) -> dual { return sin(x); };
        REQUIRE( f(x) == std::sin(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(cos(x)) );

        // Testing cos function
        f = [](dual x) -> dual { return cos(x); };
        REQUIRE( f(x) == std::cos(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(-sin(x)) );

        // Testing tan function
        f = [](dual x) -> dual { return tan(x); };
        REQUIRE( f(x) == std::tan(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(1 / (cos(x) * cos(x))) );

        // Testing sinh function
        f = [](dual x) -> dual { return sinh(x); };
        REQUIRE( f(x) == std::sinh(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(cosh(x)) );

        // Testing cosh function
        f = [](dual x) -> dual { return cosh(x); };
        REQUIRE( f(x) == std::cosh(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(sinh(x)) );

        // Testing tanh function
        f = [](dual x) -> dual { return tanh(x); };
        REQUIRE( f(x) == std::tanh(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(1 / (cosh(x) * cosh(x))) );

        // Testing asin function
        f = [](dual x) -> dual { return asin(x); };
        REQUIRE( f(x) == std::asin(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(1 / sqrt(1 - x * x)) );

        // Testing acos function
        f = [](dual x) -> dual { return acos(x); };
        REQUIRE( f(x) == std::acos(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(-1 / sqrt(1 - x * x)) );

        // Testing atan function
        f = [](dual x) -> dual { return atan(x); };
        REQUIRE( f(x) == std::atan(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(1 / (1 + x * x)) );

        // Testing exp function
        f = [](dual x) -> dual { return exp(x); };
        REQUIRE( f(x) == std::exp(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(exp(x)) );

        // Testing log function
        f = [](dual x) -> dual { return log(x); };
        REQUIRE( f(x) == std::log(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(1 / x) );

        // Testing log function
        f = [](dual x) -> dual { return log10(x); };
        REQUIRE( f(x) == std::log10(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(1 / (log(10) * x)) );

        // Testing sqrt function
        f = [](dual x) -> dual { return sqrt(x); };
        REQUIRE( f(x) == std::sqrt(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(0.5 / sqrt(x)) );

        // Testing pow function (with number exponent)
        f = [](dual x) -> dual { return pow(x, 2.0); };
        REQUIRE( f(x) == std::pow(val(x), 2.0) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(2.0 * x) );

        // Testing pow function (with dual exponent)
        f = [](dual x) -> dual { return pow(x, x); };
        REQUIRE( f(x) == std::pow(val(x), val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx((log(x) + 1) * pow(x, x)) );

        // Testing pow function (with expression exponent)
        f = [](dual x) -> dual { return pow(x, 2.0 * x); };
        REQUIRE( f(x) == std::pow(val(x), 2.0 * val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(2.0 * (log(x) + 1) * pow(x, 2.0 * x)) );

        // Testing abs function (for x > 0, x < 0, and x = 0)
        f = [](dual x) -> dual { return abs(x); };
        x = 1.0;
        REQUIRE( f(x) == std::abs(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(1.0) );
        x = -1.0;
        REQUIRE( f(x) == std::abs(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(-1.0) );
        x = 0;
        REQUIRE( f(x) == std::abs(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(0.0) );


        // Testing erf function
        f = [](dual x) -> dual { return erf(x); };
        x = 1.0;
        REQUIRE( f(x) == std::erf(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(0.415107) );
        x = -1.4;
        REQUIRE( f(x) == std::erf(val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(0.158942) );

        // Testing atan2 function on (double, dual)
        f = [](dual x) -> dual { return atan2(2.0, x); };
        x = 1.0;
        REQUIRE( f(x) == std::atan2(2.0, val(x)) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(-2.0 / (2*2 + x*x)) );

        // Testing atan2 function on (dual, double)
        f = [](dual y) -> dual { return atan2(y, 2.0); };
        x = 1.0;
        REQUIRE( f(x) == std::atan2(val(x), 2.0) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(2.0 /  (2*2 + x*x)) );

        // Testing atan2 function on (dual, dual)
        std::function<dual(dual, dual)> g = [](dual y, dual x) -> dual { return atan2(y, x); };
        x = 1.1;
        dual y = 0.9;
        REQUIRE( g(y, x) == std::atan2(val(y), val(x)) );
        REQUIRE( derivative(g, wrt(y), at(y, x)) == approx(x / (x*x + y*y)) );
        REQUIRE( derivative(g, wrt(x), at(y, x)) == approx(-y / (x*x + y*y)) );

        // Testing atan2 function on (expr, expr)
        g = [](dual y, dual x) -> dual { return 3 * atan2(sin(y), 2*x+1); };
        REQUIRE( g(y, x) == 3 * std::atan2(sin(val(y)), 2*val(x)+1) );
        REQUIRE( derivative(g, wrt(y), at(y, x)) == approx(3*(2*x+1)*cos(y) / ((2*x+1)*(2*x+1) + sin(y)*sin(y))) );
        REQUIRE( derivative(g, wrt(x), at(y, x)) == approx(3*-2*sin(y) / ((2*x+1)*(2*x+1) + sin(y)*sin(y))) );
        
        // Testing hypot function on (dual, double)
        f = [](dual y) -> dual { return hypot(y, 2.0); };
        x = 1.5;
        REQUIRE( f(x) == std::hypot(val(x), 2.0) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(x / std::hypot(val(x), 2.0)) );
        
        // Testing hypot function on (double, dual)
        f = [](dual y) -> dual { return hypot(2.0, y); };
        y = 1.5;
        REQUIRE( f(y) == std::hypot(2.0, val(y)) );
        REQUIRE( derivative(f, wrt(y), at(y)) == approx(y / std::hypot(2.0, val(y))) );
        
        // Testing hypot function on (dual, dual)
        g = [](dual x, dual y) -> dual { return hypot(x,y); };
        x = 1.1;
        y = 0.9;
        REQUIRE( g(x, y) == std::hypot(val(x), val(y)) );
        REQUIRE( derivative(g, wrt(y), at(x,y)) == approx(y / std::hypot(val(x), val(y))) );
        REQUIRE( derivative(g, wrt(x), at(x,y)) == approx(x / std::hypot(val(x), val(y))) );
        
        // Testing hypot function on (expr, expr)
        g = [](dual x, dual y) -> dual { return hypot(2.0*x,3.0*y); };
        x = 1.1;
        y = 0.9;
        REQUIRE( g(x, y) == std::hypot(2.0*val(x), 3.0*val(y)) );
        REQUIRE( derivative(g, wrt(x), at(x,y)) == approx(4.0*x / std::hypot(2.0*val(x), 3.0*val(y))) );
        REQUIRE( derivative(g, wrt(y), at(x,y)) == approx(9.0*y / std::hypot(2.0*val(x), 3.0*val(y))) );
        
        // Testing hypot function on (dual, double, double)
        f = [](dual x) -> dual { return hypot(x, 2.0, 2.0); };
        x = 1.5;
        REQUIRE( f(x) == std::hypot(val(x), 2.0, 2.0) );
        REQUIRE( derivative(f, wrt(x), at(x)) == approx(x / std::hypot(val(x), 2.0, 2.0)) );
        
        // Testing hypot function on (double, dual, double)
        f = [](dual y) -> dual { return hypot(2.0, y, 2.0); };
        y = 2.5;
        REQUIRE( f(y) == std::hypot(2.0, val(y), 2.0) );
        REQUIRE( derivative(f, wrt(y), at(y)) == approx(y / std::hypot(2.0, val(y), 2.0)) );
        
        // Testing hypot function on (double, double, dual)
        f = [](dual z) -> dual { return hypot(2.0, 2.0, z); };
        dual z = 3.5;
        REQUIRE( f(z) == std::hypot(2.0, 2.0, val(z)) );
        REQUIRE( derivative(f, wrt(z), at(z)) == approx(z / std::hypot(2.0, 2.0, val(z))) );
        
        // Testing hypot function on (dual, dual, double)
        g = [](dual x, dual y) -> dual { return hypot(x, y, 2.0); };
        x = 1.4;
        y = 2.4;
        REQUIRE( g(x,y) == std::hypot(val(x), val(y), 2.0) );
        REQUIRE( derivative(g, wrt(x), at(x,y)) == approx(x / std::hypot(val(x), val(y), 2.0)) );
        REQUIRE( derivative(g, wrt(y), at(x,y)) == approx(y / std::hypot(val(x), val(y), 2.0)) );
        
        // Testing hypot function on (double, dual, dual)
        g = [](dual y, dual z) -> dual { return hypot(2.0, y, z); };
        y = 2.4;
        z = 3.4;
        REQUIRE( g(y,z) == std::hypot(2.0, val(y), val(z)) );
        REQUIRE( derivative(g, wrt(y), at(y,z)) == approx(y / std::hypot(2.0, val(y), val(z))) );
        REQUIRE( derivative(g, wrt(z), at(y,z)) == approx(z / std::hypot(2.0, val(y), val(z))) );
        
        // Testing hypot function on (dual, double, dual)
        g = [](dual x, dual z) -> dual { return hypot(x, 2.0, z); };
        x = 3.4;
        z = 4.4;
        REQUIRE( g(x,z) == std::hypot(val(x), 2.0, val(z)) );
        REQUIRE( derivative(g, wrt(x), at(x,z)) == approx(x / std::hypot(val(x), 2.0, val(z))) );
        REQUIRE( derivative(g, wrt(z), at(x,z)) == approx(z / std::hypot(val(x), 2.0, val(z))) );
      
        // Testing hypot function on (dual, double, dual)
        std::function<dual(dual, dual, dual)> h = [](dual x, dual y, dual z) -> dual { return hypot(x, y, z); };
        x = 1.6;
        y = 2.6;
        z = 3.6;
        REQUIRE( h(x,y,z) == std::hypot(val(x), val(y), val(z)) );
        REQUIRE( derivative(h, wrt(x), at(x,y,z)) == approx(x / std::hypot(val(x), val(y), val(z))) );
        REQUIRE( derivative(h, wrt(y), at(x,y,z)) == approx(y / std::hypot(val(x), val(y), val(z))) );
        REQUIRE( derivative(h, wrt(z), at(x,y,z)) == approx(z / std::hypot(val(x), val(y), val(z))) );
      
        // Testing hypot function on (expr, expr, expr)
        h = [](dual x, dual y, dual z) -> dual { return hypot(2.0*x, 3.0*y, 4.0*z); };
        x = 2.6;
        y = 3.6;
        z = 4.6;
        REQUIRE( h(x,y,z) == std::hypot(2.0*val(x), 3.0*val(y), 4.0*val(z)) );
        REQUIRE( derivative(h, wrt(x), at(x,y,z)) == approx(4.0*x / std::hypot(2.0*val(x), 3.0*val(y), 4.0*val(z))) );
        REQUIRE( derivative(h, wrt(y), at(x,y,z)) == approx(9.0*y / std::hypot(2.0*val(x), 3.0*val(y), 4.0*val(z))) );
        REQUIRE( derivative(h, wrt(z), at(x,y,z)) == approx(16.*z / std::hypot(2.0*val(x), 3.0*val(y), 4.0*val(z))) );
    }

    SECTION("testing complex expressions")
    {
        std::function<dual(dual, dual)> f;

        x = 0.5;
        y = 0.8;

        // Testing complex function involving sin, cos, and tan
        f = [](dual x, dual y) -> dual { return sin(x + y) * cos(x / y) + tan(2.0 * x * y) - sin(4*(x + y)*2/8) * cos(x*x / (y*y) * y/x) - tan((x + y) * (x + y) - x*x - y*y); };
        REQUIRE( val(f(x, y)) == approx(0.0) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(0.0) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        // Testing complex function involving log, exp, pow, and sqrt
        f = [](dual x, dual y) -> dual { return log(x + y) * exp(x / y) + sqrt(2.0 * x * y) - 1 / pow(x, x + y) - exp(x*x / (y*y) * y/x) * log(4*(x + y)*2/8) - 4 * sqrt((x + y) * (x + y) - x*x - y*y) * 0.5 * 0.5 + 2 / pow(2.0 * x - x, y + x) * 0.5; };
        REQUIRE( val(f(x, y)) == approx(0.0) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(0.0) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(0.0) );
    }


    SECTION("testing higher order derivatives")
    {
        using dual2nd = HigherOrderDual<2>;

        std::function<dual2nd(dual2nd, dual2nd)> f;

        dual2nd x = 5;
        dual2nd y = 7;

        // Testing simpler functions
        f = [](dual2nd x, dual2nd y) -> dual2nd
        {
            return x*x + x*y + y*y;
        };

        REQUIRE( val(derivative(f, wrt(x), at(x, y))) == Approx(17.0) );
        REQUIRE( derivative(f, wrt<2>(x), at(x, y)) == Approx(2.0) );
        REQUIRE( derivative(f, wrt(x, y), at(x, y)) == Approx(1.0) );
        REQUIRE( derivative(f, wrt(y, x), at(x, y)) == Approx(1.0) );
        REQUIRE( derivative(f, wrt<2>(y), at(x, y)) == Approx(2.0) );

        // Testing complex function involving log
        x = 2.0;
        y = 1.0;

        f = [](dual2nd x, dual2nd y) -> dual2nd
        {
            return 1 + x + y + x * y + y / x + log(x / y);
        };

        REQUIRE( val(f(x, y)) == Approx( 1 + val(x) + val(y) + val(x) * val(y) + val(y) / val(x) + log(val(x) / val(y)) ) );
        REQUIRE( val(derivative(f, wrt(x), at(x, y))) == Approx( 1 + val(y) - val(y) / (val(x) * val(x)) + 1.0/val(x) - log(val(y)) ) );
        REQUIRE( val(derivative(f, wrt(y), at(x, y))) == Approx( 1 + val(x) + 1.0 / val(x) - 1.0/val(y) ) );
        REQUIRE( val(derivative(f, wrt<2>(x), at(x, y))) == Approx( 2 * val(y) / (val(x) * val(x) * val(x)) + -1.0/(val(x) * val(x)) ) );
        REQUIRE( val(derivative(f, wrt(y, y), at(x, y))) == Approx( 1.0/(val(y)*val(y)) ) );
        REQUIRE( val(derivative(f, wrt(y, x), at(x, y))) == Approx( 1 - 1.0 / (val(x) * val(x)) ) );
        REQUIRE( val(derivative(f, wrt(x, y), at(x, y))) == Approx( 1 - 1.0 / (val(x) * val(x)) ) );
    }

    SECTION("testing higher order derivatives")
    {
        using dual3rd = HigherOrderDual<3>;

        dual3rd x = 5;
        dual3rd y = 7;

        // Testing complex function involving sin and cos
        auto f = [](dual3rd x, dual3rd y) -> dual3rd
        {
            return (x + y)*(x + y)*(x + y);
        };

        REQUIRE( derivative(f, wrt<3>(x), at(x, y)) == Approx(6.0) );
        REQUIRE( derivative(f, wrt(x, x, x), at(x, y)) == Approx(6.0) );
        REQUIRE( derivative(f, wrt(x, x, y), at(x, y)) == Approx(6.0) );
        REQUIRE( derivative(f, wrt(x, y, y), at(x, y)) == Approx(6.0) );
        REQUIRE( derivative(f, wrt<3>(y), at(x, y)) == Approx(6.0) );
    }

    SECTION("testing reference to unary and binary expression nodes are not present")
    {
        x = -0.3;
        y =  0.5;

        auto pow2 = [](const auto& x)
        {
            return x*x;
        };

        auto rosenbrock = [&](const auto& x, const auto& y)
        {
            return 100.0 * pow2(x*x - y) + pow2(1.0 - x);
        };

        auto f = [&](const auto& x, const auto& y) -> dual
        {
            return rosenbrock(x, y);
        };

        REQUIRE( val(f(x, y)) == approx(100 * (x*x - y)*(x*x - y) + (1 - x)*(1 - x)) );
        REQUIRE( derivative(f, wrt(x), at(x, y)) == approx(400*(x*x - y)*x - 2*(1 - x)) );
        REQUIRE( derivative(f, wrt(y), at(x, y)) == approx(-200*(x*x - y)) );
    }
}

TEST_CASE("Eigen::VectorXdual tests", "[dual]")
{
    SECTION("testing gradient derivatives")
    {
        // Testing complex function involving sin and cos
        auto f = [](const VectorXdual& x) -> dual
        {
            return 0.5 * ( x.array() * x.array() ).sum();
        };

        VectorXdual x(3);
        x << 1.0, 2.0, 3.0;

        VectorXd g = gradient(f, wrt(x), at(x));

        REQUIRE( g[0] == approx(x[0]) );
        REQUIRE( g[1] == approx(x[1]) );
        REQUIRE( g[2] == approx(x[2]) );
    }

    SECTION("testing gradient derivatives of only the last two variables")
    {
        // Testing complex function involving sin and cos
        auto f = [](const VectorXdual& x) -> dual
        {
            return 0.5 * ( x.array() * x.array() ).sum();
        };

        VectorXdual x(3);
        x << 1.0, 2.0, 3.0;

        VectorXd g = gradient(f, wrt(x.tail(2)), at(x));

        REQUIRE( g[0] == approx(x[1]) );
        REQUIRE( g[1] == approx(x[2]) );
    }

    SECTION("testing jacobian derivatives")
    {
        // Testing complex function involving sin and cos
        auto f = [](const VectorXdual& x) -> VectorXdual
        {
            return x / x.array().sum();
        };

        VectorXdual x(3);
        x << 0.5, 0.2, 0.3;

        VectorXdual F;

        const MatrixXd J = jacobian(f, wrt(x), at(x), F);

        for(auto i = 0; i < 3; ++i)
            for(auto j = 0; j < 3; ++j)
                REQUIRE( J(i, j) == approx(-F[i] + ((i == j) ? 1.0 : 0.0)) );
    }

    SECTION("testing jacobian derivatives of only the last two variables")
    {
        // Testing complex function involving sin and cos
        auto f = [](const VectorXdual& x) -> VectorXdual
        {
            return x / x.array().sum();
        };

        VectorXdual x(3);
        x << 0.5, 0.2, 0.3;

        VectorXdual F;

        const MatrixXd J = jacobian(f, wrt(x.tail(2)), at(x), F);

        for(auto i = 0; i < 3; ++i)
            for(auto j = 0; j < 2; ++j)
                REQUIRE( J(i, j) == approx(-F[i] + ((i == j + 1) ? 1.0 : 0.0)) );
    }

    SECTION("testing casting to VectorXd")
    {
        VectorXdual x(3);
        x << 0.5, 0.2, 0.3;

        VectorXd y = x.cast<double>();

        for(auto i = 0; i < 3; ++i)
            REQUIRE( x(i) == approx(y(i)) );
    }

    SECTION("testing casting to VectorXf")
    {
    	MatrixXdual x(2, 2);
        x << 0.5, 0.2, 0.3, 0.7;
        MatrixXd y = x.cast<double>();
        for(auto i = 0; i < 2; ++i)
            for(auto j = 0; j < 2; ++j)
                REQUIRE( x(i, j) == approx(y(i, j)) );
    }

    SECTION("test gradient size with respect to few arguments")
    {
        // Testing complex function involving sin and cos
        auto f = [](const VectorXdual& x, dual y, const VectorXdual& z) -> dual
        {
            return 0.5 * (( x.array() * x.array() ).sum() + y * y + (z.array() * z.array()).sum());
        };

        VectorXdual x(3);
        x << 1.0, 2.0, 3.0;

        dual y = 2;

        VectorXdual z(4);
        z << 1.0, 2.0, 3.0, 4.0;

        VectorXd g = gradient(f, wrtpack(x.tail(2), y, z), at(x, y, z));

        REQUIRE( g.size() == 7 );
    }

    SECTION("testing gradient derivatives wrt pack variables")
    {
        auto f = [](const VectorXdual& x, dual y, const VectorXdual& z) -> dual
        {
            return 0.5 * (( x.array() * x.array() ).sum() + y * y + (z.array() * z.array()).sum());
        };

        VectorXdual x(2);
        x << 1.0, 2.0;

        dual y = 3.0;

        VectorXdual z(1);
        z << 4.0;

        VectorXd g = gradient(f, wrtpack(x, y, z), at(x, y, z));

        REQUIRE(g[0] == approx(x[0]));
        REQUIRE(g[1] == approx(x[1]));
        REQUIRE(g[2] == approx(y));
        REQUIRE(g[3] == approx(z[0]));
    }

    SECTION("test jacobian size with respect to few arguments")
    {
        auto f = [](const VectorXdual& x, dual y, const VectorXdual& z) -> VectorXdual
        {
            VectorXdual ret(x.size() + z.size());
            ret.head(x.size()) = x * y / x.array().sum();
            ret.tail(z.size()) = y * z;

            return ret;
        };

        VectorXdual x(3);
        x << 0.5, 0.2, 0.3;

        dual y = 2.0;

        VectorXdual z(3);
        z << 1.0, 2.0, 3.0;

        VectorXdual F;

        const MatrixXd J = jacobian(f, wrtpack(x.tail(2), y), at(x, y, z), F);

        REQUIRE(J.rows() == 6);
        REQUIRE(J.cols() == 3);
    }

    SECTION("test jacobian size with respect to few arguments")
    {
        auto f = [](const VectorXdual& x, dual y, const VectorXdual& z) -> VectorXdual
        {
            VectorXdual ret(x.size() + z.size());
            ret.head(x.size()) = x * y / x.array().sum();
            ret.tail(z.size()) = y * z;

            return ret;
        };

        VectorXdual x(3);
        x << 0.5, 0.2, 0.3;

        dual y = 2.0;

        VectorXdual z(3);
        z << 1.0, 2.0, 3.0;

        VectorXdual F;

        const MatrixXd J = jacobian(f, wrtpack(x, y, z), at(x, y, z), F);

        for (auto i = 0; i < 3; ++i)
            for (auto j = 0; j < 3; ++j)
                REQUIRE(J(i, j) == approx(-F[i] + ((i == j) ? y.val : 0.0)));

        for (auto i = 0; i < 3; ++i)
            for (auto j = 0; j < 3; ++j)
                REQUIRE(J(i + 3, j) == approx(0.0));

        for (auto i = 0; i < 6; ++i)
                REQUIRE(J(i, 3) == approx( i < 3 ? x(i) : z(i - 3)));

        for (auto i = 0; i < 3; ++i)
            for (auto j = 0; j < 3; ++j)
                REQUIRE(J(i, j + 4) == approx(0.0));

        for (auto i = 0; i < 3; ++i)
            for (auto j = 0; j < 3; ++j)
                REQUIRE(J(i + 3, j + 4) == approx((i == j) ? y.val : 0.0));
    }
}
