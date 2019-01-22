// C++ includes
#include <iostream>

// Catch includes
#include "catch.hpp"

#include <eigen3/Eigen/Core>
using namespace Eigen;

// autodiff includes
#include <autodiff/forward.hpp>
#include <autodiff/eigen.hpp>
using namespace autodiff;
using namespace autodiff::forward;

auto approx(double value)
{
    const double zero = std::numeric_limits<double>::epsilon();
    return Approx(value).margin(zero);
}

template<typename R, enableif<isExpr<R>>...>
auto approx(R&& expr) -> Approx
{
    return approx(val(std::forward<R>(expr)));
}

bool operator==(const dual& l, double r) { return val(l) == approx(r); }
bool operator==(double l, const dual& r) { return approx(l) == val(r); }

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

    SECTION("testing unary operators")
    {
        std::function<dual(dual)> f;

        // Testing positive operator
        f = [](dual x) -> dual { return +x; };
        REQUIRE( f(x) == x );
        REQUIRE( derivative(f, wrt(x), x) == 1.0 );

        // Testing negative operator
        f = [](dual x) -> dual { return -x; };
        REQUIRE( f(x) == -x );
        REQUIRE( derivative(f, wrt(x), x) == -1.0 );

        // Testing negative operator on a negative expression
        f = [](dual x) -> dual { return -(-x); };
        REQUIRE( f(x) == x );
        REQUIRE( derivative(f, wrt(x), x) == 1.0 );

        // Testing negative operator on a scaling expression expression
        f = [](dual x) -> dual { return -(2 * x); };
        REQUIRE( f(x) == -2 * val(x) );
        REQUIRE( derivative(f, wrt(x), x) == -2.0 );
    }

    SECTION("testing binary addition operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing addition operator on a `scalar + dual` expression
        f = [](dual x, dual y) -> dual { return 1 + x; };
        REQUIRE( f(x, y) == 1 + val(x) );
        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 );
        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );

        // Testing addition operator on a `dual + scalar` expression
        f = [](dual x, dual y) -> dual { return x + 1; };
        REQUIRE( f(x, y) == val(x) + 1 );
        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 );
        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );

        // Testing addition operator on a `(-dual) + (-dual)` expression
        f = [](dual x, dual y) -> dual { return (-x) + (-y); };
        REQUIRE( f(x, y) == -(val(x) + val(y)) );
        REQUIRE( derivative(f, wrt(x), x, y) == -1.0 );
        REQUIRE( derivative(f, wrt(y), x, y) == -1.0 );
    }

    SECTION("testing binary subtraction operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing subtraction operator on a `scalar - dual` expression
        f = [](dual x, dual y) -> dual { return 1 - x; };
        REQUIRE( f(x, y) == 1 - val(x) );
        REQUIRE( derivative(f, wrt(x), x, y) == -1.0 );
        REQUIRE( derivative(f, wrt(y), x, y) ==  0.0 );

        // Testing subtraction operator on a `dual - scalar` expression
        f = [](dual x, dual y) -> dual { return x - 1; };
        REQUIRE( f(x, y) == val(x) - 1 );
        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 );
        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );

        // Testing subtraction operator on a `(-dual) - (-dual)` expression
        f = [](dual x, dual y) -> dual { return (-x) - (-y); };
        REQUIRE( f(x, y) == -(val(x) - val(y)) );
        REQUIRE( derivative(f, wrt(x), x, y) == -1.0 );
        REQUIRE( derivative(f, wrt(y), x, y) ==  1.0 );
    }

    SECTION("testing binary multiplication operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing multiplication operator on a `scalar * dual` expression
        f = [](dual x, dual y) -> dual { return 2 * x; };
        REQUIRE( f(x, y) == 2 * val(x) );
        REQUIRE( derivative(f, wrt(x), x, y) == 2.0 );
        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );

        // Testing multiplication operator on a `dual * scalar` expression
        f = [](dual x, dual y) -> dual { return x * 2; };
        REQUIRE( f(x, y) == 2 * val(x) );
        REQUIRE( derivative(f, wrt(x), x, y) == 2.0 );
        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );

        // Testing multiplication operator on a `dual * dual` expression
        f = [](dual x, dual y) -> dual { return x * y; };
        REQUIRE( f(x, y) == val(x) * val(y) );
        REQUIRE( derivative(f, wrt(x), x, y) == y );
        REQUIRE( derivative(f, wrt(y), x, y) == x );

        // Testing multiplication operator on a `scalar * (scalar * dual)` expression
        f = [](dual x, dual y) -> dual { return 5 * (2 * x); };
        REQUIRE( f(x, y) == 10 * val(x) );
        REQUIRE( derivative(f, wrt(x), x, y) == 10.0 );
        REQUIRE( derivative(f, wrt(y), x, y) ==  0.0 );

        // Testing multiplication operator on a `(dual * scalar) * scalar` expression
        f = [](dual x, dual y) -> dual { return (x * 2) * 5; };
        REQUIRE( f(x, y) == 10 * val(x) );
        REQUIRE( derivative(f, wrt(x), x, y) == 10.0 );
        REQUIRE( derivative(f, wrt(y), x, y) ==  0.0 );

        // Testing multiplication operator on a `scalar * (-dual)` expression
        f = [](dual x, dual y) -> dual { return 2 * (-x); };
        REQUIRE( f(x, y) == -2 * val(x) );
        REQUIRE( derivative(f, wrt(x), x, y) == -2.0 );
        REQUIRE( derivative(f, wrt(y), x, y) ==  0.0 );

        // Testing multiplication operator on a `(-dual) * scalar` expression
        f = [](dual x, dual y) -> dual { return (-x) * 2; };
        REQUIRE( f(x, y) == -2 * val(x) );
        REQUIRE( derivative(f, wrt(x), x, y) == -2.0 );
        REQUIRE( derivative(f, wrt(y), x, y) ==  0.0 );

        // Testing multiplication operator on a `(-dual) * (-dual)` expression
        f = [](dual x, dual y) -> dual { return (-x) * (-y); };
        REQUIRE( f(x, y) == val(x) * val(y) );
        REQUIRE( derivative(f, wrt(x), x, y) == y );
        REQUIRE( derivative(f, wrt(y), x, y) == x );

        // Testing multiplication operator on a `(1/dual) * (1/dual)` expression
        f = [](dual x, dual y) -> dual { return (1 / x) * (1 / y); };
        REQUIRE( f(x, y) == 1 / (val(x) * val(y)) );
        REQUIRE( derivative(f, wrt(x), x, y) == approx(-1/(x * x * y)) );
        REQUIRE( derivative(f, wrt(y), x, y) == approx(-1/(x * y * y)) );
    }

    SECTION("testing binary division operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing division operator on a `scalar / dual` expression
        f = [](dual x, dual y) -> dual { return 1 / x; };
        REQUIRE( f(x, y) == 1 / val(x) );
        REQUIRE( derivative(f, wrt(x), x, y) == approx(-1.0 / (x * x)) );
        REQUIRE( derivative(f, wrt(y), x, y) == approx(0.0) );

        // Testing division operator on a `dual / scalar` expression
        f = [](dual x, dual y) -> dual { return x / 2; };
        REQUIRE( f(x, y) == val(x) / 2 );
        REQUIRE( derivative(f, wrt(x), x, y) == approx(0.5) );
        REQUIRE( derivative(f, wrt(y), x, y) == approx(0.0) );

        // Testing division operator on a `dual / dual` expression
        f = [](dual x, dual y) -> dual { return x / y; };
        REQUIRE( f(x, y) == val(x) / val(y) );
        REQUIRE( derivative(f, wrt(x), x, y) == approx(1 / y) );
        REQUIRE( derivative(f, wrt(y), x, y) == approx(-x / (y * y)) );

        // Testing division operator on a `1 / (1 / dual)` expression
        f = [](dual x, dual y) -> dual { return 1 / (1 / x); };
        REQUIRE( f(x, y) == val(x) );
        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 );
        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
    }

    SECTION("testing combination of operations")
    {
        std::function<dual(dual, dual)> f;

        // Testing multiplication with addition
        f = [](dual x, dual y) -> dual { return 2 * x + y; };
        REQUIRE( f(x, y) == 2 * val(x) + val(y) );
        REQUIRE( derivative(f, wrt(x), x, y) == approx(2.0) );
        REQUIRE( derivative(f, wrt(y), x, y) == approx(1.0) );

        // Testing a complex expression that is actually equivalent to one
        f = [](dual x, dual y) -> dual { return (2 * x * x - x * y + x/y) / (x * (2 * x - y + 1/y)); };
        REQUIRE( f(x, y) == 1.0 );
        REQUIRE( derivative(f, wrt(x), x, y) == approx(0.0) );
        REQUIRE( derivative(f, wrt(y), x, y) == approx(0.0) );
    }

    SECTION("testing mathematical functions")
    {
        std::function<dual(dual)> f;

        x = 0.5;

        // Testing sin function
        f = [](dual x) -> dual { return sin(x); };
        REQUIRE( f(x) == std::sin(val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx(cos(x)) );

        // Testing cos function
        f = [](dual x) -> dual { return cos(x); };
        REQUIRE( f(x) == std::cos(val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx(-sin(x)) );

        // Testing tan function
        f = [](dual x) -> dual { return tan(x); };
        REQUIRE( f(x) == std::tan(val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx(1 / (cos(x) * cos(x))) );

        // Testing asin function
        f = [](dual x) -> dual { return asin(x); };
        REQUIRE( f(x) == std::asin(val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx(1 / sqrt(1 - x * x)) );

        // Testing acos function
        f = [](dual x) -> dual { return acos(x); };
        REQUIRE( f(x) == std::acos(val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx(-1 / sqrt(1 - x * x)) );

        // Testing atan function
        f = [](dual x) -> dual { return atan(x); };
        REQUIRE( f(x) == std::atan(val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx(1 / (1 + x * x)) );

        // Testing exp function
        f = [](dual x) -> dual { return exp(x); };
        REQUIRE( f(x) == std::exp(val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx(exp(x)) );

        // Testing log function
        f = [](dual x) -> dual { return log(x); };
        REQUIRE( f(x) == std::log(val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx(1 / x) );

        // Testing log function
        f = [](dual x) -> dual { return log10(x); };
        REQUIRE( f(x) == std::log10(val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx(1 / (log(10) * x)) );

        // Testing sqrt function
        f = [](dual x) -> dual { return sqrt(x); };
        REQUIRE( f(x) == std::sqrt(val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx(0.5 / sqrt(x)) );

        // Testing pow function (with scalar exponent)
        f = [](dual x) -> dual { return pow(x, 2.0); };
        REQUIRE( f(x) == std::pow(val(x), 2.0) );
        REQUIRE( derivative(f, wrt(x), x) == approx(2 * x) );

        // Testing pow function (with dual exponent)
        f = [](dual x) -> dual { return pow(x, x); };
        REQUIRE( f(x) == std::pow(val(x), val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx((log(x) + 1) * pow(x, x)) );

        // Testing pow function (with expression exponent)
        f = [](dual x) -> dual { return pow(x, 2 * x); };
        REQUIRE( f(x) == std::pow(val(x), 2 * val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx(2 * (log(x) + 1) * pow(x, 2 * x)) );

        // Testing abs function (when x > 0 and when x < 0)
        f = [](dual x) -> dual { return abs(x); };
        x = 1.0;
        REQUIRE( f(x) == std::abs(val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx(1.0) );
        x = -1.0;
        REQUIRE( f(x) == std::abs(val(x)) );
        REQUIRE( derivative(f, wrt(x), x) == approx(-1.0) );
    }

    SECTION("testing complex expressions")
    {
        std::function<dual(dual, dual)> f;

        x = 0.5;
        y = 0.8;

        // Testing complex function involving sin, cos, and tan
        f = [](dual x, dual y) -> dual { return sin(x + y) * cos(x / y) + tan(2 * x * y) - sin(4*(x + y)*2/8) * cos(x*x / (y*y) * y/x) - tan((x + y) * (x + y) - x*x - y*y); };
        REQUIRE( val(f(x, y)) == approx(0.0) );
        REQUIRE( derivative(f, wrt(x), x, y) == approx(0.0) );
        REQUIRE( derivative(f, wrt(y), x, y) == approx(0.0) );

        // Testing complex function involving log, exp, pow, and sqrt
        f = [](dual x, dual y) -> dual { return log(x + y) * exp(x / y) + sqrt(2 * x * y) - 1 / pow(x, x + y) - exp(x*x / (y*y) * y/x) * log(4*(x + y)*2/8) - 4 * sqrt((x + y) * (x + y) - x*x - y*y) * 0.5 * 0.5 + 2 / pow(2 * x - x, y + x) * 0.5; };
        REQUIRE( val(f(x, y)) == approx(0.0) );
        REQUIRE( derivative(f, wrt(x), x, y) == approx(0.0) );
        REQUIRE( derivative(f, wrt(y), x, y) == approx(0.0) );
    }

    SECTION("testing higher order derivatives")
    {
        using dual2nd = Dual<Dual<double>>;

        dual2nd x = 0.5;
        dual2nd y = 0.8;

        // Testing complex function involving sin and cos
        auto f = [](dual2nd x, dual2nd y) -> dual2nd { return sin(x + y) - sin(x + y); };

//        REQUIRE( f(x, y) == approx(0.0) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(0.0) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(0.0) );

        // // Testing complex function involving log, exp, pow, and sqrt
        // f = [](dual x, dual y) -> dual { return log(x + y) * exp(x / y) + sqrt(2 * x * y) - 1 / pow(x, x + y) - exp(x*x / (y*y) * y/x) * log(4*(x + y)*2/8) - 4 * sqrt((x + y) * (x + y) - x*x - y*y) * 0.5 * 0.5 + 2 / pow(2 * x - x, y + x) * 0.5; };
        // REQUIRE( val(f(x, y)) == approx(0.0) );
        // REQUIRE( derivative(f, wrt(x), x, y) == approx(0.0) );
        // REQUIRE( derivative(f, wrt(y), x, y) == approx(0.0) );
    }
}
