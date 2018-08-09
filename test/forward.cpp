// C++ includes
#include <iostream>

// Catch includes
#include "catch.hpp"

// autodiff includes
#include <autodiff/forward.hpp>
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
}
