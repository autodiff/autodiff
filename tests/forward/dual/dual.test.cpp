// Catch includes
#include <catch2/catch.hpp>

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

// autodiff includes
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

template<typename T>
auto approx(T&& expr) -> Approx
{
    const double zero = std::numeric_limits<double>::epsilon();
    return Approx(val(std::forward<T>(expr))).margin(zero);
}

TEST_CASE("testing autodiff::dual", "[forward][dual]")
{
    dual x = 100;
    dual y = 10;

    SECTION("trivial tests")
    {
        CHECK( x == 100 );
        x += 1;
        CHECK( x == 101 );
        x -= 1;
        CHECK( x == 100 );
        x *= 2;
        CHECK( x == 200 );
        x /= 20;
        CHECK( x == 10 );
    }

    SECTION("aliasing tests")
    {
        x = 1; x = x + 3*x - 2*x + x;
        CHECK( x == 3 );

        x = 1; x += x + 3*x - 2*x + x;
        CHECK( x == 4 );

        x = 1; x -= x + 3*x - 2*x + x;
        CHECK( x == -2 );

        x = 1; x *= x + 3*x - 2*x + x;
        CHECK( x == 3 );

        x = 1; x /= x + x;
        CHECK( x == 0.5 );
    }

    SECTION("testing comparison operators")
    {
        x = 6;
        y = 5;

        CHECK( x == 6 );
        CHECK( 6 == x );
        CHECK( x == x );

        CHECK( x != 5 );
        CHECK( 5 != x );
        CHECK( x != y );

        CHECK( x > 5 );
        CHECK( x > y );

        CHECK( x >= 6 );
        CHECK( x >= x );
        CHECK( x >= y );

        CHECK( 5 < x );
        CHECK( y < x );

        CHECK( 6 <= x );
        CHECK( x <= x );
        CHECK( y <= x );
    }

    SECTION("testing unary negative operator")
    {
        std::function<dual(dual)> f;

        // Testing positive operator on a dual
        f = [](dual x) -> dual { return +x; };
        CHECK( f(x) == x );
        CHECK( derivative(f, wrt(x), at(x)) == 1.0 );

        // Testing positive operator on an expression
        f = [](dual x) -> dual { return +(x * x); };
        CHECK( f(x) == x * x );
        CHECK( derivative(f, wrt(x), at(x)) == 2.0 * x );

        // Testing negative operator on a dual
        f = [](dual x) -> dual { return -x; };
        CHECK( f(x) == -x );
        CHECK( derivative(f, wrt(x), at(x)) == -1.0 );

        // Testing negative operator on a negative expression
        f = [](dual x) -> dual { return -(-x); };
        CHECK( f(x) == x );
        CHECK( derivative(f, wrt(x), at(x)) == 1.0 );

        // Testing negative operator on a scaling expression expression
        f = [](dual x) -> dual { return -(2.0 * x); };
        CHECK( f(x) == -2.0 * x );
        CHECK( derivative(f, wrt(x), at(x)) == -2.0 );

        // Testing negative operator on a more complex expression
        f = [](dual x) -> dual { return -x - (2*x); };
        CHECK( f(x) == -val(x) - 2.0 * val(x) );
        CHECK( derivative(f, wrt(x), at(x)) == -3.0 );

        // Testing negative operator on a more complex expression
        f = [](dual x) -> dual { return -x - 2.0 * log(1.0 + exp(-x)); };
        CHECK( f(x) == -val(x) - 2.0 * std::log(1.0 + std::exp(-val(x))) );
        CHECK( derivative(f, wrt(x), at(x)) == -1.0 - 2.0/(1.0 + std::exp(-val(x))) * (-std::exp(-val(x))) );

        // Testing negative operator on a more complex expression
        f = [](dual x) -> dual { return -x - log(1.0 + exp(-x)); };
        CHECK( f(x) == -val(x) - std::log(1.0 + std::exp(-val(x))) );
        CHECK( derivative(f, wrt(x), at(x)) == -1.0 - 1.0/(1.0 + std::exp(-val(x))) * (-std::exp(-val(x))) );
    }

    SECTION("testing unary inverse operator")
    {
        std::function<dual(dual)> f;

        // Testing inverse operator on a dual
        f = [](dual x) -> dual { return 1.0 / x; };
        CHECK( f(x) == 1/x );
        CHECK( derivative(f, wrt(x), at(x)) == -1.0 / (x * x) );

        // Testing inverse operator on a trivial expression
        f = [](dual x) -> dual { return x / x; };
        CHECK( f(x) == 1.0 );
        CHECK( derivative(f, wrt(x), at(x)) == 0.0 );

        // Testing inverse operator on an inverse expression
        f = [](dual x) -> dual { return 1.0 / (1.0 / x); };
        CHECK( f(x) == x );
        CHECK( derivative(f, wrt(x), at(x)) == 1.0 );

        // Testing inverse operator on a scaling expression
        f = [](dual x) -> dual { return 1.0 / (2.0 * x); };
        CHECK( f(x) == 0.5 / x );
        CHECK( derivative(f, wrt(x), at(x)) == -0.5 / (x * x) );
    }

    SECTION("testing binary addition operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing addition operator on a `number + dual` expression
        f = [](dual x, dual y) -> dual { return 1 + x; };
        CHECK( f(x, y) == 1 + val(x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 0.0 );

        // Testing addition operator on a `dual + number` expression
        f = [](dual x, dual y) -> dual { return x + 1; };
        CHECK( f(x, y) == val(x) + 1 );
        CHECK( derivative(f, wrt(x), at(x, y)) == 1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 0.0 );

        // Testing addition operator on a `dual + dual` expression
        f = [](dual x, dual y) -> dual { return (-x) + (-y); };
        CHECK( f(x, y) == -(val(x) + val(y)) );
        CHECK( derivative(f, wrt(x), at(x, y)) == -1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == -1.0 );

        // Testing addition operator on a `(-dual) + (-dual)` expression
        f = [](dual x, dual y) -> dual { return (-x) + (-y); };
        CHECK( f(x, y) == -(val(x) + val(y)) );
        CHECK( derivative(f, wrt(x), at(x, y)) == -1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == -1.0 );

        // Testing addition operator on a `dual * dual + dual * dual` expression
        f = [](dual x, dual y) -> dual { return x * y + x * y; };
        CHECK( f(x, y) == 2.0 * (val(x) * val(y)) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx( 2.0 * y ) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx( 2.0 * x ) );

        // Testing addition operator on a `1/dual * 1/dual` expression
        f = [](dual x, dual y) -> dual { return 1.0 / x + 1.0 / y; };
        CHECK( f(x, y) == approx( 1.0 / val(x) + 1.0 / val(y)) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx( -1.0 / (x * x) ) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx( -1.0 / (y * y) ) );
    }

    SECTION("testing binary subtraction operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing subtraction operator on a `number - dual` expression
        f = [](dual x, dual y) -> dual { return 1 - x; };
        CHECK( f(x, y) == 1 - val(x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == -1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) ==  0.0 );

        // Testing subtraction operator on a `dual - number` expression
        f = [](dual x, dual y) -> dual { return x - 1; };
        CHECK( f(x, y) == val(x) - 1 );
        CHECK( derivative(f, wrt(x), at(x, y)) == 1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 0.0 );

        // Testing subtraction operator on a `(-dual) - (-dual)` expression
        f = [](dual x, dual y) -> dual { return (-x) - (-y); };
        CHECK( f(x, y) == -(val(x) - val(y)) );
        CHECK( derivative(f, wrt(x), at(x, y)) == -1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) ==  1.0 );

        // Testing subtraction operator on a `dual * dual - dual * dual` expression
        f = [](dual x, dual y) -> dual { return x * y - x * y; };
        CHECK( f(x, y) == 0.0 );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx( 0.0 ) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx( 0.0 ) );

        // Testing subtraction operator on a `1/dual * 1/dual` expression
        f = [](dual x, dual y) -> dual { return 1.0 / x - 1.0 / y; };
        CHECK( f(x, y) == approx( 1.0 / val(x) - 1.0 / val(y)) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx( -1.0 / (x * x) ) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(  1.0 / (y * y) ) );
    }

    SECTION("testing binary multiplication operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing multiplication operator on a `number * dual` expression
        f = [](dual x, dual y) -> dual { return 2.0 * x; };
        CHECK( f(x, y) == 2.0 * val(x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 2.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 0.0 );

        // Testing multiplication operator on a `dual * number` expression
        f = [](dual x, dual y) -> dual { return x * 2; };
        CHECK( f(x, y) == 2.0 * val(x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 2.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 0.0 );

        // Testing multiplication operator on a `dual * dual` expression
        f = [](dual x, dual y) -> dual { return x * y; };
        CHECK( f(x, y) == val(x) * val(y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == y );
        CHECK( derivative(f, wrt(y), at(x, y)) == x );

        // Testing multiplication operator on a `number * (number * dual)` expression
        f = [](dual x, dual y) -> dual { return 5 * (2.0 * x); };
        CHECK( f(x, y) == 10 * val(x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 10.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) ==  0.0 );

        // Testing multiplication operator on a `(dual * number) * number` expression
        f = [](dual x, dual y) -> dual { return (x * 2) * 5; };
        CHECK( f(x, y) == 10 * val(x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 10.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) ==  0.0 );

        // Testing multiplication operator on a `number * (-dual)` expression
        f = [](dual x, dual y) -> dual { return 2.0 * (-x); };
        CHECK( f(x, y) == -2.0 * val(x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == -2.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) ==  0.0 );

        // Testing multiplication operator on a `(-dual) * number` expression
        f = [](dual x, dual y) -> dual { return (-x) * 2; };
        CHECK( f(x, y) == -2.0 * val(x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == -2.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) ==  0.0 );

        // Testing multiplication operator on a `(-dual) * (-dual)` expression
        f = [](dual x, dual y) -> dual { return (-x) * (-y); };
        CHECK( f(x, y) == val(x) * val(y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == y );
        CHECK( derivative(f, wrt(y), at(x, y)) == x );

        // Testing multiplication operator on a `(1/dual) * (1/dual)` expression
        f = [](dual x, dual y) -> dual { return (1 / x) * (1 / y); };
        CHECK( f(x, y) == 1 / (val(x) * val(y)) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(-1/(x * x * y)) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(-1/(x * y * y)) );
    }

    SECTION("testing binary division operator")
    {
        std::function<dual(dual, dual)> f;

        // Testing division operator on a `number / dual` expression
        f = [](dual x, dual y) -> dual { return 1 / x; };
        CHECK( f(x, y) == 1 / val(x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(-1.0 / (x * x)) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        // Testing division operator on a `dual / number` expression
        f = [](dual x, dual y) -> dual { return x / 2; };
        CHECK( f(x, y) == val(x) / 2 );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(0.5) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        // Testing division operator on a `dual / dual` expression
        f = [](dual x, dual y) -> dual { return x / y; };
        CHECK( f(x, y) == val(x) / val(y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(1 / y) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(-x / (y * y)) );

        // Testing division operator on a `1 / (number * dual)` expression
        f = [](dual x, dual y) -> dual { return 1 / (2.0 * x); };
        CHECK( f(x, y) == approx(0.5 / x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(-0.5 / (x * x)) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        // Testing division operator on a `1 / (1 / dual)` expression
        f = [](dual x, dual y) -> dual { return 1 / (1 / x); };
        CHECK( f(x, y) == val(x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 0.0 );
    }

    SECTION("testing operator+=")
    {
        std::function<dual(dual, dual)> f;

        f = [](dual x, dual y) -> dual { return x += 1; };
        CHECK( f(x, y) == approx(x + 1) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 0.0 );

        f = [](dual x, dual y) -> dual { return x += y; };
        CHECK( f(x, y) == approx(x + y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 1.0 );

        f = [](dual x, dual y) -> dual { return x += -y; };
        CHECK( f(x, y) == approx(x - y) );
        CHECK( derivative(f, wrt(x), at(x, y)) ==  1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == -1.0 );

        f = [](dual x, dual y) -> dual { return x += -(x - y); };
        CHECK( f(x, y) == approx(y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 0.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 1.0 );

        f = [](dual x, dual y) -> dual { return x += 1.0 / y; };
        CHECK( f(x, y) == approx(x + 1.0 / y) );
        CHECK( derivative(f, wrt(x), at(x, y)) ==  1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(-1.0 / (y * y)) );

        f = [](dual x, dual y) -> dual { return x += 2.0 * y; };
        CHECK( f(x, y) == approx(x + 2.0 * y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 2.0 );

        f = [](dual x, dual y) -> dual { return x += x * y; };
        CHECK( f(x, y) == approx(x + x * y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 1.0 + y );
        CHECK( derivative(f, wrt(y), at(x, y)) == x );

        f = [](dual x, dual y) -> dual { return x += x + y; };
        CHECK( f(x, y) == approx(x + x + y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 2.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 1.0 );
    }

    SECTION("testing operator-=")
    {
        std::function<dual(dual, dual)> f;

        f = [](dual x, dual y) -> dual { return x -= 1; };
        CHECK( f(x, y) == approx(x - 1) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 0.0 );

        f = [](dual x, dual y) -> dual { return x -= y; };
        CHECK( f(x, y) == approx(x - y) );
        CHECK( derivative(f, wrt(x), at(x, y)) ==  1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == -1.0 );

        f = [](dual x, dual y) -> dual { return x -= -y; };
        CHECK( f(x, y) == approx(x + y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 1.0 );

        f = [](dual x, dual y) -> dual { return x -= -(x - y); };
        CHECK( f(x, y) == approx(2.0 * x - y) );
        CHECK( derivative(f, wrt(x), at(x, y)) ==  2.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == -1.0 );

        f = [](dual x, dual y) -> dual { return x -= 1.0 / y; };
        CHECK( f(x, y) == approx(x - 1.0 / y) );
        CHECK( derivative(f, wrt(x), at(x, y)) ==  1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(1.0 / (y * y)) );

        f = [](dual x, dual y) -> dual { return x -= 2.0 * y; };
        CHECK( f(x, y) == approx(x - 2.0 * y) );
        CHECK( derivative(f, wrt(x), at(x, y)) ==  1.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == -2.0 );

        f = [](dual x, dual y) -> dual { return x -= x * y; };
        CHECK( f(x, y) == approx(x - x * y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 1.0 - y );
        CHECK( derivative(f, wrt(y), at(x, y)) == -x );

        f = [](dual x, dual y) -> dual { return x -= x - y; };
        CHECK( f(x, y) == approx(y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == 0.0 );
        CHECK( derivative(f, wrt(y), at(x, y)) == 1.0 );
    }

    SECTION("testing operator*=")
    {
        std::function<dual(dual, dual)> f;

        f = [](dual x, dual y) -> dual { return x *= 2; };
        CHECK( f(x, y) == approx(2.0 * x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(2.0) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        f = [](dual x, dual y) -> dual { return x *= y; };
        CHECK( f(x, y) == approx(x * y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(y) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(x) );

        f = [](dual x, dual y) -> dual { return x *= -x; };
        CHECK( f(x, y) == approx(-x * x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(-2.0 * x) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        f = [](dual x, dual y) -> dual { return x *= (2.0 / y); };
        CHECK( f(x, y) == approx(2.0 * x / y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(2.0 / y) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(-2.0 * x / (y * y)) );

        f = [](dual x, dual y) -> dual { return x *= (2.0 * x); };
        CHECK( f(x, y) == approx(2.0 * x * x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(4.0 * x) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        f = [](dual x, dual y) -> dual { return x *= (2.0 * y); };
        CHECK( f(x, y) == approx(2.0 * x * y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(2.0 * y) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(2.0 * x) );

        f = [](dual x, dual y) -> dual { return x *= x + y; };
        CHECK( f(x, y) == approx(x * (x + y)) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(2.0 * x + y) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(x) );

        f = [](dual x, dual y) -> dual { return x *= x * y; };
        CHECK( f(x, y) == approx(x * (x * y)) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(2.0 * x * y) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(x * x) );
    }

    SECTION("testing operator/=")
    {
        std::function<dual(dual, dual)> f;

        f = [](dual x, dual y) -> dual { return x /= 2; };
        CHECK( f(x, y) == approx(0.5 * x) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(0.5) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        f = [](dual x, dual y) -> dual { return x /= y; };
        CHECK( f(x, y) == approx(x / y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(1.0 / y) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(-x / (y * y)) );

        f = [](dual x, dual y) -> dual { return x /= -x; };
        CHECK( f(x, y) == approx(-1.0) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(0.0) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        f = [](dual x, dual y) -> dual { return x /= (2.0 / y); };
        CHECK( f(x, y) == approx(0.5 * x * y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(0.5 * y) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(0.5 * x) );

        f = [](dual x, dual y) -> dual { return x /= (2.0 * y); };
        CHECK( f(x, y) == approx(0.5 * x / y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx( 0.5 / y) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(-0.5 * x / (y * y)) );

        f = [](dual x, dual y) -> dual { return x /= x + y; };
        CHECK( f(x, y) == approx(x / (x + y)) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(1.0 / (x + y) - x / (x + y) / (x + y)) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(-x / (x + y) / (x + y)) );

        f = [](dual x, dual y) -> dual { return x /= x * y; };
        CHECK( f(x, y) == approx(1.0 / y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(0.0) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(-1.0 / (y * y)) );
    }

    SECTION("testing combination of operations")
    {
        std::function<dual(dual, dual)> f;

        // Testing multiplication with addition
        f = [](dual x, dual y) -> dual { return 2.0 * x + y; };
        CHECK( f(x, y) == 2.0 * val(x) + val(y) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(2.0) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(1.0) );

        // Testing a complex expression that is actually equivalent to one
        f = [](dual x, dual y) -> dual { return (2.0 * x * x - x * y + x / y + x / (2.0 * y)) / (x * (2.0 * x - y + 1 / y + 1 / (2.0 * y))); };
        CHECK( f(x, y) == approx(1.0) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(0.0) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(0.0) );
    }

    SECTION("testing mathematical functions")
    {
        std::function<dual(dual)> f;

        x = 0.5;

        // Testing sin function
        f = [](dual x) -> dual { return sin(x); };
        CHECK( f(x) == std::sin(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(cos(x)) );

        // Testing cos function
        f = [](dual x) -> dual { return cos(x); };
        CHECK( f(x) == std::cos(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(-sin(x)) );

        // Testing tan function
        f = [](dual x) -> dual { return tan(x); };
        CHECK( f(x) == std::tan(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(1 / (cos(x) * cos(x))) );

        // Testing sinh function
        f = [](dual x) -> dual { return sinh(x); };
        CHECK( f(x) == std::sinh(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(cosh(x)) );

        // Testing cosh function
        f = [](dual x) -> dual { return cosh(x); };
        CHECK( f(x) == std::cosh(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(sinh(x)) );

        // Testing tanh function
        f = [](dual x) -> dual { return tanh(x); };
        CHECK( f(x) == std::tanh(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(1 / (cosh(x) * cosh(x))) );

        // Testing asin function
        f = [](dual x) -> dual { return asin(x); };
        CHECK( f(x) == std::asin(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(1 / sqrt(1 - x * x)) );

        // Testing acos function
        f = [](dual x) -> dual { return acos(x); };
        CHECK( f(x) == std::acos(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(-1 / sqrt(1 - x * x)) );

        // Testing atan function
        f = [](dual x) -> dual { return atan(x); };
        CHECK( f(x) == std::atan(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(1 / (1 + x * x)) );

        // Testing exp function
        f = [](dual x) -> dual { return exp(x); };
        CHECK( f(x) == std::exp(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(exp(x)) );

        // Testing log function
        f = [](dual x) -> dual { return log(x); };
        CHECK( f(x) == std::log(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(1 / x) );

        // Testing log function
        f = [](dual x) -> dual { return log10(x); };
        CHECK( f(x) == std::log10(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(1 / (log(10) * x)) );

        // Testing sqrt function
        f = [](dual x) -> dual { return sqrt(x); };
        CHECK( f(x) == std::sqrt(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(0.5 / sqrt(x)) );

        // Testing pow function (with number exponent)
        f = [](dual x) -> dual { return pow(x, 2.0); };
        CHECK( f(x) == std::pow(val(x), 2.0) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(2.0 * x) );

        // Testing pow function (with dual exponent)
        f = [](dual x) -> dual { return pow(x, x); };
        CHECK( f(x) == std::pow(val(x), val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx((log(x) + 1) * pow(x, x)) );

        // Testing pow function (with expression exponent)
        f = [](dual x) -> dual { return pow(x, 2.0 * x); };
        CHECK( f(x) == std::pow(val(x), 2.0 * val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(2.0 * (log(x) + 1) * pow(x, 2.0 * x)) );

        // Testing abs function (when x > 0 and when x < 0)
        f = [](dual x) -> dual { return abs(x); };
        x = 1.0;
        CHECK( f(x) == std::abs(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(1.0) );
        x = -1.0;
        CHECK( f(x) == std::abs(val(x)) );
        CHECK( derivative(f, wrt(x), at(x)) == approx(-1.0) );
    }

    SECTION("testing complex expressions")
    {
        std::function<dual(dual, dual)> f;

        x = 0.5;
        y = 0.8;

        // Testing complex function involving sin, cos, and tan
        f = [](dual x, dual y) -> dual { return sin(x + y) * cos(x / y) + tan(2.0 * x * y) - sin(4*(x + y)*2/8) * cos(x*x / (y*y) * y/x) - tan((x + y) * (x + y) - x*x - y*y); };
        CHECK( val(f(x, y)) == approx(0.0) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(0.0) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(0.0) );

        // Testing complex function involving log, exp, pow, and sqrt
        f = [](dual x, dual y) -> dual { return log(x + y) * exp(x / y) + sqrt(2.0 * x * y) - 1 / pow(x, x + y) - exp(x*x / (y*y) * y/x) * log(4*(x + y)*2/8) - 4 * sqrt((x + y) * (x + y) - x*x - y*y) * 0.5 * 0.5 + 2 / pow(2.0 * x - x, y + x) * 0.5; };
        CHECK( val(f(x, y)) == approx(0.0) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(0.0) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(0.0) );
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

        CHECK( derivative(f, wrt(x), at(x, y)) == approx(17.0) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(19.0) );
        // CHECK( derivative(f, wrt<2>(x), at(x, y)) == approx(2.0) );
        CHECK( derivative<2>(f, wrt(x, y), at(x, y)) == approx(1.0) );
        CHECK( derivative<2>(f, wrt(y, x), at(x, y)) == approx(1.0) );
        // CHECK( derivative(f, wrt<2>(y), at(x, y)) == approx(2.0) );

        // Testing complex function involving log
        x = 2.0;
        y = 1.0;

        f = [](dual2nd x, dual2nd y) -> dual2nd
        {
            return 1 + x + y + x * y + y / x + log(x / y);
        };

        CHECK( val(f(x, y)) == approx( 1 + val(x) + val(y) + val(x) * val(y) + val(y) / val(x) + log(val(x) / val(y)) ) );
        CHECK( val(derivative(f, wrt(x), at(x, y))) == approx( 1 + val(y) - val(y) / (val(x) * val(x)) + 1.0/val(x) - log(val(y)) ) );
        CHECK( val(derivative(f, wrt(y), at(x, y))) == approx( 1 + val(x) + 1.0 / val(x) - 1.0/val(y) ) );
        // CHECK( val(derivative(f, wrt<2>(x), at(x, y))) == approx( 2 * val(y) / (val(x) * val(x) * val(x)) + -1.0/(val(x) * val(x)) ) );
        CHECK( val(derivative<2>(f, wrt(y, y), at(x, y))) == approx( 1.0/(val(y)*val(y)) ) );
        CHECK( val(derivative<2>(f, wrt(y, x), at(x, y))) == approx( 1 - 1.0 / (val(x) * val(x)) ) );
        CHECK( val(derivative<2>(f, wrt(x, y), at(x, y))) == approx( 1 - 1.0 / (val(x) * val(x)) ) );
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

        // CHECK( derivative(f, wrt<3>(x), at(x, y)) == approx(6.0) );
        CHECK( derivative<3>(f, wrt(x, x, x), at(x, y)) == approx(6.0) );
        CHECK( derivative<3>(f, wrt(x, x, y), at(x, y)) == approx(6.0) );
        CHECK( derivative<3>(f, wrt(x, y, y), at(x, y)) == approx(6.0) );
        // CHECK( derivative(f, wrt<3>(y), at(x, y)) == approx(6.0) );
    }
}
