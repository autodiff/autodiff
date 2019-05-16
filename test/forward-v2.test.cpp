// Catch includes
#include "catch.hpp"

// Eigen includes
#include <eigen3/Eigen/Core>
using namespace Eigen;

// autodiff includes
#include <autodiff/forward/forward-v2.hpp>
//#include <autodiff/forward/eigen.hpp>
using namespace autodiff;
using namespace autodiff::forward2;
//
//template<typename T>
//auto approx(T&& expr) -> Approx
//{
//    const double zero = std::numeric_limits<double>::epsilon();
//    return Approx(val(std::forward<T>(expr))).margin(zero);
//}
//
//namespace autodiff::forward {
//
//template<typename U, enableif<isExpr<U>>...>
//auto operator==(U&& l, const Approx& r) { return val(std::forward<U>(l)) == r; }
//
//template<typename U, enableif<isExpr<U>>...>
//auto operator==(const Approx& l, U&& r) { return std::forward<U>(r) == l; }
//
//} // namespace autodiff::forward

TEST_CASE("dual tests", "[dual]")
{
    dual x = 100.;
    dual y = 10.;

    // auto z = head(x);

    dual a;

    // int i = x + y;

    a = 23.;
    REQUIRE(val(a) == 23);

    a = y + x;

    REQUIRE(val(a) == 110);
    // REQUIRE(grad(a).data()[0] == 0);
    // REQUIRE(grad(a).data()[1] == 0);

    // int i = grad(x);
    // int j = grad(grad(x));
    x[1] = 0.0;
    y[1] = 1.0;

    x[2] = 0.0;
    y[2] = 0.0;

    a = x * y;

    REQUIRE(a[0] == Approx( x[0] * y[0] ));
    REQUIRE(a[1] == Approx( x[1] * y[0] + x[0] * y[1] ));
    REQUIRE(a[2] == Approx( x[2] * y[0] + 2 * x[1] * y[1] + x[0] * y[2] ));

    a = x / y;

    REQUIRE(a[0] == Approx( x[0]/y[0] ));
    REQUIRE(a[1] == Approx( (x[1] - y[1]*a[0])/y[0] ));
    REQUIRE(a[2] == Approx( (x[2] - y[2]*a[0] - 2*y[1]*a[1])/y[0] ));
    // REQUIRE(grad(a).data()[0] == 0);
    // REQUIRE(grad(a).data()[1] == 0);


    // x + y;

    // z.begin[0] = 10;

    // SECTION("trivial tests")
    // {
    //     REQUIRE( x == 100 );
    //     x += 1;
    //     REQUIRE( x == 101 );
    //     x -= 1;
    //     REQUIRE( x == 100 );
    //     x *= 2;
    //     REQUIRE( x == 200 );
    //     x /= 20;
    //     REQUIRE( x == 10 );
    // }
//
//    SECTION("testing comparison operators")
//    {
//        x = 6;
//        y = 5;
//
//        REQUIRE( x == 6 );
//        REQUIRE( 6 == x );
//        REQUIRE( x == x );
//
//        REQUIRE( x != 5 );
//        REQUIRE( 5 != x );
//        REQUIRE( x != y );
//
//        REQUIRE( x > 5 );
//        REQUIRE( x > y );
//
//        REQUIRE( x >= 6 );
//        REQUIRE( x >= x );
//        REQUIRE( x >= y );
//
//        REQUIRE( 5 < x );
//        REQUIRE( y < x );
//
//        REQUIRE( 6 <= x );
//        REQUIRE( x <= x );
//        REQUIRE( y <= x );
//    }
//
//    SECTION("testing unary negative operator")
//    {
//        std::function<dual(dual)> f;
//
//        // Testing positive operator on a dual
//        f = [](dual x) -> dual { return +x; };
//        REQUIRE( f(x) == x );
//        REQUIRE( derivative(f, wrt(x), x) == 1.0 );
//
//        // Testing positive operator on an expression
//        f = [](dual x) -> dual { return +(x * x); };
//        REQUIRE( f(x) == x * x );
//        REQUIRE( derivative(f, wrt(x), x) == 2.0 * x );
//
//        // Testing negative operator on a dual
//        f = [](dual x) -> dual { return -x; };
//        REQUIRE( f(x) == -x );
//        REQUIRE( derivative(f, wrt(x), x) == -1.0 );
//
//        // Testing negative operator on a negative expression
//        f = [](dual x) -> dual { return -(-x); };
//        REQUIRE( f(x) == x );
//        REQUIRE( derivative(f, wrt(x), x) == 1.0 );
//
//        // Testing negative operator on a scaling expression expression
//        f = [](dual x) -> dual { return -(2.0 * x); };
//        REQUIRE( f(x) == -2.0 * x );
//        REQUIRE( derivative(f, wrt(x), x) == -2.0 );
//    }

//    SECTION("testing unary inverse operator")
//    {
//        std::function<dual(dual)> f;
//
//        // Testing inverse operator on a dual
//        f = [](dual x) -> dual { return 1.0 / x; };
//        REQUIRE( f(x) == 1/x );
//        REQUIRE( derivative(f, wrt(x), x) == -1.0 / (x * x) );
//
//        // Testing inverse operator on a trivial expression
//        f = [](dual x) -> dual { return x / x; };
//        REQUIRE( f(x) == 1.0 );
//        REQUIRE( derivative(f, wrt(x), x) == 0.0 );
//
//        // Testing inverse operator on an inverse expression
//        f = [](dual x) -> dual { return 1.0 / (1.0 / x); };
//        REQUIRE( f(x) == x );
//        REQUIRE( derivative(f, wrt(x), x) == 1.0 );
//
//        // Testing inverse operator on a scaling expression
//        f = [](dual x) -> dual { return 1.0 / (2.0 * x); };
//        REQUIRE( f(x) == 0.5 / x );
//        REQUIRE( derivative(f, wrt(x), x) == -0.5 / (x * x) );
//    }
//
//    SECTION("testing binary addition operator")
//    {
//        std::function<dual(dual, dual)> f;
//
//        // Testing addition operator on a `number + dual` expression
//        f = [](dual x, dual y) -> dual { return 1 + x; };
//        REQUIRE( f(x, y) == 1 + val(x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
//
//        // Testing addition operator on a `dual + number` expression
//        f = [](dual x, dual y) -> dual { return x + 1; };
//        REQUIRE( f(x, y) == val(x) + 1 );
//        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
//
//        // Testing addition operator on a `dual + dual` expression
//        f = [](dual x, dual y) -> dual { return (-x) + (-y); };
//        REQUIRE( f(x, y) == -(val(x) + val(y)) );
//        REQUIRE( derivative(f, wrt(x), x, y) == -1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == -1.0 );
//
//        // Testing addition operator on a `(-dual) + (-dual)` expression
//        f = [](dual x, dual y) -> dual { return (-x) + (-y); };
//        REQUIRE( f(x, y) == -(val(x) + val(y)) );
//        REQUIRE( derivative(f, wrt(x), x, y) == -1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == -1.0 );
//
//        // Testing addition operator on a `dual * dual + dual * dual` expression
//        f = [](dual x, dual y) -> dual { return x * y + x * y; };
//        REQUIRE( f(x, y) == 2.0 * (val(x) * val(y)) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx( 2.0 * y ) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx( 2.0 * x ) );
//
//        // Testing addition operator on a `1/dual * 1/dual` expression
//        f = [](dual x, dual y) -> dual { return 1.0 / x + 1.0 / y; };
//        REQUIRE( f(x, y) == approx( 1.0 / val(x) + 1.0 / val(y)) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx( -1.0 / (x * x) ) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx( -1.0 / (y * y) ) );
//    }
//
//    SECTION("testing binary subtraction operator")
//    {
//        std::function<dual(dual, dual)> f;
//
//        // Testing subtraction operator on a `number - dual` expression
//        f = [](dual x, dual y) -> dual { return 1 - x; };
//        REQUIRE( f(x, y) == 1 - val(x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == -1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) ==  0.0 );
//
//        // Testing subtraction operator on a `dual - number` expression
//        f = [](dual x, dual y) -> dual { return x - 1; };
//        REQUIRE( f(x, y) == val(x) - 1 );
//        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
//
//        // Testing subtraction operator on a `(-dual) - (-dual)` expression
//        f = [](dual x, dual y) -> dual { return (-x) - (-y); };
//        REQUIRE( f(x, y) == -(val(x) - val(y)) );
//        REQUIRE( derivative(f, wrt(x), x, y) == -1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) ==  1.0 );
//
//        // Testing subtraction operator on a `dual * dual - dual * dual` expression
//        f = [](dual x, dual y) -> dual { return x * y - x * y; };
//        REQUIRE( f(x, y) == 0.0 );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx( 0.0 ) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx( 0.0 ) );
//
//        // Testing subtraction operator on a `1/dual * 1/dual` expression
//        f = [](dual x, dual y) -> dual { return 1.0 / x - 1.0 / y; };
//        REQUIRE( f(x, y) == approx( 1.0 / val(x) - 1.0 / val(y)) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx( -1.0 / (x * x) ) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(  1.0 / (y * y) ) );
//    }
//
//    SECTION("testing binary multiplication operator")
//    {
//        std::function<dual(dual, dual)> f;
//
//        // Testing multiplication operator on a `number * dual` expression
//        f = [](dual x, dual y) -> dual { return 2.0 * x; };
//        REQUIRE( f(x, y) == 2.0 * val(x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 2.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
//
//        // Testing multiplication operator on a `dual * number` expression
//        f = [](dual x, dual y) -> dual { return x * 2; };
//        REQUIRE( f(x, y) == 2.0 * val(x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 2.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
//
//        // Testing multiplication operator on a `dual * dual` expression
//        f = [](dual x, dual y) -> dual { return x * y; };
//        REQUIRE( f(x, y) == val(x) * val(y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == y );
//        REQUIRE( derivative(f, wrt(y), x, y) == x );
//
//        // Testing multiplication operator on a `number * (number * dual)` expression
//        f = [](dual x, dual y) -> dual { return 5 * (2.0 * x); };
//        REQUIRE( f(x, y) == 10 * val(x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 10.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) ==  0.0 );
//
//        // Testing multiplication operator on a `(dual * number) * number` expression
//        f = [](dual x, dual y) -> dual { return (x * 2) * 5; };
//        REQUIRE( f(x, y) == 10 * val(x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 10.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) ==  0.0 );
//
//        // Testing multiplication operator on a `number * (-dual)` expression
//        f = [](dual x, dual y) -> dual { return 2.0 * (-x); };
//        REQUIRE( f(x, y) == -2.0 * val(x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == -2.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) ==  0.0 );
//
//        // Testing multiplication operator on a `(-dual) * number` expression
//        f = [](dual x, dual y) -> dual { return (-x) * 2; };
//        REQUIRE( f(x, y) == -2.0 * val(x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == -2.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) ==  0.0 );
//
//        // Testing multiplication operator on a `(-dual) * (-dual)` expression
//        f = [](dual x, dual y) -> dual { return (-x) * (-y); };
//        REQUIRE( f(x, y) == val(x) * val(y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == y );
//        REQUIRE( derivative(f, wrt(y), x, y) == x );
//
//        // Testing multiplication operator on a `(1/dual) * (1/dual)` expression
//        f = [](dual x, dual y) -> dual { return (1 / x) * (1 / y); };
//        REQUIRE( f(x, y) == 1 / (val(x) * val(y)) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(-1/(x * x * y)) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(-1/(x * y * y)) );
//    }
//
//    SECTION("testing binary division operator")
//    {
//        std::function<dual(dual, dual)> f;
//
//        // Testing division operator on a `number / dual` expression
//        f = [](dual x, dual y) -> dual { return 1 / x; };
//        REQUIRE( f(x, y) == 1 / val(x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(-1.0 / (x * x)) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(0.0) );
//
//        // Testing division operator on a `dual / number` expression
//        f = [](dual x, dual y) -> dual { return x / 2; };
//        REQUIRE( f(x, y) == val(x) / 2 );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(0.5) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(0.0) );
//
//        // Testing division operator on a `dual / dual` expression
//        f = [](dual x, dual y) -> dual { return x / y; };
//        REQUIRE( f(x, y) == val(x) / val(y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(1 / y) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(-x / (y * y)) );
//
//        // Testing division operator on a `1 / (number * dual)` expression
//        f = [](dual x, dual y) -> dual { return 1 / (2.0 * x); };
//        REQUIRE( f(x, y) == approx(0.5 / x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(-0.5 / (x * x)) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(0.0) );
//
//        // Testing division operator on a `1 / (1 / dual)` expression
//        f = [](dual x, dual y) -> dual { return 1 / (1 / x); };
//        REQUIRE( f(x, y) == val(x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
//    }
//
//    SECTION("testing operator+=")
//    {
//        std::function<dual(dual, dual)> f;
//
//        f = [](dual x, dual y) -> dual { return x += 1; };
//        REQUIRE( f(x, y) == approx(x + 1) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
//
//        f = [](dual x, dual y) -> dual { return x += y; };
//        REQUIRE( f(x, y) == approx(x + y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 1.0 );
//
//        f = [](dual x, dual y) -> dual { return x += -y; };
//        REQUIRE( f(x, y) == approx(x - y) );
//        REQUIRE( derivative(f, wrt(x), x, y) ==  1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == -1.0 );
//
//        f = [](dual x, dual y) -> dual { return x += -(x - y); };
//        REQUIRE( f(x, y) == approx(y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 0.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 1.0 );
//
//        f = [](dual x, dual y) -> dual { return x += 1.0 / y; };
//        REQUIRE( f(x, y) == approx(x + 1.0 / y) );
//        REQUIRE( derivative(f, wrt(x), x, y) ==  1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(-1.0 / (y * y)) );
//
//        f = [](dual x, dual y) -> dual { return x += 2.0 * y; };
//        REQUIRE( f(x, y) == approx(x + 2.0 * y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 2.0 );
//
//        f = [](dual x, dual y) -> dual { return x += x * y; };
//        REQUIRE( f(x, y) == approx(x + x * y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 + y );
//        REQUIRE( derivative(f, wrt(y), x, y) == x );
//
//        f = [](dual x, dual y) -> dual { return x += x + y; };
//        REQUIRE( f(x, y) == approx(x + x + y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 2.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 1.0 );
//    }
//
//    SECTION("testing operator-=")
//    {
//        std::function<dual(dual, dual)> f;
//
//        f = [](dual x, dual y) -> dual { return x -= 1; };
//        REQUIRE( f(x, y) == approx(x - 1) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
//
//        f = [](dual x, dual y) -> dual { return x -= y; };
//        REQUIRE( f(x, y) == approx(x - y) );
//        REQUIRE( derivative(f, wrt(x), x, y) ==  1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == -1.0 );
//
//        f = [](dual x, dual y) -> dual { return x -= -y; };
//        REQUIRE( f(x, y) == approx(x + y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 1.0 );
//
//        f = [](dual x, dual y) -> dual { return x -= -(x - y); };
//        REQUIRE( f(x, y) == approx(2.0 * x - y) );
//        REQUIRE( derivative(f, wrt(x), x, y) ==  2.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == -1.0 );
//
//        f = [](dual x, dual y) -> dual { return x -= 1.0 / y; };
//        REQUIRE( f(x, y) == approx(x - 1.0 / y) );
//        REQUIRE( derivative(f, wrt(x), x, y) ==  1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(1.0 / (y * y)) );
//
//        f = [](dual x, dual y) -> dual { return x -= 2.0 * y; };
//        REQUIRE( f(x, y) == approx(x - 2.0 * y) );
//        REQUIRE( derivative(f, wrt(x), x, y) ==  1.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == -2.0 );
//
//        f = [](dual x, dual y) -> dual { return x -= x * y; };
//        REQUIRE( f(x, y) == approx(x - x * y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 1.0 - y );
//        REQUIRE( derivative(f, wrt(y), x, y) == -x );
//
//        f = [](dual x, dual y) -> dual { return x -= x - y; };
//        REQUIRE( f(x, y) == approx(y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 0.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 1.0 );
//    }
//
//    SECTION("testing operator*=")
//    {
//        std::function<dual(dual, dual)> f;
//
//        f = [](dual x, dual y) -> dual { return x *= 2; };
//        REQUIRE( f(x, y) == approx(2.0 * x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 2.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
//
//        f = [](dual x, dual y) -> dual { return x *= y; };
//        REQUIRE( f(x, y) == approx(x * y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == y );
//        REQUIRE( derivative(f, wrt(y), x, y) == x );
//
//        f = [](dual x, dual y) -> dual { return x *= -x; };
//        REQUIRE( f(x, y) == approx(-x * x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(-2.0 * x) );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
//
//        f = [](dual x, dual y) -> dual { return x *= (2.0 / y); };
//        REQUIRE( f(x, y) == approx(2.0 * x / y) );
//        REQUIRE( derivative(f, wrt(x), x, y) ==  2.0 / y );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(-2.0 * x / (y * y)) );
//
//        f = [](dual x, dual y) -> dual { return x *= (2.0 * x); };
//        REQUIRE( f(x, y) == approx(2.0 * x * x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(4.0 * x) );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
//
//        f = [](dual x, dual y) -> dual { return x *= (2.0 * y); };
//        REQUIRE( f(x, y) == approx(2.0 * x * y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 2.0 * y );
//        REQUIRE( derivative(f, wrt(y), x, y) == 2.0 * x );
//
//        f = [](dual x, dual y) -> dual { return x *= x + y; };
//        REQUIRE( f(x, y) == approx(x * (x + y)) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 2.0 * x + y );
//        REQUIRE( derivative(f, wrt(y), x, y) == x );
//
//        f = [](dual x, dual y) -> dual { return x *= x * y; };
//        REQUIRE( f(x, y) == approx(x * (x * y)) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(2.0 * x * y) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(x * x) );
//    }
//
//    SECTION("testing operator/=")
//    {
//        std::function<dual(dual, dual)> f;
//
//        f = [](dual x, dual y) -> dual { return x /= 2; };
//        REQUIRE( f(x, y) == approx(0.5 * x) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 0.5 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
//
//        f = [](dual x, dual y) -> dual { return x /= y; };
//        REQUIRE( f(x, y) == approx(x / y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(1.0 / y) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(-x / (y * y)) );
//
//        f = [](dual x, dual y) -> dual { return x /= -x; };
//        REQUIRE( f(x, y) == approx(-1.0) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 0.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.0 );
//
//        f = [](dual x, dual y) -> dual { return x /= (2.0 / y); };
//        REQUIRE( f(x, y) == approx(0.5 * x * y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 0.5 * y );
//        REQUIRE( derivative(f, wrt(y), x, y) == 0.5 * x );
//
//        f = [](dual x, dual y) -> dual { return x /= (2.0 * y); };
//        REQUIRE( f(x, y) == approx(0.5 * x / y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx( 0.5 / y) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(-0.5 * x / (y * y)) );
//
//        f = [](dual x, dual y) -> dual { return x /= x + y; };
//        REQUIRE( f(x, y) == approx(x / (x + y)) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(1.0 / (x + y) - x / (x + y) / (x + y)) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(-x / (x + y) / (x + y)) );
//
//        f = [](dual x, dual y) -> dual { return x /= x * y; };
//        REQUIRE( f(x, y) == approx(1.0 / y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == 0.0 );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(-1.0 / (y * y)) );
//    }
//
//    SECTION("testing combination of operations")
//    {
//        std::function<dual(dual, dual)> f;
//
//        // Testing multiplication with addition
//        f = [](dual x, dual y) -> dual { return 2.0 * x + y; };
//        REQUIRE( f(x, y) == 2.0 * val(x) + val(y) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(2.0) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(1.0) );
//
//        // Testing a complex expression that is actually equivalent to one
//        f = [](dual x, dual y) -> dual { return (2.0 * x * x - x * y + x / y + x / (2.0 * y)) / (x * (2.0 * x - y + 1 / y + 1 / (2.0 * y))); };
//        REQUIRE( f(x, y) == approx(1.0) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(0.0) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(0.0) );
//    }
//
//    SECTION("testing mathematical functions")
//    {
//        std::function<dual(dual)> f;
//
//        x = 0.5;
//
//        // Testing sin function
//        f = [](dual x) -> dual { return sin(x); };
//        REQUIRE( f(x) == std::sin(val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(cos(x)) );
//
//        // Testing cos function
//        f = [](dual x) -> dual { return cos(x); };
//        REQUIRE( f(x) == std::cos(val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(-sin(x)) );
//
//        // Testing tan function
//        f = [](dual x) -> dual { return tan(x); };
//        REQUIRE( f(x) == std::tan(val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(1 / (cos(x) * cos(x))) );
//
//        // Testing asin function
//        f = [](dual x) -> dual { return asin(x); };
//        REQUIRE( f(x) == std::asin(val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(1 / sqrt(1 - x * x)) );
//
//        // Testing acos function
//        f = [](dual x) -> dual { return acos(x); };
//        REQUIRE( f(x) == std::acos(val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(-1 / sqrt(1 - x * x)) );
//
//        // Testing atan function
//        f = [](dual x) -> dual { return atan(x); };
//        REQUIRE( f(x) == std::atan(val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(1 / (1 + x * x)) );
//
//        // Testing exp function
//        f = [](dual x) -> dual { return exp(x); };
//        REQUIRE( f(x) == std::exp(val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(exp(x)) );
//
//        // Testing log function
//        f = [](dual x) -> dual { return log(x); };
//        REQUIRE( f(x) == std::log(val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(1 / x) );
//
//        // Testing log function
//        f = [](dual x) -> dual { return log10(x); };
//        REQUIRE( f(x) == std::log10(val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(1 / (log(10) * x)) );
//
//        // Testing sqrt function
//        f = [](dual x) -> dual { return sqrt(x); };
//        REQUIRE( f(x) == std::sqrt(val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(0.5 / sqrt(x)) );
//
//        // Testing pow function (with number exponent)
//        f = [](dual x) -> dual { return pow(x, 2.0); };
//        REQUIRE( f(x) == std::pow(val(x), 2.0) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(2.0 * x) );
//
//        // Testing pow function (with dual exponent)
//        f = [](dual x) -> dual { return pow(x, x); };
//        REQUIRE( f(x) == std::pow(val(x), val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx((log(x) + 1) * pow(x, x)) );
//
//        // Testing pow function (with expression exponent)
//        f = [](dual x) -> dual { return pow(x, 2.0 * x); };
//        REQUIRE( f(x) == std::pow(val(x), 2.0 * val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(2.0 * (log(x) + 1) * pow(x, 2.0 * x)) );
//
//        // Testing abs function (when x > 0 and when x < 0)
//        f = [](dual x) -> dual { return abs(x); };
//        x = 1.0;
//        REQUIRE( f(x) == std::abs(val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(1.0) );
//        x = -1.0;
//        REQUIRE( f(x) == std::abs(val(x)) );
//        REQUIRE( derivative(f, wrt(x), x) == approx(-1.0) );
//    }
//
//    SECTION("testing complex expressions")
//    {
//        std::function<dual(dual, dual)> f;
//
//        x = 0.5;
//        y = 0.8;
//
//        // Testing complex function involving sin, cos, and tan
//        f = [](dual x, dual y) -> dual { return sin(x + y) * cos(x / y) + tan(2.0 * x * y) - sin(4*(x + y)*2/8) * cos(x*x / (y*y) * y/x) - tan((x + y) * (x + y) - x*x - y*y); };
//        REQUIRE( val(f(x, y)) == approx(0.0) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(0.0) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(0.0) );
//
//        // Testing complex function involving log, exp, pow, and sqrt
//        f = [](dual x, dual y) -> dual { return log(x + y) * exp(x / y) + sqrt(2.0 * x * y) - 1 / pow(x, x + y) - exp(x*x / (y*y) * y/x) * log(4*(x + y)*2/8) - 4 * sqrt((x + y) * (x + y) - x*x - y*y) * 0.5 * 0.5 + 2 / pow(2.0 * x - x, y + x) * 0.5; };
//        REQUIRE( val(f(x, y)) == approx(0.0) );
//        REQUIRE( derivative(f, wrt(x), x, y) == approx(0.0) );
//        REQUIRE( derivative(f, wrt(y), x, y) == approx(0.0) );
//    }
//
//
//    SECTION("testing higher order derivatives")
//    {
//        using dual2nd = HigherOrderDual<2>;
//
//        dual2nd x = 5;
//        dual2nd y = 7;
//
//        // Testing complex function involving sin and cos
//        auto f = [](dual2nd x, dual2nd y) -> dual2nd
//        {
//            return x*x + x*y + y*y;
//        };
//
//        REQUIRE( val(derivative(f, wrt(x), x, y)) == Approx(17.0) );
//        REQUIRE( derivative(f, wrt(x, x), x, y) == Approx(2.0) );
//        REQUIRE( derivative(f, wrt(x, y), x, y) == Approx(1.0) );
//        REQUIRE( derivative(f, wrt(y, x), x, y) == Approx(1.0) );
//        REQUIRE( derivative(f, wrt(y, y), x, y) == Approx(2.0) );
//    }
//
//    SECTION("testing higher order derivatives")
//    {
//        using dual3rd = HigherOrderDual<3>;
//
//        dual3rd x = 5;
//        dual3rd y = 7;
//
//        // Testing complex function involving sin and cos
//        auto f = [](dual3rd x, dual3rd y) -> dual3rd
//        {
//            return (x + y)*(x + y)*(x + y);
//        };
//
//        REQUIRE( derivative(f, wrt(x, x, x), x, y) == Approx(6.0) );
//        REQUIRE( derivative(f, wrt(x, x, x), x, y) == Approx(6.0) );
//        REQUIRE( derivative(f, wrt(x, x, y), x, y) == Approx(6.0) );
//        REQUIRE( derivative(f, wrt(x, y, y), x, y) == Approx(6.0) );
//        REQUIRE( derivative(f, wrt(y, y, y), x, y) == Approx(6.0) );
//    }
//
//    SECTION("testing gradient derivatives")
//    {
//        // Testing complex function involving sin and cos
//        auto f = [](const VectorXdual& x) -> dual
//        {
//            return 0.5 * ( x.array() * x.array() ).sum();
//        };
//
//        VectorXdual x(3);
//        x << 1.0, 2.0, 3.0;
//
//        VectorXd g = gradient(f, x);
//
//        REQUIRE( g[0] == approx(x[0]) );
//        REQUIRE( g[1] == approx(x[1]) );
//        REQUIRE( g[2] == approx(x[2]) );
//    }
//
//    SECTION("testing jacobian derivatives")
//    {
//        // Testing complex function involving sin and cos
//        auto f = [](const VectorXdual& x) -> VectorXdual
//        {
//            return x / x.array().sum();
//        };
//
//        VectorXdual x(3);
//        x << 0.5, 0.2, 0.3;
//
//        VectorXdual u = f(x);
//
//        MatrixXd J = jacobian(f, u, x);
//
//        for(auto i = 0; i < 3; ++i)
//            for(auto j = 0; j < 3; ++j)
//                REQUIRE( J(i, j) == approx(-u[i] + ((i == j) ? 1.0 : 0.0)) );
//    }
}
