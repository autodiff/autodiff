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
    const double epsilon = std::numeric_limits<double>::epsilon() * 100;
    const double margin = 1e-12;
    return Approx(val(std::forward<T>(expr))).epsilon(epsilon).margin(margin);
}

#define CHECK_DERIVATIVES_FX(expr, u, ux)         \
{                                                 \
    auto f = [](dual x) -> dual { return expr; }; \
    auto dfdx = derivatives(f, wrt(x), at(x));    \
    CHECK( dfdx[0] == approx(u) );                \
    CHECK( dfdx[1] == approx(ux) );               \
}

#define CHECK_DERIVATIVES_FXY(expr, u, ux, uy)            \
{                                                         \
    auto f = [](dual x, dual y) -> dual { return expr; }; \
    auto dfdx = derivatives(f, wrt(x), at(x, y));         \
    CHECK( dfdx[0] == approx(u) );                        \
    CHECK( dfdx[1] == approx(ux) );                       \
    auto dfdy = derivatives(f, wrt(y), at(x, y));         \
    CHECK( dfdy[0] == approx(u) );                        \
    CHECK( dfdy[1] == approx(uy) );                       \
}

#define CHECK_DERIVATIVES_FXY_3RD_ORDER(expr, u, ux, uy, uxx, uxy, uyy, uxxx, uxxy, uxyy, uyyy) \
{                                                                                               \
    using dual3rd = HigherOrderDual<3>;                                                         \
    dual3rd x = 1;                                                                              \
    dual3rd y = 2;                                                                              \
    auto f = [](dual3rd x, dual3rd y) -> dual3rd { return expr; };                              \
    auto dfdx = derivatives(f, wrt(x), at(x, y));                                               \
    CHECK( dfdx[0] == approx(u) );                                                              \
    CHECK( dfdx[1] == approx(ux) );                                                             \
    CHECK( dfdx[2] == approx(uxx) );                                                            \
    CHECK( dfdx[3] == approx(uxxx) );                                                           \
    auto dfdy = derivatives(f, wrt(y), at(x, y));                                               \
    CHECK( dfdy[0] == approx(u) );                                                              \
    CHECK( dfdy[1] == approx(uy) );                                                             \
    CHECK( dfdy[2] == approx(uyy) );                                                            \
    CHECK( dfdy[3] == approx(uyyy) );                                                           \
    auto dfdxx = derivatives(f, wrt(x, x), at(x, y));                                           \
    CHECK( dfdxx[0] == approx(u) );                                                             \
    CHECK( dfdxx[1] == approx(ux) );                                                            \
    CHECK( dfdxx[2] == approx(uxx) );                                                           \
    CHECK( dfdxx[3] == approx(uxxx) );                                                          \
    auto dfdxy = derivatives(f, wrt(x, y), at(x, y));                                           \
    CHECK( dfdxy[0] == approx(u) );                                                             \
    CHECK( dfdxy[1] == approx(ux) );                                                            \
    CHECK( dfdxy[2] == approx(uxy) );                                                           \
    CHECK( dfdxy[3] == approx(uxyy) );                                                          \
    auto dfdyx = derivatives(f, wrt(y, x), at(x, y));                                           \
    CHECK( dfdyx[0] == approx(u) );                                                             \
    CHECK( dfdyx[1] == approx(uy) );                                                            \
    CHECK( dfdyx[2] == approx(uxy) );                                                           \
    CHECK( dfdyx[3] == approx(uxyy) );                                                          \
    auto dfdyy = derivatives(f, wrt(y, y), at(x, y));                                           \
    CHECK( dfdyy[0] == approx(u) );                                                             \
    CHECK( dfdyy[1] == approx(uy) );                                                            \
    CHECK( dfdyy[2] == approx(uyy) );                                                           \
    CHECK( dfdyy[3] == approx(uyyy) );                                                          \
    auto dfdxxx = derivatives(f, wrt(x, x, x), at(x, y));                                       \
    CHECK( dfdxxx[0] == approx(u) );                                                            \
    CHECK( dfdxxx[1] == approx(ux) );                                                           \
    CHECK( dfdxxx[2] == approx(uxx) );                                                          \
    CHECK( dfdxxx[3] == approx(uxxx) );                                                         \
    auto dfdxyx = derivatives(f, wrt(x, y, x), at(x, y));                                       \
    CHECK( dfdxyx[0] == approx(u) );                                                            \
    CHECK( dfdxyx[1] == approx(ux) );                                                           \
    CHECK( dfdxyx[2] == approx(uxy) );                                                          \
    CHECK( dfdxyx[3] == approx(uxxy) );                                                         \
    auto dfdxxy = derivatives(f, wrt(x, x, y), at(x, y));                                       \
    CHECK( dfdxxy[0] == approx(u) );                                                            \
    CHECK( dfdxxy[1] == approx(ux) );                                                           \
    CHECK( dfdxxy[2] == approx(uxx) );                                                          \
    CHECK( dfdxxy[3] == approx(uxxy) );                                                         \
    auto dfdxyy = derivatives(f, wrt(x, y, y), at(x, y));                                       \
    CHECK( dfdxyy[0] == approx(u) );                                                            \
    CHECK( dfdxyy[1] == approx(ux) );                                                           \
    CHECK( dfdxyy[2] == approx(uxy) );                                                          \
    CHECK( dfdxyy[3] == approx(uxyy) );                                                         \
    auto dfdyxx = derivatives(f, wrt(y, x, x), at(x, y));                                       \
    CHECK( dfdyxx[0] == approx(u) );                                                            \
    CHECK( dfdyxx[1] == approx(uy) );                                                           \
    CHECK( dfdyxx[2] == approx(uxy) );                                                          \
    CHECK( dfdyxx[3] == approx(uxxy) );                                                         \
    auto dfdyyx = derivatives(f, wrt(y, y, x), at(x, y));                                       \
    CHECK( dfdyyx[0] == approx(u) );                                                            \
    CHECK( dfdyyx[1] == approx(uy) );                                                           \
    CHECK( dfdyyx[2] == approx(uyy) );                                                          \
    CHECK( dfdyyx[3] == approx(uxyy) );                                                         \
    auto dfdyxy = derivatives(f, wrt(y, x, y), at(x, y));                                       \
    CHECK( dfdyxy[0] == approx(u) );                                                            \
    CHECK( dfdyxy[1] == approx(uy) );                                                           \
    CHECK( dfdyxy[2] == approx(uxy) );                                                          \
    CHECK( dfdyxy[3] == approx(uxyy) );                                                         \
    auto dfdyyy = derivatives(f, wrt(y, y, y), at(x, y));                                       \
    CHECK( dfdyyy[0] == approx(u) );                                                            \
    CHECK( dfdyyy[1] == approx(uy) );                                                           \
    CHECK( dfdyyy[2] == approx(uyy) );                                                          \
    CHECK( dfdyyy[3] == approx(uyyy) );                                                         \
}

TEST_CASE("testing autodiff::dual", "[forward][dual]")
{
    dual x = 100;
    dual y = 10;

    SECTION("trivial tests")
    {
        CHECK( x == approx(100) );
        x += 1;
        CHECK( x == approx(101) );
        x -= 1;
        CHECK( x == approx(100) );
        x *= 2;
        CHECK( x == approx(200) );
        x /= 20;
        CHECK( x == approx(10) );
    }

    SECTION("aliasing tests")
    {
        x = 1; x = x + 3*x - 2*x + x;
        CHECK( x == approx(3) );

        x = 1; x += x + 3*x - 2*x + x;
        CHECK( x == approx(4) );

        x = 1; x -= x + 3*x - 2*x + x;
        CHECK( x == approx(-2) );

        x = 1; x *= x + 3*x - 2*x + x;
        CHECK( x == approx(3) );

        x = 1; x /= x + x;
        CHECK( x == approx(0.5) );
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
        // Testing positive operator on a dual
        CHECK_DERIVATIVES_FX(+x, val(x), 1.0);

        // Testing positive operator on an expression
        CHECK_DERIVATIVES_FX(+(x * x), val(x) * val(x), 2.0 * val(x));

        // Testing negative operator on a dual
        CHECK_DERIVATIVES_FX(-x, -val(x), -1.0);

        // Testing negative operator on a negative expression
        CHECK_DERIVATIVES_FX(-(-x), val(x), 1.0);

        // Testing negative operator on a scaling expression expression
        CHECK_DERIVATIVES_FX(-(2.0 * x), -(2.0 * val(x)), -2.0);

        // Testing negative operator on a more complex expression
        CHECK_DERIVATIVES_FX(-x - (2*x), -val(x) - (2*val(x)), -3.0);

        // Testing negative operator on a more complex expression
        CHECK_DERIVATIVES_FX(-x - 2.0 * log(1.0 + exp(-x)), -val(x) - 2.0 * log(1.0 + exp(-val(x))), -1.0 - 2.0/(1.0 + exp(-val(x))) * (-exp(-val(x))));

        // Testing negative operator on a more complex expression
        CHECK_DERIVATIVES_FX(-x - log(1.0 + exp(-x)), -val(x) - log(1.0 + exp(-val(x))), -1.0 - 1.0/(1.0 + exp(-val(x))) * (-exp(-val(x))));
    }

    SECTION("testing unary inverse operator")
    {
        // Testing inverse operator on a dual
        CHECK_DERIVATIVES_FX(1.0/x, 1.0/val(x), -1.0/(x * x));

        // Testing inverse operator on a trivial expression
        CHECK_DERIVATIVES_FX(x/x, 1.0, 0.0);

        // Testing inverse operator on an inverse expression
        CHECK_DERIVATIVES_FX(1.0/(1.0/x), val(x), 1.0);

        // Testing inverse operator on a scaling expression
        CHECK_DERIVATIVES_FX(1.0/(2.0 * x), 0.5/val(x), -0.5/(x * x));
    }

    SECTION("testing binary addition operator")
    {
        // Testing addition operator on a `number + dual` expression
        CHECK_DERIVATIVES_FXY(1 + x, 1 + val(x), 1.0, 0.0);

        // Testing addition operator on a `dual + number` expression
        CHECK_DERIVATIVES_FXY(x + 1, val(x) + 1, 1.0, 0.0);

        // Testing addition operator on a `(-dual) + (-dual)` expression
        CHECK_DERIVATIVES_FXY((-x) + (-y), -(val(x) + val(y)), -1.0, -1.0);

        // Testing addition operator on a `dual * dual + dual * dual` expression
        CHECK_DERIVATIVES_FXY(x * y + x * y, 2*(val(x) * val(y)), 2.0 * y, 2.0 * x );

        // Testing addition operator on a `1/dual * 1/dual` expression
        CHECK_DERIVATIVES_FXY(1.0/x + 1.0/y, 1.0/val(x) + 1.0/val(y), -1.0/(x * x), -1.0/(y * y) );
    }

    SECTION("testing binary subtraction operator")
    {
        // Testing subtraction operator on a `number - dual` expression
        CHECK_DERIVATIVES_FXY(1 - x, 1 - val(x), -1.0,  0.0);

        // Testing subtraction operator on a `dual - number` expression
        CHECK_DERIVATIVES_FXY(x - 1, val(x) - 1, 1.0, 0.0);

        // Testing subtraction operator on a `(-dual) - (-dual)` expression
        CHECK_DERIVATIVES_FXY((-x) - (-y), val(y) - val(x), -1.0,  1.0);

        // Testing subtraction operator on a `dual * dual - dual * dual` expression
        CHECK_DERIVATIVES_FXY(x * y - x * y, 0.0, 0.0, 0.0 );

        // Testing subtraction operator on a `1/dual * 1/dual` expression
        CHECK_DERIVATIVES_FXY(1.0/x - 1.0/y, 1.0/val(x) - 1.0/val(y), -1.0/(x * x),  1.0/(y * y) );
    }

    SECTION("testing binary multiplication operator")
    {
        // Testing multiplication operator on a `number * dual` expression
        CHECK_DERIVATIVES_FXY(2.0 * x, 2.0 * val(x), 2.0, 0.0);

        // Testing multiplication operator on a `dual * number` expression
        CHECK_DERIVATIVES_FXY(x * 2.0, val(x) * 2.0, 2.0, 0.0);

        // Testing multiplication operator on a `dual * dual` expression
        CHECK_DERIVATIVES_FXY(x * y, val(x) * val(y), y, x);

        // Testing multiplication operator on a `number * (number * dual)` expression
        CHECK_DERIVATIVES_FXY(5 * (2.0 * x), 10.0 * val(x), 10.0,  0.0);

        // Testing multiplication operator on a `(dual * number) * number` expression
        CHECK_DERIVATIVES_FXY((x * 2) * 5, 10.0 * val(x), 10.0,  0.0);

        // Testing multiplication operator on a `number * (-dual)` expression
        CHECK_DERIVATIVES_FXY(2.0 * (-x), -2.0 * val(x), -2.0,  0.0);

        // Testing multiplication operator on a `(-dual) * number` expression
        CHECK_DERIVATIVES_FXY((-x) * 2, -2.0 * val(x), -2.0,  0.0);

        // Testing multiplication operator on a `(-dual) * (-dual)` expression
        CHECK_DERIVATIVES_FXY((-x) * (-y), val(x) * val(y), y, x);

        // Testing multiplication operator on a `(1/dual) * (1/dual)` expression
        CHECK_DERIVATIVES_FXY((1/x) * (1/y), 1/(val(x) * val(y)), -1/(x * x * y), -1/(x * y * y));
    }

    SECTION("testing binary division operator")
    {
        // Testing division operator on a `number/dual` expression
        CHECK_DERIVATIVES_FXY(1/x, 1/val(x), -1.0/(x * x), 0.0);

        // Testing division operator on a `dual/number` expression
        CHECK_DERIVATIVES_FXY(x/2, 0.5 * val(x), 0.5, 0.0);

        // Testing division operator on a `dual/dual` expression
        CHECK_DERIVATIVES_FXY(x/y, val(x)/val(y), 1/y, -x/(y * y));

        // Testing division operator on a `1/(number * dual)` expression
        CHECK_DERIVATIVES_FXY(1/(2.0 * x), 0.5/val(x), -0.5/(x * x), 0.0);

        // Testing division operator on a `1/(1/dual)` expression
        CHECK_DERIVATIVES_FXY(1/(1/x), val(x), 1.0, 0.0);
    }

    SECTION("testing operator+=")
    {
        CHECK_DERIVATIVES_FXY(x += 1, x + 1, 1.0, 0.0);
        CHECK_DERIVATIVES_FXY(x += y, x + y, 1.0, 1.0);
        CHECK_DERIVATIVES_FXY(x += -y, x - y,  1.0, -1.0);
        CHECK_DERIVATIVES_FXY(x += -(x - y), y, 0.0, 1.0);
        CHECK_DERIVATIVES_FXY(x += 1.0 / y, x + 1.0 / y,  1.0, -1.0 / (y * y));
        CHECK_DERIVATIVES_FXY(x += 2.0 * y, x + 2.0 * y, 1.0, 2.0);
        CHECK_DERIVATIVES_FXY(x += x * y, x + x * y, 1.0 + y, x);
        CHECK_DERIVATIVES_FXY(x += x + y, x + x + y, 2.0, 1.0);
    }

    SECTION("testing operator-=")
    {
        CHECK_DERIVATIVES_FXY(x -= 1, x - 1, 1.0, 0.0);
        CHECK_DERIVATIVES_FXY(x -= y, x - y,  1.0, -1.0);
        CHECK_DERIVATIVES_FXY(x -= -y, x + y, 1.0, 1.0);
        CHECK_DERIVATIVES_FXY(x -= -(x - y), 2.0 * x - y,  2.0, -1.0);
        CHECK_DERIVATIVES_FXY(x -= 1.0 / y, x - 1.0 / y,  1.0, 1.0 / (y * y));
        CHECK_DERIVATIVES_FXY(x -= 2.0 * y, x - 2.0 * y,  1.0, -2.0);
        CHECK_DERIVATIVES_FXY(x -= x * y, x - x * y, 1.0 - y, -x);
        CHECK_DERIVATIVES_FXY(x -= x - y, y, 0.0, 1.0);
    }

    SECTION("testing operator*=")
    {
        CHECK_DERIVATIVES_FXY(x *= 2, 2.0 * x, 2.0, 0.0);
        CHECK_DERIVATIVES_FXY(x *= y, x * y, y, x);
        CHECK_DERIVATIVES_FXY(x *= -x, -x * x, -2.0 * x, 0.0);
        CHECK_DERIVATIVES_FXY(x *= (2.0 / y), 2.0 * x / y, 2.0 / y, -2.0 * x / (y * y));
        CHECK_DERIVATIVES_FXY(x *= (2.0 * x), 2.0 * x * x, 4.0 * x, 0.0);
        CHECK_DERIVATIVES_FXY(x *= (2.0 * y), 2.0 * x * y, 2.0 * y, 2.0 * x);
        CHECK_DERIVATIVES_FXY(x *= x + y, x * (x + y), 2.0 * x + y, x);
        CHECK_DERIVATIVES_FXY(x *= x * y, x * (x * y), 2.0 * x * y, x * x);
    }

    SECTION("testing operator/=")
    {
        CHECK_DERIVATIVES_FXY(x /= 2, 0.5 * x, 0.5, 0.0);
        CHECK_DERIVATIVES_FXY(x /= y, x / y, 1.0 / y, -x / (y * y));
        CHECK_DERIVATIVES_FXY(x /= -x, -1.0, 0.0, 0.0);
        CHECK_DERIVATIVES_FXY(x /= (2.0 / y), 0.5 * x * y, 0.5 * y, 0.5 * x);
        CHECK_DERIVATIVES_FXY(x /= (2.0 * y), 0.5 * x / y,  0.5 / y, -0.5 * x / (y * y));
        CHECK_DERIVATIVES_FXY(x /= x + y, x / (x + y), 1.0 / (x + y) - x / (x + y) / (x + y), -x / (x + y) / (x + y));
        CHECK_DERIVATIVES_FXY(x /= x * y, 1.0 / y, 0.0, -1.0 / (y * y));
    }

    SECTION("testing combination of operations")
    {
        // Testing multiplication with addition
        CHECK_DERIVATIVES_FXY(2.0 * x + y, 2.0 * val(x) + val(y), 2.0, 1.0);

        // Testing a complex expression that is actually equivalent to one
        CHECK_DERIVATIVES_FXY((2.0 * x * x - x * y + x / y + x / (2.0 * y)) / (x * (2.0 * x - y + 1 / y + 1 / (2.0 * y))), 1.0, 0.0, 0.0);
    }

    SECTION("testing mathematical functions")
    {
        x = 0.5;

        // Testing sin function
        CHECK_DERIVATIVES_FX(sin(x), sin(val(x)), cos(x));

        // Testing cos function
        CHECK_DERIVATIVES_FX(cos(x), cos(val(x)), -sin(x));

        // Testing tan function
        CHECK_DERIVATIVES_FX(tan(x), tan(val(x)), 1 / (cos(x) * cos(x)));

        // Testing sinh function
        CHECK_DERIVATIVES_FX(sinh(x), sinh(val(x)), cosh(x));

        // Testing cosh function
        CHECK_DERIVATIVES_FX(cosh(x), cosh(val(x)), sinh(x));

        // Testing tanh function
        CHECK_DERIVATIVES_FX(tanh(x), tanh(val(x)), 1 / (cosh(x) * cosh(x)));

        // Testing asin function
        CHECK_DERIVATIVES_FX(asin(x), asin(val(x)), 1 / sqrt(1 - x * x));

        // Testing acos function
        CHECK_DERIVATIVES_FX(acos(x), acos(val(x)), -1 / sqrt(1 - x * x));

        // Testing atan function
        CHECK_DERIVATIVES_FX(atan(x), atan(val(x)), 1 / (1 + x * x));

        // Testing exp function
        CHECK_DERIVATIVES_FX(exp(x), exp(val(x)), exp(x));

        // Testing log function
        CHECK_DERIVATIVES_FX(log(x), log(val(x)), 1 / x);

        // Testing log function
        CHECK_DERIVATIVES_FX(log10(x), log10(val(x)), 1 / (log(10) * x));

        // Testing sqrt function
        CHECK_DERIVATIVES_FX(sqrt(x), sqrt(val(x)), 0.5 / sqrt(x));

        // Testing pow function (with number exponent)
        CHECK_DERIVATIVES_FX(pow(x, 2.0), pow(val(x), 2.0), 2.0 * x);

        // Testing pow function (with dual exponent)
        CHECK_DERIVATIVES_FX(pow(x, x), pow(val(x), val(x)), (log(x) + 1) * pow(x, x));

        // Testing pow function (with expression exponent)
        CHECK_DERIVATIVES_FX(pow(x, 2.0 * x), pow(val(x), 2.0 * val(x)), 2.0 * (log(x) + 1) * pow(x, 2.0 * x));

        // Testing abs function (when x > 0)
        x = 1.0; CHECK_DERIVATIVES_FX(abs(x), abs(val(x)), 1.0);

        // Testing abs function (when x < 0)
        x = -1.0; CHECK_DERIVATIVES_FX(abs(x), abs(val(x)), -1.0);
    }

    SECTION("testing complex expressions")
    {
        x = 0.5;
        y = 0.8;

        // Testing complex function involving sin, cos, and tan
        CHECK_DERIVATIVES_FXY(sin(x + y) * cos(x / y) + tan(2.0 * x * y) - sin(4*(x + y)*2/8) * cos(x*x / (y*y) * y/x) - tan((x + y) * (x + y) - x*x - y*y), 0.0, 0.0, 0.0);

        // Testing complex function involving log, exp, pow, and sqrt
        CHECK_DERIVATIVES_FXY(log(x + y) * exp(x / y) + sqrt(2.0 * x * y) - 1 / pow(x, x + y) - exp(x*x / (y*y) * y/x) * log(4*(x + y)*2/8) - 4 * sqrt((x + y) * (x + y) - x*x - y*y) * 0.5 * 0.5 + 2 / pow(2.0 * x - x, y + x) * 0.5, 0.0, 0.0, 0.0);
    }

    // SECTION("testing higher order derivatives")
    // {
    //     using dual2nd = HigherOrderDual<2>;

    //     std::function<dual2nd(dual2nd, dual2nd)> f;

    //     dual2nd x = 5;
    //     dual2nd y = 7;

    //     // Testing simpler functions
    //     f = [](dual2nd x, dual2nd y) -> dual2nd
    //     {
    //         return x*x + x*y + y*y;
    //     };

    //     CHECK( derivative(f, wrt(x), at(x, y)) == approx(17.0) );
    //     CHECK( derivative(f, wrt(y), at(x, y)) == approx(19.0) );
    //     // CHECK( derivative(f, wrt<2>(x), at(x, y)) == approx(2.0) );
    //     CHECK( derivative<2>(f, wrt(x, y), at(x, y)) == approx(1.0) );
    //     CHECK( derivative<2>(f, wrt(y, x), at(x, y)) == approx(1.0) );
    //     // CHECK( derivative(f, wrt<2>(y), at(x, y)) == approx(2.0) );

    //     // Testing complex function involving log
    //     x = 2.0;
    //     y = 1.0;

    //     f = [](dual2nd x, dual2nd y) -> dual2nd
    //     {
    //         return 1 + x + y + x * y + y / x + log(x / y);
    //     };

    //     CHECK( val(f(x, y)) == approx( 1 + val(x) + val(y) + val(x) * val(y) + val(y) / val(x) + log(val(x) / val(y)) ) );
    //     CHECK( val(derivative(f, wrt(x), at(x, y))) == approx( 1 + val(y) - val(y) / (val(x) * val(x)) + 1.0/val(x) - log(val(y)) ) );
    //     CHECK( val(derivative(f, wrt(y), at(x, y))) == approx( 1 + val(x) + 1.0 / val(x) - 1.0/val(y) ) );
    //     // CHECK( val(derivative(f, wrt<2>(x), at(x, y))) == approx( 2 * val(y) / (val(x) * val(x) * val(x)) + -1.0/(val(x) * val(x)) ) );
    //     CHECK( val(derivative<2>(f, wrt(y, y), at(x, y))) == approx( 1.0/(val(y)*val(y)) ) );
    //     CHECK( val(derivative<2>(f, wrt(y, x), at(x, y))) == approx( 1 - 1.0 / (val(x) * val(x)) ) );
    //     CHECK( val(derivative<2>(f, wrt(x, y), at(x, y))) == approx( 1 - 1.0 / (val(x) * val(x)) ) );
    // }

    SECTION("testing higher order derivatives")
    {
        Catch::StringMaker<double>::precision = 15;

        CHECK_DERIVATIVES_FXY_3RD_ORDER((x + y)*(x + y)*(x + y),
            (val(x) + val(y)) * (val(x) + val(y)) * (val(x) + val(y)), // u
            3.0 * (x + y) * (x + y),                                   // ux
            3.0 * (x + y) * (x + y),                                   // uy
            6.0 * (x + y),                                             // uxx
            6.0 * (x + y),                                             // uxy
            6.0 * (x + y),                                             // uyy
            6.0,                                                       // uxxx
            6.0,                                                       // uxxy
            6.0,                                                       // uxyy
            6.0                                                        // uyyy
        );

        CHECK_DERIVATIVES_FXY_3RD_ORDER(exp(log(x * y)), // expression identical to x*y
            val(x) * val(y), // u
            y,               // ux
            x,               // uy
            0.0,             // uxx
            1.0,             // uxy
            0.0,             // uyy
            0.0,             // uxxx
            0.0,             // uxxy
            0.0,             // uxyy
            0.0              // uyyy
        );

        CHECK_DERIVATIVES_FXY_3RD_ORDER(sin(x * y) * sin(x * y) + cos(x * y) * cos(x * y), // expression identical to 1
            1.0, // u
            0.0, // ux
            0.0, // uy
            0.0, // uxx
            0.0, // uxy
            0.0, // uyy
            0.0, // uxxx
            0.0, // uxxy
            0.0, // uxyy
            0.0  // uyyy
        );

        CHECK_DERIVATIVES_FXY_3RD_ORDER(tan(log(x) * exp(y)) - sin(exp(y) * log(x)) / cos(log(x) * exp(y)), // expression identical to 0
            0.0, // u
            0.0, // ux
            0.0, // uy
            0.0, // uxx
            0.0, // uxy
            0.0, // uyy
            0.0, // uxxx
            0.0, // uxxy
            0.0, // uxyy
            0.0  // uyyy
        );
    }
}
