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
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

template<typename T>
auto approx(T&& expr) -> Catch::Approx
{
    return Catch::Approx(val(std::forward<T>(expr))).margin(1e-12);
}

#define CHECK_DERIVATIVES_FX(expr, u, ux)         \
{                                                 \
    auto f = [](dual x) -> dual { return expr; }; \
    auto dfdx = derivatives(f, wrt(x), at(x));    \
    CHECK( dfdx[0] == approx(val(u)) );           \
    CHECK( dfdx[1] == approx(val(ux)) );          \
}

#define CHECK_DERIVATIVES_FXY(expr, u, ux, uy)            \
{                                                         \
    auto f = [](dual x, dual y) -> dual { return expr; }; \
    auto dfdx = derivatives(f, wrt(x), at(x, y));         \
    CHECK( dfdx[0] == approx(val(u)) );                   \
    CHECK( dfdx[1] == approx(val(ux)) );                  \
    auto dfdy = derivatives(f, wrt(y), at(x, y));         \
    CHECK( dfdy[0] == approx(val(u)) );                   \
    CHECK( dfdy[1] == approx(val(uy)) );                  \
}

#define CHECK_DERIVATIVES_FXYZ(expr, u, ux, uy, uz)               \
{                                                                 \
    auto f = [](dual x, dual y, dual z) -> dual { return expr; }; \
    auto dfdx = derivatives(f, wrt(x), at(x, y, z));              \
    CHECK( dfdx[0] == approx(val(u)) );                           \
    CHECK( dfdx[1] == approx(val(ux)) );                          \
    auto dfdy = derivatives(f, wrt(y), at(x, y, z));              \
    CHECK( dfdy[0] == approx(val(u)) );                           \
    CHECK( dfdy[1] == approx(val(uy)) );                          \
    auto dfdz = derivatives(f, wrt(z), at(x, y, z));              \
    CHECK( dfdz[0] == approx(val(u)) );                           \
    CHECK( dfdz[1] == approx(val(uz)) );                          \
}

#define CHECK_DERIVATIVES_FXY_3RD_ORDER(expr, u, ux, uy, uxx, uxy, uyy, uxxx, uxxy, uxyy, uyyy) \
{                                                                                               \
    dual3rd x = 1;                                                                              \
    dual3rd y = 2;                                                                              \
    auto f = [](dual3rd x, dual3rd y) -> dual3rd { return expr; };                              \
    auto dfdx = derivatives(f, wrt(x), at(x, y));                                               \
    CHECK( dfdx[0] == approx(val(u)) );                                                         \
    CHECK( dfdx[1] == approx(val(ux)) );                                                        \
    CHECK( dfdx[2] == approx(val(uxx)) );                                                       \
    CHECK( dfdx[3] == approx(val(uxxx)) );                                                      \
    auto dfdy = derivatives(f, wrt(y), at(x, y));                                               \
    CHECK( dfdy[0] == approx(val(u)) );                                                         \
    CHECK( dfdy[1] == approx(val(uy)) );                                                        \
    CHECK( dfdy[2] == approx(val(uyy)) );                                                       \
    CHECK( dfdy[3] == approx(val(uyyy)) );                                                      \
    auto dfdxx = derivatives(f, wrt(x, x), at(x, y));                                           \
    CHECK( dfdxx[0] == approx(val(u)) );                                                        \
    CHECK( dfdxx[1] == approx(val(ux)) );                                                       \
    CHECK( dfdxx[2] == approx(val(uxx)) );                                                      \
    CHECK( dfdxx[3] == approx(val(uxxx)) );                                                     \
    auto dfdxy = derivatives(f, wrt(x, y), at(x, y));                                           \
    CHECK( dfdxy[0] == approx(val(u)) );                                                        \
    CHECK( dfdxy[1] == approx(val(ux)) );                                                       \
    CHECK( dfdxy[2] == approx(val(uxy)) );                                                      \
    CHECK( dfdxy[3] == approx(val(uxyy)) );                                                     \
    auto dfdyx = derivatives(f, wrt(y, x), at(x, y));                                           \
    CHECK( dfdyx[0] == approx(val(u)) );                                                        \
    CHECK( dfdyx[1] == approx(val(uy)) );                                                       \
    CHECK( dfdyx[2] == approx(val(uxy)) );                                                      \
    CHECK( dfdyx[3] == approx(val(uxyy)) );                                                     \
    auto dfdyy = derivatives(f, wrt(y, y), at(x, y));                                           \
    CHECK( dfdyy[0] == approx(val(u)) );                                                        \
    CHECK( dfdyy[1] == approx(val(uy)) );                                                       \
    CHECK( dfdyy[2] == approx(val(uyy)) );                                                      \
    CHECK( dfdyy[3] == approx(val(uyyy)) );                                                     \
    auto dfdxxx = derivatives(f, wrt(x, x, x), at(x, y));                                       \
    CHECK( dfdxxx[0] == approx(val(u)) );                                                       \
    CHECK( dfdxxx[1] == approx(val(ux)) );                                                      \
    CHECK( dfdxxx[2] == approx(val(uxx)) );                                                     \
    CHECK( dfdxxx[3] == approx(val(uxxx)) );                                                    \
    auto dfdxyx = derivatives(f, wrt(x, y, x), at(x, y));                                       \
    CHECK( dfdxyx[0] == approx(val(u)) );                                                       \
    CHECK( dfdxyx[1] == approx(val(ux)) );                                                      \
    CHECK( dfdxyx[2] == approx(val(uxy)) );                                                     \
    CHECK( dfdxyx[3] == approx(val(uxxy)) );                                                    \
    auto dfdxxy = derivatives(f, wrt(x, x, y), at(x, y));                                       \
    CHECK( dfdxxy[0] == approx(val(u)) );                                                       \
    CHECK( dfdxxy[1] == approx(val(ux)) );                                                      \
    CHECK( dfdxxy[2] == approx(val(uxx)) );                                                     \
    CHECK( dfdxxy[3] == approx(val(uxxy)) );                                                    \
    auto dfdxyy = derivatives(f, wrt(x, y, y), at(x, y));                                       \
    CHECK( dfdxyy[0] == approx(val(u)) );                                                       \
    CHECK( dfdxyy[1] == approx(val(ux)) );                                                      \
    CHECK( dfdxyy[2] == approx(val(uxy)) );                                                     \
    CHECK( dfdxyy[3] == approx(val(uxyy)) );                                                    \
    auto dfdyxx = derivatives(f, wrt(y, x, x), at(x, y));                                       \
    CHECK( dfdyxx[0] == approx(val(u)) );                                                       \
    CHECK( dfdyxx[1] == approx(val(uy)) );                                                      \
    CHECK( dfdyxx[2] == approx(val(uxy)) );                                                     \
    CHECK( dfdyxx[3] == approx(val(uxxy)) );                                                    \
    auto dfdyyx = derivatives(f, wrt(y, y, x), at(x, y));                                       \
    CHECK( dfdyyx[0] == approx(val(u)) );                                                       \
    CHECK( dfdyyx[1] == approx(val(uy)) );                                                      \
    CHECK( dfdyyx[2] == approx(val(uyy)) );                                                     \
    CHECK( dfdyyx[3] == approx(val(uxyy)) );                                                    \
    auto dfdyxy = derivatives(f, wrt(y, x, y), at(x, y));                                       \
    CHECK( dfdyxy[0] == approx(val(u)) );                                                       \
    CHECK( dfdyxy[1] == approx(val(uy)) );                                                      \
    CHECK( dfdyxy[2] == approx(val(uxy)) );                                                     \
    CHECK( dfdyxy[3] == approx(val(uxyy)) );                                                    \
    auto dfdyyy = derivatives(f, wrt(y, y, y), at(x, y));                                       \
    CHECK( dfdyyy[0] == approx(val(u)) );                                                       \
    CHECK( dfdyyy[1] == approx(val(uy)) );                                                      \
    CHECK( dfdyyy[2] == approx(val(uyy)) );                                                     \
    CHECK( dfdyyy[3] == approx(val(uyyy)) );                                                    \
}

TEST_CASE("testing autodiff::dual", "[forward][dual]")
{
    dual x = 100;
    dual y = 10;
    dual z = 1;

    SECTION("trivial tests")
    {
        CHECK( val(x) == approx(100) );
        CHECK( grad(x) == 0.0 );
        x += 1;
        CHECK( val(x) == approx(101) );
        CHECK( grad(x) == 0.0 );
        x -= 1;
        CHECK( val(x) == approx(100) );
        CHECK( grad(x) == 0.0 );
        x *= 2;
        CHECK( val(x) == approx(200) );
        CHECK( grad(x) == 0.0 );
        x /= 20;
        CHECK( val(x) == approx(10) );
        CHECK( grad(x) == 0.0 );
    }

    SECTION("aliasing tests")
    {
        x = 1; x = x + 3*x - 2*x + x;
        CHECK( val(x) == approx(3) );
        CHECK( grad(x) == 0.0 );

        x = 1; x += x + 3*x - 2*x + x;
        CHECK( val(x) == approx(4) );
        CHECK( grad(x) == 0.0 );

        x = 1; x -= x + 3*x - 2*x + x;
        CHECK( val(x) == approx(-2) );
        CHECK( grad(x) == 0.0 );

        x = 1; x *= x + 3*x - 2*x + x;
        CHECK( val(x) == approx(3) );
        CHECK( grad(x) == 0.0 );

        x = 1; x /= x + x;
        CHECK( val(x) == approx(0.5) );
        CHECK( grad(x) == 0.0 );
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
        y = 0.8;

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
        x = 1.0; CHECK_DERIVATIVES_FX(abs(x), std::abs(val(x)), 1.0);

        // Testing abs function (when x < 0)
        x = -1.0; CHECK_DERIVATIVES_FX(abs(x), std::abs(val(x)), -1.0);

        // Testing erf function (when x = 1.0)
        x =  1.0; CHECK_DERIVATIVES_FX(erf(x), erf(val(x)), 0.4151074974);

        // Testing erf function (when x = -1.4)
        x = -1.4; CHECK_DERIVATIVES_FX(erf(x), erf(val(x)), 0.1589417077);

        // Testing atan2 function on (double, dual)
        x = 1.0; CHECK_DERIVATIVES_FX(atan2(2.0, x), atan2(2.0, val(x)), -2.0 / (2*2 + x*x));

        // Testing atan2 function on (dual, double)
        x = 1.0; CHECK_DERIVATIVES_FX(atan2(x, 2.0), atan2(val(x), 2.0), 2.0 / (2*2 + x*x));

        // Testing atan2 function on (dual, dual)
        x = 1.1;
        y = 0.9;
        CHECK_DERIVATIVES_FXY(atan2(y, x), atan2(val(y), val(x)), -y/(x*x + y*y), x/(x*x + y*y));

        // // Testing atan2 function on (expr, expr)
        CHECK_DERIVATIVES_FXY(3 * atan2(sin(x), 2*y+1), 3 * atan2(sin(val(x)), 2*val(y)+1), 3*(2*y+1)*cos(x) / ((2*y+1)*(2*y+1) + sin(x)*sin(x)), 3*-2*sin(x) / ((2*y+1)*(2*y+1) + sin(x)*sin(x)));

        // Testing hypot function on (dual, double)
        x = 1.5;
        CHECK_DERIVATIVES_FX(hypot(x, 2.0), hypot(val(x), 2.0), x/hypot(val(x), 2.0));

        // Testing hypot function on (double, dual)
        CHECK_DERIVATIVES_FX(hypot(2.0, x), hypot(2.0, val(x)), x/hypot(2.0, val(x)));

        // Testing hypot function on (dual, dual)
        x = 1.1;
        y = 0.9;
        CHECK_DERIVATIVES_FXY(hypot(x, y), hypot(val(x), val(y)), x/hypot(val(x), val(y)), y/hypot(val(x), val(y)));

        // Testing hypot function on (expr, expr)
        CHECK_DERIVATIVES_FXY(hypot(2.0*x,3.0*y), hypot(2.0*val(x), 3.0*val(y)), 4.0*x/hypot(2.0*val(x), 3.0*val(y)), 9.0*y/hypot(2.0*val(x), 3.0*val(y)));

        // Testing hypot function on (dual, double, double)
        x = 1.5;
        CHECK_DERIVATIVES_FX(hypot(x, 2.0, 2.0), std::hypot(val(x), 2.0, 2.0), x/std::hypot(val(x), 2.0, 2.0));

        // Testing hypot function on (double, dual, double)
        x = 2.5;
        CHECK_DERIVATIVES_FX(hypot(2.0, x, 2.0), std::hypot(2.0, val(x), 2.0), x/std::hypot(2.0, val(x), 2.0));

        // Testing hypot function on (double, double, dual)
        x = 3.5;
        CHECK_DERIVATIVES_FX(hypot(2.0, 2.0, x), std::hypot(2.0, 2.0, val(x)), x/std::hypot(2.0, 2.0, val(x)));

        // Testing hypot function on (dual, dual, double)
        x = 1.4;
        y = 2.4;
        CHECK_DERIVATIVES_FXY(hypot(x, y, 2.0), std::hypot(val(x), val(y), 2.0), x/std::hypot(val(x), val(y), 2.0), y/std::hypot(val(x), val(y), 2.0));

        // Testing hypot function on (double, dual, dual)
        x = 2.4;
        y = 3.4;
        CHECK_DERIVATIVES_FXY(hypot(2.0, x, y), std::hypot(2.0, val(x), val(y)), x/std::hypot(2.0, val(x), val(y)), y/std::hypot(2.0, val(x), val(y)));

        // Testing hypot function on (dual, double, dual)
        x = 3.4;
        y = 4.4;
        CHECK_DERIVATIVES_FXY(hypot(x, 2.0, y), std::hypot(val(x), 2.0, val(y)), x/std::hypot(val(x), 2.0, val(y)), y/std::hypot(val(x), 2.0, val(y)));

        // Testing hypot function on (dual, double, dual)
        x = 1.6;
        y = 2.6;
        z = 3.6;
        CHECK_DERIVATIVES_FXYZ(hypot(x, y, z), std::hypot(val(x), val(y), val(z)),
            x/std::hypot(val(x), val(y), val(z)),
            y/std::hypot(val(x), val(y), val(z)),
            z/std::hypot(val(x), val(y), val(z)));

        // Testing hypot function on (expr, expr, expr)
        x = 2.6;
        y = 3.6;
        z = 4.6;
        CHECK_DERIVATIVES_FXYZ(hypot(2.0*x, 3.0*y, 4.0*z), std::hypot(2.0*val(x), 3.0*val(y), 4.0*val(z)),
            4.0*x/std::hypot(2.0*val(x), 3.0*val(y), 4.0*val(z)),
            9.0*y/std::hypot(2.0*val(x), 3.0*val(y), 4.0*val(z)),
            16.*z/std::hypot(2.0*val(x), 3.0*val(y), 4.0*val(z)));
    }

    SECTION("testing min and max functions")
    {
        x = 0.5;
        y = 0.8;

        CHECK( min(x, y) == x );
        CHECK( min(y, x) == x );
        CHECK( max(x, y) == y );
        CHECK( max(y, x) == y );

        x = 1.1;
        y = 1.1;

        CHECK( min(x, y) == x );
        CHECK( min(y, x) == x );
        CHECK( max(x, y) == x );
        CHECK( max(y, x) == x );

        x = -7.1;
        y = -9.1;

        CHECK( min(x, y) == y );
        CHECK( min(y, x) == y );
        CHECK( max(x, y) == x );
        CHECK( max(y, x) == x );
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

    SECTION("testing array-unpacking of derivatives for dual number")
    {
        dual4th x;
        detail::seed<0>(x, 2.0);
        detail::seed<1>(x, 3.0);
        detail::seed<2>(x, 4.0);
        detail::seed<3>(x, 5.0);
        detail::seed<4>(x, 6.0);

        auto [x0, x1, x2, x3, x4] = derivatives(x);

        CHECK( x0 == approx(derivative<0>(x)) );
        CHECK( x1 == approx(derivative<1>(x)) );
        CHECK( x2 == approx(derivative<2>(x)) );
        CHECK( x3 == approx(derivative<3>(x)) );
        CHECK( x4 == approx(derivative<4>(x)) );
    }

    SECTION("testing array-unpacking of derivatives for vector of dual numbers")
    {
        dual4th x;
        detail::seed<0>(x, 2.0);
        detail::seed<1>(x, 3.0);
        detail::seed<2>(x, 4.0);
        detail::seed<3>(x, 5.0);
        detail::seed<4>(x, 6.0);

        dual4th y;
        detail::seed<0>(y, 3.0);
        detail::seed<1>(y, 4.0);
        detail::seed<2>(y, 5.0);
        detail::seed<3>(y, 6.0);
        detail::seed<4>(y, 7.0);

        dual4th z;
        detail::seed<0>(z, 4.0);
        detail::seed<1>(z, 5.0);
        detail::seed<2>(z, 6.0);
        detail::seed<3>(z, 7.0);
        detail::seed<4>(z, 8.0);

        std::vector<dual4th> u = { x, y, z };

        auto [u0, u1, u2, u3, u4] = derivatives(u);

        CHECK( u0[0] == approx(derivative<0>(x)) );
        CHECK( u0[1] == approx(derivative<0>(y)) );
        CHECK( u0[2] == approx(derivative<0>(z)) );

        CHECK( u1[0] == approx(derivative<1>(x)) );
        CHECK( u1[1] == approx(derivative<1>(y)) );
        CHECK( u1[2] == approx(derivative<1>(z)) );

        CHECK( u2[0] == approx(derivative<2>(x)) );
        CHECK( u2[1] == approx(derivative<2>(y)) );
        CHECK( u2[2] == approx(derivative<2>(z)) );

        CHECK( u3[0] == approx(derivative<3>(x)) );
        CHECK( u3[1] == approx(derivative<3>(y)) );
        CHECK( u3[2] == approx(derivative<3>(z)) );

        CHECK( u4[0] == approx(derivative<4>(x)) );
        CHECK( u4[1] == approx(derivative<4>(y)) );
        CHECK( u4[2] == approx(derivative<4>(z)) );
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

        CHECK( val(f(x, y)) == approx(val(100 * (x*x - y)*(x*x - y) + (1 - x)*(1 - x))) );
        CHECK( derivative(f, wrt(x), at(x, y)) == approx(val(400*(x*x - y)*x - 2*(1 - x))) );
        CHECK( derivative(f, wrt(y), at(x, y)) == approx(val(-200*(x*x - y))) );
    }
}
