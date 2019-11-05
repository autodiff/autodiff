//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2019 Allan Leal
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

// autodiff includes
#include <autodiff/forward/real.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;


#define CHECK_4TH_ORDER_REAL_NUMBERS(a, b) \
    CHECK_APPROX( a[0], b[0] );            \
    CHECK_APPROX( a[1], b[1] );            \
    CHECK_APPROX( a[2], b[2] );            \
    CHECK_APPROX( a[3], b[3] );            \
    CHECK_APPROX( a[4], b[4] );            \

#define CHECK_DERIVATIVES_REAL4TH_WRT(expr)                                       \
{                                                                                 \
    real4th x = 5, y = 7;                                                         \
    auto f = [](const real4th& x, const real4th& y) -> real4th { return expr; };  \
    /* Check directional derivatives of f(x,y) along direction (3, 5) */          \
    auto dfdv = derivatives(f, along(3, 5), at(x, y));                            \
    x[1] = 3.0; y[1] = 5.0; u = expr; x[1] = 0.0; y[1] = 0.0;                     \
    CHECK_APPROX( dfdv[0], u[0] );                                                \
    CHECK_APPROX( dfdv[1], u[1] );                                                \
    CHECK_APPROX( dfdv[2], u[2] );                                                \
    CHECK_APPROX( dfdv[3], u[3] );                                                \
    CHECK_APPROX( dfdv[4], u[4] );                                                \
}

// #define CHECK_DERIVATIVES_REAL4TH_WRT(expr)                                       \
// {                                                                                 \
//     real4th x = 5, y = 7;                                                         \
//     auto f = [](const real4th& x, const real4th& y) -> real4th { return expr; };  \
//     /* Check directional derivatives of f(x,y) wrt x */                           \
//     auto dfdx = derivatives(f, wrt(x), at(x, y));                                 \
//     x[1] = 1.0; u = expr; x[1] = 0.0;                                             \
//     CHECK_APPROX( dfdx[0], u[0] );                                                \
//     CHECK_APPROX( dfdx[1], u[1] );                                                \
//     /* Check directional derivatives of f(x,y) wrt y */                           \
//     auto dfdy = derivatives(f, wrt(y), at(x, y));                                 \
//     y[1] = 1.0; u = expr; y[1] = 0.0;                                             \
//     CHECK_APPROX( dfdy[0], u[0] );                                                \
//     CHECK_APPROX( dfdy[1], u[1] );                                                \
//     /* Check directional derivatives of f(x,y) along direction (3, 5) */          \
//     auto dfdv = derivatives(f, along(3, 5), at(x, y));                            \
//     x[1] = 3.0; y[1] = 5.0; u = expr; x[1] = 0.0; y[1] = 0.0;                     \
//     CHECK_APPROX( dfdv[0], u[0] );                                                \
//     CHECK_APPROX( dfdv[1], u[1] );                                                \
//     CHECK_APPROX( dfdv[2], u[2] );                                                \
//     CHECK_APPROX( dfdv[3], u[3] );                                                \
//     CHECK_APPROX( dfdv[4], u[4] );                                                \
// }

// Auxiliary constants
const auto ln10 = 2.302585092994046;
const auto pi = 3.14159265359;


TEST_CASE("", "")
{
    {
        real x, y;
        seed(at(x, y), along(2, 3));

        CHECK(x[1] == 2.0);
        CHECK(y[1] == 3.0);
    }

    {
        std::vector<real> x(4);
        real y;

        std::vector<double> v = {2.0, 3.0, 4.0, 5.0};

        seed(at(x, y), along(v, 7.0));

        CHECK(x[0][1] == 2.0);
        CHECK(x[1][1] == 3.0);
        CHECK(x[2][1] == 4.0);
        CHECK(x[3][1] == 5.0);
        CHECK(y[1] == 7.0);

        unseed(at(x, y));

        CHECK(x[0][1] == 0.0);
        CHECK(x[1][1] == 0.0);
        CHECK(x[2][1] == 0.0);
        CHECK(x[3][1] == 0.0);
        CHECK(y[1] == 0.0);
    }

    // auto aux1 = at(x, y);
    // auto aux2 = along(2.0, 3.0);

    // CHECK(aux1.numArgs == aux2.numArgs);

}


/**/
TEST_CASE("testing autodiff::real", "[forward][real]")
{
    real4th x, y, z, u, v, w;

    x = 1.0;

    CHECK_APPROX( x[0],  1.0 );
    CHECK_APPROX( x[1],  0.0 );
    CHECK_APPROX( x[2],  0.0 );
    CHECK_APPROX( x[3],  0.0 );
    CHECK_APPROX( x[4],  0.0 );

    x = {0.5, 3.0, -5.0, -15.0, 11.0};

    CHECK_APPROX( x[0],   0.5 );
    CHECK_APPROX( x[1],   3.0 );
    CHECK_APPROX( x[2],  -5.0 );
    CHECK_APPROX( x[3], -15.0 );
    CHECK_APPROX( x[4],  11.0 );

    y = +x;

    CHECK_APPROX( y[0],  x[0] );
    CHECK_APPROX( y[1],  x[1] );
    CHECK_APPROX( y[2],  x[2] );
    CHECK_APPROX( y[3],  x[3] );
    CHECK_APPROX( y[4],  x[4] );

    y = -x;

    CHECK_APPROX( y[0],  -x[0] );
    CHECK_APPROX( y[1],  -x[1] );
    CHECK_APPROX( y[2],  -x[2] );
    CHECK_APPROX( y[3],  -x[3] );
    CHECK_APPROX( y[4],  -x[4] );

    z = x + y;

    CHECK_APPROX( z[0],  x[0] + y[0] );
    CHECK_APPROX( z[1],  x[1] + y[1] );
    CHECK_APPROX( z[2],  x[2] + y[2] );
    CHECK_APPROX( z[3],  x[3] + y[3] );
    CHECK_APPROX( z[4],  x[4] + y[4] );

    z = x + 1.0;

    CHECK_APPROX( z[0],  x[0] + 1.0 );
    CHECK_APPROX( z[1],  x[1] );
    CHECK_APPROX( z[2],  x[2] );
    CHECK_APPROX( z[3],  x[3] );
    CHECK_APPROX( z[4],  x[4] );

    z = 1.0 + x;

    CHECK_APPROX( z[0],  x[0] + 1.0 );
    CHECK_APPROX( z[1],  x[1] );
    CHECK_APPROX( z[2],  x[2] );
    CHECK_APPROX( z[3],  x[3] );
    CHECK_APPROX( z[4],  x[4] );

    z = x - y;

    CHECK_APPROX( z[0],  x[0] - y[0] );
    CHECK_APPROX( z[1],  x[1] - y[1] );
    CHECK_APPROX( z[2],  x[2] - y[2] );
    CHECK_APPROX( z[3],  x[3] - y[3] );
    CHECK_APPROX( z[4],  x[4] - y[4] );

    z = x * y;

    CHECK_APPROX( z[0],  x[0]*y[0] );
    CHECK_APPROX( z[1],  x[1]*y[0] + x[0]*y[1] );
    CHECK_APPROX( z[2],  x[2]*y[0] + 2*x[1]*y[1] + x[0]*y[2] );
    CHECK_APPROX( z[3],  x[3]*y[0] + 3*x[2]*y[1] + 3*x[1]*y[2] + x[0]*y[3] );
    CHECK_APPROX( z[4],  x[4]*y[0] + 4*x[3]*y[1] + 6*x[2]*y[2] + 4*x[1]*y[3] + x[0]*y[4] );

    z = x / y;

    CHECK_APPROX( z[0],  (x[0])/y[0] );
    CHECK_APPROX( z[1],  (x[1] - y[1]*z[0])/y[0] );
    CHECK_APPROX( z[2],  (x[2] - y[2]*z[0] - 2*y[1]*z[1])/y[0] );
    CHECK_APPROX( z[3],  (x[3] - y[3]*z[0] - 3*y[2]*z[1] - 3*y[1]*z[2])/y[0] );
    CHECK_APPROX( z[4],  (x[4] - y[4]*z[0] - 4*y[3]*z[1] - 6*y[2]*z[2] - 4*y[1]*z[3])/y[0] );

    z = 3.0 / y;

    CHECK_APPROX( z[0],  3.0/y[0] );
    CHECK_APPROX( z[1],  -(y[1]*z[0])/y[0] );
    CHECK_APPROX( z[2],  -(y[2]*z[0] + 2*y[1]*z[1])/y[0] );
    CHECK_APPROX( z[3],  -(y[3]*z[0] + 3*y[2]*z[1] + 3*y[1]*z[2])/y[0] );
    CHECK_APPROX( z[4],  -(y[4]*z[0] + 4*y[3]*z[1] + 6*y[2]*z[2] + 4*y[1]*z[3])/y[0] );

    //=====================================================================================================================
    //
    // TESTING EXPONENTIAL AND LOGARITHMIC FUNCTIONS
    //
    //=====================================================================================================================

    y = exp(x);

    CHECK_APPROX( y[0], exp(x[0]) );
    CHECK_APPROX( y[1], x[1] * y[0] );
    CHECK_APPROX( y[2], x[2] * y[0] + x[1] * y[1] );
    CHECK_APPROX( y[3], x[3] * y[0] + 2*x[2] * y[1] + x[1] * y[2] );
    CHECK_APPROX( y[4], x[4] * y[0] + 3*x[3] * y[1] + 3*x[2] * y[2] + x[1] * y[3] );

    y = log(x);

    CHECK_APPROX( y[0], log(x[0]) );
    CHECK_APPROX( y[1], (x[1]) / x[0] );
    CHECK_APPROX( y[2], (x[2] - x[1]*y[1]) / x[0] );
    CHECK_APPROX( y[3], (x[3] - x[2]*y[1] - 2*x[1]*y[2]) / x[0] );
    CHECK_APPROX( y[4], (x[4] - x[3]*y[1] - 3*x[2]*y[2] - 3*x[1]*y[3]) / x[0] );

    y = log10(x);
    z = log(x)/ln10;

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    y = sqrt(x);
    z = exp(0.5 * log(x));

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    y = pow(x, x);
    z = exp(x * log(x));

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    y = pow(x, pi);
    z = exp(pi * log(x));

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    y = pow(pi, x);
    z = exp(x * log(pi));

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    //=====================================================================================================================
    //
    // TESTING TRIGONOMETRIC FUNCTIONS
    //
    //=====================================================================================================================

    y = sin(x);
    z = cos(x);

    CHECK_APPROX( y[0],  sin(x[0]) );
    CHECK_APPROX( z[0],  cos(x[0]) );
    CHECK_APPROX( y[1],   x[1] * z[0] );
    CHECK_APPROX( z[1],  -x[1] * y[0] );
    CHECK_APPROX( y[2],   x[2] * z[0] + x[1] * z[1] );
    CHECK_APPROX( z[2],  -x[2] * y[0] - x[1] * y[1] );
    CHECK_APPROX( y[3],   x[3] * z[0] + 2*x[2] * z[1] + x[1] * z[2] );
    CHECK_APPROX( z[3],  -x[3] * y[0] - 2*x[2] * y[1] - x[1] * y[2] );
    CHECK_APPROX( y[4],   x[4] * z[0] + 3*x[3] * z[1] + 3*x[2] * z[2] + x[1] * z[3] );
    CHECK_APPROX( z[4],  -x[4] * y[0] - 3*x[3] * y[1] - 3*x[2] * y[2] - x[1] * y[3] );

    y = tan(x);
    z = sin(x)/cos(x);

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    y = asin(x);
    z = 1/sqrt(1 - x*x);

    CHECK_APPROX( y[0], asin(x[0]) );
    CHECK_APPROX( y[1], z[0] );
    CHECK_APPROX( y[2], z[1] );
    CHECK_APPROX( y[3], z[2] );
    CHECK_APPROX( y[4], z[3] );

    y = acos(x);
    z = -1/sqrt(1 - x*x);

    CHECK_APPROX( y[0], acos(x[0]) );
    CHECK_APPROX( y[1], z[0] );
    CHECK_APPROX( y[2], z[1] );
    CHECK_APPROX( y[3], z[2] );
    CHECK_APPROX( y[4], z[3] );

    y = atan(x);
    z = 1/(1 + x*x);

    CHECK_APPROX( y[0], atan(x[0]) );
    CHECK_APPROX( y[1], z[0] );
    CHECK_APPROX( y[2], z[1] );
    CHECK_APPROX( y[3], z[2] );
    CHECK_APPROX( y[4], z[3] );

    //=====================================================================================================================
    //
    // TESTING HYPERBOLIC FUNCTIONS
    //
    //=====================================================================================================================
    y = sinh(x);
    z = cosh(x);

    CHECK_APPROX( y[0],  sinh(x[0]) );
    CHECK_APPROX( z[0],  cosh(x[0]) );
    CHECK_APPROX( y[1],  x[1] * z[0] );
    CHECK_APPROX( z[1],  x[1] * y[0] );
    CHECK_APPROX( y[2],  x[2] * z[0] + x[1] * z[1] );
    CHECK_APPROX( z[2],  x[2] * y[0] + x[1] * y[1] );
    CHECK_APPROX( y[3],  x[3] * z[0] + 2*x[2] * z[1] + x[1] * z[2] );
    CHECK_APPROX( z[3],  x[3] * y[0] + 2*x[2] * y[1] + x[1] * y[2] );
    CHECK_APPROX( y[4],  x[4] * z[0] + 3*x[3] * z[1] + 3*x[2] * z[2] + x[1] * z[3] );
    CHECK_APPROX( z[4],  x[4] * y[0] + 3*x[3] * y[1] + 3*x[2] * y[2] + x[1] * y[3] );

    y = tanh(x);
    z = sinh(x)/cosh(x);

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    y = asinh(x);
    z = 1/sqrt(x*x + 1);

    CHECK_APPROX( y[0], asinh(x[0]) );
    CHECK_APPROX( y[1], z[0] );
    CHECK_APPROX( y[2], z[1] );
    CHECK_APPROX( y[3], z[2] );
    CHECK_APPROX( y[4], z[3] );

    y = acosh(10*x); // acosh requires x > 1
    z = 1/sqrt(100*x*x - 1);

    CHECK_APPROX( y[0], acosh(10*x[0]) );
    CHECK_APPROX( y[1], z[0] );
    CHECK_APPROX( y[2], z[1] );
    CHECK_APPROX( y[3], z[2] );
    CHECK_APPROX( y[4], z[3] );

    y = atanh(x);
    z = 1/(1 - x*x);

    CHECK_APPROX( y[0], atanh(x[0]) );
    CHECK_APPROX( y[1], z[0] );
    CHECK_APPROX( y[2], z[1] );
    CHECK_APPROX( y[3], z[2] );
    CHECK_APPROX( y[4], z[3] );

    //=====================================================================================================================
    //
    // TESTING OTHER FUNCTIONS
    //
    //=====================================================================================================================

    y = abs(x);

    CHECK_APPROX( y[0], abs(x[0]) );
    CHECK_APPROX( y[1], abs(x[0])/x[0] * x[1] );
    CHECK_APPROX( y[2], abs(x[0])/x[0] * x[2] );
    CHECK_APPROX( y[3], abs(x[0])/x[0] * x[3] );
    CHECK_APPROX( y[4], abs(x[0])/x[0] * x[4] );

    y = -x;
    z = abs(y);

    CHECK_APPROX( z[0], abs(y[0]) );
    CHECK_APPROX( z[1], abs(y[0])/(y[0]) * y[1] );
    CHECK_APPROX( z[2], abs(y[0])/(y[0]) * y[2] );
    CHECK_APPROX( z[3], abs(y[0])/(y[0]) * y[3] );
    CHECK_APPROX( z[4], abs(y[0])/(y[0]) * y[4] );

    //=====================================================================================================================
    //
    // TESTING DERIVATIVE CALCULATIONS
    //
    //=====================================================================================================================

    CHECK_DERIVATIVES_REAL4TH_WRT( exp(log(2*x + 3*y)) );
    CHECK_DERIVATIVES_REAL4TH_WRT( sin(2*x + 3*y) );
    CHECK_DERIVATIVES_REAL4TH_WRT( exp(2*x + 3*y) * log(x/y) );

    SECTION("testing array-unpacking of derivatives for real number")
    {
        real4th x = {{2.0, 3.0, 4.0, 5.0, 6.0}};

        auto [x0, x1, x2, x3, x4] = derivatives(x);

        CHECK_APPROX( x0, x[0] );
        CHECK_APPROX( x1, x[1] );
        CHECK_APPROX( x2, x[2] );
        CHECK_APPROX( x3, x[3] );
        CHECK_APPROX( x4, x[4] );
    }

    SECTION("testing array-unpacking of derivatives for vector of real numbers")
    {
        real4th x = {{2.0, 3.0, 4.0, 5.0, 6.0}};
        real4th y = {{3.0, 4.0, 5.0, 6.0, 7.0}};
        real4th z = {{4.0, 5.0, 6.0, 7.0, 8.0}};

        std::vector<real4th> u = { x, y, z };

        auto [u0, u1, u2, u3, u4] = derivatives(u);

        CHECK_APPROX( u0[0], x[0] ); CHECK_APPROX( u1[0], x[1] ); CHECK_APPROX( u2[0], x[2] ); CHECK_APPROX( u3[0], x[3] ); CHECK_APPROX( u4[0], x[4] );
        CHECK_APPROX( u0[1], y[0] ); CHECK_APPROX( u1[1], y[1] ); CHECK_APPROX( u2[1], y[2] ); CHECK_APPROX( u3[1], y[3] ); CHECK_APPROX( u4[1], y[4] );
        CHECK_APPROX( u0[2], z[0] ); CHECK_APPROX( u1[2], z[1] ); CHECK_APPROX( u2[2], z[2] ); CHECK_APPROX( u3[2], z[3] ); CHECK_APPROX( u4[2], z[4] );
    }
}
