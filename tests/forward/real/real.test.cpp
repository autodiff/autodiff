//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2022 Allan Leal
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
#include <array>
#include <autodiff/forward/real.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <tests/utils/catch.hpp>

using namespace autodiff;

#define CHECK_4TH_ORDER_REAL_NUMBERS(a, b) \
    CHECK_APPROX(a[0], b[0]);              \
    CHECK_APPROX(a[1], b[1]);              \
    CHECK_APPROX(a[2], b[2]);              \
    CHECK_APPROX(a[3], b[3]);              \
    CHECK_APPROX(a[4], b[4]);

constexpr bool equalWithinAbs(const real4th a, const real4th b, const double tol = 1e-15)
{
    for(int i = 0; i < 5; ++i)
        if(std::abs(a[i] - b[i]) > tol)
            return false;
    return true;
}

#define CHECK_DERIVATIVES_REAL4TH_WRT(expr)                                          \
    {                                                                                \
        real4th x = 5, y = 7;                                                        \
        auto f = [](const real4th& x, const real4th& y) -> real4th { return expr; }; \
        /* Check directional derivatives of f(x,y) along direction (3, 5) */         \
        auto dfdv = derivatives(f, along(3, 5), at(x, y));                           \
        x[1] = 3.0;                                                                  \
        y[1] = 5.0;                                                                  \
        u = expr;                                                                    \
        x[1] = 0.0;                                                                  \
        y[1] = 0.0;                                                                  \
        CHECK_APPROX(dfdv[0], u[0]);                                                 \
        CHECK_APPROX(dfdv[1], u[1]);                                                 \
        CHECK_APPROX(dfdv[2], u[2]);                                                 \
        CHECK_APPROX(dfdv[3], u[3]);                                                 \
        CHECK_APPROX(dfdv[4], u[4]);                                                 \
    }

// Auxiliary constants
constexpr auto ln10 = 2.302585092994046;
constexpr auto pi = 3.14159265359;

TEST_CASE("testing autodiff::real", "[forward][real]")
{
    SECTION("Constructors")
    {
        constexpr auto tmp = real4th();

        constexpr auto x = real4th(2);
        CHECK(x[0] == 2);
        CHECK(x[1] == 0);
        CHECK(x[2] == 0);
        CHECK(x[3] == 0);
        CHECK(x[4] == 0);

        constexpr auto y = real4th({1, -3, 5, -7, 11});
        CHECK(y[0] == 1);
        CHECK(y[1] == -3);
        CHECK(y[2] == 5);
        CHECK(y[3] == -7);
        CHECK(y[4] == 11);

        constexpr auto z = real3rd(Real<4, float>({1, -3, 5, -7, 11}));
        CHECK(z[0] == 1);
        CHECK(z[1] == -3);
        CHECK(z[2] == 5);
        CHECK(z[3] == -7);
    }

    SECTION("Equality Comparison")
    {
        constexpr auto x = real4th({1, -3, 5, -7, 11});
        CHECK(x == x);

        for(int i = 0; i < 5; ++i) {
            auto y = x;
            y[i] += 1e-12;
            CHECK_FALSE(x == y);
        }
    }

    // explicitly tested ctor + equality comparison -> now we can trust checks by equality comparisons of temporaries

    SECTION("Assignments")
    {
        real4th x, y;

        x = 1;
        CHECK(x == real4th({1, 0, 0, 0, 0}));

        x = {0.5, 3.0, -5.0, -15.0, 11.0};
        CHECK(x == real4th({0.5, 3.0, -5.0, -15.0, 11.0}));

        y = +x;
        CHECK(y == x);

        y = -x;
        CHECK(y == -x);
    }

    SECTION("Unary plus/minus")
    {
        constexpr auto x = real4th({1, -3, 5, -7, 11});

        constexpr auto plusX = +x;
        CHECK(plusX == x);

        constexpr auto minusX = -x;
        CHECK(minusX == real4th({-1, 3, -5, 7, -11}));
    }

    SECTION("Addition")
    {
        constexpr auto x = real4th({1, -3, 5, -7, 11});
        constexpr auto y = real4th({0.5, 3.0, -5.0, -15.0, 11.0});

        constexpr auto realAddReal = x + y;
        CHECK(realAddReal == real4th({1.5, 0.0, 0.0, -22.0, 22.0}));

        constexpr auto realAddNum = x + 1;
        CHECK(realAddNum == real4th({2, -3, 5, -7, 11}));

        constexpr auto numAddReal = 1 + x;
        CHECK(numAddReal == x + 1);
    }

    SECTION("Subtraction")
    {
        constexpr auto x = real4th({1, -3, 5, -7, 11});
        constexpr auto y = real4th({0.5, 3.0, -5.0, -15.0, 11.0});

        constexpr auto realSubReal = x - y;
        CHECK(realSubReal == real4th({0.5, -6.0, 10.0, 8.0, 0.0}));

        constexpr auto realSubNum = x - 1;
        CHECK(realSubNum == real4th({0, -3, 5, -7, 11}));

        constexpr auto numSubReal = 1 - x;
        CHECK(numSubReal == -(x - 1));
    }

    SECTION("Multiplication")
    {
        using Catch::Matchers::WithinAbs;

        constexpr auto x = real4th({1, -3, 5, -7, 11});
        constexpr auto y = real4th({0.5, 3.0, -5.0, -15.0, 11.0});

        constexpr auto realMulReal = x * y;
        constexpr auto z0 = x[0] * y[0];
        constexpr auto z1 = x[1] * y[0] + x[0] * y[1];
        constexpr auto z2 = x[2] * y[0] + 2 * x[1] * y[1] + x[0] * y[2];
        constexpr auto z3 = x[3] * y[0] + 3 * x[2] * y[1] + 3 * x[1] * y[2] + x[0] * y[3];
        constexpr auto z4 = x[4] * y[0] + 4 * x[3] * y[1] + 6 * x[2] * y[2] + 4 * x[1] * y[3] + x[0] * y[4];
        CHECK(equalWithinAbs(realMulReal, real4th({z0, z1, z2, z3, z4})));

        constexpr auto realMulNum = x * 3;
        CHECK(realMulNum == real4th({3, -9, 15, -21, 33}));

        constexpr auto numMulReal = 5 * x;
        CHECK(numMulReal == real4th({5, -15, 25, -35, 55}));
    }

    SECTION("Division")
    {
        using Catch::Matchers::WithinAbs;

        constexpr auto x = real4th({1, -3, 5, -7, 11});
        constexpr auto y = real4th({0.5, 3.0, -5.0, -15.0, 11.0});

        constexpr auto realDivReal = x / y;
        constexpr auto z0 = x[0] / y[0];
        constexpr auto z1 = (x[1] - y[1] * z0) / y[0];
        constexpr auto z2 = (x[2] - y[2] * z0 - 2 * y[1] * z1) / y[0];
        constexpr auto z3 = (x[3] - y[3] * z0 - 3 * y[2] * z1 - 3 * y[1] * z2) / y[0];
        constexpr auto z4 = (x[4] - y[4] * z0 - 4 * y[3] * z1 - 6 * y[2] * z2 - 4 * y[1] * z3) / y[0];
        CHECK(equalWithinAbs(realDivReal, real4th({z0, z1, z2, z3, z4})));

        constexpr auto numDivReal = 3 / y;
        constexpr auto a0 = 3.0 / y[0];
        constexpr auto a1 = -(y[1] * a0) / y[0];
        constexpr auto a2 = -(y[2] * a0 + 2 * y[1] * a1) / y[0];
        constexpr auto a3 = -(y[3] * a0 + 3 * y[2] * a1 + 3 * y[1] * a2) / y[0];
        constexpr auto a4 = -(y[4] * a0 + 4 * y[3] * a1 + 6 * y[2] * a2 + 4 * y[1] * a3) / y[0];
        CHECK(equalWithinAbs(numDivReal, real4th({a0, a1, a2, a3, a4})));

        constexpr auto realDivNum = y / 5;
        CHECK(equalWithinAbs(realDivNum, real4th({0.1, 0.6, -1.0, -3.0, 2.2})));
    }

    //=====================================================================================================================
    //
    // TESTING EXPONENTIAL AND LOGARITHMIC FUNCTIONS
    //
    //=====================================================================================================================

    SECTION("exp function")
    {
        CHECK(exp(real4th(1.234)) == real4th(std::exp(1.234)));

        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});
        const auto z0 = exp(x[0]); // not constexpr
        const auto z1 = x[1] * z0;
        const auto z2 = x[2] * z0 + x[1] * z1;
        const auto z3 = x[3] * z0 + 2 * x[2] * z1 + x[1] * z2;
        const auto z4 = x[4] * z0 + 3 * x[3] * z1 + 3 * x[2] * z2 + x[1] * z3;
        CHECK(equalWithinAbs(exp(x), real4th({z0, z1, z2, z3, z4})));
    }

    SECTION("log, log10 functions")
    {
        CHECK(log(real4th(1.234)) == real4th(std::log(1.234)));

        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});
        const auto z0 = log(x[0]); // not constexpr
        const auto z1 = x[1] / x[0];
        const auto z2 = (x[2] - x[1] * z1) / x[0];
        const auto z3 = (x[3] - x[2] * z1 - 2 * x[1] * z2) / x[0];
        const auto z4 = (x[4] - x[3] * z1 - 3 * x[2] * z2 - 3 * x[1] * z3) / x[0];
        CHECK(equalWithinAbs(log(x), real4th({z0, z1, z2, z3, z4})));

        CHECK(equalWithinAbs(log10(real4th(1.234)), real4th(std::log10(1.234)), 5e-16));
        CHECK(equalWithinAbs(log10(x), log(x) / ln10));
    }

    SECTION("sqrt function")
    {
        CHECK(sqrt(real4th(1.234)) == real4th(std::sqrt(1.234)));

        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});
        CHECK(equalWithinAbs(sqrt(x), exp(0.5 * log(x))));
    }

    SECTION("cbrt function")
    {
        CHECK(equalWithinAbs(cbrt(real4th(1.234)), real4th(std::cbrt(1.234)), 5e-16));

        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});
        CHECK(equalWithinAbs(cbrt(x), exp(1.0 / 3.0 * log(x))));
    }

    SECTION("pow functions")
    {
        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});

        // real^real
        CHECK(equalWithinAbs(pow(real4th(1.234), real4th(1.234)), real4th(std::pow(1.234, 1.234)), 5e-16));
        CHECK(equalWithinAbs(pow(x, x), exp(x * log(x))));

        // real^number
        CHECK(equalWithinAbs(pow(real4th(1.234), 1.234), real4th(std::pow(1.234, 1.234)), 5e-16));
        CHECK(equalWithinAbs(pow(x, pi), exp(pi * log(x)), 1e-12));

        // number^real
        CHECK(equalWithinAbs(pow(1.234, real4th(1.234)), real4th(std::pow(1.234, 1.234)), 5e-16));
        CHECK(equalWithinAbs(pow(pi, x), exp(x * log(pi)), 1e-12));
    }

    //=====================================================================================================================
    //
    // TESTING TRIGONOMETRIC FUNCTIONS
    //
    //=====================================================================================================================

    SECTION("sin, cos functions")
    {
        CHECK(sin(real4th(1.234)) == real4th(std::sin(1.234)));
        CHECK(cos(real4th(1.234)) == real4th(std::cos(1.234)));

        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});
        const auto y0 = sin(x[0]);
        const auto z0 = cos(x[0]);
        const auto y1 = x[1] * z0;
        const auto z1 = -x[1] * y0;
        const auto y2 = x[2] * z0 + x[1] * z1;
        const auto z2 = -x[2] * y0 - x[1] * y1;
        const auto y3 = x[3] * z0 + 2 * x[2] * z1 + x[1] * z2;
        const auto z3 = -x[3] * y0 - 2 * x[2] * y1 - x[1] * y2;
        const auto y4 = x[4] * z0 + 3 * x[3] * z1 + 3 * x[2] * z2 + x[1] * z3;
        const auto z4 = -x[4] * y0 - 3 * x[3] * y1 - 3 * x[2] * y2 - x[1] * y3;
        CHECK(equalWithinAbs(sin(x), real4th({y0, y1, y2, y3, y4})));
        CHECK(equalWithinAbs(cos(x), real4th({z0, z1, z2, z3, z4})));
    }

    SECTION("tan function")
    {
        CHECK(tan(real4th(1.234)) == real4th(std::tan(1.234)));
        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});
        CHECK(equalWithinAbs(tan(x), sin(x) / cos(x), 1e-13));
    }

    SECTION("asin function")
    {
        CHECK(asin(real4th(0.1234)) == real4th(std::asin(0.1234)));

        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});
        constexpr auto xPrime = real3rd({x[1], x[2], x[3], x[4]});
        const auto asinPrime = xPrime / sqrt(1 - real3rd(x) * real3rd(x));
        real4th expected = asin(x[0]);
        for(int i = 1; i < 5; ++i)
            expected[i] = asinPrime[i - 1];
        CHECK(equalWithinAbs(asin(x), expected, 1e-12));
    }

    SECTION("acos function")
    {
        CHECK(acos(real4th(0.1234)) == real4th(std::acos(0.1234)));

        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});
        constexpr auto xPrime = real3rd({x[1], x[2], x[3], x[4]});
        const auto acosPrime = -xPrime / sqrt(1 - real3rd(x) * real3rd(x));
        real4th expected = acos(x[0]);
        for(int i = 1; i < 5; ++i)
            expected[i] = acosPrime[i - 1];
        CHECK(equalWithinAbs(acos(x), expected, 1e-12));
    }

    SECTION("atan function")
    {
        CHECK(atan(real4th(1.234)) == real4th(std::atan(1.234)));

        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});
        constexpr auto xPrime = real3rd({x[1], x[2], x[3], x[4]});
        const auto atanPrime = xPrime / (1 + real3rd(x) * real3rd(x));
        real4th expected = atan(x[0]);
        for(int i = 1; i < 5; ++i)
            expected[i] = atanPrime[i - 1];
        CHECK(equalWithinAbs(atan(x), expected, 1e-12));
    }

    SECTION("atan2 functions")
    {
        CHECK(atan2(real4th(1.234), real4th(5.678)) == real4th(std::atan2(1.234, 5.678)));

        constexpr auto c = 2.0;
        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});
        constexpr auto xPrime = real3rd({x[1], x[2], x[3], x[4]});

        // atan2(double, real4th)
        {
            real4th expected = atan2(c, x[0]);
            constexpr auto atan2Prime = xPrime * (-c / (c * c + real3rd(x) * real3rd(x)));
            for(int i = 1; i < 5; ++i)
                expected[i] = atan2Prime[i - 1];
            CHECK(equalWithinAbs(atan2(c, x), expected, 1e-12));
        }

        // atan2(real4th, double)
        {
            real4th expected = atan2(x[0], c);
            constexpr auto atan2Prime = xPrime * (c / (c * c + real3rd(x) * real3rd(x)));
            for(int i = 1; i < 5; ++i)
                expected[i] = atan2Prime[i - 1];
            CHECK(equalWithinAbs(atan2(x, c), expected, 1e-12));
        }

        // atan2(real4th, real4th)
        {
            constexpr auto y = real4th({1, -3, 5, -7, 11});
            constexpr auto yPrime = real3rd({y[1], y[2], y[3], y[4]});

            real4th expected = atan2(y[0], x[0]);
            constexpr auto atan2Prime = (x[0] * yPrime - y[0] * xPrime) / (x[0] * x[0] + y[0] * y[0]);
            for(int i = 1; i < 5; ++i)
                expected[i] = atan2Prime[i - 1];
            CHECK(equalWithinAbs(atan2(y, x), expected, 1e-12));
        }
    }

    //=====================================================================================================================
    //
    // TESTING HYPERBOLIC FUNCTIONS
    //
    //=====================================================================================================================

    // const auto a = acos(x);
    // const auto b = ;
    // for(int i = 0; i < 5; ++i)
    //     std::cout << a[i] << ' ' << b[i] << ' ' << a[i] - b[i] << '\n';
    real4th x, y, z, u, v, w;

    x = {0.5, 3.0, -5.0, -15.0, 11.0};
    y = -x;
    z = y / 5.0;

    y = sinh(x);
    z = cosh(x);

    CHECK_APPROX(y[0], sinh(x[0]));
    CHECK_APPROX(z[0], cosh(x[0]));
    CHECK_APPROX(y[1], x[1] * z[0]);
    CHECK_APPROX(z[1], x[1] * y[0]);
    CHECK_APPROX(y[2], x[2] * z[0] + x[1] * z[1]);
    CHECK_APPROX(z[2], x[2] * y[0] + x[1] * y[1]);
    CHECK_APPROX(y[3], x[3] * z[0] + 2 * x[2] * z[1] + x[1] * z[2]);
    CHECK_APPROX(z[3], x[3] * y[0] + 2 * x[2] * y[1] + x[1] * y[2]);
    CHECK_APPROX(y[4], x[4] * z[0] + 3 * x[3] * z[1] + 3 * x[2] * z[2] + x[1] * z[3]);
    CHECK_APPROX(z[4], x[4] * y[0] + 3 * x[3] * y[1] + 3 * x[2] * y[2] + x[1] * y[3]);

    y = tanh(x);
    z = sinh(x) / cosh(x);

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    y = asinh(x);
    z = 1 / sqrt(x * x + 1);

    CHECK_APPROX(y[0], asinh(x[0]));
    CHECK_APPROX(y[1], z[0]);
    CHECK_APPROX(y[2], z[1]);
    CHECK_APPROX(y[3], z[2]);
    CHECK_APPROX(y[4], z[3]);

    y = acosh(10 * x); // acosh requires x > 1
    z = 1 / sqrt(100 * x * x - 1);

    CHECK_APPROX(y[0], acosh(10 * x[0]));
    CHECK_APPROX(y[1], z[0]);
    CHECK_APPROX(y[2], z[1]);
    CHECK_APPROX(y[3], z[2]);
    CHECK_APPROX(y[4], z[3]);

    y = atanh(x);
    z = 1 / (1 - x * x);

    CHECK_APPROX(y[0], atanh(x[0]));
    CHECK_APPROX(y[1], z[0]);
    CHECK_APPROX(y[2], z[1]);
    CHECK_APPROX(y[3], z[2]);
    CHECK_APPROX(y[4], z[3]);

    //=====================================================================================================================
    //
    // TESTING OTHER FUNCTIONS
    //
    //=====================================================================================================================

    y = abs(x);

    CHECK_APPROX(y[0], std::abs(x[0]));
    CHECK_APPROX(y[1], std::abs(x[0]) / x[0] * x[1]);
    CHECK_APPROX(y[2], std::abs(x[0]) / x[0] * x[2]);
    CHECK_APPROX(y[3], std::abs(x[0]) / x[0] * x[3]);
    CHECK_APPROX(y[4], std::abs(x[0]) / x[0] * x[4]);

    y = -x;
    z = abs(y);

    CHECK_APPROX(z[0], std::abs(y[0]));
    CHECK_APPROX(z[1], std::abs(y[0]) / (y[0]) * y[1]);
    CHECK_APPROX(z[2], std::abs(y[0]) / (y[0]) * y[2]);
    CHECK_APPROX(z[3], std::abs(y[0]) / (y[0]) * y[3]);
    CHECK_APPROX(z[4], std::abs(y[0]) / (y[0]) * y[4]);

    //=====================================================================================================================
    //
    // TESTING MIN/MAX FUNCTIONS
    //
    //=====================================================================================================================

    x = {0.5, 3.0, -5.0, -15.0, 11.0};
    y = {4.5, 3.0, -5.0, -15.0, 11.0};

    z = min(x, y);
    CHECK(z == x);
    z = min(y, x);
    CHECK(z == x);
    z = min(x, 0.1);
    CHECK(z == real4th(0.1));
    z = min(0.2, x);
    CHECK(z == real4th(0.2));
    z = min(0.5, x);
    CHECK(z == x);
    z = min(x, 0.5);
    CHECK(z == x);
    z = min(3.5, x);
    CHECK(z == x);
    z = min(x, 3.5);
    CHECK(z == x);
    z = max(x, y);
    CHECK(z == y);
    z = max(y, x);
    CHECK(z == y);
    z = max(x, 0.1);
    CHECK(z == x);
    z = max(0.2, x);
    CHECK(z == x);
    z = max(0.5, x);
    CHECK(z == x);
    z = max(x, 0.5);
    CHECK(z == x);
    z = max(8.5, x);
    CHECK(z == real4th(8.5));
    z = max(x, 8.5);
    CHECK(z == real4th(8.5));

    //=====================================================================================================================
    //
    // TESTING COMPARISON OPERATORS
    //
    //=====================================================================================================================

    x = {0.5, 3.0, -5.0, -15.0, 11.0};

    // Check equality not only on value but also on the derivatives
    CHECK(x == real4th({0.5, 3.0, -5.0, -15.0, 11.0}));

    // Check equality against plain numeric types (double) do not require check against derivatives
    CHECK_FALSE(x == 0.6);
    CHECK(x == 0.5);

    // Check inequalities
    CHECK_FALSE(x == real4th({0.5, 3.1, -5.0, -15.0, 11.0}));
    CHECK(x != real4th({0.5, 3.1, -5.0, -15.0, 11.0}));
    CHECK(x != 1.0);
    CHECK(x < 1.0);
    CHECK(x > 0.1);
    CHECK(x <= 1.0);
    CHECK(x >= 0.1);
    CHECK(x <= 0.5);
    CHECK(x >= 0.5);

    //=====================================================================================================================
    //
    // TESTING DERIVATIVE CALCULATIONS
    //
    //=====================================================================================================================

    CHECK_DERIVATIVES_REAL4TH_WRT(exp(log(2 * x + 3 * y)));
    CHECK_DERIVATIVES_REAL4TH_WRT(sin(2 * x + 3 * y));
    CHECK_DERIVATIVES_REAL4TH_WRT(exp(2 * x + 3 * y) * log(x / y));

    // Testing array-unpacking of derivatives for real number
    {
        real4th x = {{2.0, 3.0, 4.0, 5.0, 6.0}};

        auto [x0, x1, x2, x3, x4] = derivatives(x);

        CHECK_APPROX(x0, x[0]);
        CHECK_APPROX(x1, x[1]);
        CHECK_APPROX(x2, x[2]);
        CHECK_APPROX(x3, x[3]);
        CHECK_APPROX(x4, x[4]);
    }

    // Testing array-unpacking of derivatives for vector of real numbers
    {
        real4th x = {{2.0, 3.0, 4.0, 5.0, 6.0}};
        real4th y = {{3.0, 4.0, 5.0, 6.0, 7.0}};
        real4th z = {{4.0, 5.0, 6.0, 7.0, 8.0}};

        std::vector<real4th> u = {x, y, z};

        auto [u0, u1, u2, u3, u4] = derivatives(u);

        CHECK_APPROX(u0[0], x[0]);
        CHECK_APPROX(u1[0], x[1]);
        CHECK_APPROX(u2[0], x[2]);
        CHECK_APPROX(u3[0], x[3]);
        CHECK_APPROX(u4[0], x[4]);
        CHECK_APPROX(u0[1], y[0]);
        CHECK_APPROX(u1[1], y[1]);
        CHECK_APPROX(u2[1], y[2]);
        CHECK_APPROX(u3[1], y[3]);
        CHECK_APPROX(u4[1], y[4]);
        CHECK_APPROX(u0[2], z[0]);
        CHECK_APPROX(u1[2], z[1]);
        CHECK_APPROX(u2[2], z[2]);
        CHECK_APPROX(u3[2], z[3]);
        CHECK_APPROX(u4[2], z[4]);
    }
}
