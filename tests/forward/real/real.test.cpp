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

// autodiff includes
#include <array>
#include <autodiff/common/meta.hpp>
#include <autodiff/forward/real.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <functional>
#include <tests/utils/catch.hpp>
#include <vector>

using namespace autodiff;

constexpr bool equalWithinAbs(const real4th a, const real4th b, const double tol = 1e-15)
{
    for(int i = 0; i < 5; ++i)
        if(std::abs(a[i] - b[i]) > tol)
            return false;
    return true;
}

// Auxiliary constants
constexpr auto ln10 = 2.302585092994046;
constexpr auto pi = 3.14159265359;

TEST_CASE("testing autodiff::real", "[forward][real]")
{
    // test ctor + equality comparison first -> then we can trust checks by equality comparisons of temporaries

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

    SECTION("sinh, cosh functions")
    {
        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});

        const auto y0 = sinh(x[0]);
        const auto z0 = cosh(x[0]);
        const auto y1 = x[1] * z0;
        const auto z1 = x[1] * y0;
        const auto y2 = x[2] * z0 + x[1] * z1;
        const auto z2 = x[2] * y0 + x[1] * y1;
        const auto y3 = x[3] * z0 + 2 * x[2] * z1 + x[1] * z2;
        const auto z3 = x[3] * y0 + 2 * x[2] * y1 + x[1] * y2;
        const auto y4 = x[4] * z0 + 3 * x[3] * z1 + 3 * x[2] * z2 + x[1] * z3;
        const auto z4 = x[4] * y0 + 3 * x[3] * y1 + 3 * x[2] * y2 + x[1] * y3;

        CHECK(equalWithinAbs(sinh(x), real4th({y0, y1, y2, y3, y4})));
        CHECK(equalWithinAbs(cosh(x), real4th({z0, z1, z2, z3, z4})));
    }

    SECTION("tanh function")
    {
        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});
        CHECK(equalWithinAbs(tanh(x), sinh(x) / cosh(x), 1e-12));
    }

    SECTION("asinh function")
    {
        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});
        const auto asinhPrime = 1 / sqrt(x * x + 1);
        real4th expected = asinh(x[0]);
        for(int i = 1; i < 5; ++i)
            expected[i] = asinhPrime[i - 1];
        CHECK(equalWithinAbs(asinh(x), expected, 1e-12));
    }

    SECTION("acosh function")
    {
        constexpr auto x = real4th({1.5, 3.0, +5.0, +15.0, 11.0}); // acosh requires x > 1
        const auto acoshPrime = 1 / sqrt(x * x - 1);
        real4th expected = acosh(x[0]);
        for(int i = 1; i < 5; ++i)
            expected[i] = acoshPrime[i - 1];
        CHECK(equalWithinAbs(acosh(x), expected, 1e-12));
    }

    SECTION("atanh function")
    {
        constexpr auto x = real4th({0.5, 0.3, -0.5, -0.15, 0.11}); // atanh requires |x| < 1
        const auto atanhPrime = 1 / (1 - x * x);
        real4th expected = atanh(x[0]);
        for(int i = 1; i < 5; ++i)
            expected[i] = atanhPrime[i - 1];
        CHECK(equalWithinAbs(atanh(x), expected, 1e-12));
    }

    //=====================================================================================================================
    //
    // TESTING OTHER FUNCTIONS
    //      abs, min, max
    //=====================================================================================================================

    SECTION("abs function")
    {
        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});
        CHECK(abs(x) == x);
        CHECK(abs(-x) == x);
    }

    SECTION("min function")
    {
        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});

        // (real, real)
        {
            constexpr auto y = real4th({4.5, 3.0, -5.0, -15.0, 11.0});
            constexpr auto a = min(x, y);
            constexpr auto b = min(y, x);
            CHECK(a == x);
            CHECK(b == x);
        }

        // (real, smaller scalar)
        {
            constexpr auto y = real4th(0.2);
            constexpr auto a = min(x, y);
            constexpr auto b = min(y, x);
            CHECK(a == y);
            CHECK(b == y);
        }

        // (real, same scalar)
        {
            constexpr auto x0 = real4th(x[0]);
            constexpr auto a = min(x, x0);
            constexpr auto b = min(x0, x);
            CHECK(a == x);
            CHECK(b == x0);
        }

        // (real, larger scalar)
        {
            constexpr auto a = min(x, 3.5);
            constexpr auto b = min(3.5, x);
            CHECK(a == x);
            CHECK(b == x);
        }
    }

    SECTION("max function")
    {
        constexpr auto x = real4th({0.5, 3.0, -5.0, -15.0, 11.0});

        // (real, real)
        {
            constexpr auto y = real4th({4.5, 3.0, -5.0, -15.0, 11.0});
            constexpr auto a = max(x, y);
            constexpr auto b = max(y, x);
            CHECK(a == y);
            CHECK(b == y);
        }

        // (real, smaller scalar)
        {
            constexpr auto a = max(x, 0.2);
            constexpr auto b = max(0.2, x);
            CHECK(a == x);
            CHECK(b == x);
        }

        // (real, same scalar)
        {
            constexpr auto x0 = real4th(x[0]);
            constexpr auto a = max(x, x0);
            constexpr auto b = max(x0, x);
            CHECK(a == x);
            CHECK(b == x0);
        }

        // (real, larger scalar)
        {
            constexpr auto y = real4th(3.5);
            constexpr auto a = max(x, y);
            constexpr auto b = max(y, x);
            CHECK(a == y);
            CHECK(b == y);
        }
    }

    //=====================================================================================================================
    //
    // TESTING COMPARISON OPERATORS
    //
    //=====================================================================================================================

    SECTION("(Non-)Equality comparisons: {real, real}")
    {
        constexpr auto x = real4th({1, -3, 5, -7, 11});

        // compare: x == x
        {
            constexpr auto eq = x == x;
            constexpr auto neq = x != x;
            CHECK(eq);
            CHECK_FALSE(neq);
        }

        const auto unitDeriv = [](int i) constexpr
        {
            real4th unit = 0;
            unit[i] = 1;
            return unit;
        };

        // compare: x == x + unitDeriv(i)*1e-12
        detail::For<5>(
            [&x, unitDeriv](auto i) {
                constexpr auto y = x + unitDeriv(i) * 1e-12;
                constexpr auto eq = x == y;
                constexpr auto neq = x != y;
                CHECK_FALSE(eq);
                CHECK(neq);
            });
    }

    SECTION("(Non-)Equality comparisons: {real, number}")
    {
        constexpr auto x = real4th({1, -3, 5, -7, 11});

        // equal values
        {
            constexpr auto eqRealNum = x == x[0];
            constexpr auto eqNumReal = x[0] == x;
            CHECK(eqRealNum);
            CHECK(eqNumReal);

            constexpr auto neqRealNum = x != x[0];
            constexpr auto neqNumReal = x[0] != x;
            CHECK_FALSE(neqRealNum);
            CHECK_FALSE(neqNumReal);
        }

        // non-equal values
        {
            constexpr auto y0 = x[0] + 1;

            constexpr auto eqRealNum = x == y0;
            constexpr auto eqNumReal = y0 == x;
            CHECK_FALSE(eqRealNum);
            CHECK_FALSE(eqNumReal);

            constexpr auto neqRealNum = x != y0;
            constexpr auto neqNumReal = y0 != x;
            CHECK(neqRealNum);
            CHECK(neqNumReal);
        }
    }

    SECTION("Inequality comparisons: {real, real}")
    {
        constexpr auto x = real4th({1, -3, 5, -7, 11});

        const auto unitDeriv = [](int i) constexpr
        {
            real4th unit = 0;
            unit[i] = 1;
            return unit;
        };

        // idea: perturb "x" in each component and then check all inequalities.
        // all combined in nested loops. a bit complex but results in shorter code ;)
        detail::For<3>(
            [&](auto i_dx0) {
                // y: perturb x just in 0th component. only component important for comparison
                constexpr auto dx0 = static_cast<int>(i_dx0) - 1;
                constexpr auto y = x + unitDeriv(0) * dx0;

                constexpr auto less = x < y;
                CHECK(less == (dx0 > 0));

                constexpr auto greater = x > y;
                CHECK(greater == (dx0 < 0));

                constexpr auto lessEqual = x <= y;
                CHECK(lessEqual == (dx0 >= 0));

                constexpr auto greaterEqual = x >= y;
                CHECK(greaterEqual == (dx0 <= 0));

                // z: perturb y in 1st-4th component. should not change comparison
                detail::For<1, 5>(
                    [&x, &y, unitDeriv](auto i) {
                        detail::For<3>(
                            [&x, &y, i, unitDeriv](auto i_dxi) {
                                constexpr auto dxi = static_cast<int>(i_dxi) - 1; // dxi: -1, 0, +1
                                constexpr auto z = y + unitDeriv(i) * dxi;
                                CHECK((x < y) == (x < z));
                                CHECK((x > y) == (x > z));
                                CHECK((x <= y) == (x <= z));
                                CHECK((x >= y) == (x >= z));
                            });
                    });
            });
    }

    SECTION("Inequality comparisons: {real, number}")
    {
        constexpr auto x = real4th({1, -3, 5, -7, 11});

        const auto unitDeriv = [](int i) constexpr
        {
            real4th unit = 0;
            unit[i] = 1;
            return unit;
        };

        detail::For<3>(
            [&](auto i_dx0) {
                constexpr auto dx0 = static_cast<int>(i_dx0) - 1;
                constexpr auto y0 = x[0] + dx0;

                constexpr auto realLessNum = x < y0;
                constexpr auto numLessReal = y0 < x;
                CHECK(realLessNum == (dx0 > 0));
                CHECK(numLessReal == (dx0 < 0));

                constexpr auto realGreaterNum = x > y0;
                constexpr auto numGreaterReal = y0 > x;
                CHECK(realGreaterNum == dx0 < 0);
                CHECK(numGreaterReal == dx0 > 0);

                constexpr auto realLessEqualNum = x <= y0;
                constexpr auto numLessEqualReal = y0 <= x;
                CHECK(realLessEqualNum == (dx0 >= 0));
                CHECK(numLessEqualReal == (dx0 <= 0));

                constexpr auto realGreaterEqualNum = x >= y0;
                constexpr auto numGreaterEqualReal = y0 >= x;
                CHECK(realGreaterEqualNum == dx0 <= 0);
                CHECK(numGreaterEqualReal == dx0 >= 0);
            });
    }

    //=====================================================================================================================
    //
    // TESTING DERIVATIVE CALCULATIONS
    //
    //=====================================================================================================================

    SECTION("directional derivative of f(x,y)=exp(log(2*x+3*y))")
    {
        constexpr auto x0 = 5, y0 = 7, x1 = 3, y1 = 5;
        const auto f = [](const real4th& x, const real4th& y) -> real4th { return exp(log(2 * x + 3 * y)); };

        const auto dfdv = derivatives(f, along(x1, y1), at(real4th(x0), real4th(y0)));
        const auto u = f(real4th({x0, x1, 0, 0, 0}), real4th({y0, y1, 0, 0, 0}));
        CHECK(equalWithinAbs(u, dfdv, 1e-12));
    }

    SECTION("directional derivative of f(x,y)=sin(2*x+3*y)")
    {
        constexpr auto x0 = 5, y0 = 7, x1 = 3, y1 = 5;
        const auto f = [](const real4th& x, const real4th& y) -> real4th { return sin(2 * x + 3 * y); };

        const auto dfdv = derivatives(f, along(x1, y1), at(real4th(x0), real4th(y0)));
        const auto u = f(real4th({x0, x1, 0, 0, 0}), real4th({y0, y1, 0, 0, 0}));
        CHECK(equalWithinAbs(u, dfdv, 1e-12));
    }

    SECTION("directional derivative of f(x,y)=exp(2*x+3*y)*log(x/y)")
    {
        constexpr auto x0 = 5, y0 = 7, x1 = 3, y1 = 5;
        const auto f = [](const real4th& x, const real4th& y) -> real4th { return exp(2 * x + 3 * y) * log(x / y); };

        const auto dfdv = derivatives(f, along(x1, y1), at(real4th(x0), real4th(y0)));
        const auto u = f(real4th({x0, x1, 0, 0, 0}), real4th({y0, y1, 0, 0, 0}));
        CHECK(equalWithinAbs(u, dfdv, 1e-12));
    }

    SECTION("Testing array-unpacking of derivatives for real number")
    {
        constexpr auto x = real4th({2.0, 3.0, 4.0, 5.0, 6.0});

        const auto [x0, x1, x2, x3, x4] = derivatives(x);
        CHECK(x == real4th({x0, x1, x2, x3, x4}));
    }

    SECTION("Testing array-unpacking of derivatives for vector of real numbers")
    {
        constexpr auto x = real4th({2.0, 3.0, 4.0, 5.0, 6.0});
        constexpr auto y = real4th({3.0, 4.0, 5.0, 6.0, 7.0});
        constexpr auto z = real4th({4.0, 5.0, 6.0, 7.0, 8.0});

        const auto u = std::vector{x, y, z};

        const auto [u0, u1, u2, u3, u4] = derivatives(u);

        for(int i = 0; i < 3; ++i)
            CHECK(u[i] == real4th({u0[i], u1[i], u2[i], u3[i], u4[i]}));
    }
}
