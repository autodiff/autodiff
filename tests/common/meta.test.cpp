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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Catch includes
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// autodiff includes
#include <array>
#include <autodiff/common/meta.hpp>

using namespace autodiff::detail;

TEST_CASE("testing autodiff meta utilities", "[utils][meta]")
{
    SECTION("Index")
    {
        CHECK(7 == std::size_t(Index<7>{}));
    }

    SECTION("For")
    {
        auto arr = std::array<int, 5>{};
        arr.fill(0);
        auto counter = 0;
        auto setArrayElement = [&arr, &counter](auto i) { arr[i] = ++counter; };
        For<2, 5>(setArrayElement);
        CHECK(arr == std::array{0, 0, 1, 2, 3});
    }

    SECTION("ReverseFor")
    {
        auto arr = std::array<int, 5>{};
        arr.fill(0);
        auto counter = 0;
        auto setArrayElement = [&arr, &counter](auto i) { arr[i] = ++counter; };
        ReverseFor<2, 5>(setArrayElement);
        CHECK(arr == std::array{0, 0, 3, 2, 1});
    }

    SECTION("ForEach_oneTuple")
    {
        constexpr auto tuple = std::make_tuple(1.0, -2.0f, 5);
        auto sum = 0.0;
        auto addToSum = [&sum](auto x) { sum += x; };
        ForEach(tuple, addToSum);
        CHECK_THAT(4.0, Catch::Matchers::WithinAbs(sum, 1e-10));
    }

    SECTION("ForEach_twoTuples")
    {
        constexpr auto tuple0 = std::make_tuple(1.0, -2.0f, 5);
        constexpr auto tuple1 = std::make_tuple(2, -5.0, -7.0f);
        auto sum = 0.0;
        auto addFirstSubtractSecond = [&sum](auto x, auto y) { sum += x - y; };
        ForEach(tuple0, tuple1, addFirstSubtractSecond);
        constexpr auto expectedSum = (1 - 2 + 5) - (2 - 5 - 7);
        CHECK_THAT(expectedSum, Catch::Matchers::WithinAbs(sum, 1e-10));
    }

    SECTION("Sum")
    {
        auto doSquare = [](auto x) { return x * x; };
        constexpr auto sum = Sum<2, 7>(doSquare);
        constexpr auto expectedSum = 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6;
        CHECK(expectedSum == sum);
    }

    SECTION("Reduce")
    {
        constexpr auto tuple = std::make_tuple(1.0, -2.0f, 5);
        auto doSquare = [](auto x) { return x * x; };
        constexpr auto res = Reduce(tuple, doSquare);
        constexpr auto expectedRes = 1 * 1 + 2 * 2 + 5 * 5;
        CHECK(expectedRes == res);
    }
}
