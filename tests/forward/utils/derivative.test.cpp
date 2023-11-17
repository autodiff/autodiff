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

// C++ includes
#include <vector>

// Catch includes
#include <catch2/catch_test_macros.hpp>

// autodiff includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;


TEST_CASE("testing forward derivative module", "[forward][utils][derivative]")
{
    SECTION("testing seed operations for higher-order cross derivatives...")
    {
        dual4th x, y;

        seed(wrt(x, y, x, y));

        CHECK( val(x.grad) == 1.0 );
        CHECK( val(x.val.val.grad) == 1.0 );

        CHECK( val(y.val.grad) == 1.0 );
        CHECK( val(y.val.val.val.grad) == 1.0 );
    }

    SECTION("testing seed operations for higher-order directional derivatives using real4th...")
    {
        real4th x, y;

        seed(at(x, y), along(2, 3));

        CHECK( x[1] == 2.0 );
        CHECK( y[1] == 3.0 );
    }

    SECTION("testing seed operations for higher-order directional derivatives using dual4th...")
    {
        dual4th x, y;

        seed(at(x, y), along(2, 3));

        CHECK( derivative<1>(x) == 2.0 );
        CHECK( derivative<1>(y) == 3.0 );
    }

    SECTION("testing seed operations for higher-order directional derivatives using std::vector...")
    {
        std::vector<real4th> x(4);

        real4th y;

        std::vector<double> v = {2.0, 3.0, 4.0, 5.0};

        seed(at(x, y), along(v, 7.0));

        CHECK( x[0][1] == 2.0 );
        CHECK( x[1][1] == 3.0 );
        CHECK( x[2][1] == 4.0 );
        CHECK( x[3][1] == 5.0 );
        CHECK( y[1] == 7.0 );

        unseed(at(x, y));

        CHECK( x[0][1] == 0.0 );
        CHECK( x[1][1] == 0.0 );
        CHECK( x[2][1] == 0.0 );
        CHECK( x[3][1] == 0.0 );
        CHECK( y[1] == 0.0 );
    }
}
