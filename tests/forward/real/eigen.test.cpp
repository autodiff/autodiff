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

// Catch includes
#include <catch2/catch.hpp>

// autodiff includes
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

template<typename T>
auto approx(T&& val) -> Approx
{
    const double epsilon = std::numeric_limits<double>::epsilon() * 100;
    const double margin = 1e-12;
    return Approx(std::forward<T>(val)).epsilon(epsilon).margin(margin);
}


TEST_CASE("testing autodiff::real (with eigen)", "[forward][real][eigen]")
{
    using Eigen::ArrayXd;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    MatrixXd M(3, 3);
    M << 1.0, 2.0, 3.0,
         4.0, 1.0, 6.0,
         7.0, 3.0, 9.0;

    ArrayXreal x(3), y(3), z(3);
    x << 1.0, 2.0, 3.0;

    SECTION("testing gradient computation")
    {
        CHECK_GRADIENT( x.sum() );
        CHECK_GRADIENT( x.exp().sum() );
        CHECK_GRADIENT( x.log().sum() );
        CHECK_GRADIENT( x.tan().sum() );
        CHECK_GRADIENT( (x * x).sum() );
        CHECK_GRADIENT( (x.sin() * x.exp()).sum() );
        CHECK_GRADIENT( (x * x.log()).sum() );
        CHECK_GRADIENT( (x.sin() * x.cos()).sum() );
    }

    SECTION("testing jacobian computation")
    {
        CHECK_JACOBIAN( x );
        CHECK_JACOBIAN( x.exp() );
        CHECK_JACOBIAN( x.log() );
        CHECK_JACOBIAN( x.tan() );
        CHECK_JACOBIAN( (x * x) );
        CHECK_JACOBIAN( (x.sin() * x.exp()) );
        CHECK_JACOBIAN( (x * x.log()) );
        CHECK_JACOBIAN( (x.sin() * x.cos()) );
    }
}
