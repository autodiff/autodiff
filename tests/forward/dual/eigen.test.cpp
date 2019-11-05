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
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;


TEST_CASE("testing autodiff::dual (with eigen)", "[forward][dual][eigen]")
{
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    SECTION("testing array-unpacking of derivatives for eigen vector of dual numbers")
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

        VectorXdual4th u(3);
        u << x, y, z;

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

    SECTION("testing casting to VectorXd")
    {
        VectorXdual x(3);
        x << 0.5, 0.2, 0.3;

        VectorXd y = x.cast<double>();

        for(auto i = 0; i < 3; ++i)
            CHECK( x[i] == approx(y(i)) );
    }

    SECTION("testing casting to MatriXd")
    {
        MatrixXdual x(2, 2);
        x << 0.5, 0.2, 0.3, 0.7;

        MatrixXd y = x.cast<double>();

        for(auto i = 0; i < 2; ++i)
            for(auto j = 0; j < 2; ++j)
                CHECK(x(i, j) == approx(y(i, j)));
    }

    SECTION("testing multiplication of VectorXdual by MatrixXd")
    {
        MatrixXd A(2, 2);
        A << 1.0, 3.0, 5.0, 7.0;

        VectorXdual x(2);

        detail::seed<0>(x[0], 1.0);
        detail::seed<0>(x[1], 2.0);

        detail::seed<1>(x[0], 3.0);
        detail::seed<1>(x[1], 5.0);

        VectorXdual b = A * x;

        CHECK( derivative<0>(b[0]) == approx(7.0) );
        CHECK( derivative<0>(b[1]) == approx(19.0) );

        CHECK( derivative<1>(b[0]) == approx(18.0) );
        CHECK( derivative<1>(b[1]) == approx(50.0) );
    }
}
