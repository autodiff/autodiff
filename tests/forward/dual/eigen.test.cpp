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
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

template<typename T>
auto approx(T&& expr) -> Approx
{
    const double zero = std::numeric_limits<double>::epsilon();
    return Approx(val(std::forward<T>(expr))).margin(zero);
}

TEST_CASE("testing autodiff::dual (with eigen)", "[forward][dual][eigen]")
{
    using Eigen::VectorXd;
    using Eigen::MatrixXd;

    SECTION("testing gradient derivatives")
    {
        // Testing complex function involving sin and cos
        auto f = [](const VectorXdual& x) -> dual
        {
            return 0.5 * ( x.array() * x.array() ).sum();
        };

        VectorXdual x(3);
        x << 1.0, 2.0, 3.0;

        VectorXd g = gradient(f, wrt(x), at(x));

        REQUIRE( g[0] == approx(x[0]) );
        REQUIRE( g[1] == approx(x[1]) );
        REQUIRE( g[2] == approx(x[2]) );
    }

    SECTION("testing gradient derivatives of only the last two variables")
    {
        // Testing complex function involving sin and cos
        auto f = [](const VectorXdual& x) -> dual
        {
            return 0.5 * ( x.array() * x.array() ).sum();
        };

        VectorXdual x(3);
        x << 1.0, 2.0, 3.0;

        VectorXd g = gradient(f, wrt(x.tail(2)), at(x));

        REQUIRE( g[0] == approx(x[1]) );
        REQUIRE( g[1] == approx(x[2]) );
    }

    SECTION("testing jacobian derivatives")
    {
        // Testing complex function involving sin and cos
        auto f = [](const VectorXdual& x) -> VectorXdual
        {
            return x / x.array().sum();
        };

        VectorXdual x(3);
        x << 0.5, 0.2, 0.3;

        VectorXdual F;

        const MatrixXd J = jacobian(f, wrt(x), at(x), F);

        for(auto i = 0; i < 3; ++i)
            for(auto j = 0; j < 3; ++j)
                REQUIRE( J(i, j) == approx(-F[i] + ((i == j) ? 1.0 : 0.0)) );
    }

    SECTION("testing jacobian derivatives of only the last two variables")
    {
        // Testing complex function involving sin and cos
        auto f = [](const VectorXdual& x) -> VectorXdual
        {
            return x / x.array().sum();
        };

        VectorXdual x(3);
        x << 0.5, 0.2, 0.3;

        VectorXdual F;

        const MatrixXd J = jacobian(f, wrt(x.tail(2)), at(x), F);

        for(auto i = 0; i < 3; ++i)
            for(auto j = 0; j < 2; ++j)
                REQUIRE( J(i, j) == approx(-F[i] + ((i == j + 1) ? 1.0 : 0.0)) );
    }

    SECTION("testing casting to VectorXd")
    {
	VectorXdual x(3);
        x << 0.5, 0.2, 0.3;
	VectorXd y = x.cast<double>();

	for(auto i = 0; i < 3; ++i)
	    REQUIRE( x(i) == approx(y(i)) );
    }

    SECTION("testing casting to VectorXf")
    {
        MatrixXdual x(2, 2);
        x << 0.5, 0.2, 0.3, 0.7;
        MatrixXd y = x.cast<double>();
        for(auto i = 0; i < 2; ++i)
            for(auto j = 0; j < 2; ++j)
                REQUIRE(x(i, j) == approx(y(i, j)));
    }

    SECTION("testing array-unpacking of derivatives for eigen vector of dual numbers")
    {
        using dual4th = HigherOrderDual<4>;

        AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(dual4th, dual4th);

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
}
