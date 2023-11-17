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
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;


TEST_CASE("testing autodiff::real (with eigen)", "[forward][real][eigen]")
{
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    SECTION("testing array-unpacking of derivatives for eigen vector of real numbers")
    {
        real4th x = {{ 2.0, 3.0, 4.0, 5.0, 6.0 }};
        real4th y = {{ 3.0, 4.0, 5.0, 6.0, 7.0 }};
        real4th z = {{ 4.0, 5.0, 6.0, 7.0, 8.0 }};

        VectorXreal4th u(3);
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

        auto u0th = derivative<0>(u);
        auto u1th = derivative<1>(u);
        auto u2th = derivative<2>(u);
        auto u3th = derivative<3>(u);
        auto u4th = derivative<4>(u);

        CHECK( u0th.size() == 3);
        CHECK( u1th.size() == 3);
        CHECK( u2th.size() == 3);
        CHECK( u3th.size() == 3);
        CHECK( u4th.size() == 3);

        CHECK( u0th[0] == derivative<0>(x) );
        CHECK( u0th[1] == derivative<0>(y) );
        CHECK( u0th[2] == derivative<0>(z) );

        CHECK( u1th[0] == derivative<1>(x) );
        CHECK( u1th[1] == derivative<1>(y) );
        CHECK( u1th[2] == derivative<1>(z) );

        CHECK( u2th[0] == derivative<2>(x) );
        CHECK( u2th[1] == derivative<2>(y) );
        CHECK( u2th[2] == derivative<2>(z) );

        CHECK( u3th[0] == derivative<3>(x) );
        CHECK( u3th[1] == derivative<3>(y) );
        CHECK( u3th[2] == derivative<3>(z) );

        CHECK( u4th[0] == derivative<4>(x) );
        CHECK( u4th[1] == derivative<4>(y) );
        CHECK( u4th[2] == derivative<4>(z) );

        CHECK( u0th[0] == grad<0>(x) );
        CHECK( u0th[1] == grad<0>(y) );
        CHECK( u0th[2] == grad<0>(z) );

        CHECK( u1th[0] == grad<1>(x) );
        CHECK( u1th[1] == grad<1>(y) );
        CHECK( u1th[2] == grad<1>(z) );

        CHECK( u2th[0] == grad<2>(x) );
        CHECK( u2th[1] == grad<2>(y) );
        CHECK( u2th[2] == grad<2>(z) );

        CHECK( u3th[0] == grad<3>(x) );
        CHECK( u3th[1] == grad<3>(y) );
        CHECK( u3th[2] == grad<3>(z) );

        CHECK( u4th[0] == grad<4>(x) );
        CHECK( u4th[1] == grad<4>(y) );
        CHECK( u4th[2] == grad<4>(z) );
    }

    SECTION("testing casting to VectorXd")
    {
        VectorXreal x(3);
        x << 0.5, 0.2, 0.3;

        VectorXd y = x.cast<double>();

        for(auto i = 0; i < 3; ++i)
            CHECK( x[i] == approx(y(i)) );
    }

    SECTION("testing casting to MatriXd")
    {
        MatrixXreal x(2, 2);
        x << 0.5, 0.2, 0.3, 0.7;

        MatrixXd y = x.cast<double>();

        for(auto i = 0; i < 2; ++i)
            for(auto j = 0; j < 2; ++j)
                CHECK(x(i, j) == approx(y(i, j)));
    }

    SECTION("testing multiplication of VectorXreal by MatrixXd")
    {
        MatrixXd A(2, 2);
        A << 1.0, 3.0, 5.0, 7.0;

        VectorXreal x(2);
        x[0] = { 1.0, 3.0 };
        x[1] = { 2.0, 5.0 };

        VectorXreal b = A * x;

        CHECK( derivative<0>(b[0]) == approx(7.0) );
        CHECK( derivative<0>(b[1]) == approx(19.0) );

        CHECK( derivative<1>(b[0]) == approx(18.0) );
        CHECK( derivative<1>(b[1]) == approx(50.0) );
    }

    SECTION("testing gradient derivatives")
    {
        SECTION("using VectorXreal")
        {
            auto f = [](const VectorXreal& x) -> real
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorXreal x(3);
            x << 1.0, 2.0, 3.0;

            VectorXd g = gradient(f, wrt(x), at(x));

            CHECK( g[0] == approx(x[0]) );
            CHECK( g[1] == approx(x[1]) );
            CHECK( g[2] == approx(x[2]) );
        }

        SECTION("using Eigen::Map<VectorXreal>")
        {
            auto f = [](const VectorXreal& x) -> real
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorXreal vec(3);
            vec << 1.0, 2.0, 3.0;

            auto x = Eigen::Map<VectorXreal>(vec.data(), vec.size());

            VectorXd g = gradient(f, wrt(x), at(x));

            CHECK( g[0] == approx(x[0]) );
            CHECK( g[1] == approx(x[1]) );
            CHECK( g[2] == approx(x[2]) );
        }
    }

    SECTION("testing jacobian derivatives")
    {
        SECTION("using VectorXreal")
        {
            auto f = [](const VectorXreal& x) -> VectorXreal
            {
                return x / x.array().sum();
            };

            VectorXreal vec(3);
            vec << 0.5, 0.2, 0.3;

            auto x = Eigen::Map<VectorXreal>(vec.data(), vec.size());

            VectorXreal F;

            const MatrixXd J = jacobian(f, wrt(x), at(x), F);

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 3; ++j)
                    CHECK( J(i, j) == approx(-F[i] + ((i == j) ? 1.0 : 0.0)) );
        }

        SECTION("using Eigen::Map<VectorXreal>")
        {
            auto f = [](const VectorXreal& x) -> VectorXreal
            {
                return x / x.array().sum();
            };

            VectorXreal x(3);
            x << 0.5, 0.2, 0.3;

            VectorXreal F;

            const MatrixXd J = jacobian(f, wrt(x), at(x), F);

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 3; ++j)
                    CHECK( J(i, j) == approx(-F[i] + ((i == j) ? 1.0 : 0.0)) );
        }
    }
}
