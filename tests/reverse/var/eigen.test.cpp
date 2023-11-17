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
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

using autodiff::gradient;
using autodiff::hessian;
using autodiff::val;
using autodiff::var;
using autodiff::VectorXvar;

using Eigen::MatrixXd;
using Eigen::VectorXd;

template<typename Var>
auto approx(const Var& x) -> Catch::Approx
{
    return Catch::Approx(val(x));
}

TEST_CASE("testing autodiff::var (with eigen)", "[reverse][var][eigen]")
{
    var y;
    VectorXd g;
    MatrixXd H;
    VectorXvar x(5);
    x << 1, 2, 3, 4, 5;
    // x.setConstant(3.0); // In Eigen 3.4.0+, this call causes all elements in x to be the same var object, and thus the tests below fail!

    //--------------------------------------------------------------------------
    // TESTING GRADIENT AND HESSIAN WHEN y = sum(x)
    //--------------------------------------------------------------------------
    y = x.sum();
    g = gradient(y, x);

    CHECK( val(y) == approx(15.0) );
    for(auto i = 0; i < x.size(); ++i)
        CHECK( g[i] == approx(1.0) );

    H = hessian(y, x, g);
    for(auto i = 0; i < x.size(); ++i) {
        CHECK( val(g[i]) == approx(1.0) );
        for(auto j = 0; j < x.size(); ++j)
            CHECK( H(i, j) == approx(0.0) );
    }

    //--------------------------------------------------------------------------
    // TESTING GRADIENT AND HESSIAN WHEN y = ||x||^2
    //--------------------------------------------------------------------------
    x << 1, 2, 3, 4, 5;
    y = x.cwiseProduct(x).sum();
    g = gradient(y, x);

    CHECK( val(y) == approx(1 + 2*2 + 3*3 + 4*4 + 5*5) );
    for(auto i = 0; i < x.size(); ++i)
        CHECK( val(g[i]) == approx(2 * x[i]) );

    H = hessian(y, x, g);
    for(auto i = 0; i < x.size(); ++i) {
        CHECK( val(g[i]) == approx(2 * x[i]) );
        for(auto j = 0; j < x.size(); ++j)
            CHECK( H(i, j) == approx(i == j ? 2.0 : 0.0) );
    }

    //--------------------------------------------------------------------------
    // TESTING GRADIENT AND HESSIAN WHEN y = prod(sin(x))
    //--------------------------------------------------------------------------
    y = x.array().sin().prod();
    g = gradient(y, x);

    CHECK( val(y) == approx(sin(1) * sin(2) * sin(3) * sin(4) * sin(5)) );
    for(auto i = 0; i < x.size(); ++i)
        CHECK( val(g[i]) == approx(y / tan(x[i])) );

    H = hessian(y, x, g);
    for(auto i = 0; i < x.size(); ++i) {
        CHECK( val(g[i]) == approx(y / tan(x[i])) );
        for(auto j = 0; j < x.size(); ++j)
            if(i == j)
                CHECK( H(i, j) == Catch::Approx(val(g[i] / tan(x[i]) * (1.0 - 1.0/(cos(x[i]) * cos(x[i]))))) );
            else
                CHECK( H(i, j) == Catch::Approx(val(g[j] / tan(x[i]))) );
    }

    //--------------------------------------------------------------------------
    // TESTING GRADIENT AND HESSIAN WHEN y = sum(diff(x).^2)
    //--------------------------------------------------------------------------
    x << 1, 1, 1, 1, 1;
    y = (x.head(4) - x.tail(4)).array().pow(2).sum();
    g = gradient(y, x);

    CHECK( val(y) == approx(0.0) );
    CHECK( val(g[0]) == approx( 2*x[0] - 2*x[1]) );
    CHECK( val(g[1]) == approx(-2*x[0] + 4*x[1] - 2*x[2]) );
    CHECK( val(g[2]) == approx(-2*x[1] + 4*x[2] - 2*x[3]) );
    CHECK( val(g[3]) == approx(-2*x[2] + 4*x[3] - 2*x[4]) );
    CHECK( val(g[4]) == approx(-2*x[3] + 2*x[4]) );

    H = hessian(y, x, g);
    CHECK( val(g[0]) == approx( 2*x[0] - 2*x[1]) );
    CHECK( val(g[1]) == approx(-2*x[0] + 4*x[1] - 2*x[2]) );
    CHECK( val(g[2]) == approx(-2*x[1] + 4*x[2] - 2*x[3]) );
    CHECK( val(g[3]) == approx(-2*x[2] + 4*x[3] - 2*x[4]) );
    CHECK( val(g[4]) == approx(-2*x[3] + 2*x[4]) );

    CHECK( H(0, 0) == approx( 2.0) );
    CHECK( H(0, 1) == approx(-2.0) );
    CHECK( H(0, 2) == approx( 0.0) );
    CHECK( H(0, 3) == approx( 0.0) );
    CHECK( H(0, 4) == approx( 0.0) );
    CHECK( H(1, 0) == approx(-2.0) );
    CHECK( H(1, 1) == approx( 4.0) );
    CHECK( H(1, 2) == approx(-2.0) );
    CHECK( H(1, 3) == approx( 0.0) );
    CHECK( H(1, 4) == approx( 0.0) );
    CHECK( H(2, 0) == approx( 0.0) );
    CHECK( H(2, 1) == approx(-2.0) );
    CHECK( H(2, 2) == approx( 4.0) );
    CHECK( H(2, 3) == approx(-2.0) );
    CHECK( H(2, 4) == approx( 0.0) );
    CHECK( H(3, 0) == approx( 0.0) );
    CHECK( H(3, 1) == approx( 0.0) );
    CHECK( H(3, 2) == approx(-2.0) );
    CHECK( H(3, 3) == approx( 4.0) );
    CHECK( H(3, 4) == approx(-2.0) );
    CHECK( H(4, 0) == approx( 0.0) );
    CHECK( H(4, 1) == approx( 0.0) );
    CHECK( H(4, 2) == approx( 0.0) );
    CHECK( H(4, 3) == approx(-2.0) );
    CHECK( H(4, 4) == approx( 2.0) );
}
