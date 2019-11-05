//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2020 Allan Leal
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
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;


#define CHECK_GRADIENT(type, expr) \
{ \
    /* Define vector x */ \
    ArrayX##type x(5); \
    x << 2.0, 3.0, 5.0, 7.0, 9.0; \
    /* Define functions f(x) = expr/expr - 1 == 0 and g(x) = expr */ \
    std::function<type(const ArrayX##type&)> f, g; \
    f = [](const ArrayX##type& x) -> type { return expr/expr - 1.0; }; \
    g = [](const ArrayX##type& x) -> type { return expr; }; \
    /* Auxiliary vectors dfdx, dgdx, dgdw where w is a vector with a combination of x entries */ \
    Eigen::VectorXd dfdx, dgdx, dgdw; \
    /* Compute dfdx which is identical to zero by construction */ \
    dfdx = gradient(f, wrt(x), at(x)); \
    CHECK( dfdx.size() == 5 ); \
    CHECK( dfdx.squaredNorm() == approx(0.0) ); \
    /* Compute dgdx to be used as reference when checking againts dgdw below for different orderings of x entries in w */ \
    dgdx = gradient(g, wrt(x), at(x)); \
    /* Compute dgdw where w = (x1, x2, x3, x4, x0) */ \
    dgdw = gradient(g, wrt(x.tail(4), x[0]), at(x)); \
    CHECK( dgdw.size() == 5 ); \
    CHECK( dgdw[0] == approx(dgdx[1]) ); \
    CHECK( dgdw[1] == approx(dgdx[2]) ); \
    CHECK( dgdw[2] == approx(dgdx[3]) ); \
    CHECK( dgdw[3] == approx(dgdx[4]) ); \
    CHECK( dgdw[4] == approx(dgdx[0]) ); \
    /* Compute dgdw where w = (x3, x0, x4) */ \
    dgdw = gradient(g, wrt(x[3], x[0], x[4]), at(x)); \
    CHECK( dgdw.size() == 3 ); \
    CHECK( dgdw[0] == approx(dgdx[3]) ); \
    CHECK( dgdw[1] == approx(dgdx[0]) ); \
    CHECK( dgdw[2] == approx(dgdx[4]) ); \
    /* Compute dgdw where w = (x3) */ \
    dgdw = gradient(g, wrt(x[3]), at(x)); \
    CHECK( dgdw.size() == 1 ); \
    CHECK( dgdw[0] == approx(dgdx[3]) ); \
    /* Compute dgdw where w = (x0, x1, x2, x3, x4, x0, x1, x2, x3, x4) */ \
    dgdw = gradient(g, wrt(x, x), at(x)); \
    CHECK( dgdw.size() == 10 ); \
    CHECK( dgdw[0] == approx(dgdx[0]) ); \
    CHECK( dgdw[1] == approx(dgdx[1]) ); \
    CHECK( dgdw[2] == approx(dgdx[2]) ); \
    CHECK( dgdw[3] == approx(dgdx[3]) ); \
    CHECK( dgdw[4] == approx(dgdx[4]) ); \
    CHECK( dgdw[5] == approx(dgdx[0]) ); \
    CHECK( dgdw[6] == approx(dgdx[1]) ); \
    CHECK( dgdw[7] == approx(dgdx[2]) ); \
    CHECK( dgdw[8] == approx(dgdx[3]) ); \
    CHECK( dgdw[9] == approx(dgdx[4]) ); \
}

#define CHECK_JACOBIAN(type, expr) \
{ \
    /* Define vector x */ \
    ArrayX##type x(5); \
    x << 2.0, 3.0, 5.0, 7.0, 9.0; \
    /* Define functions f(x) = expr/expr - 1 == 0 and g(x) = expr */ \
    std::function<ArrayX##type(const ArrayX##type&)> f, g; \
    f = [](const ArrayX##type& x) -> ArrayX##type { return expr/expr - 1.0; }; \
    g = [](const ArrayX##type& x) -> ArrayX##type { return expr; }; \
    /* Auxiliary matrices dfdx, dgdx, dgdw where w is a vector with a combination of x entries */ \
    Eigen::MatrixXd dfdx, dgdx, dgdw; \
    /* Compute dfdx which is identical to zero by construction */ \
    dfdx = jacobian(f, wrt(x), at(x)); \
    CHECK( dfdx.rows() == 5 ); \
    CHECK( dfdx.cols() == 5 ); \
    CHECK( dfdx.squaredNorm() == approx(0.0) ); \
    /* Compute dgdx to be used as reference when checking againts dgdw below for different orderings of x entries in w */ \
    dgdx = jacobian(g, wrt(x), at(x)); \
    /* Compute dgdw where w = (x1, x2, x3, x4, x0) */ \
    dgdw = jacobian(g, wrt(x.tail(4), x[0]), at(x)); \
    CHECK( dgdw.rows() == 5 ); \
    CHECK( dgdw.cols() == 5 ); \
    CHECK( dgdw.col(0).isApprox(dgdx.col(1)) ); \
    CHECK( dgdw.col(1).isApprox(dgdx.col(2)) ); \
    CHECK( dgdw.col(2).isApprox(dgdx.col(3)) ); \
    CHECK( dgdw.col(3).isApprox(dgdx.col(4)) ); \
    CHECK( dgdw.col(4).isApprox(dgdx.col(0)) ); \
    /* Compute dgdw where w = (x3, x0, x4) */ \
    dgdw = jacobian(g, wrt(x[3], x[0], x[4]), at(x)); \
    CHECK( dgdw.rows() == 5 ); \
    CHECK( dgdw.cols() == 3 ); \
    CHECK( dgdw.col(0).isApprox(dgdx.col(3)) ); \
    CHECK( dgdw.col(1).isApprox(dgdx.col(0)) ); \
    CHECK( dgdw.col(2).isApprox(dgdx.col(4)) ); \
    /* Compute dgdw where w = (x3) */ \
    dgdw = jacobian(g, wrt(x[3]), at(x)); \
    CHECK( dgdw.rows() == 5 ); \
    CHECK( dgdw.cols() == 1 ); \
    CHECK( dgdw.col(0).isApprox(dgdx.col(3)) ); \
    /* Compute dgdw where w = (x0, x1, x2, x3, x4, x0, x1, x2, x3, x4) */ \
    dgdw = jacobian(g, wrt(x, x), at(x)); \
    CHECK( dgdw.rows() == 5 ); \
    CHECK( dgdw.cols() == 10 ); \
    CHECK( dgdw.col(0).isApprox(dgdx.col(0)) ); \
    CHECK( dgdw.col(1).isApprox(dgdx.col(1)) ); \
    CHECK( dgdw.col(2).isApprox(dgdx.col(2)) ); \
    CHECK( dgdw.col(3).isApprox(dgdx.col(3)) ); \
    CHECK( dgdw.col(4).isApprox(dgdx.col(4)) ); \
    CHECK( dgdw.col(5).isApprox(dgdx.col(0)) ); \
    CHECK( dgdw.col(6).isApprox(dgdx.col(1)) ); \
    CHECK( dgdw.col(7).isApprox(dgdx.col(2)) ); \
    CHECK( dgdw.col(8).isApprox(dgdx.col(3)) ); \
    CHECK( dgdw.col(9).isApprox(dgdx.col(4)) ); \
}

TEST_CASE("testing forward gradient module", "[forward][utils][gradient]")
{
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    SECTION("testing gradient computation")
    {
        CHECK_GRADIENT( real, x.sum() );
        CHECK_GRADIENT( real, x.exp().sum() );
        CHECK_GRADIENT( real, x.log().sum() );
        CHECK_GRADIENT( real, x.tan().sum() );
        CHECK_GRADIENT( real, (x * x).sum() );
        CHECK_GRADIENT( real, (x.sin() * x.exp()).sum() );
        CHECK_GRADIENT( real, (x * x.log()).sum() );
        CHECK_GRADIENT( real, (x.sin() * x.cos()).sum() );

        CHECK_GRADIENT( dual, x.sum() );
        CHECK_GRADIENT( dual, x.exp().sum() );
        CHECK_GRADIENT( dual, x.log().sum() );
        CHECK_GRADIENT( dual, x.tan().sum() );
        CHECK_GRADIENT( dual, (x * x).sum() );
        CHECK_GRADIENT( dual, (x.sin() * x.exp()).sum() );
        CHECK_GRADIENT( dual, (x * x.log()).sum() );
        CHECK_GRADIENT( dual, (x.sin() * x.cos()).sum() );
    }

    SECTION("testing jacobian computation")
    {
        CHECK_JACOBIAN( real, x );
        CHECK_JACOBIAN( real, x.exp() );
        CHECK_JACOBIAN( real, x.log() );
        CHECK_JACOBIAN( real, x.tan() );
        CHECK_JACOBIAN( real, (x * x) );
        CHECK_JACOBIAN( real, (x.sin() * x.exp()) );
        CHECK_JACOBIAN( real, (x * x.log()) );
        CHECK_JACOBIAN( real, (x.sin() * x.cos()) );

        CHECK_JACOBIAN( dual, x );
        CHECK_JACOBIAN( dual, x.exp() );
        CHECK_JACOBIAN( dual, x.log() );
        CHECK_JACOBIAN( dual, x.tan() );
        CHECK_JACOBIAN( dual, (x * x) );
        CHECK_JACOBIAN( dual, (x.sin() * x.exp()) );
        CHECK_JACOBIAN( dual, (x * x.log()) );
        CHECK_JACOBIAN( dual, (x.sin() * x.cos()) );
    }
}
