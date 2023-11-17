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

#include <exception>

// autodiff includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;

#define CHECK_GRADIENT(type, expr) \
{ \
    /* Define vector x and y */ \
    double y[5]; \
    ArrayX##type x(5); \
    x << 2.0, 3.0, 5.0, 7.0, 9.0; \
    /* Define functions f(x) = expr/expr - 1 == 0 and g(x) = expr */ \
    std::function<type(const ArrayX##type&)> f, g; \
    f = [](const ArrayX##type& x) -> type { return expr/expr - 1.0; }; \
    g = [](const ArrayX##type& x) -> type { return expr; }; \
    /* Auxiliary vectors dfdx, dgdx, dgdw where w is a vector with a combination of x entries */ \
    Eigen::VectorXd dfdx, dgdx, dgdw; \
    Eigen::Map<Eigen::VectorXd> map_5(y, 5); \
    /* Compute dfdx which is identical to zero by construction */ \
    dfdx = gradient(f, wrt(x), at(x)); \
    CHECK( dfdx.size() == 5 ); \
    CHECK( dfdx.squaredNorm() == approx(0.0) ); \
    /* Compute dgdx to be used as reference when checking againts dgdw below for different orderings of x entries in w */ \
    dgdx = gradient(g, wrt(x), at(x)); \
    /* Compute gradient using pre-allocated storage. */ \
    type u; \
    gradient(g, wrt(x), at(x), u, map_5); \
    CHECK( y[0] == approx(dgdx[0]) ); \
    CHECK( y[1] == approx(dgdx[1]) ); \
    CHECK( y[2] == approx(dgdx[2]) ); \
    CHECK( y[3] == approx(dgdx[3]) ); \
    CHECK( y[4] == approx(dgdx[4]) ); \
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

#define CHECK_HESSIAN(type, expr) \
{ \
    /* Define vector x and matrix y */ \
    double y[25]; \
    ArrayX##type x(5); \
    x << 2.0, 3.0, 5.0, 7.0, 9.0; \
    /* Define functions f(x) = expr/expr - 1 == 0 and g(x) = expr */ \
    std::function<type(const ArrayX##type&)> f, g; \
    f = [](const ArrayX##type& x) -> type { return expr/expr - 1.0; }; \
    g = [](const ArrayX##type& x) -> type { return expr; }; \
    /* Auxiliary matrices fxx, gxx, gww where w is a vector with a combination of x entries */ \
    Eigen::MatrixXd fxx, gxx, gww; \
    Eigen::Map<Eigen::MatrixXd> map_5x5(y, 5, 5); \
    /* The indices of the x variables in the w vector */ \
    std::vector<size_t> iw; \
    /* Compute fxx which is identical to zero by construction */ \
    fxx = hessian(f, wrt(x), at(x)); \
    CHECK( fxx.rows() == 5 ); \
    CHECK( fxx.cols() == 5 ); \
    CHECK( fxx.squaredNorm() == approx(0.0) ); \
    /* Compute gxx to be used as reference when checking againts gww below for different orderings of x entries in w */ \
    gxx = hessian(g, wrt(x), at(x)); \
    /* Compute hessian using pre-allocated storage */ \
    type u; \
    VectorXd grad; \
    hessian(g, wrt(x), at(x), u, grad, map_5x5); \
    for(size_t i = 0; i < gxx.rows(); ++i) \
        for(size_t j = 0; j < gxx.cols(); ++j) \
            CHECK( gxx(i, j) == approx(map_5x5(i, j)) ); \
    /* Compute gww where w = (x1, x2, x3, x4, x0) */ \
    gww = hessian(g, wrt(x.tail(4), x[0]), at(x)); \
    iw = {1, 2, 3, 4, 0}; \
    CHECK( gww.rows() == iw.size() ); \
    CHECK( gww.rows() == gww.cols() ); \
    for(size_t i = 0; i < iw.size(); ++i) \
        for(size_t j = 0; j < iw.size(); ++j) \
            CHECK( gxx(iw[i], iw[j]) == approx(gww(i, j)) ); \
    /* Compute gww where w = (x3, x0, x4) */ \
    gww = hessian(g, wrt(x[3], x[0], x[4]), at(x)); \
    iw = {3, 0, 4}; \
    CHECK( gww.rows() == iw.size() ); \
    CHECK( gww.rows() == gww.cols() ); \
    for(size_t i = 0; i < iw.size(); ++i) \
        for(size_t j = 0; j < iw.size(); ++j) \
            CHECK( gxx(iw[i], iw[j]) == approx(gww(i, j)) ); \
    /* Compute gww where w = (x3) */ \
    gww = hessian(g, wrt(x[3]), at(x)); \
    iw = {3}; \
    CHECK( gww.rows() == iw.size() ); \
    CHECK( gww.rows() == gww.cols() ); \
    for(size_t i = 0; i < iw.size(); ++i) \
        for(size_t j = 0; j < iw.size(); ++j) \
            CHECK( gxx(iw[i], iw[j]) == approx(gww(i, j)) ); \
    /* Compute gww where w = (x0, x1, x2, x3, x4, x0, x1, x2, x3, x4) */ \
    gww = hessian(g, wrt(x, x), at(x)); \
    iw = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4}; \
    CHECK( gww.rows() == iw.size() ); \
    CHECK( gww.rows() == gww.cols() ); \
    for(size_t i = 0; i < iw.size(); ++i) \
        for(size_t j = 0; j < iw.size(); ++j) \
            CHECK( gxx(iw[i], iw[j]) == approx(gww(i, j)) ); \
}

#define CHECK_JACOBIAN(type, expr) \
{ \
    /* Define vector x and matrix y */ \
    double y[25]; \
    ArrayX##type x(5); \
    x << 2.0, 3.0, 5.0, 7.0, 9.0; \
    /* Define functions f(x) = expr/expr - 1 == 0 and g(x) = expr */ \
    std::function<ArrayX##type(const ArrayX##type&)> f, g; \
    f = [](const ArrayX##type& x) -> ArrayX##type { return expr/expr - 1.0; }; \
    g = [](const ArrayX##type& x) -> ArrayX##type { return expr; }; \
    /* Auxiliary matrices dfdx, dgdx, dgdw where w is a vector with a combination of x entries */ \
    Eigen::MatrixXd dfdx, dgdx, dgdw; \
    Eigen::Map<Eigen::MatrixXd> map_5x5(y, 5, 5); \
    Eigen::Map<Eigen::MatrixXd> map_5x3(y, 5, 3); \
    /* Compute dfdx which is identical to zero by construction */ \
    dfdx = jacobian(f, wrt(x), at(x)); \
    CHECK( dfdx.rows() == 5 ); \
    CHECK( dfdx.cols() == 5 ); \
    CHECK( dfdx.squaredNorm() == approx(0.0) ); \
    /* Compute dgdx to be used as reference when checking againts dgdw below for different orderings of x entries in w */ \
    dgdx = jacobian(g, wrt(x), at(x)); \
    /* Compute square jacobian using pre-allocated storage */ \
    VectorX##type Gval; \
    jacobian(g, wrt(x), at(x), Gval, map_5x5); \
    CHECK( dgdx.col(0).isApprox(map_5x5.col(0)) ); \
    CHECK( dgdx.col(1).isApprox(map_5x5.col(1)) ); \
    CHECK( dgdx.col(2).isApprox(map_5x5.col(2)) ); \
    CHECK( dgdx.col(3).isApprox(map_5x5.col(3)) ); \
    CHECK( dgdx.col(4).isApprox(map_5x5.col(4)) ); \
    /* Compute rectangular jacobian using pre-allocated storage */ \
    jacobian(g, wrt(x[0], x[1], x[2]), at(x), Gval, map_5x3); \
    CHECK( dgdx.col(0).isApprox(map_5x3.col(0)) ); \
    CHECK( dgdx.col(1).isApprox(map_5x3.col(1)) ); \
    CHECK( dgdx.col(2).isApprox(map_5x3.col(2)) ); \
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

    SECTION("testing gradient computations")
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

    SECTION("testing hessian computations")
    {
        CHECK_HESSIAN( dual2nd, x.sum() );
        // CHECK_HESSIAN( dual2nd, x.exp().sum() );
        // CHECK_HESSIAN( dual2nd, x.log().sum() );
        // CHECK_HESSIAN( dual2nd, x.tan().sum() );
        // CHECK_HESSIAN( dual2nd, (x * x).sum() );
        // CHECK_HESSIAN( dual2nd, (x.sin() * x.exp()).sum() );
        // CHECK_HESSIAN( dual2nd, (x * x.log()).sum() );
        // CHECK_HESSIAN( dual2nd, (x.sin() * x.cos()).sum() );
    }

    SECTION("testing jacobian computations")
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
