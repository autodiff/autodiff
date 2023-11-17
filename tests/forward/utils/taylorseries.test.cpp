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
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;

/// Return the Taylor projection of *f(x)* along vector *v* at *t*.
template<typename Array>
auto taylor_projection_at(const Array& dfdv, double t)
{
    auto size = dfdv.size();
    auto res = dfdv[0];
    double factor = t;
    for(size_t i = 1; i < size; ++i) {
        res += factor * dfdv[i];
        factor *= t/static_cast<double>(i + 1);
    }
    return res;
}

#define CHECK_TAYLOR_SERIES_SCALAR_FUN(type, expr)                                                                                          \
{                                                                                                                                           \
    auto f = [](const ArrayX##type& x) -> type { return expr; };                                                                            \
                                                                                                                                            \
    ArrayX##type x(3);                                                                                                                      \
    x << 1.0, 2.0, 3.0;                                                                                                                     \
                                                                                                                                            \
    ArrayXd v(3);                                                                                                                           \
    v << 3.0, 5.0, 7.0;                                                                                                                     \
                                                                                                                                            \
    auto g = taylorseries(f, along(v), at(x));                                                                                              \
                                                                                                                                            \
    auto dfdv = g.derivatives();                                                                                                            \
                                                                                                                                            \
    CHECK( dfdv.size() == detail::Order<type> + 1 );                                                                                        \
                                                                                                                                            \
    CHECK_APPROX( g(0.0), taylor_projection_at(dfdv, 0.0) );                                                                                  \
                                                                                                                                            \
    CHECK_APPROX( g(1.0), taylor_projection_at(dfdv, 1.0) );                                                                                  \
    CHECK_APPROX( g(2.0), taylor_projection_at(dfdv, 2.0) );                                                                                  \
                                                                                                                                            \
    CHECK_APPROX( g(-1.0), taylor_projection_at(dfdv, -1.0) );                                                                                \
    CHECK_APPROX( g(-2.0), taylor_projection_at(dfdv, -2.0) );                                                                                \
}

#define CHECK_TAYLOR_SERIES_VECTOR_FUN(type, expr)                                                                                          \
{                                                                                                                                           \
    auto f = [](const ArrayX##type& x) -> ArrayX##type { return expr; };                                                                    \
                                                                                                                                            \
    ArrayX##type x(3);                                                                                                                      \
    x << 1.0, 2.0, 3.0;                                                                                                                     \
                                                                                                                                            \
    ArrayXd v(3);                                                                                                                           \
    v << 3.0, 5.0, 7.0;                                                                                                                     \
                                                                                                                                            \
    auto g = taylorseries(f, along(v), at(x));                                                                                              \
                                                                                                                                            \
    auto dfdv = g.derivatives();                                                                                                            \
                                                                                                                                            \
    CHECK( dfdv.size() == detail::Order<type> + 1 );                                                                                        \
                                                                                                                                            \
    CHECK( g(0.0).isApprox(taylor_projection_at(dfdv, 0.0)) );                                                                              \
                                                                                                                                            \
    CHECK( g(1.0).isApprox(taylor_projection_at(dfdv, 1.0)) );                                                                              \
    CHECK( g(2.0).isApprox(taylor_projection_at(dfdv, 2.0)) );                                                                              \
                                                                                                                                            \
    CHECK( g(-1.0).isApprox(taylor_projection_at(dfdv, -1.0)) );                                                                            \
    CHECK( g(-2.0).isApprox(taylor_projection_at(dfdv, -2.0)) );                                                                            \
}

TEST_CASE("testing forward taylorseries module", "[forward][utils][taylorseries]")
{
    using Eigen::ArrayXd;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    CHECK_TAYLOR_SERIES_SCALAR_FUN( real1st, (x.sin() * x.exp()).sum() );
    CHECK_TAYLOR_SERIES_SCALAR_FUN( real2nd, (x.sin() * x.exp()).sum() );
    CHECK_TAYLOR_SERIES_SCALAR_FUN( real3rd, (x.sin() * x.exp()).sum() );
    CHECK_TAYLOR_SERIES_SCALAR_FUN( real4th, (x.sin() * x.exp()).sum() );

    CHECK_TAYLOR_SERIES_SCALAR_FUN( dual1st, (x.sin() * x.exp()).sum() );
    CHECK_TAYLOR_SERIES_SCALAR_FUN( dual2nd, (x.sin() * x.exp()).sum() );
    CHECK_TAYLOR_SERIES_SCALAR_FUN( dual3rd, (x.sin() * x.exp()).sum() );
    CHECK_TAYLOR_SERIES_SCALAR_FUN( dual4th, (x.sin() * x.exp()).sum() );

    CHECK_TAYLOR_SERIES_VECTOR_FUN( real1st, x.sin() * x.exp() );
    CHECK_TAYLOR_SERIES_VECTOR_FUN( real2nd, x.sin() * x.exp() );
    CHECK_TAYLOR_SERIES_VECTOR_FUN( real3rd, x.sin() * x.exp() );
    CHECK_TAYLOR_SERIES_VECTOR_FUN( real4th, x.sin() * x.exp() );

    CHECK_TAYLOR_SERIES_VECTOR_FUN( dual1st, x.sin() * x.exp() );
    CHECK_TAYLOR_SERIES_VECTOR_FUN( dual2nd, x.sin() * x.exp() );
    CHECK_TAYLOR_SERIES_VECTOR_FUN( dual3rd, x.sin() * x.exp() );
    CHECK_TAYLOR_SERIES_VECTOR_FUN( dual4th, x.sin() * x.exp() );
}
