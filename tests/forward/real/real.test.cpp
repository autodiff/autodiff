// Catch includes
#include <catch2/catch.hpp>

// autodiff includes
#include <autodiff/forward/real.hpp>
#include <autodiff/utils/traits.hpp>
using namespace autodiff;
using namespace autodiff::detail;

#define CHECK_APPROX(a, b) CHECK(a == Approx(b).epsilon(1e-14).margin(1e-12))

#define CHECK_4TH_ORDER_REAL_NUMBERS(a, b) \
    CHECK_APPROX( a[0], b[0] );            \
    CHECK_APPROX( a[1], b[1] );            \
    CHECK_APPROX( a[2], b[2] );            \
    CHECK_APPROX( a[3], b[3] );            \
    CHECK_APPROX( a[4], b[4] );            \

// #define CHECK_DERIVATIVES_REAL4TH_WRT(expr, u, ux, uxx, uxxx, uxxxx, uy, uyy, uyyy, uyyyy) \
// {                                                                                          \
//     auto f = [](const real4th& x, const real4th& y) -> real4th { return expr; };           \
//     auto dfdx = derivatives(f, wrt(x), at(x, y));                                          \
//     CHECK_APPROX( dfdx[0], u );                                                            \
//     CHECK_APPROX( dfdx[1], ux );                                                           \
//     CHECK_APPROX( dfdx[2], uxx );                                                          \
//     CHECK_APPROX( dfdx[3], uxxx );                                                         \
//     CHECK_APPROX( dfdx[4], uxxxx );                                                        \
//     auto dfdy = derivatives(f, wrt(y), at(x, y));                                          \
//     CHECK_APPROX( dfdy[0], u );                                                            \
//     CHECK_APPROX( dfdy[1], uy );                                                           \
//     CHECK_APPROX( dfdy[2], uyy );                                                          \
//     CHECK_APPROX( dfdy[3], uyyy );                                                         \
//     CHECK_APPROX( dfdy[4], uyyyy );                                                        \
// }

#define CHECK_DERIVATIVES_REAL4TH_WRT(expr)                                       \
{                                                                                 \
    real4th x, y, z;                                                              \
    x = 5;                                                                        \
    y = 7;                                                                        \
    auto f = [](const real4th& x, const real4th& y) -> real4th { return expr; };  \
    auto dfdx = derivatives(f, wrt(x), at(x, y));                                 \
    x[1] = 1.0; u = expr; x[1] = 0.0;                                             \
    CHECK_APPROX( dfdx[0], u[0] );                                                \
    CHECK_APPROX( dfdx[1], u[1] );                                                \
    auto dfdy = derivatives(f, wrt(y), at(x, y));                                 \
    y[1] = 1.0; u = expr; y[1] = 0.0;                                             \
    CHECK_APPROX( dfdy[0], u[0] );                                                \
    CHECK_APPROX( dfdy[1], u[1] );                                                \
    auto dfdv = derivatives(f, along(3, 5), at(x, y));                            \
    x[1] = 3.0; y[1] = 5.0; u = expr; x[1] = 0.0; y[1] = 0.0;                     \
    CHECK_APPROX( dfdv[0], u[0] );                                                \
    CHECK_APPROX( dfdv[1], u[1] );                                                \
    CHECK_APPROX( dfdv[2], u[2] );                                                \
    CHECK_APPROX( dfdv[3], u[3] );                                                \
    CHECK_APPROX( dfdv[4], u[4] );                                                \
}

// Auxiliary constants
const auto ln10 = 2.302585092994046;
const auto pi = 3.14159265359;


TEST_CASE("", "")
{
    {
        real x, y;
        seed(at(x, y), along(2, 3));

        CHECK(x[1] == 2.0);
        CHECK(y[1] == 3.0);
    }

    {
        std::vector<real> x(4);
        real y;

        std::vector<double> v = {2.0, 3.0, 4.0, 5.0};

        seed(at(x, y), along(v, 7.0));

        CHECK(x[0][1] == 2.0);
        CHECK(x[1][1] == 3.0);
        CHECK(x[2][1] == 4.0);
        CHECK(x[3][1] == 5.0);
        CHECK(y[1] == 7.0);

        unseed(at(x, y));

        CHECK(x[0][1] == 0.0);
        CHECK(x[1][1] == 0.0);
        CHECK(x[2][1] == 0.0);
        CHECK(x[3][1] == 0.0);
        CHECK(y[1] == 0.0);
    }

    // auto aux1 = at(x, y);
    // auto aux2 = along(2.0, 3.0);

    // CHECK(aux1.numArgs == aux2.numArgs);

}


/**/
TEST_CASE("testing autodiff::real", "[forward][real]")
{
    real4th x, y, z, u, v, w;

    x = 1.0;

    CHECK_APPROX( x[0],  1.0 );
    CHECK_APPROX( x[1],  0.0 );
    CHECK_APPROX( x[2],  0.0 );
    CHECK_APPROX( x[3],  0.0 );
    CHECK_APPROX( x[4],  0.0 );

    x = {0.5, 3.0, -5.0, -15.0, 11.0};

    CHECK_APPROX( x[0],   0.5 );
    CHECK_APPROX( x[1],   3.0 );
    CHECK_APPROX( x[2],  -5.0 );
    CHECK_APPROX( x[3], -15.0 );
    CHECK_APPROX( x[4],  11.0 );

    y = +x;

    CHECK_APPROX( y[0],  x[0] );
    CHECK_APPROX( y[1],  x[1] );
    CHECK_APPROX( y[2],  x[2] );
    CHECK_APPROX( y[3],  x[3] );
    CHECK_APPROX( y[4],  x[4] );

    y = -x;

    CHECK_APPROX( y[0],  -x[0] );
    CHECK_APPROX( y[1],  -x[1] );
    CHECK_APPROX( y[2],  -x[2] );
    CHECK_APPROX( y[3],  -x[3] );
    CHECK_APPROX( y[4],  -x[4] );

    z = x + y;

    CHECK_APPROX( z[0],  x[0] + y[0] );
    CHECK_APPROX( z[1],  x[1] + y[1] );
    CHECK_APPROX( z[2],  x[2] + y[2] );
    CHECK_APPROX( z[3],  x[3] + y[3] );
    CHECK_APPROX( z[4],  x[4] + y[4] );

    z = x + 1.0;

    CHECK_APPROX( z[0],  x[0] + 1.0 );
    CHECK_APPROX( z[1],  x[1] );
    CHECK_APPROX( z[2],  x[2] );
    CHECK_APPROX( z[3],  x[3] );
    CHECK_APPROX( z[4],  x[4] );

    z = 1.0 + x;

    CHECK_APPROX( z[0],  x[0] + 1.0 );
    CHECK_APPROX( z[1],  x[1] );
    CHECK_APPROX( z[2],  x[2] );
    CHECK_APPROX( z[3],  x[3] );
    CHECK_APPROX( z[4],  x[4] );

    z = x - y;

    CHECK_APPROX( z[0],  x[0] - y[0] );
    CHECK_APPROX( z[1],  x[1] - y[1] );
    CHECK_APPROX( z[2],  x[2] - y[2] );
    CHECK_APPROX( z[3],  x[3] - y[3] );
    CHECK_APPROX( z[4],  x[4] - y[4] );

    z = x * y;

    CHECK_APPROX( z[0],  x[0]*y[0] );
    CHECK_APPROX( z[1],  x[1]*y[0] + x[0]*y[1] );
    CHECK_APPROX( z[2],  x[2]*y[0] + 2*x[1]*y[1] + x[0]*y[2] );
    CHECK_APPROX( z[3],  x[3]*y[0] + 3*x[2]*y[1] + 3*x[1]*y[2] + x[0]*y[3] );
    CHECK_APPROX( z[4],  x[4]*y[0] + 4*x[3]*y[1] + 6*x[2]*y[2] + 4*x[1]*y[3] + x[0]*y[4] );

    z = x / y;

    CHECK_APPROX( z[0],  (x[0])/y[0] );
    CHECK_APPROX( z[1],  (x[1] - y[1]*z[0])/y[0] );
    CHECK_APPROX( z[2],  (x[2] - y[2]*z[0] - 2*y[1]*z[1])/y[0] );
    CHECK_APPROX( z[3],  (x[3] - y[3]*z[0] - 3*y[2]*z[1] - 3*y[1]*z[2])/y[0] );
    CHECK_APPROX( z[4],  (x[4] - y[4]*z[0] - 4*y[3]*z[1] - 6*y[2]*z[2] - 4*y[1]*z[3])/y[0] );

    z = 3.0 / y;

    CHECK_APPROX( z[0],  3.0/y[0] );
    CHECK_APPROX( z[1],  -(y[1]*z[0])/y[0] );
    CHECK_APPROX( z[2],  -(y[2]*z[0] + 2*y[1]*z[1])/y[0] );
    CHECK_APPROX( z[3],  -(y[3]*z[0] + 3*y[2]*z[1] + 3*y[1]*z[2])/y[0] );
    CHECK_APPROX( z[4],  -(y[4]*z[0] + 4*y[3]*z[1] + 6*y[2]*z[2] + 4*y[1]*z[3])/y[0] );

    //=====================================================================================================================
    //
    // TESTING EXPONENTIAL AND LOGARITHMIC FUNCTIONS
    //
    //=====================================================================================================================

    y = exp(x);

    CHECK_APPROX( y[0], exp(x[0]) );
    CHECK_APPROX( y[1], x[1] * y[0] );
    CHECK_APPROX( y[2], x[2] * y[0] + x[1] * y[1] );
    CHECK_APPROX( y[3], x[3] * y[0] + 2*x[2] * y[1] + x[1] * y[2] );
    CHECK_APPROX( y[4], x[4] * y[0] + 3*x[3] * y[1] + 3*x[2] * y[2] + x[1] * y[3] );

    y = log(x);

    CHECK_APPROX( y[0], log(x[0]) );
    CHECK_APPROX( y[1], (x[1]) / x[0] );
    CHECK_APPROX( y[2], (x[2] - x[1]*y[1]) / x[0] );
    CHECK_APPROX( y[3], (x[3] - x[2]*y[1] - 2*x[1]*y[2]) / x[0] );
    CHECK_APPROX( y[4], (x[4] - x[3]*y[1] - 3*x[2]*y[2] - 3*x[1]*y[3]) / x[0] );

    y = log10(x);
    z = log(x)/ln10;

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    y = sqrt(x);
    z = exp(0.5 * log(x));

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    y = pow(x, x);
    z = exp(x * log(x));

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    y = pow(x, pi);
    z = exp(pi * log(x));

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    y = pow(pi, x);
    z = exp(x * log(pi));

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    //=====================================================================================================================
    //
    // TESTING TRIGONOMETRIC FUNCTIONS
    //
    //=====================================================================================================================

    y = sin(x);
    z = cos(x);

    CHECK_APPROX( y[0],  sin(x[0]) );
    CHECK_APPROX( z[0],  cos(x[0]) );
    CHECK_APPROX( y[1],   x[1] * z[0] );
    CHECK_APPROX( z[1],  -x[1] * y[0] );
    CHECK_APPROX( y[2],   x[2] * z[0] + x[1] * z[1] );
    CHECK_APPROX( z[2],  -x[2] * y[0] - x[1] * y[1] );
    CHECK_APPROX( y[3],   x[3] * z[0] + 2*x[2] * z[1] + x[1] * z[2] );
    CHECK_APPROX( z[3],  -x[3] * y[0] - 2*x[2] * y[1] - x[1] * y[2] );
    CHECK_APPROX( y[4],   x[4] * z[0] + 3*x[3] * z[1] + 3*x[2] * z[2] + x[1] * z[3] );
    CHECK_APPROX( z[4],  -x[4] * y[0] - 3*x[3] * y[1] - 3*x[2] * y[2] - x[1] * y[3] );

    y = tan(x);
    z = sin(x)/cos(x);

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    y = asin(x);
    z = 1/sqrt(1 - x*x);

    CHECK_APPROX( y[0], asin(x[0]) );
    CHECK_APPROX( y[1], z[0] );
    CHECK_APPROX( y[2], z[1] );
    CHECK_APPROX( y[3], z[2] );
    CHECK_APPROX( y[4], z[3] );

    y = acos(x);
    z = -1/sqrt(1 - x*x);

    CHECK_APPROX( y[0], acos(x[0]) );
    CHECK_APPROX( y[1], z[0] );
    CHECK_APPROX( y[2], z[1] );
    CHECK_APPROX( y[3], z[2] );
    CHECK_APPROX( y[4], z[3] );

    y = atan(x);
    z = 1/(1 + x*x);

    CHECK_APPROX( y[0], atan(x[0]) );
    CHECK_APPROX( y[1], z[0] );
    CHECK_APPROX( y[2], z[1] );
    CHECK_APPROX( y[3], z[2] );
    CHECK_APPROX( y[4], z[3] );

    //=====================================================================================================================
    //
    // TESTING HYPERBOLIC FUNCTIONS
    //
    //=====================================================================================================================
    y = sinh(x);
    z = cosh(x);

    CHECK_APPROX( y[0],  sinh(x[0]) );
    CHECK_APPROX( z[0],  cosh(x[0]) );
    CHECK_APPROX( y[1],  x[1] * z[0] );
    CHECK_APPROX( z[1],  x[1] * y[0] );
    CHECK_APPROX( y[2],  x[2] * z[0] + x[1] * z[1] );
    CHECK_APPROX( z[2],  x[2] * y[0] + x[1] * y[1] );
    CHECK_APPROX( y[3],  x[3] * z[0] + 2*x[2] * z[1] + x[1] * z[2] );
    CHECK_APPROX( z[3],  x[3] * y[0] + 2*x[2] * y[1] + x[1] * y[2] );
    CHECK_APPROX( y[4],  x[4] * z[0] + 3*x[3] * z[1] + 3*x[2] * z[2] + x[1] * z[3] );
    CHECK_APPROX( z[4],  x[4] * y[0] + 3*x[3] * y[1] + 3*x[2] * y[2] + x[1] * y[3] );

    y = tanh(x);
    z = sinh(x)/cosh(x);

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    y = asinh(x);
    z = 1/sqrt(x*x + 1);

    CHECK_APPROX( y[0], asinh(x[0]) );
    CHECK_APPROX( y[1], z[0] );
    CHECK_APPROX( y[2], z[1] );
    CHECK_APPROX( y[3], z[2] );
    CHECK_APPROX( y[4], z[3] );

    y = acosh(10*x); // acosh requires x > 1
    z = 1/sqrt(100*x*x - 1);

    CHECK_APPROX( y[0], acosh(10*x[0]) );
    CHECK_APPROX( y[1], z[0] );
    CHECK_APPROX( y[2], z[1] );
    CHECK_APPROX( y[3], z[2] );
    CHECK_APPROX( y[4], z[3] );

    y = atanh(x);
    z = 1/(1 - x*x);

    CHECK_APPROX( y[0], atanh(x[0]) );
    CHECK_APPROX( y[1], z[0] );
    CHECK_APPROX( y[2], z[1] );
    CHECK_APPROX( y[3], z[2] );
    CHECK_APPROX( y[4], z[3] );

    //=====================================================================================================================
    //
    // TESTING OTHER FUNCTIONS
    //
    //=====================================================================================================================

    y = abs(x);

    CHECK_APPROX( y[0], abs(x[0]) );
    CHECK_APPROX( y[1], abs(x[0])/x[0] * x[1] );
    CHECK_APPROX( y[2], abs(x[0])/x[0] * x[2] );
    CHECK_APPROX( y[3], abs(x[0])/x[0] * x[3] );
    CHECK_APPROX( y[4], abs(x[0])/x[0] * x[4] );

    y = -x;
    z = abs(y);

    CHECK_APPROX( z[0], abs(y[0]) );
    CHECK_APPROX( z[1], abs(y[0])/(y[0]) * y[1] );
    CHECK_APPROX( z[2], abs(y[0])/(y[0]) * y[2] );
    CHECK_APPROX( z[3], abs(y[0])/(y[0]) * y[3] );
    CHECK_APPROX( z[4], abs(y[0])/(y[0]) * y[4] );



    // seed(at(x, y), along(2.0, 3.0));

    auto aux1 = at(x, y);
    auto aux2 = along(2.0, 3.0);

    CHECK(aux1.numArgs == aux2.numArgs);



    //=====================================================================================================================
    //
    // TESTING DERIVATIVE CALCULATIONS
    //
    //=====================================================================================================================
    std::function<real4th(const real4th&, const real4th&)> f, g, h;

    x = 5.0;
    y = 7.0;

    //---------------------------------------------------------------------------------------------------------------------
    // f(x, y) = exp(log(2x + 3y))
    //---------------------------------------------------------------------------------------------------------------------
    CHECK_DERIVATIVES_REAL4TH_WRT( exp(log(2*x + 3*y)) );

    // z = derivatives(f, wrt(x), at(x, y));
    //
    // CHECK_APPROX( z[0], 2*x[0] + 3*y[0] );
    // CHECK_APPROX( z[1], 2.0 );
    // CHECK_APPROX( z[2], 0.0 );
    // CHECK_APPROX( z[3], 0.0 );
    // CHECK_APPROX( z[4], 0.0 );

    // z = derivatives(f, wrt(y), at(x, y));

    // CHECK_APPROX( z[0], 2*x[0] + 3*y[0] );
    // CHECK_APPROX( z[1], 3.0 );
    // CHECK_APPROX( z[2], 0.0 );
    // CHECK_APPROX( z[3], 0.0 );
    // CHECK_APPROX( z[4], 0.0 );

    // //---------------------------------------------------------------------------------------------------------------------
    // // f(x, y) = sin(2x + 3y)
    // //---------------------------------------------------------------------------------------------------------------------

    // f = [](const real4th& x, const real4th& y) {
    //     return sin(2*x + 3*y);
    // };

    // z = derivatives(f, wrt(x), at(x, y));

    // CHECK_APPROX( z[0], sin(2*x[0] + 3*y[0]) );
    // CHECK_APPROX( z[1], cos(2*x[0] + 3*y[0])*2.0 );
    // CHECK_APPROX( z[2], -sin(2*x[0] + 3*y[0])*4.0 );
    // CHECK_APPROX( z[3], -cos(2*x[0] + 3*y[0])*8.0 );
    // CHECK_APPROX( z[4], sin(2*x[0] + 3*y[0])*16.0 );

    // z = derivatives(f, wrt(y), at(x, y));

    // CHECK_APPROX( z[0], sin(2*x[0] + 3*y[0]) );
    // CHECK_APPROX( z[1], cos(2*x[0] + 3*y[0])*3.0 );
    // CHECK_APPROX( z[2], -sin(2*x[0] + 3*y[0])*9.0 );
    // CHECK_APPROX( z[3], -cos(2*x[0] + 3*y[0])*27.0 );
    // CHECK_APPROX( z[4], sin(2*x[0] + 3*y[0])*81.0 );

    // //---------------------------------------------------------------------------------------------------------------------
    // // f(x, y) = exp(2x + 3y) * log(x/y)
    // //---------------------------------------------------------------------------------------------------------------------

    // g = [](const real4th& x, const real4th& y) {
    //     return exp(2*x + 3*y);
    // };

    // h = [](const real4th& x, const real4th& y) {
    //     return log(x/y);
    // };

    // f = [&](const real4th& x, const real4th& y) {
    //     return g(x, y) * h(x, y);
    // };

    // // --- derivatives along x ---

    // u = derivatives(g, wrt(x), at(x, y));
    // v = derivatives(h, wrt(x), at(x, y));
    // w = u * v;

    // z = derivatives(f, wrt(x), at(x, y));

    // CHECK_APPROX( z[0], w[0] );
    // CHECK_APPROX( z[1], w[1] );
    // CHECK_APPROX( z[2], w[2] );
    // CHECK_APPROX( z[3], w[3] );
    // CHECK_APPROX( z[4], w[4] );

    // // --- derivatives along y ---

    // u = derivatives(g, wrt(y), at(x, y));
    // v = derivatives(h, wrt(y), at(x, y));
    // w = u * v;

    // z = derivatives(f, wrt(y), at(x, y));

    // CHECK_APPROX( z[0], w[0] );
    // CHECK_APPROX( z[1], w[1] );
    // CHECK_APPROX( z[2], w[2] );
    // CHECK_APPROX( z[3], w[3] );
    // CHECK_APPROX( z[4], w[4] );
}
//*/