// Catch includes
#include <catch2/catch.hpp>

// autodiff includes
#include <autodiff/real/real.hpp>
using namespace autodiff;

#define CHECK_APPROX(a, b) CHECK(a == Approx(b))

#define CHECK_4TH_ORDER_REAL_NUMBERS(a, b) \
    CHECK_APPROX( y[0], z[0] );            \
    CHECK_APPROX( y[1], z[1] );            \
    CHECK_APPROX( y[2], z[2] );            \
    CHECK_APPROX( y[3], z[3] );            \
    CHECK_APPROX( y[4], z[4] );            \

// Auxiliary constants
const auto ln10 = 2.302585092994046;
const auto pi = 3.14159265359;

TEST_CASE("real tests", "[real]")
{
    real4th x, y, z;

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

    //=====================================================================================================================
    //
    // TESTING DERIVATIVE CALCULATIONS
    //
    //=====================================================================================================================
    std::function<real4th(const real4th&, const real4th&)> f;

    x = 5.0;
    y = 7.0;

    f = [](const real4th& x, const real4th& y) {
        return sin(2*x + 3*y);
    };

    z = derivatives(f, along(x), at(x, y));

    CHECK_APPROX( z[0], sin(2*x[0] + 3*y[0]) );
    CHECK_APPROX( z[1], cos(2*x[0] + 3*y[0])*2.0 );
    CHECK_APPROX( z[2], -sin(2*x[0] + 3*y[0])*4.0 );
    CHECK_APPROX( z[3], -cos(2*x[0] + 3*y[0])*8.0 );
    CHECK_APPROX( z[4], sin(2*x[0] + 3*y[0])*16.0 );

    z = derivatives(f, along(y), at(x, y));

    CHECK_APPROX( z[0], sin(2*x[0] + 3*y[0]) );
    CHECK_APPROX( z[1], cos(2*x[0] + 3*y[0])*3.0 );
    CHECK_APPROX( z[2], -sin(2*x[0] + 3*y[0])*9.0 );
    CHECK_APPROX( z[3], -cos(2*x[0] + 3*y[0])*27.0 );
    CHECK_APPROX( z[4], sin(2*x[0] + 3*y[0])*81.0 );
}
