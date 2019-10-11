// Catch includes
#include <catch2/catch.hpp>

// autodiff includes
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

TEST_CASE("testing autodiff::real (with eigen)", "[forward][real][eigen]")
{
    Eigen::MatrixXd M(3, 3);
    M << 1.0, 2.0, 3.0,
         4.0, 1.0, 6.0,
         7.0, 3.0, 9.0;

    VectorXreal x(3), y(3), z(3);
    x << 1.0, 2.0, 3.0;
    x[0][1] = 1.0;

    z = M * x;
    // bool zz = M * x;
    // z = x + y;

    auto f = [](const VectorXreal& x) {
        return x.cwiseProduct(x).sum();
    };

    // auto dfdx = gradient(f, wrt(x), at(x));
    // auto dfdx = gradient(f, wrt(x), at(x));
    // auto dfdx = gradient(f, wrt(x[0], x[1], x[2]), at(x));

    // CHECK( dfdx[0] == Approx(2*x[0][0]) );
    // CHECK( dfdx[1] == Approx(2*x[1][0]) );
    // CHECK( dfdx[2] == Approx(2*x[2][0]) );

    x[0][1] = 0.0;   // TODO: IMPLEMENT A FUNCTION THAT ZERO OUT ALL DERIVATIVE SEED VALUES

    auto dfdx = gradient(f, wrt(x[1], x[2], x[0]), at(x));


    CHECK( dfdx[0] == Approx(2*x[1][0]) );
    CHECK( dfdx[1] == Approx(2*x[2][0]) );
    CHECK( dfdx[2] == Approx(2*x[0][0]) );


}
