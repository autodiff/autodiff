// Catch includes
#include <catch2/catch.hpp>

// autodiff includes
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

TEST_CASE("testing autodiff::real (with eigen)", "[forward][real][eigen]")
{
    using Eigen::ArrayXd;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    MatrixXd M(3, 3);
    M << 1.0, 2.0, 3.0,
         4.0, 1.0, 6.0,
         7.0, 3.0, 9.0;

    ArrayXreal x(3), y(3), z(3);
    x << 1.0, 2.0, 3.0;

    SECTION("testing gradient computation")
    {
        std::function<real(const ArrayXreal&)> f;
        ArrayXd g;

        //---------------------------------------------------------------------------------------------------------------------
        // f(x) = tr(x) * x
        //---------------------------------------------------------------------------------------------------------------------

        f = [](const ArrayXreal& x) {
            return (x * x).sum();
        };

        g = gradient(f, wrt(x), at(x));

        CHECK( g[0] == Approx(2*x[0]) );
        CHECK( g[1] == Approx(2*x[1]) );
        CHECK( g[2] == Approx(2*x[2]) );

        g = gradient(f, wrt(x[1], x[2], x[0]), at(x));

        CHECK( g[0] == Approx(2*x[1]) );
        CHECK( g[1] == Approx(2*x[2]) );
        CHECK( g[2] == Approx(2*x[0]) );

        g = gradient(f, wrt(x[1]), at(x));

        CHECK( g[0] == Approx(2*x[1]) );

        //---------------------------------------------------------------------------------------------------------------------
        // f(x) = sum(sin(x))
        //---------------------------------------------------------------------------------------------------------------------

        f = [](const ArrayXreal& x) {
            return (x.sin() * x.cos()).sum();
        };

        g = gradient(f, wrt(x), at(x));

        CHECK( g[0] == Approx( cos(x[0])*cos(x[0]) - sin(x[0])*sin(x[0]) ) );
        CHECK( g[1] == Approx( cos(x[1])*cos(x[1]) - sin(x[1])*sin(x[1]) ) );
        CHECK( g[2] == Approx( cos(x[2])*cos(x[2]) - sin(x[2])*sin(x[2]) ) );

        g = gradient(f, wrt(x[1], x[2], x[0]), at(x));

        CHECK( g[0] == Approx( cos(x[1])*cos(x[1]) - sin(x[1])*sin(x[1]) ) );
        CHECK( g[1] == Approx( cos(x[2])*cos(x[2]) - sin(x[2])*sin(x[2]) ) );
        CHECK( g[2] == Approx( cos(x[0])*cos(x[0]) - sin(x[0])*sin(x[0]) ) );

        g = gradient(f, wrt(x[1]), at(x));

        CHECK( g[0] == Approx( cos(x[1])*cos(x[1]) - sin(x[1])*sin(x[1]) ) );
    }

    SECTION("testing jacobian computation")
    {
        std::function<ArrayXreal(const ArrayXreal&)> F;

        MatrixXd J, Jx;

        //---------------------------------------------------------------------------------------------------------------------
        // F(x) = x * x
        //---------------------------------------------------------------------------------------------------------------------

        F = [](const ArrayXreal& x) {
            return x * x;
        };

        // J = jacobian(F, wrt(x), at(x));
        // Jx = 2*x.matrix().asDiagonal();

        // J = detail::jacobian_aux(F, x[0], wrt(x[1], x[2]), at(x), z);
        // J = jacobian(F, wrt(x[0], x[1], x[2]), at(x));
        J = jacobian(F, wrt(x), at(x));
        // CHECK( x[0][1] == Approx( 1.0 ) );
        // CHECK( x[1][1] == Approx( 2.0 ) );
        // CHECK( x[2][1] == Approx( 3.0 ) );


        CHECK( J(0, 0) == Approx( 2*x[0] ) );
        CHECK( J(0, 1) == Approx( 0.0 ) );
        CHECK( J(0, 2) == Approx( 0.0 ) );

        CHECK( J(1, 0) == Approx( 0.0 ) );
        CHECK( J(1, 1) == Approx( 2*x[1] ) );
        CHECK( J(1, 2) == Approx( 0.0 ) );

        CHECK( J(2, 0) == Approx( 0.0 ) );
        CHECK( J(2, 1) == Approx( 0.0 ) );
        CHECK( J(2, 2) == Approx( 2*x[2] ) );
    }
}
