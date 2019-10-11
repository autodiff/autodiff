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
}
