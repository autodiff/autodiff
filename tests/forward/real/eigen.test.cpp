// Catch includes
#include <catch2/catch.hpp>

// autodiff includes
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

template<typename T>
auto approx(T&& val) -> Approx
{
    const double epsilon = std::numeric_limits<double>::epsilon() * 100;
    const double margin = 1e-12;
    return Approx(std::forward<T>(val)).epsilon(epsilon).margin(margin);
}

#define CHECK_GRADIENT(expr) \
{ \
    /* Define vector x */ \
    ArrayXreal x(5); \
    x << 2.0, 3.0, 5.0, 7.0, 9.0; \
    /* Define functions f(x) = expr/expr - 1 == 0 and g(x) = expr */ \
    std::function<real(const ArrayXreal&)> f, g; \
    f = [](const ArrayXreal& x) -> real { return expr/expr - 1.0; }; \
    g = [](const ArrayXreal& x) -> real { return expr; }; \
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

#define CHECK_JACOBIAN(expr) \
{ \
    /* Define vector x */ \
    ArrayXreal x(5); \
    x << 2.0, 3.0, 5.0, 7.0, 9.0; \
    /* Define functions f(x) = expr/expr - 1 == 0 and g(x) = expr */ \
    std::function<ArrayXreal(const ArrayXreal&)> f, g; \
    f = [](const ArrayXreal& x) -> ArrayXreal { return expr/expr - 1.0; }; \
    g = [](const ArrayXreal& x) -> ArrayXreal { return expr; }; \
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
        CHECK_GRADIENT( x.sum() );
        CHECK_GRADIENT( x.exp().sum() );
        CHECK_GRADIENT( x.log().sum() );
        CHECK_GRADIENT( x.tan().sum() );
        CHECK_GRADIENT( (x * x).sum() );
        CHECK_GRADIENT( (x.sin() * x.exp()).sum() );
        CHECK_GRADIENT( (x * x.log()).sum() );
        CHECK_GRADIENT( (x.sin() * x.cos()).sum() );
    }

    SECTION("testing jacobian computation")
    {
        CHECK_JACOBIAN( x );
        CHECK_JACOBIAN( x.exp() );
        CHECK_JACOBIAN( x.log() );
        CHECK_JACOBIAN( x.tan() );
        CHECK_JACOBIAN( (x * x) );
        CHECK_JACOBIAN( (x.sin() * x.exp()) );
        CHECK_JACOBIAN( (x * x.log()) );
        CHECK_JACOBIAN( (x.sin() * x.cos()) );
    }
}
