// Catch includes
#include "catch.hpp"

// autodiff includes
#include <autodiff/autodiff.hpp>
using namespace autodiff;

auto f(var a, var b, var t) -> var
{
    return ((2*a + b) + b + 3*b + t) + 1.0;
}

TEST_CASE("autodiff tests", "[autodiff]")
{
    var a = 10;
    var b = 20;

    REQUIRE( grad(a, a) == 1 );
    REQUIRE( grad(a, b) == 0 );

    var c = a;

    REQUIRE( grad(c, a) == 1 );

    c = +a;

    REQUIRE( grad(c, a) == 1 );

    c = -a;

    REQUIRE( grad(c, a) == -1 );

    var t = a;
    c = a + t;

    REQUIRE( grad(c, a) == 2 );
    REQUIRE( grad(c, t) == 2 );

    c = -2*a;

    REQUIRE( grad(c, a) == -2 );

    c = a / 3;

    REQUIRE( grad(c, a) == 1.0/3.0 );

    c = a + b;

    REQUIRE( grad(c, a) == 1.0 );
    REQUIRE( grad(c, b) == 1.0 );

    c = a - b;

    REQUIRE( grad(c, a) ==  1.0 );
    REQUIRE( grad(c, b) == -1.0 );

    c = -a + b;

    REQUIRE( grad(c, a) == -1.0 );
    REQUIRE( grad(c, b) ==  1.0 );

    c = a + 1;

    REQUIRE( grad(c, a) == 1.0 );

//    c = std::exp(a);
//
//    REQUIRE( grad(c, a) == c );
}
