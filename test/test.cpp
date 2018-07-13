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
    var t = a;
    var d = 200;
    var c = f(a, b, t);

    REQUIRE( grad(c, a) == 3 );
    REQUIRE( grad(c, t) == 3 );
    REQUIRE( grad(c, b) == 5 );
    REQUIRE( grad(c, d) == 0 );
}
