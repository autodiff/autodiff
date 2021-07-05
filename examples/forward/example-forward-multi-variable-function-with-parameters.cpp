// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// A type defining parameters for a function of interest
struct Params
{
    dual a;
    dual b;
    dual c;
};

// The function that depends on parameters for which derivatives are needed
dual f(dual x, const Params& params)
{
    return params.a * sin(x) + params.b * cos(x) + params.c * sin(x)*cos(x);
}

int main()
{
    Params params;   // initialize the parameter variables
    params.a = 1.0;  // the parameter a of type dual, not double!
    params.b = 2.0;  // the parameter b of type dual, not double!
    params.c = 3.0;  // the parameter c of type dual, not double!

    dual x = 0.5;  // the input variable x

    dual u = f(x, params);  // the output variable u

    double dudx = derivative(f, wrt(x), at(x, params));        // evaluate the derivative du/dx
    double duda = derivative(f, wrt(params.a), at(x, params)); // evaluate the derivative du/da
    double dudb = derivative(f, wrt(params.b), at(x, params)); // evaluate the derivative du/db
    double dudc = derivative(f, wrt(params.c), at(x, params)); // evaluate the derivative du/dc

    std::cout << "u = " << u << std::endl;         // print the evaluated output u
    std::cout << "du/dx = " << dudx << std::endl;  // print the evaluated derivative du/dx
    std::cout << "du/da = " << duda << std::endl;  // print the evaluated derivative du/da
    std::cout << "du/db = " << dudb << std::endl;  // print the evaluated derivative du/db
    std::cout << "du/dc = " << dudc << std::endl;  // print the evaluated derivative du/dc
}

/*-------------------------------------------------------------------------------------------------
=== Note ===
---------------------------------------------------------------------------------------------------
This example would also work if real was used instead of dual. Should you
need higher-order cross derivatives, however, e.g.,:

    double d2udxda = derivative(f, wrt(x, params.a), at(x, params));

then higher-order dual types are the right choicesince real types are
optimally designed for higher-order directional derivatives.
-------------------------------------------------------------------------------------------------*/
