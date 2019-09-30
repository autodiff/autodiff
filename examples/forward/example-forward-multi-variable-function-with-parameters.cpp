// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/forward.hpp>
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

    cout << "u = " << u << endl;         // print the evaluated output u
    cout << "du/dx = " << dudx << endl;  // print the evaluated derivative du/dx
    cout << "du/da = " << duda << endl;  // print the evaluated derivative du/da
    cout << "du/db = " << dudb << endl;  // print the evaluated derivative du/db
    cout << "du/dc = " << dudc << endl;  // print the evaluated derivative du/dc
}
