// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// A type defining parameters for a function of interest
struct Params
{
    var a;
    var b;
    var c;
};

// The function that depends on parameters for which derivatives are needed
var f(var x, const Params& params)
{
    return params.a * sin(x) + params.b * cos(x) + params.c * sin(x)*cos(x);
}

int main()
{
    Params params;   // initialize the parameter variables
    params.a = 1.0;  // the parameter a of type var, not double!
    params.b = 2.0;  // the parameter b of type var, not double!
    params.c = 3.0;  // the parameter c of type var, not double!

    var x = 0.5;  // the input variable x
    var u = f(x, params);  // the output variable u

    auto [ux, ua, ub, uc] = derivatives(u, wrt(x, params.a, params.b, params.c)); // evaluate the derivatives of u with respect to x and parameters a, b, c

    cout << "u = " << u << endl;    // print the evaluated output u
    cout << "ux = " << ux << endl;  // print the evaluated derivative du/dx
    cout << "ua = " << ua << endl;  // print the evaluated derivative du/da
    cout << "ub = " << ub << endl;  // print the evaluated derivative du/db
    cout << "uc = " << uc << endl;  // print the evaluated derivative du/dc
}
