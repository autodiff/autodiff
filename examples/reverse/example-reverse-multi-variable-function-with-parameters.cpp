// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse.hpp>
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
    Params params;                       // initialize the parameter variables
    params.a = 1.0;                      // the parameter a of type var, not double!
    params.b = 2.0;                      // the parameter b of type var, not double!
    params.c = 3.0;                      // the parameter c of type var, not double!

    var x = 0.5;                         // the input variable x
    var u = f(x, params);                // the output variable u

    Derivatives dud = derivatives(u);    // evaluate all derivatives of u

    var dudx = dud(x);                   // extract the derivative du/dx
    var duda = dud(params.a);            // extract the derivative du/da
    var dudb = dud(params.b);            // extract the derivative du/db
    var dudc = dud(params.c);            // extract the derivative du/dc

    cout << "u = " << u << endl;         // print the evaluated output u
    cout << "du/dx = " << dudx << endl;  // print the evaluated derivative du/dx
    cout << "du/da = " << duda << endl;  // print the evaluated derivative du/da
    cout << "du/db = " << dudb << endl;  // print the evaluated derivative du/db
    cout << "du/dc = " << dudc << endl;  // print the evaluated derivative du/dc
}
