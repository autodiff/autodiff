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
    var y = f(x, params);                // the output variable y

    var dydx = grad(y, x);               // evaluate the derivative du/dx
    var dyda = grad(y, params.a);        // evaluate the derivative du/da
    var dydb = grad(y, params.b);        // evaluate the derivative du/db
    var dydc = grad(y, params.c);        // evaluate the derivative du/dc

    cout << "y = " << y << endl;         // print the evaluated output y
    cout << "dy/dx = " << dydx << endl;  // print the evaluated derivative dy/dx
    cout << "dy/da = " << dyda << endl;  // print the evaluated derivative dy/da
    cout << "dy/db = " << dydb << endl;  // print the evaluated derivative dy/db
    cout << "dy/dc = " << dydc << endl;  // print the evaluated derivative dy/dc
}
