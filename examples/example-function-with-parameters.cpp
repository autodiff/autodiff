// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff.hpp>
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
    params.a = 1.0;                      // a - parameter variable
    params.b = 2.0;                      // b - parameter variable
    params.c = 3.0;                      // c - parameter variable

    var x = 0.5;                         // x - input variable
    var y = f(x, params);                // y - output variable

    double dydx = grad(y, x);            // evaluate derivative du/dx
    double dyda = grad(y, params.a);     // evaluate derivative du/da
    double dydb = grad(y, params.b);     // evaluate derivative du/db
    double dydc = grad(y, params.c);     // evaluate derivative du/dc

    cout << "y = " << y << endl;         // print evaluated output y
    cout << "dy/dx = " << dydx << endl;  // print evaluated derivative dy/dx
    cout << "dy/da = " << dyda << endl;  // print evaluated derivative dy/da
    cout << "dy/db = " << dydb << endl;  // print evaluated derivative dy/db
    cout << "dy/dc = " << dydc << endl;  // print evaluated derivative dy/dc
}
