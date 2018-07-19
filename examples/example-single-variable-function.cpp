// C++ includes
#include <iostream>
using namespace std;

// autodiff includes
#include <autodiff.hpp>
using namespace autodiff;

// The single-variable function for which derivatives are needed
var f(var x)
{
    return 1 + x + x*x + 1/x + log(x);
}

int main()
{
    var x = 2.0;                         // x - input variable of type autodiff::var
    var y = f(x);                        // y - output variable of type autodiff::var

    double dydx = grad(y, x);            // evaluate derivative dy/dx using autodiff::grad function

    cout << "y = " << y << endl;         // print evaluated output y
    cout << "dy/dx = " << dydx << endl;  // print evaluated derivative dy/dx
}
