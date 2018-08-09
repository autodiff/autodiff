// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse.hpp>
using namespace autodiff;

// The single-variable function for which derivatives are needed
var f(var x)
{
    return 1 + x + x*x + 1/x + log(x);
}

int main()
{
    var x = 2.0;                         // the input variable x
    var y = f(x);                        // the output variable y

    var dydx = grad(y, x);               // evaluate the derivative dy/dx

    cout << "y = " << y << endl;         // print the evaluated output y
    cout << "dy/dx = " << dydx << endl;  // print the evaluated derivative dy/dx
}
