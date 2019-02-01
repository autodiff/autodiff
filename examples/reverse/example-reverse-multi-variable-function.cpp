// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse.hpp>
using namespace autodiff;

// The multi-variable function for which derivatives are needed
var f(var x, var y, var z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

int main()
{
    var x = 1.0;                         // the input variable x
    var y = 2.0;                         // the input variable y
    var z = 3.0;                         // the input variable z
    var u = f(x, y, z);                  // the output variable u

    Derivatives dud = derivatives(u);    // evaluate all derivatives of u

    var dudx = dud(x);                   // extract the derivative du/dx
    var dudy = dud(y);                   // extract the derivative du/dy
    var dudz = dud(z);                   // extract the derivative du/dz

    cout << "u = " << u << endl;         // print the evaluated output u
    cout << "du/dx = " << dudx << endl;  // print the evaluated derivative du/dx
    cout << "du/dy = " << dudy << endl;  // print the evaluated derivative du/dy
    cout << "du/dz = " << dudz << endl;  // print the evaluated derivative du/dz
}
