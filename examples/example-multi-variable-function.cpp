// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff.hpp>
using namespace autodiff;

// The multi-variable function for which derivatives are needed
var f(var x, var y, var z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

int main()
{
    var x = 1.0;                         // x - input variable
    var y = 2.0;                         // y - input variable
    var z = 3.0;                         // z - input variable
    var u = f(x, y, z);                  // u - output variable

    var dudx = grad(u, x);               // evaluate the derivative du/dx
    var dudy = grad(u, y);               // evaluate the derivative du/dy
    var dudz = grad(u, z);               // evaluate the derivative du/dz

    cout << "u = " << u << endl;         // print the evaluated output u
    cout << "du/dx = " << dudx << endl;  // print the evaluated derivative du/dx
    cout << "du/dy = " << dudy << endl;  // print the evaluated derivative du/dy
    cout << "du/dz = " << dudz << endl;  // print the evaluated derivative du/dz
}
