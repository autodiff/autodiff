// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// The multi-variable function for which derivatives are needed
var f(var x, var y, var z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

int main()
{
    var x = 1.0;         // the input variable x
    var y = 2.0;         // the input variable y
    var z = 3.0;         // the input variable z
    var u = f(x, y, z);  // the output variable u

    auto [ux, uy, uz] = derivatives(u, wrt(x, y, z)); // evaluate the derivatives of u with respect to x, y, z

    cout << "u = " << u << endl;    // print the evaluated output u
    cout << "ux = " << ux << endl;  // print the evaluated derivative ux
    cout << "uy = " << uy << endl;  // print the evaluated derivative uy
    cout << "uz = " << uz << endl;  // print the evaluated derivative uz
}
