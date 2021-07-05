// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

int main()
{
    var x = 0.5;  // the input variable x
    var u = sin(x) * cos(x);  // the output variable u

    auto [ux] = derivativesx(u, wrt(x));  // evaluate the first order derivatives of u
    auto [uxx] = derivativesx(ux, wrt(x));  // evaluate the second order derivatives of ux

    cout << "u = " << u << endl;  // print the evaluated output variable u
    cout << "ux(autodiff) = " << ux << endl;  // print the evaluated first order derivative ux
    cout << "ux(exact) = " << 1 - 2*sin(x)*sin(x) << endl;  // print the exact first order derivative ux
    cout << "uxx(autodiff) = " << uxx << endl;  // print the evaluated second order derivative uxx
    cout << "uxx(exact) = " << -4*cos(x)*sin(x) << endl;  // print the exact second order derivative uxx
}

/*===============================================================================
Output:
=================================================================================
u = 0.420735
ux(autodiff) = 0.540302
ux(exact) = 0.540302
uxx(autodiff) = -1.68294
uxx(exact) = -1.68294
===============================================================================*/
