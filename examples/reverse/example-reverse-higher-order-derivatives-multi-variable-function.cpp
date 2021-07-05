// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

int main()
{
    var x = 1.0;  // the input variable x
    var y = 0.5;  // the input variable y
    var z = 2.0;  // the input variable z

    var u = x * log(y) * exp(z);  // the output variable u

    auto [ux, uy, uz] = derivativesx(u, wrt(x, y, z));  // evaluate the derivatives of u with respect to x, y, z.

    auto [uxx, uxy, uxz] = derivativesx(ux, wrt(x, y, z)); // evaluate the derivatives of ux with respect to x, y, z.
    auto [uyx, uyy, uyz] = derivativesx(uy, wrt(x, y, z)); // evaluate the derivatives of uy with respect to x, y, z.
    auto [uzx, uzy, uzz] = derivativesx(uz, wrt(x, y, z)); // evaluate the derivatives of uz with respect to x, y, z.

    cout << "u = " << u << endl;  // print the evaluated output variable u

    cout << "ux = " << ux << endl;  // print the evaluated first order derivative ux
    cout << "uy = " << uy << endl;  // print the evaluated first order derivative uy
    cout << "uz = " << uz << endl;  // print the evaluated first order derivative uz

    cout << "uxx = " << uxx << endl;  // print the evaluated second order derivative uxx
    cout << "uxy = " << uxy << endl;  // print the evaluated second order derivative uxy
    cout << "uxz = " << uxz << endl;  // print the evaluated second order derivative uxz

    cout << "uyx = " << uyx << endl;  // print the evaluated second order derivative uyx
    cout << "uyy = " << uyy << endl;  // print the evaluated second order derivative uyy
    cout << "uyz = " << uyz << endl;  // print the evaluated second order derivative uyz

    cout << "uzx = " << uzx << endl;  // print the evaluated second order derivative uzx
    cout << "uzy = " << uzy << endl;  // print the evaluated second order derivative uzy
    cout << "uzz = " << uzz << endl;  // print the evaluated second order derivative uzz
}
