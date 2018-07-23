// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff.hpp>
using namespace autodiff;

int main()
{
    var x = 1.0;                              // the input variable x
    var y = 0.5;                              // the input variable y
    var z = 2.0;                              // the input variable z

    var u = x * log(y) * exp(z);              // the output variable u
    
    var dudx = gradx(u, x);                   // the first order derivative du/dx of type var, not double!
    var dudy = gradx(u, y);                   // the first order derivative du/dy of type var, not double!
    var dudz = gradx(u, z);                   // the first order derivative du/dz of type var, not double!

    var d2udx2 = gradx(dudx, x);              // the second order derivative d2u/dx2 of type var, not double!
    var d2udy2 = gradx(dudy, y);              // the second order derivative d2u/dy2 of type var, not double!
    var d2udz2 = gradx(dudz, z);              // the second order derivative d2u/dz2 of type var, not double!

    var d2udxdy = gradx(dudx, y);             // the second order derivative d2u/dxdy of type var, not double!
    var d2udydz = gradx(dudy, z);             // the second order derivative d2u/dydz of type var, not double!
    var d2udzdx = gradx(dudz, x);             // the second order derivative d2u/dzdx of type var, not double!

    cout << "u = " << u << endl;              // print the evaluated output variable u

    cout << "du/dx = " << dudx << endl;       // print the evaluated first order derivative du/dx
    cout << "du/dy = " << dudy << endl;       // print the evaluated first order derivative du/dy
    cout << "du/dz = " << dudz << endl;       // print the evaluated first order derivative du/dz
    
    cout << "d2u/dx2 = " << d2udx2 << endl;   // print the second order derivative d2u/dx2
    cout << "d2u/dy2 = " << d2udy2 << endl;   // print the second order derivative d2u/dy2
    cout << "d2u/dz2 = " << d2udz2 << endl;   // print the second order derivative d2u/dz2

    cout << "d2u/dxdy = " << d2udxdy << endl; // print the second order derivative d2u/dxdy
    cout << "d2u/dydz = " << d2udydz << endl; // print the second order derivative d2u/dydz
    cout << "d2u/dzdx = " << d2udzdx << endl; // print the second order derivative d2u/dzdx
}
