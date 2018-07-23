// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff.hpp>
using namespace autodiff;

int main()
{
    var x = 0.5;                             // the input variable x

    var y = sin(x) * cos(x);                 // the output variable y
    var dydx = gradx(y, x);                  // the first order derivative dy/dx of type var, not double!
    var d2ydx2 = gradx(dydx, x);             // the second order derivative d2y/dx2 of type var, not double!

    cout << "y = " << y << endl;             // print the evaluated output variable y
    cout << "dy/dx = " << dydx << endl;      // print the evaluated first order derivative dy/dx
    cout << "d2y/dx2 = " << d2ydx2 << endl;  // print the evaluated second order derivative d2y/dx2
}
