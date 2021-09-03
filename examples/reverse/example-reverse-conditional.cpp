// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// A two-variable piecewise function for which derivatives are needed
var f(var x, var y) { return condition(x < y, x * y, x * x); }

int main()
{
    var x = 1.0;   // the input variable x
    var y = 2.0;   // the input variable y
    var u = f(x, y);  // the output variable u
    auto [ux, uy] = derivatives(u, wrt(x, y)); // evaluate the derivatives of u

    cout << "x = " << x << ", y = " << y << endl;
    cout << "u = " << u << endl;  // x = 1, y = 2, so x < y, so x * y = 2
    cout << "ux = " << ux << endl;  // d/dx x * y = y = 2
    cout << "uy = " << uy << endl;  // d/dy x * y = x = 1

    x.update(3.0); // Change the value of x in the expression tree
    u.update(); // Update the expression tree in a sweep
    auto [ux2, uy2] = derivatives(u, wrt(x, y)); // re-evaluate the derivatives

    cout << "x = " << x << ", y = " << y << endl;
    cout << "u = " << u << endl;  // Now x > y, so x * x = 9
    cout << "ux = " << ux2 << endl;  // d/dx x * x = 2x = 6
    cout << "uy = " << uy2 << endl;  // d/dy x * x = 0

    // condition-associated functions
    cout << "min(x, y) = " << min(x, y) << endl;
    cout << "max(x, y) = " << max(x, y) << endl;
    cout << "sgn(x) = " << sgn(x) << endl;
}
