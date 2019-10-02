// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/forward.hpp>
using namespace autodiff;

// Define a 2nd order dual type using HigherOrderDual<N> construct.
using dual2nd = HigherOrderDual<2>;

// The single-variable function for which derivatives are needed
dual2nd f(dual2nd x, dual2nd y)
{
    return 1 + x + y + x*y + y/x + log(x/y);
}

int main()
{
    dual2nd x = 2.0;      // the input variable x
    dual2nd y = 1.0;      // the input variable y
    dual2nd u = f(x, y);  // the output variable u

    dual ux = derivative(f, wrt(x), at(x, y));  // evaluate the derivative du/dx
    dual uy = derivative(f, wrt(y), at(x, y));  // evaluate the derivative du/dy

    double uxx = derivative(f, wrt<2>(x), at(x, y));  // evaluate the derivative d²u/dxdx
    double uxy = derivative(f, wrt(x, y), at(x, y));  // evaluate the derivative d²u/dxdy
    double uyx = derivative(f, wrt(y, x), at(x, y));  // evaluate the derivative d²u/dydx
    double uyy = derivative(f, wrt<2>(y), at(x, y));  // evaluate the derivative d²u/dydy

    cout << "u   = " << u << endl;    // print the evaluated output u
    cout << "ux  = " << ux << endl;   // print the evaluated derivative du/dx
    cout << "uy  = " << uy << endl;   // print the evaluated derivative du/dy
    cout << "uxx = " << uxx << endl;  // print the evaluated derivative d²u/dxdx
    cout << "uxy = " << uxy << endl;  // print the evaluated derivative d²u/dxdy
    cout << "uyx = " << uyx << endl;  // print the evaluated derivative d²u/dydx
    cout << "uyy = " << uyy << endl;  // print the evaluated derivative d²u/dydy
}
