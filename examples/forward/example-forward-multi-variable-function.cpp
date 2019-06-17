// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/forward.hpp>
using namespace autodiff;

// The multi-variable function for which derivatives are needed
dual f(dual x, dual y, dual z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

int main()
{
    dual x = 1.0;  // the input variable x
    dual y = 2.0;  // the input variable y
    dual z = 3.0;  // the input variable z

    dual u = f(x, y, z);  // the output scalar u = f(x, y, z)

    double dudx = derivative(f, wrt(x), at(x, y, z));  // evaluate du/dx
    double dudy = derivative(f, wrt(y), at(x, y, z));  // evaluate du/dy
    double dudz = derivative(f, wrt(z), at(x, y, z));  // evaluate du/dz

    cout << "u = " << u << endl;         // print the evaluated output u
    cout << "du/dx = " << dudx << endl;  // print the evaluated derivative du/dx
    cout << "du/dy = " << dudy << endl;  // print the evaluated derivative du/dy
    cout << "du/dz = " << dudz << endl;  // print the evaluated derivative du/dz
}
