// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/forward.hpp>
using namespace autodiff;

// The single-variable function for which derivatives are needed
dual f(dual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

int main()
{
    dual x = 2.0;   // the input variable x
    dual u = f(x);  // the output variable u

    double dudx = derivative(f, wrt(x), at(x));  // evaluate the derivative du/dx

    cout << "u = " << u << endl;         // print the evaluated output u
    cout << "du/dx = " << dudx << endl;  // print the evaluated derivative du/dx
}
