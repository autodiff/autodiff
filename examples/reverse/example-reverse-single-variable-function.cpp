// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// The single-variable function for which derivatives are needed
var f(var x)
{
    return 1 + x + x*x + 1/x + log(x);
}

int main()
{
    var x = 2.0;                         // the input variable x
    var u = f(x);                        // the output variable u

    Derivatives dud = derivatives(u);    // evaluate all derivatives of u

    var dudx = dud(x);                   // extract the derivative du/dx

    std::cout << "u = " << u << std::endl;         // print the evaluated output u
    std::cout << "du/dx = " << dudx << std::endl;  // print the evaluated derivative du/dx
}
