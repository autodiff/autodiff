// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

int main()
{
    var x = 0.5;                              // the input variable x
    var u = sin(x) * cos(x);                  // the output variable u

    DerivativesX dud = derivativesx(u);       // evaluate the first order derivatives of u

    var dudx = dud(x);                        // extract the first order derivative du/dx of type var, not double!

    DerivativesX d2udxd = derivativesx(dudx); // evaluate the second order derivatives of du/dx

    var d2udxdx = d2udxd(x);                  // extract the second order derivative d2u/dxdx of type var, not double!

    std::cout << "u = " << u << std::endl;              // print the evaluated output variable u
    std::cout << "du/dx = " << dudx << std::endl;       // print the evaluated first order derivative du/dx
    std::cout << "d2u/dx2 = " << d2udxdx << std::endl;  // print the evaluated second order derivative d2u/dxdx
}
