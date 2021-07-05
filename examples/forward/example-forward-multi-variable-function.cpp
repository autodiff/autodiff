// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// The multi-variable function for which derivatives are needed
dual f(dual x, dual y, dual z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

int main()
{
    dual x = 1.0;
    dual y = 2.0;
    dual z = 3.0;

    dual u = f(x, y, z);

    double dudx = derivative(f, wrt(x), at(x, y, z));
    double dudy = derivative(f, wrt(y), at(x, y, z));
    double dudz = derivative(f, wrt(z), at(x, y, z));

    std::cout << "u = " << u << std::endl;         // print the evaluated output u = f(x, y, z)
    std::cout << "du/dx = " << dudx << std::endl;  // print the evaluated derivative du/dx
    std::cout << "du/dy = " << dudy << std::endl;  // print the evaluated derivative du/dy
    std::cout << "du/dz = " << dudz << std::endl;  // print the evaluated derivative du/dz
}
