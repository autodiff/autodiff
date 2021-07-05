// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
using namespace autodiff;

// The multi-variable function for which higher-order derivatives are needed (up to 4th order)
real4th f(real4th x, real4th y, real4th z)
{
    return sin(x) * cos(y) * exp(z);
}

int main()
{
    real4th x = 1.0;
    real4th y = 2.0;
    real4th z = 3.0;

    auto dfdv = derivatives(f, along(1.0, 1.0, 2.0), at(x, y, z)); // the directional derivatives of f along direction v = (1, 1, 2) at (x, y, z) = (1, 2, 3)

    std::cout << "dfdv[0] = " << dfdv[0] << std::endl; // print the evaluated 0th order directional derivative of f along v (equivalent to f(x, y, z))
    std::cout << "dfdv[1] = " << dfdv[1] << std::endl; // print the evaluated 1st order directional derivative of f along v
    std::cout << "dfdv[2] = " << dfdv[2] << std::endl; // print the evaluated 2nd order directional derivative of f along v
    std::cout << "dfdv[3] = " << dfdv[3] << std::endl; // print the evaluated 3rd order directional derivative of f along v
    std::cout << "dfdv[4] = " << dfdv[4] << std::endl; // print the evaluated 4th order directional derivative of f along v
}

/*-------------------------------------------------------------------------------------------------
=== Note ===
---------------------------------------------------------------------------------------------------
This example would also work if dual was used instead of real. However, real
types are your best option for directional derivatives, as they were optimally
designed for this kind of derivatives.
-------------------------------------------------------------------------------------------------*/
