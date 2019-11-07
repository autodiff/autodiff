// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// The multi-variable function for which higher-order derivatives are needed (up to 4th order)
dual4th f(dual4th x, dual4th y, dual4th z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

int main()
{
    dual4th x = 1.0;
    dual4th y = 2.0;
    dual4th z = 3.0;

    auto [u0, ux, uxy, uxyx, uxyxz] = derivatives(f, wrt(x, y, x, z), at(x, y, z));

    std::cout << "u0 = " << u0 << std::endl;       // print the evaluated value of u = f(x, y, z)
    std::cout << "ux = " << ux << std::endl;       // print the evaluated derivative du/dx
    std::cout << "uxy = " << uxy << std::endl;     // print the evaluated derivative d²u/dxdy
    std::cout << "uxyx = " << uxyx << std::endl;   // print the evaluated derivative d³u/dx²dy
    std::cout << "uxyxz = " << uxyxz << std::endl; // print the evaluated derivative d⁴u/dx²dydz
}

/*-------------------------------------------------------------------------------------------------
=== Note ===
---------------------------------------------------------------------------------------------------
In most cases, dual can be replaced by real, as commented in other examples.
However, computing higher-order cross derivatives has definitely to be done
using higher-order dual types (e.g., dual3rd, dual4th)! This is because real
types (e.g., real2nd, real3rd, real4th) are optimally designed for computing
higher-order directional derivatives.
-------------------------------------------------------------------------------------------------*/
