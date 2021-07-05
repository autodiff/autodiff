// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// The scalar function for which a 4th order directional Taylor series will be computed.
real4th f(const real4th& x, const real4th& y, const real4th& z)
{
    return sin(x * y) * cos(x * z) * exp(z);
}

int main()
{
    real4th x = 1.0;                                       // the input vector x
    real4th y = 2.0;                                       // the input vector y
    real4th z = 3.0;                                       // the input vector z

    auto g = taylorseries(f, along(1, 1, 2), at(x, y, z)); // the function g(t) as a 4th order Taylor approximation of f(x + t, y + t, z + 2t)

    double t = 0.1;                                        // the step length used to evaluate g(t), the Taylor approximation of f(x + t, y + t, z + 2t)

    real4th u = f(x + t, y + t, z + 2*t);                  // the exact value of f(x + t, y + t, z + 2t)

    double utaylor = g(t);                                 // the 4th order Taylor estimate of f(x + t, y + t, z + 2t)

    std::cout << std::fixed;
    std::cout << "Comparison between exact evaluation and 4th order Taylor estimate of f(x + t, y + t, z + 2t):" << std::endl;
    std::cout << "u(exact)  = " << u << std::endl;
    std::cout << "u(taylor) = " << utaylor << std::endl;
}

/*-------------------------------------------------------------------------------------------------
=== Output ===
---------------------------------------------------------------------------------------------------
Comparison between exact evaluation and 4th order Taylor estimate of f(x + t, y + t, z + 2t):
u(exact)  = -16.847071
u(taylor) = -16.793986
-------------------------------------------------------------------------------------------------*/
