// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// The vector function for which a 4th order directional Taylor series will be computed.
ArrayXreal4th f(const ArrayXreal4th& x)
{
    return x.sin() / x;
}

int main()
{
    using Eigen::ArrayXd;

    ArrayXreal4th x(5);                        // the input vector x
    x << 1.0, 2.0, 3.0, 4.0, 5.0;

    ArrayXd v(5);                              // the direction vector v
    v << 1.0, 1.0, 1.0, 1.0, 1.0;

    auto g = taylorseries(f, along(v), at(x)); // the function g(t) as a 4th order Taylor approximation of f(x + t·v)

    double t = 0.1;                            // the step length used to evaluate g(t), the Taylor approximation of f(x + t·v)

    ArrayXreal4th u = f(x + t * v);            // the exact value of f(x + t·v)

    ArrayXd utaylor = g(t);                    // the 4th order Taylor estimate of f(x + t·v)

    std::cout << std::fixed;
    std::cout << "Comparison between exact evaluation and 4th order Taylor estimate of f(x + t·v):" << std::endl;
    std::cout << "u(exact)  = " << u.transpose() << std::endl;
    std::cout << "u(taylor) = " << utaylor.transpose() << std::endl;
}

/*-------------------------------------------------------------------------------------------------
=== Output ===
---------------------------------------------------------------------------------------------------
Comparison between exact evaluation and 4th order Taylor estimate of f(x + t·v):
u(exact)  =  0.810189  0.411052  0.013413 -0.199580 -0.181532
u(taylor) =  0.810189  0.411052  0.013413 -0.199580 -0.181532
-------------------------------------------------------------------------------------------------*/
