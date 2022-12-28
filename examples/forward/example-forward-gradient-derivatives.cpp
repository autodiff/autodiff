// C++ includes
#include <iostream>
#include <array>
#include <numeric>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

using VectorXr = std::vector<real>;

// The scalar function for which the gradient is needed
real f(VectorXr x)
{
    std::transform(x.begin(), x.end(), x.begin(), [](const real& r){ return r * exp(r); });
    return std::accumulate(x.begin(), x.end(), real(0.));
}

int main()
{
    VectorXr x{1, 2, 3, 4, 5};                  // the input array x with 5 variables

    real u;                                     // the output scalar u = f(x) evaluated together with gradient below

    Eigen::VectorXd g = gradient(f, wrt(x), at(x), u); // evaluate the function value u and its gradient vector g = du/dx

    std::cout << "u = " << u << std::endl;      // print the evaluated output u
    std::cout << "g = \n" << g << std::endl;    // print the evaluated gradient vector g = du/dx
}
