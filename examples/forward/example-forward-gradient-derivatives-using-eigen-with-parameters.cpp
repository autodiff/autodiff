// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// The scalar function for which the gradient is needed
real f(const ArrayXreal& x, const ArrayXreal& p)
{
    return (x * x).sum() * exp(p.sum()); // sum([xi * xi for i = 1:5]) * exp(sum(p))
}

int main()
{
    using Eigen::VectorXd;

    VectorXreal x(5);    // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    VectorXreal p(3);    // the input parameter vector p with 3 variables
    p << 1, 2, 3;        // p = [1, 2, 3]

    real u;  // the output scalar u = f(x, p) evaluated together with gradient below

    VectorXd gx = gradient(f, wrt(x), at(x, p), u);  // evaluate the function value u and its gradient vector gx = du/dx
    VectorXd gp = gradient(f, wrt(p), at(x, p), u);  // evaluate the function value u and its gradient vector gp = du/dp

    std::cout << "u = " << u << std::endl;      // print the evaluated output u
    std::cout << "gx = \n" << gx << std::endl;  // print the evaluated gradient vector gx = du/dx
    std::cout << "gp = \n" << gp << std::endl;  // print the evaluated gradient vector gp = du/dp
}

//-------------------------------------------------------------------------------------------------
// Note
//-------------------------------------------------------------------------------------------------
// This example would also work if dual was used instead. However, if gradient,
// Jacobian, and directional derivatives are all you need, then real types are
// your best option. You want to use dual types when evaluating higher-order
// cross derivatives, which is not supported for real types.
//-------------------------------------------------------------------------------------------------
