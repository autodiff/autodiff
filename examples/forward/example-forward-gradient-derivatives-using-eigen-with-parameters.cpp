// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// The scalar function for which the gradient is needed
real f(const ArrayXreal& x, const ArrayXreal& p, const real& q)
{
    return (x * x).sum() * p.sum() * exp(q); // sum([xi * xi for i = 1:5]) * sum(p) * exp(q)
}

int main()
{
    using Eigen::VectorXd;

    ArrayXreal x(5);    // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5; // x = [1, 2, 3, 4, 5]

    ArrayXreal p(3);    // the input parameter vector p with 3 variables
    p << 1, 2, 3;       // p = [1, 2, 3]

    real q = -2;        // the input parameter q as a single variable

    real u;             // the output scalar u = f(x, p, q) evaluated together with gradient below

    VectorXd gx   = gradient(f, wrt(x), at(x, p, q), u);       // evaluate the function value u and its gradient vector gx = du/dx
    VectorXd gp   = gradient(f, wrt(p), at(x, p, q), u);       // evaluate the function value u and its gradient vector gp = du/dp
    VectorXd gq   = gradient(f, wrt(q), at(x, p, q), u);       // evaluate the function value u and its gradient vector gq = du/dq
    VectorXd gqpx = gradient(f, wrt(q, p, x), at(x, p, q), u); // evaluate the function value u and its gradient vector gqpx = [du/dq, du/dp, du/dx]

    std::cout << "u = " << u << std::endl;       // print the evaluated output u
    std::cout << "gx = \n" << gx << std::endl;   // print the evaluated gradient vector gx = du/dx
    std::cout << "gp = \n" << gp << std::endl;   // print the evaluated gradient vector gp = du/dp
    std::cout << "gq = \n" << gq << std::endl;   // print the evaluated gradient vector gq = du/dq
    std::cout << "gqpx = \n" << gqpx << std::endl; // print the evaluated gradient vector gqpx = [du/dq, du/dp, du/dx]
}

/*-------------------------------------------------------------------------------------------------
=== Note ===
---------------------------------------------------------------------------------------------------
This example would also work if dual was used instead. However, if gradient,
Jacobian, and directional derivatives are all you need, then real types are your
best option. You want to use dual types when evaluating higher-order cross
derivatives, which is not supported for real types.
-------------------------------------------------------------------------------------------------*/
