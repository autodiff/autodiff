// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// The vector function with parameters for which the Jacobian is needed
VectorXreal f(const VectorXreal& x, const VectorXreal& p)
{
    return x * exp(p.sum());
}

int main()
{
    using Eigen::MatrixXd;

    VectorXreal x(5);    // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    VectorXreal p(3);    // the input parameter vector p with 3 variables
    p << 1, 2, 3;        // p = [1, 2, 3]

    VectorXreal F;  // the output vector F = f(x, p) evaluated together with Jacobian below

    MatrixXd Jx = jacobian(f, wrt(x), at(x, p), F);     // evaluate the function and the Jacobian matrix Jx = dF/dx
    MatrixXd Jp = jacobian(f, wrt(p), at(x, p), F);     // evaluate the function and the Jacobian matrix Jp = dF/dp
    MatrixXd Jpx = jacobian(f, wrt(p, x), at(x, p), F); // evaluate the function and the Jacobian matrix Jpx = [dF/dp, dF/dx]

    std::cout << "F = \n" << F << std::endl;     // print the evaluated output vector F
    std::cout << "Jx = \n" << Jx << std::endl;   // print the evaluated Jacobian matrix dF/dx
    std::cout << "Jp = \n" << Jp << std::endl;   // print the evaluated Jacobian matrix dF/dp
    std::cout << "Jpx = \n" << Jpx << std::endl; // print the evaluated Jacobian matrix [dF/dp, dF/dx]
}
