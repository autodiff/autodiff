// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

// The vector function with parameters for which the Jacobian is needed
VectorXdual f(const VectorXdual& x, const VectorXdual& p)
{
    return x * exp(p.sum());
}

int main()
{
    using Eigen::MatrixXd;

    VectorXdual x(5);    // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    VectorXdual p(3);    // the input parameter vector p with 3 variables
    p << 1, 2, 3;        // p = [1, 2, 3]

    VectorXdual F;  // the output vector F = f(x, p) evaluated together with Jacobian below

    MatrixXd Jx = jacobian(f, wrt(x), at(x, p), F);  // evaluate the function and the Jacobian matrix dF/dx
    MatrixXd Jp = jacobian(f, wrt(p), at(x, p), F);  // evaluate the function and the Jacobian matrix dF/dp

    std::cout << "F = \n" << F << std::endl;    // print the evaluated output vector F
    std::cout << "Jx = \n" << Jx << std::endl;  // print the evaluated Jacobian matrix dF/dx
    std::cout << "Jp = \n" << Jp << std::endl;  // print the evaluated Jacobian matrix dF/dp
}
