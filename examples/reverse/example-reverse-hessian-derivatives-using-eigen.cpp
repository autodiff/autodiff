// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

// The scalar function for which the gradient is needed
var f(const ArrayXvar& x)
{
    return sqrt((x * x).sum()); // sqrt(sum([xi * xi for i = 1:5]))
}

int main()
{
    using Eigen::MatrixXd;

    VectorXvar x(5);                       // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;                    // x = [1, 2, 3, 4, 5]

    var u = f(x);                          // the output variable u

    MatrixXd dudx = hessian(u, x);         // evaluate the Hessian matrix d2u/dx2

    std::cout << "u = " << u << std::endl;           // print the evaluated output u
    std::cout << "du/dx = \n" << dudx << std::endl;  // print the evaluated gradient vector du/dx
}
