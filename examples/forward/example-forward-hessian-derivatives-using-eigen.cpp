// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

// The scalar function for which the gradient is needed
dual2nd f(const VectorXdual2nd& x)
{
    return x.cwiseProduct(x).sum(); // sum([x(i) * x(i) for i = 1:5])
}

int main()
{
    using Eigen::MatrixXd;

    VectorXdual2nd x(3); // the input vector x with 3 variables
    x << 1, 2, 3; // x = [1, 2, 3]

    dual2nd u; // the output scalar u = f(x) evaluated together with Hessian below
    VectorXdual g;

    MatrixXd H = hessian(f, wrt(x), at(x), u, g); // evaluate the function value u and its Hessian matrix H

    std::cout << "u = "   << u << std::endl; // print the evaluated output u
    std::cout << "g = \n" << g << std::endl; // print the evaluated gradient vector g = du/dx
    std::cout << "H = \n" << H << std::endl; // print the evaluated Hessian matrix H = d²u/dx²
}
