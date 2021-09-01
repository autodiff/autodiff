// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// The vector function for which the Jacobian is needed
VectorXreal f(const VectorXreal& x)
{
    return x * x.sum();
}

int main()
{
    using Eigen::Map;
    using Eigen::MatrixXd;

    VectorXreal x(5);                           // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;                         // x = [1, 2, 3, 4, 5]
    double y[25];                               // the output Jacobian as a flat array

    VectorXreal F;                              // the output vector F = f(x) evaluated together with Jacobian matrix below
    Map<MatrixXd> J(y, 5, 5);                   // the output Jacobian dF/dx mapped onto the flat array

    jacobian(f, wrt(x), at(x), F, J);           // evaluate the output vector F and the Jacobian matrix dF/dx

    std::cout << "F = \n" << F << std::endl;    // print the evaluated output vector F
    std::cout << "J = \n" << J << std::endl;    // print the evaluated Jacobian matrix dF/dx
    std::cout << "y = ";                        // print the flat array
    for(int i = 0 ; i < 25 ; ++i)
        std::cout << y[i] << " ";
    std::cout << std::endl;
}
