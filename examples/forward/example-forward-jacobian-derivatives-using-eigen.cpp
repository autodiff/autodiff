// C++ includes
#include <iostream>
using namespace std;

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

// autodiff include
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
using namespace autodiff;

// The vector function for which the Jacobian is needed
VectorXdual f(const VectorXdual& x)
{
    return x * x.sum();
}

int main()
{
    VectorXdual x(5);    // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    VectorXdual F;  // the output vector F = f(x) evaluated together with Jacobian matrix below

    MatrixXd J = jacobian(f, wrt(x), at(x), F);  // evaluate the output vector F and the Jacobian matrix dF/dx

    cout << "F = \n" << F << endl;  // print the evaluated output vector F
    cout << "J = \n" << J << endl;  // print the evaluated Jacobian matrix dF/dx
}
