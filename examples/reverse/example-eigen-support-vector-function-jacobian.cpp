// C++ includes
#include <iostream>
using namespace std;

// Eigen includes
#include <eigen3/Eigen/Core>
using namespace Eigen;

// autodiff include
#define AUTODIFF_ENABLE_EIGEN_SUPPORT
#include <autodiff/reverse.hpp>
using namespace autodiff;

// The vector function for which the Jacobian is needed
VectorXv f(const VectorXv& x)
{
    return (x / x.sum()).array().log(); // log(x / sum(x))
}

int main()
{
    VectorXv x(5);                         // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;                    // x = [1, 2, 3, 4, 5]

    VectorXv y = f(x);                     // the output vector y

    MatrixXd dydx = grad(y, x);            // evaluate the Jacobian matrix dy/dx

    cout << "y = \n" << y << endl;         // print the evaluated output vector y
    cout << "dy/dx = \n" << dydx << endl;  // print the evaluated Jacobian matrix dy/dx
}
