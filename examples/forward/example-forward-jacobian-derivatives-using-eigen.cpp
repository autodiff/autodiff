// C++ includes
#include <iostream>
using namespace std;

// Eigen includes
#include <eigen3/Eigen/Core>
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

    VectorXdual u = f(x);  // the output vector u

    MatrixXd dudx = jacobian(f, u, x);  // evaluate the Jacobian matrix du/dx

    cout << "u = \n" << u << endl;         // print the evaluated output vector u
    cout << "du/dx = \n" << dudx << endl;  // print the evaluated Jacobian matrix du/dx
}
