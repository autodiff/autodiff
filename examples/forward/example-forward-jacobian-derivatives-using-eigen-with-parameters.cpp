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

// The vector function with parameters for which the Jacobian is needed
VectorXdual f(const VectorXdual& x, const VectorXdual& p)
{
    return x * exp(p.sum());
}

int main()
{
    VectorXdual x(5);    // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    VectorXdual p(3);    // the input parameter vector p with 3 variables
    p << 1, 2, 3;        // p = [1, 2, 3]

    VectorXdual F;  // the output vector F = f(x, p) evaluated together with Jacobian below

    MatrixXd Jx = jacobian(f, wrt(x), at(x, p), F);  // evaluate the function and the Jacobian matrix dF/dx
    MatrixXd Jp = jacobian(f, wrt(p), at(x, p), F);  // evaluate the function and the Jacobian matrix dF/dp
    MatrixXd Jpx = jacobian(f, wrtpack(p, x), at(x, p), F);  // evaluate the function and the Jacobian matrix [dF/dp, dF/dx]

    cout << "F = \n" << F << endl;    // print the evaluated output vector F
    cout << "Jx = \n" << Jx << endl;  // print the evaluated Jacobian matrix dF/dx
    cout << "Jp = \n" << Jp << endl;  // print the evaluated Jacobian matrix dF/dp
    cout << "Jpx = \n" << Jpx << endl;  // print the evaluated Jacobian matrix [dF/dp, dF/dx]
}
