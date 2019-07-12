// C++ includes
#include <iostream>
using namespace std;

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

// autodiff include
#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>
using namespace autodiff;

// The scalar function for which the gradient is needed
var f(const VectorXvar& x)
{
    return sqrt(x.cwiseProduct(x).sum()); // sqrt(sum([x(i) * x(i) for i = 1:5]))
}

int main()
{
    VectorXvar x(5);                       // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;                    // x = [1, 2, 3, 4, 5]

    var u = f(x);                          // the output variable u

    MatrixXd dudx = hessian(u, x);         // evaluate the Hessian matrix d2u/dx2

    cout << "u = " << u << endl;           // print the evaluated output u
    cout << "du/dx = \n" << dudx << endl;  // print the evaluated gradient vector du/dx
}
