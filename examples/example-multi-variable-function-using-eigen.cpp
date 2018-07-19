// C++ includes
#include <iostream>
using namespace std;

// Eigen includes
#include <eigen3/Eigen/Core>
using namespace Eigen;

// autodiff include
#define AUTODIFF_ENABLE_EIGEN_SUPPORT
#include <autodiff.hpp>
using namespace autodiff;

// The multi-variable function with a vector input and a scalar output
var f(const VectorXv& x)
{
    return sqrt(x.cwiseProduct(x).sum()); // sqrt(sum([x(i) * x(i) for i = 1:5]))
}

// The multi-variable function with a vector input and a vector output
VectorXv g(const VectorXv& x)
{
//    return (x / x.sum()).array().log(); // log(x / sum(x))
    return (x / x.sum()); // log(x / sum(x))
}

int main()
{
    VectorXv x(5);                       // x - input vector with 5 variables
    x << 1, 2, 3, 4, 5;                  // x = [1, 2, 3, 4, 5]

    var y = f(x);                        // y - output variable

    RowVectorXd dydx = grad(y, x);       // evaluate the derivative dy/dx - a row vector

    cout << "y = " << y << endl;         // print the evaluated output y
    cout << "dy/dx = " << dydx << endl;  // print the evaluated derivatives dy/dx

    VectorXv u = g(x);                   // u - output variable

    MatrixXd dudx = grad(u, x);          // evaluate the derivative du/dx - a matrix

    cout << "u = \n" << u << endl;         // print the evaluated output u
    cout << "du/dx \n =" << dudx << endl;  // print the evaluated derivatives du/dx
}
