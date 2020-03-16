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
    VectorXvar x(5);     // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    var u = f(x);  // the output variable u

    VectorXd g;  // the gradient vector to be computed in method `hessian`
    MatrixXd H = hessian(u, x, g);  // evaluate the Hessian matrix H and the gradient vector g of u

    cout << "u = " << u << endl;    // print the evaluated output variable u
    cout << "g = \n" << g << endl;  // print the evaluated gradient vector of u
    cout << "H = \n" << H << endl;  // print the evaluated Hessian matrix of u
}
