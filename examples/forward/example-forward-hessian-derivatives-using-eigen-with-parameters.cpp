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

// Define a 2nd order dual type using HigherOrderDual<N> construct.
using dual2nd = HigherOrderDual<2>;

// The scalar function for which the Hessian is needed
dual2nd f(const VectorXdual2nd& x, const VectorXdual2nd& y)
{
    return sqrt(x.cwiseProduct(x).sum() + y.cwiseProduct(y).sum());
}

int main()
{
    VectorXdual2nd x(3); // the input vector x with 3 variables
    x << 1, 2, 3; // x = [1, 2, 3]

    VectorXdual2nd y(2); // the input vector y with 2 variables
    y << 4, 5; // x = [1, 2, 3]

    dual2nd u; // the output scalar u = f(x) evaluated together with Hessian below
    VectorXdual g; // gradient of f(x) evaluated together with Hessian below

    MatrixXd H = hessian(f, wrtpack(x, y), at(x, y), u, g); // evaluate the function value u and its Hessian matrix H

    cout << "u = " << u << endl; // print the evaluated output u
    cout << "Hessian = \n" << H << endl; // print the evaluated Hessian matrix H
    cout << "g =\n" << g << endl; // print the evaluated gradient vector g = [du/dx, du/dy]
}
