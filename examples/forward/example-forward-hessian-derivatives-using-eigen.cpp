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

// The scalar function for which the gradient is needed
dual2nd f(const VectorXdual2nd& x)
{
    return x.cwiseProduct(x).sum(); // sum([x(i) * x(i) for i = 1:5])
}

int main()
{
    VectorXdual2nd x(3); // the input vector x with 3 variables
    x << 1, 2, 3; // x = [1, 2, 3]

    dual2nd u; // the output scalar u = f(x) evaluated together with Hessian below
    VectorXdual g;

    MatrixXd H = hessian(f, wrt(x), at(x), u, g); // evaluate the function value u and its Hessian matrix H

    cout << "u = " << u << endl; // print the evaluated output u
    cout << "Hessian = \n" << H << endl; // print the evaluated Hessian matrix H
    cout << "g =\n" << g << endl; // print the evaluated gradient vector gp = [du/dx]
}
