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
dual2nd f(const VectorXdual2nd& x, dual2nd p)
{
    return x.cwiseProduct(x).sum() * exp(p); // sum([x(i) * x(i) for i = 1:5]) * exp(p)
}

int main()
{
    VectorXdual2nd x(5);    // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    dual2nd p = 3;    // the input parameter vector p with 3 variables

    dual2nd u;  // the output scalar u = f(x, p) evaluated together with gradient below
    VectorXd g; // gradient of f(x, p) evaluated together with Hessian below

    VectorXd H = hessian(f, wrtpack(p, x), at(x, p), u, g);  // evaluate the function value u and its gradient vector gp = [du/dp, du/dx]

    cout << "u = " << u << endl;    // print the evaluated output u
    cout << "Hessian = \n" << H << endl;  // print the evaluated gradient vector gp = [du/dp, du/dx]
    cout << "g =\n" << g << endl; // print the evaluated gradient vector gp = [du/dp, du/dx]
}
