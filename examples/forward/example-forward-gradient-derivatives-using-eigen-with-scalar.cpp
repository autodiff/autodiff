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

// The scalar function for which the gradient is needed
dual f(const VectorXdual& x, dual p)
{
    return x.cwiseProduct(x).sum() * exp(p); // sum([x(i) * x(i) for i = 1:5]) * exp(p)
}

int main()
{
    VectorXdual x(5);    // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    dual p = 3;    // the input parameter vector p with 3 variables

    dual u;  // the output scalar u = f(x, p) evaluated together with gradient below

    VectorXd gpx = gradient(f, wrtpack(p, x), at(x, p), u);  // evaluate the function value u and its gradient vector gp = [du/dp, du/dx]

    cout << "u = " << u << endl;    // print the evaluated output u
    cout << "gpx = \n" << gpx << endl;  // print the evaluated gradient vector gp = [du/dp, du/dx]
}
