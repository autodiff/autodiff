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

// The scalar function for which the gradient is needed
dual f(const VectorXdual& x)
{
    return x.cwiseProduct(x).sum(); // sum([x(i) * x(i) for i = 1:5])
}

int main()
{
    VectorXdual x(5);    // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    dual u = f(x);  // the output variable u

    VectorXd dudx = gradient(f, x);  // evaluate the gradient vector du/dx

    cout << "u = " << u << endl;             // print the evaluated output u
    cout << "grad(u) = \n" << dudx << endl;  // print the evaluated gradient vector du/dx
}
