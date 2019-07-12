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
dual f(const VectorXdual& x)
{
    return x.cwiseProduct(x).sum(); // sum([x(i) * x(i) for i = 1:5])
}

int main()
{
    VectorXdual x(5);    // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    dual u;  // the output scalar u = f(x) evaluated together with gradient below

    VectorXd g = gradient(f, wrt(x), at(x), u);  // evaluate the function value u and its gradient vector g = du/dx

    cout << "u = " << u << endl;    // print the evaluated output u
    cout << "g = \n" << g << endl;  // print the evaluated gradient vector g = du/dx
}
