// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

// The scalar function for which the Hessian is needed
dual2nd f(const ArrayXdual2nd& x, const ArrayXdual2nd& p, const dual2nd& q)
{
    return (x * x).sum() + (p * p).sum() * q; // sum(x*x) + sum(p*p) + q
}

int main()
{
    using Eigen::VectorXd;
    using Eigen::MatrixXd;

    ArrayXdual2nd x(3);  // the input vector x with 3 variables
    x << 1, 2, 3;

    ArrayXdual2nd p(2);  // the input parameter vector p with 2 variables
    p << 4, 5;

    dual2nd q = -2;      // the input parameter q as a single variable

    dual2nd u; // the output scalar u = f(x) evaluated together with Hessian below
    VectorXdual g; // gradient of f(x) evaluated together with Hessian below

    MatrixXd H = hessian(f, wrt(x, p, q), at(x, p, q), u, g); // evaluate the function value u, its gradient vector g, and its Hessian matrix H with respect to (x, p, q)

    std::cout << "u = " << u << std::endl; // print the evaluated output u
    std::cout << "g =\n" << g << std::endl; // print the evaluated gradient vector g = [du/dx, du/dp, du/dq]
    std::cout << "H = \n" << H << std::endl; // print the evaluated Hessian matrix H = d²u/d[x, p, q]²
}
