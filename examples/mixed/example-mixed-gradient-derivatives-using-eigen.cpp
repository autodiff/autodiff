// C++ includes
#include <iostream>
using namespace std;

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

// autodiff include
#include <autodiff/mixed.hpp>
#include <autodiff/mixed/eigen.hpp>
using namespace autodiff::taperep;

// The scalar function for which the gradient is needed
auto f(const VectorXvar& x)
{
    return x.cwiseProduct(x).sum(); // sum([x(i) * x(i) for i = 1 : 5])
}

int main()
{
    // For now this is one from that differences from reverse mode, we need to create this functor
    // We can make tape as singleton, but I'm not sure if we need it.
    // Probably we can support both types (this, where we need to create lambda,
    // and singleton).
    auto var = [tape = tape_storage<1>{ }](auto value) mutable { return tape.variable(value); };

    // We can't create var from numerical type as is,
    // since we want to know tape.
    VectorXvar x(5);                       // the input vector x with 5 variables
    x << var(1), var(2), var(3),
         var(4), var(5);                   // x = [1, 2, 3, 4, 5]

    auto y = f(x);                         // the output variable y

    VectorXd dydx = gradient(y, x);        // evaluate the gradient vector dy/dx

    cout << "y = " << y << endl;           // print the evaluated output y
    cout << "dy/dx = \n" << dydx << endl;  // print the evaluated gradient vector dy/dx
}
