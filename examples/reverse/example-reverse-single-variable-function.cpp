// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// The single-variable function for which derivatives are needed
var f(var x)
{
    return 1 + x + x*x + 1/x + log(x);
}

int main()
{
    var x = 2.0;   // the input variable x
    var u = f(x);  // the output variable u

    auto [ux] = derivatives(u, wrt(x)); // evaluate the derivative of u with respect to x

    cout << "u = " << u << endl;  // print the evaluated output variable u
    cout << "ux = " << ux << endl;  // print the evaluated derivative ux
}
