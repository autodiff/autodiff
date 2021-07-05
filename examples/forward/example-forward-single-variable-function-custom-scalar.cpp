// C++ includes
#include <iostream>
#include <complex>
using namespace std;

// autodiff include
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// Specialize isArithmetic for complex to make it compatible with dual
namespace autodiff::detail {

template<typename T>
struct ArithmeticTraits<complex<T>> : ArithmeticTraits<T> {};

} // autodiff::detail

using cxdual = Dual<complex<double>, complex<double>>;

// The single-variable function for which derivatives are needed
cxdual f(cxdual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

int main()
{
    cxdual x = 2.0;   // the input variable x
    cxdual u = f(x);  // the output variable u

    cxdual dudx = derivative(f, wrt(x), at(x));  // evaluate the derivative du/dx

    cout << "u = " << u << endl;         // print the evaluated output u
    cout << "du/dx = " << dudx << endl;  // print the evaluated derivative du/dx
}
