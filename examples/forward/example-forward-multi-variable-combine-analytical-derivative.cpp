// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// Define functions A, Ax, Ay using double; analytical derivatives are available.
double  A(double x, double y) { return x*y; }
double Ax(double x, double y) { return x; }
double Ay(double x, double y) { return y; }

// Define functions B, Bx, By using double; analytical derivatives are available.
double  B(double x, double y) { return x + y; }
double Bx(double x, double y) { return 1.0; }
double By(double x, double y) { return 1.0; }

// Wrap A into Adual function so that it can be used within autodiff-enabled codes.
dual Adual(dual const& x, dual const& y)
{
    dual res = A(x.val, y.val);

    if(x.grad != 0.0)
        res.grad += x.grad * Ax(x.val, y.val);

    if(y.grad != 0.0)
        res.grad += y.grad * Ay(x.val, y.val);

    return res;
}

// Wrap B into Bdual function so that it can be used within autodiff-enabled codes.
dual Bdual(dual const& x, dual const& y)
{
    dual res = B(x.val, y.val);

    if(x.grad != 0.0)
        res.grad += x.grad * Bx(x.val, y.val);

    if(y.grad != 0.0)
        res.grad += y.grad * By(x.val, y.val);

    return res;
}

// Define autodiff-enabled C function that relies on Adual and Bdual
dual C(dual const& x, dual const& y)
{
    const auto A = Adual(x, y);
    const auto B = Bdual(x, y);
    return A*A + B;
}

int main()
{
    dual x = 1.0;
    dual y = 2.0;

    auto C0 = C(x, y);

    // Compute derivatives of C with respect to x and y using autodiff!
    auto Cx = derivative(C, wrt(x), at(x, y));
    auto Cy = derivative(C, wrt(y), at(x, y));

    // Compute expected analytical derivatives of C with respect to x and y
    auto x0 = x.val;
    auto y0 = y.val;
    auto expectedCx = 2.0*A(x0, y0)*Ax(x0, y0) + Bx(x0, y0);
    auto expectedCy = 2.0*A(x0, y0)*Ay(x0, y0) + By(x0, y0);

    std::cout << "C0 = " << C0 << "\n";

    std::cout << "Cx(computed) = " << Cx << "\n";
    std::cout << "Cx(expected) = " << expectedCx << "\n";

    std::cout << "Cy(computed) = " << Cy << "\n";
    std::cout << "Cy(expected) = " << expectedCy << "\n";
}

// Output:
// C0 = 7
// Cx(computed) = 5
// Cx(expected) = 5
// Cy(computed) = 9
// Cy(expected) = 9
