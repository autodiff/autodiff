#include <iostream>

#include <autodiff/forward/dual.hpp>
using namespace autodiff;

dual f(dual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

int main()
{
    dual x = 1.0;
    dual u = f(x);
    double dudx = derivative(f, wrt(x), at(x));

    std::cout << "u = " << u << std::endl;
    std::cout << "du/dx = " << dudx << std::endl;
}
