#include <iostream>
using namespace std;

#include <autodiff/forward.hpp>
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

    cout << "u = " << u << endl;
    cout << "du/dx = " << dudx << endl;
}
