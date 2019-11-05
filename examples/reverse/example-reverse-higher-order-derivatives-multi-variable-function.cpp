// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

int main()
{
    var x = 1.0;                                 // the input variable x
    var y = 0.5;                                 // the input variable y
    var z = 2.0;                                 // the input variable z

    var u = x * log(y) * exp(z);                 // the output variable u

    DerivativesX dud = derivativesx(u);          // evaluate all derivatives of u using autodiff::derivativesx!

    var dudx = dud(x);                           // extract the first order derivative du/dx of type var, not double!
    var dudy = dud(y);                           // extract the first order derivative du/dy of type var, not double!
    var dudz = dud(z);                           // extract the first order derivative du/dz of type var, not double!

    DerivativesX d2udxd = derivativesx(dudx);    // evaluate all derivatives of dudx using autodiff::derivativesx!
    DerivativesX d2udyd = derivativesx(dudy);    // evaluate all derivatives of dudy using autodiff::derivativesx!
    DerivativesX d2udzd = derivativesx(dudz);    // evaluate all derivatives of dudz using autodiff::derivativesx!

    var d2udxdx = d2udxd(x);                     // extract the second order derivative d2u/dxdx of type var, not double!
    var d2udxdy = d2udxd(y);                     // extract the second order derivative d2u/dxdy of type var, not double!
    var d2udxdz = d2udxd(z);                     // extract the second order derivative d2u/dxdz of type var, not double!

    var d2udydx = d2udyd(x);                     // extract the second order derivative d2u/dydx of type var, not double!
    var d2udydy = d2udyd(y);                     // extract the second order derivative d2u/dydy of type var, not double!
    var d2udydz = d2udyd(z);                     // extract the second order derivative d2u/dydz of type var, not double!

    var d2udzdx = d2udzd(x);                     // extract the second order derivative d2u/dzdx of type var, not double!
    var d2udzdy = d2udzd(y);                     // extract the second order derivative d2u/dzdy of type var, not double!
    var d2udzdz = d2udzd(z);                     // extract the second order derivative d2u/dzdz of type var, not double!

    std::cout << "u = " << u << std::endl;                 // print the evaluated output variable u

    std::cout << "du/dx = " << dudx << std::endl;          // print the evaluated first order derivative du/dx
    std::cout << "du/dy = " << dudy << std::endl;          // print the evaluated first order derivative du/dy
    std::cout << "du/dz = " << dudz << std::endl;          // print the evaluated first order derivative du/dz

    std::cout << "d2udxdx = " << d2udxdx << std::endl;     // print the evaluated second order derivative d2u/dxdx
    std::cout << "d2udxdy = " << d2udxdy << std::endl;     // print the evaluated second order derivative d2u/dxdy
    std::cout << "d2udxdz = " << d2udxdz << std::endl;     // print the evaluated second order derivative d2u/dxdz

    std::cout << "d2udydx = " << d2udydx << std::endl;     // print the evaluated second order derivative d2u/dydx
    std::cout << "d2udydy = " << d2udydy << std::endl;     // print the evaluated second order derivative d2u/dydy
    std::cout << "d2udydz = " << d2udydz << std::endl;     // print the evaluated second order derivative d2u/dydz

    std::cout << "d2udzdx = " << d2udzdx << std::endl;     // print the evaluated second order derivative d2u/dzdx
    std::cout << "d2udzdy = " << d2udzdy << std::endl;     // print the evaluated second order derivative d2u/dzdy
    std::cout << "d2udzdz = " << d2udzdz << std::endl;     // print the evaluated second order derivative d2u/dzdz
}
