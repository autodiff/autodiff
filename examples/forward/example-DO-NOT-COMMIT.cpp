// #include <iostream>

// // autodiff include
// #include "Eigen/Dense"
// #include <autodiff/forward.hpp>
// #include <autodiff/forward/eigen.hpp>

// template <typename T>
// auto pow2(const T& x0) { return x0 * x0; }


// // template <typename T>
// // auto pow2(T&& x0) { return std::forward<T>(x0) * std::forward<T>(x0); }


// struct Bar
// {};

// template<typename T>
// struct Foo
// {
//     T val;
// };

// template<typename T>
// auto createFoo(T&& data)
// {
//     return Foo<T>{ std::forward<T>(data) };
// }

// template <typename T>
// auto Rosenbrock(const T& x0, const T& x1)
// {
//     // bool a = pow2(pow2(x0) - x1);
//     // bool a = pow2(pow2(x0));
//     // bool a = pow2(pow2(x0) - x1);
//     // bool a = ((x0*x0) - x1)*((x0*x0) - x1);
//     // bool a = (x0 - (x1+x0));
//     // bool a = (x0 - autodiff::dual(x1));
//     // bool a = (x0 - x1);
//     // return 100.0 * pow2(pow2(x0) - x1) + pow2(1.0 - x0);
//     // return 100.0 * (x0*x0 - x1)*(x0*x0 - x1) + (1.0 - x0)*(1.0 - x0);
//     // return 100.0 * (pow2(x0) - x1)*(pow2(x0) - x1) + (1.0 - x0)*(1.0 - x0);
//     return 100.0 * pow2(pow2(x0) - x1) + (1.0 - x0)*(1.0 - x0);
// }

// // Analytic gradient
// Eigen::VectorXd Rosenbrock_exact_gradient(const double x0, const double x1)
// {
//     Eigen::VectorXd o(2);
//     o(0) = 200 * (pow2(x0) - x1) * (2 * x0) - 2 * (1 - x0);
//     o(1) = 200 * (pow2(x0) - x1) * -1;
//     return o;
// }

// // This one returns zero gradient always (BAD)
// autodiff::dual Rosenbrockvec(const Eigen::VectorXdual &x)
// {
//     return Rosenbrock(x[0], x[1]);
// }

// // This one is fine
// autodiff::dual Rosenbrockvec1(const Eigen::VectorXdual &x)
// {
//     return 100.0 * pow2(x(0) * x(0) - x(1)) + pow2(1.0 - x(0));
// }

// int main()
// {
//     // Bar bar;
//     // bool a = createFoo(Bar{});
//     // bool b = createFoo(bar);

//     Eigen::VectorXd x0(2);
//     x0 << -0.3, 0.5;
//     Eigen::ArrayXd gexact = Rosenbrock_exact_gradient(x0(0), x0(1));
//     std::cout << gexact << std::endl;

//     Eigen::VectorXdual x0dual = x0.cast<autodiff::dual>();
//     Eigen::VectorXd gnest = autodiff::forward::gradient(Rosenbrockvec, autodiff::wrt(x0dual), autodiff::forward::at(x0dual));
//     Eigen::VectorXd gnonest = autodiff::forward::gradient(Rosenbrockvec1, autodiff::wrt(x0dual), autodiff::forward::at(x0dual));

//     std::cout << gnest.array() - gexact.array() << " must be zero\n";
//     std::cout << gnonest.array() - gexact.array() << " must be zero\n";
// }

#include <iostream>

#include <autodiff/forward.hpp>
using namespace autodiff;

template <typename T>
auto pow2(T&& x)
{
    return std::forward<T>(x) * std::forward<T>(x);
}

int main()
{
    dual x = 2.0;

    bool b = pow2(dual(2));
    dual y = pow2(pow2(dual(2)));

    std::cout << y << std::endl;
}
