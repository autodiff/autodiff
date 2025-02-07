// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

// The scalar function for which the gradient is needed
VectorXvar f(const VectorXvar& x)
{
    VectorXvar y(2);
    y << x.transpose() * x, (x + x).transpose() * (x + x);
    return y;
}

int main()
{
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    MatrixXd J;
    VectorXvar x(4);
    x << 1.0, 2.0, 3.0, 4.0;

    VectorXvar y = f(x);
    J = jacbian(y, x);

    std::cout << "Jacbian rows number is : " << J.rows() << "\n";
    std::cout << "Jacbian cols number is : " << J.cols() << "\n";

    for(auto i = 0; i < J.rows(); ++i) {
        for(auto j = 0; j < J.cols(); ++j) {
            std::cout << J(i, j) << ", ";
        }
        std::cout << "\n";
    }
}
