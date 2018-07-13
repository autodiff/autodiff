#include <autodiff/autodiff.hpp>
using namespace autodiff;

auto f(var a, var b, var t) -> var
{
    return ((2*a + b) + b + 3*b + t) + 1.0;
}

int main(int argc, char const *argv[])
{
    var a = 10;
    var b = 20;
    var t = a;
    var d = 200;
    var c = f(a, b, t);

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    std::cout << grad(c, a) << std::endl;
    std::cout << grad(c, t) << std::endl;
    std::cout << grad(c, b) << std::endl;
    std::cout << grad(c, d) << std::endl;

    return 0;
}
