// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/forward.hpp>
using namespace autodiff;





/// TODO remove
#include <memory>
#include <map>
#include <tuple>


// The single-variable function for which derivatives are needed
dual f(dual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

template<typename Ret, typename... Args>
auto memoize_helper(std::function<Ret(Args...)> f) -> std::function<Ret(Args...)>
{
    auto cache = std::make_shared<std::map<std::tuple<Args...>, Ret>>();
    return [=](Args... args) mutable -> Ret
    {
        std::tuple<Args...> t(args...);
        if(cache->find(t) == cache->end())
            (*cache)[t] = f(args...);
        return (*cache)[t];
    };
}

template<typename Function>
auto memoize(const Function& f)
{
    return memoize_helper(std::function{f});
}

//template<typename Function, typename... Args>
//auto grad22(const Function& f) -> std::function<dual(Args...)>
//{
//    auto g = [=](Args... args) -> dual {
//        return f(args...);
//    };
//    return g;
//}

auto func1(double x) -> double
{
    return 1;
};


int main()
{
    using Func = std::function<double(double)>;

    auto func2 = [](double x) -> double
    {
        return 1 + x + x*x + 1/x + log(x);
    };

    Func func3 = func2;

//    auto g = grad22(f);
    auto g1 = memoize(func1);
    auto g2 = memoize(func2);
    auto g3 = memoize(func3);
//    var x = 2.0;                         // the input variable x
//    var y = f(x);                        // the output variable y

//    var dydx = grad(y, x);               // evaluate the derivative dy/dx

//    cout << "y = " << y << endl;         // print the evaluated output y
//    cout << "dy/dx = " << dydx << endl;  // print the evaluated derivative dy/dx
}
