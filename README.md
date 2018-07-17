[![autodiff](art/autodiff-header.svg)][autodiff]

# Overview

You most likely arrived here because:

- you have complicated functions for which you would like to compute derivatives;
- you have learned that manually deriving the analytical expressions of these derivatives is a tedious and error-prone task;
- you understand the accuracy and performance disadvantages of numerical derivatives using finite differences; and
- you are looking for a straightforward solution to calculate derivatives in C++ using **automatic differentiation**.

Here is how [autodiff][autodiff] can help you. Let's assume you have the following code:

```js
double x = 0.0;
double y = f(x);
```

for which you want to calculate `dydx`, that is, the derivative of output variable `y` with respect to input variable `x` when `x = 0.0`. Instead of using type `double` for `x` and `y`, as well as all for all intermediate variables inside `f`, if applicable, use `autodiff::var` instead:

```js
var x = 0.0;
var y = f(x);
```

and calculate your derivative `dydx` using the `autodiff::grad` function as follows:

```cpp
double dydx = grad(y, x);
```

That's it!

# Examples


## Example 1: Derivative of a single-variable function

~~~c++
#include <autodiff.hpp>
using namespace autodiff;

var f(var x) 
{
    return 1 + x + x*x + 1/x + log(x); 
}

int main()
{
    var x = 2.0;
    var y = f(x);

    double dydx = grad(y, x);

    std::cout << "y = " << y << std::endl;
    std::cout << "dy/dx = " << dydx << std::endl;
}
~~~

## Example 2: Derivatives of a multi-variable function

~~~c++
#include <autodiff.hpp>
using namespace autodiff;

var f(var x, var y, var z) 
{ 
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

int main()
{
    var x = 1.0;
    var y = 2.0;
    var z = 3.0;
    var u = f(x, y, z);

    double dudx = grad(u, x);
    double dudy = grad(u, y);
    double dudz = grad(u, z);

    std::cout << "u = " << u << std::endl;
    std::cout << "du/dx = " << dudx << std::endl;
    std::cout << "du/dy = " << dudy << std::endl;
    std::cout << "du/dz = " << dudz << std::endl;
}
~~~

## Example 3: Derivatives with respect to function parameters

~~~c++
#include <autodiff.hpp>
using namespace autodiff;

struct Params
{
    var a;
    var b;
    var c;
};

var f(var x, const Params& params)
{
    return params.a * sin(x) + params.b * cos(x) + params.c * sin(x)*cos(x);
}

int main()
{
    Params params;
    params.a = 1.0;
    params.b = 2.0;
    params.c = 3.0;

    var x = 0.5;
    var y = f(x, params);

    double dydx = grad(y, x);
    double dyda = grad(y, params.a);
    double dydb = grad(y, params.b);
    double dydc = grad(y, params.c);

    std::cout << "y = " << y << std::endl;
    std::cout << "dy/dx = " << dydx << std::endl;
    std::cout << "dy/da = " << dyda << std::endl;
    std::cout << "dy/db = " << dydb << std::endl;
    std::cout << "dy/dc = " << dydc << std::endl;
}
~~~

[autodiff]: https://github.com/reaktoro/autodiff "autodiff"
