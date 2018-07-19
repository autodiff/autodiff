[![autodiff](art/autodiff-header.svg)][autodiff]


# Overview

You most likely arrived here because:

- you have complicated functions for which you would like to compute derivatives;
- you understand that manually deriving and implementing the analytical expressions of these derivatives is a tedious and error-prone task;
- you are aware that numerical derivatives using finite differences can be inaccurate and inefficient; and
- you are looking for a straightforward solution to calculate derivatives in C++ using **automatic differentiation**.

Here is how [autodiff][autodiff] can help you. Let's assume you have the following code:

```c++
double x = 0.0;
double y = f(x);
```

for which you want to calculate `dydx`, that is, the derivative of output variable `y` with respect to input variable `x` when `x = 0.0`. Instead of using type `double` for `x` and `y`, as well as all for all intermediate variables inside `f`, if applicable, use `autodiff::var` instead:

```c++
var x = 0.0;
var y = f(x);
```

and calculate your derivative `dydx` using the `autodiff::grad` function as follows:

```cpp
double dydx = grad(y, x);
```

That's it -- simple and straightforward!

# Examples

The previous demonstration might be too simple for your needs. Below you find many complete C++ examples for different types of functions and circumstances.

## Example 1: Derivative of a single-variable function

The example below demonstrates how to use `autodiff` for calculating derivatives of a single-variable function.

~~~c++
// C++ includes
#include <iostream>
using namespace std;

// autodiff includes
#include <autodiff.hpp>
using namespace autodiff;

// The single-variable function for which derivatives are needed
var f(var x) 
{
    return 1 + x + x*x + 1/x + log(x); 
}

int main()
{
    var x = 2.0;                         // x - input variable of type autodiff::var
    var y = f(x);                        // y - output variable of type autodiff::var

    double dydx = grad(y, x);            // evaluate derivative dy/dx using autodiff::grad function

    cout << "y = " << y << endl;         // print evaluated output y
    cout << "dy/dx = " << dydx << endl;  // print evaluated derivative dy/dx
}
~~~

After compiling and executing this example, you should see the following output:

~~~
y = 8.19315
dy/dx = 5.25
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

Executing this example produces:

~~~
u     = 27.2113
du/dx = 13.6056
du/dy = 8.26761
du/dz = 5.28638
~~~

## Example 3: Derivatives with respect to function parameters

Sometimes, it is necessary to understand how sensitive an output is with respect to some parameters. For example, a mathematical model that computes the density of a substance at different temperatures and pressures will depend on some parameters. One might want to optimize the values of such parameters so that the model is accurate relative to some new experimental measurements. In such cases, the parameter optimization calculation can greatly benefit from available derivatives of the density with respect to every parameter in the model. 

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

The example above introduces a new type `Params` with three data members of type `autodiff::var`: `a`, `b`, and `c`. These can be seen as parameters for the mathematical model function `f`. The example then demonstrates how to calculate not only the derivative of the output variable `y` with respect to input variable `x`, given by `dydx`, but also with respect to every parameter in `Params`: `dyda`, `dydb`, and `dydx`.

This example outputs the following results:
~~~
y = 3.4968
dy/dx = 1.53964
dy/da = 0.479426
dy/db = 0.877583
dy/dc = 0.420735
~~~

## Example 4: Derivatives of a function with a vector input

# What is missing?

1. Combine autodiff with C++ linear algebra library [Eigen][Eigen].
2. Evaluate the performance of autodiff for many functions with different complexity.

# License

MIT License

Copyright (c) 2018 autodiff

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[autodiff]: https://github.com/reaktoro/autodiff "autodiff"
[autodiff-hpp]: TODO "autodiff.hpp"
[Eigen]: http://eigen.tuxfamily.org "Eigen"
