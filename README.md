<p align="center">
    <img src="art/autodiff-header.svg" style="max-width:100%;" title="autodiff">
</p>

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

// autodiff include
#include <autodiff/reverse.hpp>
using namespace autodiff;

// The single-variable function for which derivatives are needed
var f(var x)
{
    return 1 + x + x*x + 1/x + log(x);
}

int main()
{
    var x = 2.0;                         // the input variable x
    var y = f(x);                        // the output variable y

    double dydx = grad(y, x);            // evaluate the derivative dy/dx

    cout << "y = " << y << endl;         // print the evaluated output y
    cout << "dy/dx = " << dydx << endl;  // print the evaluated derivative dy/dx
}
~~~

After compiling and executing this example, you should see the following output:

~~~
y = 8.19315
dy/dx = 5.25
~~~

## Example 2: Derivatives of a multi-variable function

~~~c++
// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse.hpp>
using namespace autodiff;

// The multi-variable function for which derivatives are needed
var f(var x, var y, var z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

int main()
{
    var x = 1.0;                         // the input variable x
    var y = 2.0;                         // the input variable y
    var z = 3.0;                         // the input variable z
    var u = f(x, y, z);                  // the output variable u

    double dudx = grad(u, x);            // evaluate the derivative du/dx
    double dudy = grad(u, y);            // evaluate the derivative du/dy
    double dudz = grad(u, z);            // evaluate the derivative du/dz

    cout << "u = " << u << endl;         // print the evaluated output u
    cout << "du/dx = " << dudx << endl;  // print the evaluated derivative du/dx
    cout << "du/dy = " << dudy << endl;  // print the evaluated derivative du/dy
    cout << "du/dz = " << dudz << endl;  // print the evaluated derivative du/dz
}
~~~

Executing this example produces:

~~~
u = 27.2113
du/dx = 13.6056
du/dy = 8.26761
du/dz = 5.28638
~~~

## Example 3: Derivatives with respect to function parameters

Sometimes, it is necessary to understand how sensitive an output is with respect to some parameters. For example, a mathematical model that computes the density of a substance at different temperatures and pressures will depend on some parameters. One might want to optimize the values of such parameters so that the model is accurate relative to some new experimental measurements. In such cases, the parameter optimization calculation can greatly benefit from available derivatives of the density with respect to every parameter in the model. 

~~~c++
// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse.hpp>
using namespace autodiff;

// A type defining parameters for a function of interest
struct Params
{
    var a;
    var b;
    var c;
};

// The function that depends on parameters for which derivatives are needed
var f(var x, const Params& params)
{
    return params.a * sin(x) + params.b * cos(x) + params.c * sin(x)*cos(x);
}

int main()
{
    Params params;                       // initialize the parameter variables
    params.a = 1.0;                      // the parameter a of type var, not double!
    params.b = 2.0;                      // the parameter b of type var, not double!
    params.c = 3.0;                      // the parameter c of type var, not double!

    var x = 0.5;                         // the input variable x
    var y = f(x, params);                // the output variable y

    double dydx = grad(y, x);            // evaluate derivative du/dx
    double dyda = grad(y, params.a);     // evaluate derivative du/da
    double dydb = grad(y, params.b);     // evaluate derivative du/db
    double dydc = grad(y, params.c);     // evaluate derivative du/dc

    cout << "y = " << y << endl;         // print evaluated output y
    cout << "dy/dx = " << dydx << endl;  // print evaluated derivative dy/dx
    cout << "dy/da = " << dyda << endl;  // print evaluated derivative dy/da
    cout << "dy/db = " << dydb << endl;  // print evaluated derivative dy/db
    cout << "dy/dc = " << dydc << endl;  // print evaluated derivative dy/dc
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

## Example 4: Gradient of a scalar function

~~~c++
// C++ includes
#include <iostream>
using namespace std;

// Eigen includes
#include <eigen3/Eigen/Core>
using namespace Eigen;

// autodiff include
#define AUTODIFF_ENABLE_EIGEN_SUPPORT
#include <autodiff/reverse.hpp>
using namespace autodiff;

// The scalar function for which the gradient is needed
var f(const VectorXv& x)
{
    return sqrt(x.cwiseProduct(x).sum()); // sqrt(sum([x(i) * x(i) for i = 1:5]))
}

int main()
{
    VectorXv x(5);                         // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;                    // x = [1, 2, 3, 4, 5]

    var y = f(x);                          // the output variable y

    VectorXd dydx = grad(y, x);            // evaluate the gradient vector dy/dx

    cout << "y = " << y << endl;           // print the evaluated output y
    cout << "dy/dx = \n" << dydx << endl;  // print the evaluated gradient vector dy/dx
}
~~~

## Example 5: Jacobian of a vector function

~~~c++
// C++ includes
#include <iostream>
using namespace std;

// Eigen includes
#include <eigen3/Eigen/Core>
using namespace Eigen;

// autodiff include
#define AUTODIFF_ENABLE_EIGEN_SUPPORT
#include <autodiff/reverse.hpp>
using namespace autodiff;

// The vector function for which the Jacobian is needed
VectorXv f(const VectorXv& x)
{
    return (x / x.sum()).array().log(); // log(x / sum(x))
}

int main()
{
    VectorXv x(5);                         // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;                    // x = [1, 2, 3, 4, 5]

    VectorXv y = f(x);                     // the output vector y

    MatrixXd dydx = grad(y, x);            // evaluate the Jacobian matrix dy/dx

    cout << "y = \n" << y << endl;         // print the evaluated output vector y
    cout << "dy/dx = \n" << dydx << endl;  // print the evaluated Jacobian matrix dy/dx
}
~~~

## Example 6: Higher-order derivatives of a single-variable function

~~~c++
// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse.hpp>
using namespace autodiff;

int main()
{
    var x = 0.5;                             // the input variable x

    var y = sin(x) * cos(x);                 // the output variable y
    var dydx = gradx(y, x);                  // the first order derivative dy/dx of type var, not double!
    var d2ydx2 = gradx(dydx, x);             // the second order derivative d2y/dx2 of type var, not double!

    cout << "y = " << y << endl;             // print the evaluated output variable y
    cout << "dy/dx = " << dydx << endl;      // print the evaluated first order derivative dy/dx
    cout << "d2y/dx2 = " << d2ydx2 << endl;  // print the evaluated second order derivative d2y/dx2
}
~~~

## Example 7: Higher-order derivatives of a multi-variable function

~~~c++
// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <autodiff/reverse.hpp>
using namespace autodiff;

int main()
{
    var x = 1.0;                              // the input variable x
    var y = 0.5;                              // the input variable y
    var z = 2.0;                              // the input variable z

    var u = x * log(y) * exp(z);              // the output variable u
    
    var dudx = gradx(u, x);                   // the first order derivative du/dx of type var, not double!
    var dudy = gradx(u, y);                   // the first order derivative du/dy of type var, not double!
    var dudz = gradx(u, z);                   // the first order derivative du/dz of type var, not double!

    var d2udx2 = gradx(dudx, x);              // the second order derivative d2u/dx2 of type var, not double!
    var d2udy2 = gradx(dudy, y);              // the second order derivative d2u/dy2 of type var, not double!
    var d2udz2 = gradx(dudz, z);              // the second order derivative d2u/dz2 of type var, not double!

    var d2udxdy = gradx(dudx, y);             // the second order derivative d2u/dxdy of type var, not double!
    var d2udydz = gradx(dudy, z);             // the second order derivative d2u/dydz of type var, not double!
    var d2udzdx = gradx(dudz, x);             // the second order derivative d2u/dzdx of type var, not double!

    cout << "u = " << u << endl;              // print the evaluated output variable u

    cout << "du/dx = " << dudx << endl;       // print the evaluated first order derivative du/dx
    cout << "du/dy = " << dudy << endl;       // print the evaluated first order derivative du/dy
    cout << "du/dz = " << dudz << endl;       // print the evaluated first order derivative du/dz
    
    cout << "d2u/dx2 = " << d2udx2 << endl;   // print the second order derivative d2u/dx2
    cout << "d2u/dy2 = " << d2udy2 << endl;   // print the second order derivative d2u/dy2
    cout << "d2u/dz2 = " << d2udz2 << endl;   // print the second order derivative d2u/dz2

    cout << "d2u/dxdy = " << d2udxdy << endl; // print the second order derivative d2u/dxdy
    cout << "d2u/dydz = " << d2udydz << endl; // print the second order derivative d2u/dydz
    cout << "d2u/dzdx = " << d2udzdx << endl; // print the second order derivative d2u/dzdx
}
~~~

# How does it work?

`autodiff` works its magic in calculating *a posteriori* derivatives of **any `var` variable** with respect to **any `var` variable** by generating an **expression tree** of the entire evaluation procedure. For example, in the code below:

~~~c++
var x = 2.0;
var y = 1.0;
var u = (x + y) * log(x - y)
~~~

the variable `u` not only knows its value, given by `(2.0 + 1.0) * log(2.0 - 1.0)`, but also how this value is computed step-by-step by storing the following expression tree:

<p align="center">
    <img src="art/expression-tree-diagram.svg" style="max-width:100%;" title="Expression tree diagram.">
</p>

Every time an operation is performed on a variable of type `var` (e.g., via mathematical functions `sin`, `cos`, `log`, or via arithmetic operators `+`, `-`, `*`, `/`) a new unary or binary expression tree is formed. Each newly formed expression tree is a combination of previously created ones. For example, `(x + y) * log(x - y)` is a combination of the expression trees resulting from the binary operations `x + y` and `x - y` as well as the unary operation `log(x - y)`. Note that the tree is then constructed from bottom to top as illustrated in the diagram. At the end of this tree construction, the output variable of interest (our `u` variable in the example) has been fully evaluated, producing not only its actual numerical value but alto the corresponding expression tree that defines the algebraic steps for its calculation. By traversing this expression tree and applying rules of differentiation to each sub-expression (e.g., derivative rules for arithmetic operations and common mathematical functions), it is possible to compute derivatives of any order, with respect to any available `var` variable.

# What is missing?

1. Implement forward mode automatic differentiation using template meta-programming - implement type `dual` in addition to `var`.
1. Evaluate the performance of autodiff for many functions with different complexity.
1. Take advantage of r-values and move semantics in C++11 to avoid unnecessary memory allocations as the expression tree is constructed.

# License

MIT License

Copyright (c) 2018 Allan Leal

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

[autodiff]: https://github.com/autodiff/autodiff/releases "autodiff"
[Eigen]: http://eigen.tuxfamily.org "Eigen"
