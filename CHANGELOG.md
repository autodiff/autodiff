# Change Log

This file documents important changes to this project, which uses [Semantic Versioning](http://semver.org/).

#[v0.5](https://github.com/autodiff/autodiff/releases/tag/v0.5.0) (June 17, 2019)
[Full Changelog](https://github.com/autodiff/autodiff/compare/v0.5.0...v0.4.2)

This release introduces a **BREAKING CHANGE**!

From now on, methods `derivative`, `gradient`, and `jacobian` require the
use of auxiliary functions `wrt` and the newly introduced one `at`.

Examples:

~~~c++
// f = f(x)
double dudx = derivative(f, wrt(x), at(x));
~~~

~~~c++
// f = f(x, y, z)
double dudx = derivative(f, wrt(x), at(x, y, z));
double dudy = derivative(f, wrt(y), at(x, y, z));
double dudz = derivative(f, wrt(z), at(x, y, z));
~~~

~~~c++
// f = f(x), scalar function, where x is an Eigen vector
VectorXd g = gradient(f, wrt(x), at(x));

// Compuring gradient with respect to only some variables
VectorXd gpartial = gradient(f, wrt(x.tail(5)), at(x));
~~~

~~~c++
// F = F(x), vector function, where x is an Eigen vector
MatrixXd J = jacobian(f, wrt(x), at(x));

// F = F(x, p), vector function with params, where x and p are Eigen vectors
MatrixXd Jx = jacobian(f, wrt(x), at(x, p));
MatrixXd Jp = jacobian(f, wrt(p), at(x, p));

// Compuring Jacobian with respect to only some variables
MatrixXd Jpartial = jacobian(f, wrt(x.tail(5)), at(x));
~~~

This release also permits one to retrieve the evaluated value of function during
a call to the methods `derivative`, `gradient`, and `jacobian`:

~~~c++
// f = f(x)
dual u;
double dudx = derivative(f, wrt(x), at(x), u);
~~~

~~~c++
// f = f(x), scalar function, where x is an Eigen vector
dual u;
VectorXd g = gradient(f, wrt(x), at(x), u);
~~~

~~~c++
// F = F(x), vector function, where x is an Eigen vector
VectorXdual F;
MatrixXd J = jacobian(f, wrt(x), at(x), F);
~~~

#[v0.4.2](https://github.com/autodiff/autodiff/releases/tag/v0.4.2) (Mar 28, 2019)
[Full Changelog](https://github.com/autodiff/autodiff/compare/v0.4.2...v0.4.1)

This is to force conda-forge to produce a new version (now 0.4.2) since the last
one (0.4.1) did not work.

#[v0.4.1](https://github.com/autodiff/autodiff/releases/tag/v0.4.1) (Mar 26, 2019)
[Full Changelog](https://github.com/autodiff/autodiff/compare/v0.4.1...v0.4.0)

This release fixes a bug in the computation of Jacobian matrices when the input
and output vectors in a vector-valued function have different dimensions (see
issue [#24](https://github.com/autodiff/autodiff/issues/24)).

#[v0.4.0](https://github.com/autodiff/autodiff/releases/tag/v0.4.0) (Feb 20, 2019)
[Full Changelog](https://github.com/autodiff/autodiff/compare/v0.4.0...v0.3.0)

This release contains changes that enable autodiff to be successfully compiled
in Linux, macOS, and Windows. Compilers tested were GCC 7, Clang 9, and Visual
Studio 2017. Compilers should support C++17.

#[v0.3.0](https://github.com/autodiff/autodiff/releases/tag/v0.3.0) (Feb 5, 2019)
[Full Changelog](https://github.com/autodiff/autodiff/compare/v0.3.0...v0.2.0)

This release improves the forward mode algorithm to compute derivatives of any
order. It also introduces a proper website containing a more detailed
documentation of autodiff library: https://autodiff.github.io

#[v0.2.0](https://github.com/autodiff/autodiff/releases/tag/v0.2.0) (Jul 26, 2018)
[Full Changelog](https://github.com/autodiff/autodiff/compare/v0.2.0...v0.1.0)

This release permits higher order derivatives to be computed with
`autodiff::gradx` function and it also enables the use of `autodiff::var` type
with Eigen vector and matrix types.

#[v0.1.0](https://github.com/autodiff/autodiff/releases/tag/v0.1.0) (Jul 19, 2018)
[Full Changelog](https://github.com/autodiff/autodiff/compare/v3.6.0...v3.6.1)

This is the first release of autodiff. Please note breaking changes might be
introduced, but not something that would take you more than a few minutes to
correct.
