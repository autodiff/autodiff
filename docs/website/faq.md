## Why should I consider automatic differentiation?

The following are the reasons why you should consider automatic differentiation
in your computational project:

- your functions are extremely complicated;
- manually implementing analytical derivatives is a tedious and error-prone
  task; and
- computing derivatives using finite differences can be inaccurate and
  inefficient.

## What is the difference between reverse and forward modes?

Here is a brief, practical, and qualitative discussion on the differences
between these two automatic differentiation algorithms.

In a forward mode algorithm, each function evaluation produces not only its
output value but also its derivative. The evaluation of a vector function, for
example, computes both the output vector as well as the derivative of this
vector, in general, with respect to one of the input variables (e.g., *∂u/∂x*,
*∂u/∂y*). However, the forward mode algorithm can also be used to compute
directional derivatives. In this case, the derivative of the output vector with
respect to a given direction vector is performed. We could say that,
fundamentally, the forward mode always computes directional derivatives. The
simplest case of a derivative with respect to a single input variable
corresponds thus with a directional derivative whose direction vector is a unit
vector along the input variable of interest (e.g., the unit vector along *x*,
or along *y*, and so forth).

In a reverse mode algorithm, each function evaluation produces a complete
expression tree that contains the sequence of all mathematical operations
between the input variables to produce the output variable (a scalar). Once
this expression tree is constructed, it is then used to compute the derivatives
of the scalar output variable with respect to all input variables.

## Is one algorithm always faster than another?

Even though the reverse mode algorithm requires a single function evaluation
and all derivatives are subsequently computed in a single pass, as the
constructed expression tree is traversed, **the forward algorithm can still be
a much more efficient choice for your application**, despite its multiple,
repeated function evaluations, one for each input variable. This is because the
implementation of the forward mode algorithm in {{autodiff}} uses *template
meta-programming* techniques to avoid as much as possible temporary memory
allocations and to optimize the calculations in certain cases. The reverse mode
algorithm, on the other hand, requires the construction of an expression tree
at runtime, and, for this, several dynamic memory allocations are needed, which
can be costly. We plan to implement alternative versions of this algorithm, in
which memory allocation could be done in advance to decrease the number of
subsequent allocations. This, however, will require a slightly more complicated
usage than it is already provided by the reverse mode algorithm implemented in
{{autodiff}}.

## Which automatic differentiation algorithm should I use?

Ideally, you should try both algorithms for your specific needs, benchmark
them, and then make an informed decision about which one to use.

If you're in a hurry, consider:

**forward mode**: if you have a vector function, or a scalar function with not
many input variables.

**reverse mode**: if you have a scalar function with many (thousands or more)
input variables.

Have in mind this is a very simplistic rule, and you should definitely try both
algorithms whenever possible, since the forward mode could still be faster than
reverse mode even when many input variables are considered for a function of
interest.

## How do I cite {{autodiff}}?

We appreciate your intention of citing {{autodiff}} in your publications. Please
use the following BibTeX reference entry:

~~~bibtex
@misc{autodiff,
    author = {Leal, Allan M. M.},
    title = {autodiff, a modern, fast and expressive {C++} library for automatic differentiation},
    url = {https://autodiff.github.io},
    howpublished = {\texttt{https://autodiff.github.io}},
    year = {2018}
}
~~~

This should produce a formatted citation that looks more or less the following:

> Leal, A.M.M. (2018). *autodiff, a modern, fast and expressive C++ library for automatic differentiation*.
> [https://autodiff.github.io](https://autodiff.github.io)

Please ensure the website URL is displayed.
