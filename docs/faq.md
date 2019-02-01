## Why should I consider automatic differentiation?

The following are the reasons why you should consider automatic differentiation
in your computational project:

- your functions are extremely complicated;
- manually implementing analytical derivatives is a tedious and error-prone
  task; and
- computing derivatives using finite differences can be inaccurate and
  inefficient.

## What is the difference between reverse and forward modes?

Let's refer to *f* as the function we want to compute the derivatives.

In a forward mode algorithm, we need to evaluate the same function we want to compute derivatives for
need to be called once for each input variable. Each time, a se

In the reverse mode, the `autodiff::derivatives` function computed all
derivatives of `u` in a single pass. For this, function `f` was evaluated
before, only once, during which an expression tree was constructed to record
all mathematical operations between the input variables *(x, y, z)*. This
expression tree was then stored in the output variable `u`, and later used by
`autodiff::derivatives` to compute the derivatives.

Above is just a simple observation of the practical differences of forward and
reverse algorithms applied to a simple function. A more in-depth discussion of
both automatic differentiation algorithms will be presented later.

## Is one algorithm always faster than another?

Even though the reverse algorithm requires a single function evaluation and all
derivatives are subsequently computed in a single pass, as the constructed
expression tree is traversed, **the forward algorithm can still be a much more
efficient choice for your application**, despite its multiple, repeated
function evaluations, one for each input variable. This is because the
implementation of the forward mode algorithm in {{autodiff}} uses *template
meta-programming* techniques to avoid as much as possible temporary memory
allocations and to optimize the calculations in certain cases. The reverse mode
algorithm, on the other hand, requires the construction of an expression tree
at runtime, and, for this, several dynamic memory allocations are needed, as
well as the need to perform pointer dereferences, which can be costly.

## Which automatic differentiation algorithm should I use?

Ideally, you should try both algorithms for your application, benchmark, and
then make an informed decision about which algorithm to use.