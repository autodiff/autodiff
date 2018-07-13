[![autodiff](art/autodiff-header.svg)](https://github.com/reaktoro/autodiff)

# Overview

This is how easy to calculate derivatives using `autodiff`:

```cpp
var x = 10;
var y = f(x);

double dydx = grad(y, x);
```
