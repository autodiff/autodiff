# Tutorials

We present here some examples demonstrating the use of {{autodiff}} for
computing different types of derivatives. We welcome any contribution towards
improving and expanding this list of examples. We would also love to hear your
suggestions on how to better demonstrate the capabilities of {{autodiff}}.

## Forward mode

### Derivatives of a single-variable function

{{ inputcpp('examples/forward/example-forward-single-variable-function.cpp') }}

### Derivatives of a single-variable function using a custom scalar (complex)

{{ inputcpp('examples/forward/example-forward-single-variable-function-custom-scalar.cpp') }}

### Derivatives of a multi-variable function

{{ inputcpp('examples/forward/example-forward-multi-variable-function.cpp') }}

### Derivatives of a multi-variable function with parameters

{{ inputcpp('examples/forward/example-forward-multi-variable-function-with-parameters.cpp') }}

### Derivatives of a multi-variable function that also relies on analytical derivatives

{{ inputcpp('examples/forward/example-forward-multi-variable-combine-analytical-derivative.cpp') }}

### Gradient vector of a scalar function

{{ inputcpp('examples/forward/example-forward-gradient-derivatives-using-eigen.cpp') }}

### Gradient vector of a scalar function with parameters

{{ inputcpp('examples/forward/example-forward-gradient-derivatives-using-eigen-with-parameters.cpp') }}

### Jacobian matrix of a vector function

{{ inputcpp('examples/forward/example-forward-jacobian-derivatives-using-eigen.cpp') }}

### Jacobian matrix of a vector function with parameters

{{ inputcpp('examples/forward/example-forward-jacobian-derivatives-using-eigen-with-parameters.cpp') }}

### Jacobian matrix of a vector function using memory maps

{{ inputcpp('examples/forward/example-forward-jacobian-derivatives-using-eigen-map.cpp') }}

### Higher-order cross derivatives of a scalar function

{{ inputcpp('examples/forward/example-forward-higher-order-cross-derivatives.cpp') }}

### Higher-order directional derivatives of a scalar function

{{ inputcpp('examples/forward/example-forward-higher-order-directional-derivatives.cpp') }}

### Higher-order directional derivatives of a vector function

{{ inputcpp('examples/forward/example-forward-higher-order-directional-derivatives-using-eigen.cpp') }}

### Taylor series of a scalar function along a direction

{{ inputcpp('examples/forward/example-forward-directional-taylor-series.cpp') }}

### Taylor series of a vector function along a direction

{{ inputcpp('examples/forward/example-forward-directional-taylor-series-using-eigen.cpp') }}

## Reverse mode

### Single-variable function

{{ inputcpp('examples/reverse/example-reverse-single-variable-function.cpp') }}

### Multi-variable function

{{ inputcpp('examples/reverse/example-reverse-multi-variable-function.cpp') }}

### Multi-variable function with conditional

{{ inputcpp('examples/reverse/example-reverse-conditional.cpp') }}

### Multi-variable function with parameters

{{ inputcpp('examples/reverse/example-reverse-multi-variable-function-with-parameters.cpp') }}

### Gradient of a scalar function

{{ inputcpp('examples/reverse/example-reverse-gradient-derivatives-using-eigen.cpp') }}

### Hessian of a scalar function

{{ inputcpp('examples/reverse/example-reverse-hessian-derivatives-using-eigen.cpp') }}

### Higher-order derivatives of a single-variable function

{{ inputcpp('examples/reverse/example-reverse-higher-order-derivatives-single-variable-function.cpp') }}

### Higher-order derivatives of a multi-variable function

{{ inputcpp('examples/reverse/example-reverse-higher-order-derivatives-multi-variable-function.cpp') }}

## Integration with CMake-based projects

Integrating {{autodiff}} in a CMake-based project is very simple as shown next.

Let's assume our CMake-based project consists of two files: `main.cpp` and
`CMakeLists.txt`, whose contents are shown below:

----

**main.cpp**
{{ inputcpp('examples/cmake-project/main.cpp') }}

**CMakeLists.txt**

{{ inputcode('examples/cmake-project/CMakeLists.txt', 'cmake') }}

----

In the `CMakeLists.txt` file, note the use of the command:

```cmake
find_package(autodiff)
```

to find the header files of the {{autodiff}} library, and the command:

```cmake
target_link_libraries(app autodiff::autodiff)
```
to link the executable target `app` against the {{autodiff}} library
(`autodiff::autodiff`) using CMake's modern target-based design.

To build the application, do:

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/autodiff/install/dir
make
```

!!! attention

    If {{autodiff}} has been installed system-wide, then the CMake argument
    `CMAKE_PREFIX_PATH` should not be needed. Otherwise, you will need to specify
    where {{autodiff}} is installed in your machine. For example:

    ```bash
    cmake .. -DCMAKE_PREFIX_PATH=$HOME/local
    ```

    assuming directory `$HOME/local` is where {{autodiff}} was installed to, which
    should then contain the following directory:

    ```
    $HOME/local/include/autodiff/
    ```

    where the {{autodiff}} header files are located.

To execute the application, do:

```bash
./app
```
