
# CMake-Based Project Using autodiff

## Introduction
This example demonstrates how CMake's command `find_package` can be used to
resolve the dependency of an executable `app` on **autodiff**, a header-only
C++17 library.

The source file `main.cpp` includes the header-file `autodiff/forward.hpp` and
uses a forward mode automatic differentiation algorithm to compute the derivatives of a scalar function.

The `CMakeLists.txt` file uses the command:

```cmake
find_package(autodiff)
```

to find the **autodiff** header files. The executable target `app` is then
linked against the imported target `autodiff::autodiff`:

```cmake
target_link_libraries(app autodiff::autodiff)
```

## Building and Executing the Application
To build the application, do:

```bash
cd cmake-project
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/autodiff/install/dir
make
```

To execute the application, do:

```bash
./app
```

Note: If **autodiff** has been installed system-wide, then the CMake argument
`CMAKE_PREFIX_PATH` should not be needed. Otherwise, you will need to specify
where **autodiff** is installed in your machine. For example:

```cmake
cmake .. -DCMAKE_PREFIX_PATH=$HOME/local
```

assuming directory `$HOME/local` is where **autodiff** was installed to, which should then contain the following directory:

```
$HOME/local/include/autodiff/
```

where the **autodiff** header files are located.
