# Installation

Installing {{autodiff}} is easy, since it is a *header-only library*. Follow
the steps below.

## Download

Download {{autodiff}} by either git cloning its [GitHub repository][github]:

~~~
git clone https://github.com/autodiff/autodiff
~~~

or by [clicking here][zip] to start the download of a zip file, which
you should extract to a directory of your choice.

## Installation by conda

The easiest way of installing `autodiff` is via conda:

~~~
conda install conda-forge autodiff
~~~

For this, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
with Python 3.8+. You should be familiar with conda though.

## Installation by copying

Assuming the git cloned repository or extracted source code resides in a
directory named `autodiff`, you can now copy the directory `autodiff/autodiff`
to somewhere in your project directory and directly use {{autodiff}}.

This quick and dirty solution might suffices in most cases. If this solution
bothers you, read the next section!

## Installation using CMake

If you have `cmake` installed in your system, you can then install {{autodiff}}
(and also build its tests and examples) as follows:

~~~
mkdir .build && cd .build
cmake ..
cmake --build . --target install
~~~

!!! attention

    We assume above that you are in the root of the source code directory, under
    `autodiff`! The build directory will be created at `autodiff/.build`.
    We use `.build` here instead of the more usual `build` because there is a
    file called `BUILD` that provides support to [Bazel](https://bazel.build/) build system.
    In operating systems that treats file and directory names as case insensitive,
    you may not be able to create a `build` directory.

The previous installation commands will require administrative rights in most
systems. To install {{autodiff}} locally, use:

~~~
cmake .. -DCMAKE_INSTALL_PREFIX=/some/local/dir
~~~


[github]: https://github.com/autodiff/autodiff
[zip]: https://github.com/autodiff/autodiff/archive/master.zip

## Installation failed. What do I do?

Create a [new issue][issues], and let us know what happened and possibly howe
we can improve the installation process of {{autodiff}}.


[issues]: https://github.com/autodiff/autodiff/issues/new
