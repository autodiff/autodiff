# Installation

Installing {{autodiff}} is easy, since it is a *header-only library*. Follow
the steps below.

## Installation using Conda

If you have [Anaconda] or [Miniconda] installed (with Python 3.7+), you can
install {{autodiff}} with a single command:

~~~
conda install conda-forge::autodiff
~~~

This will install {{autodiff}} in the conda environment that is active at the
moment (e.g., the default conda environment is named `base`).

### Development Environment

[Anaconda] or [Miniconda] can be used to create a full development environment using [conda-devenv].

You can install [conda-devenv] and build the development environment defined in `environment.devenv.yml` as follows:
~~~
conda install conda-devenv
conda devenv
~~~

The above commands will produce a {{conda}} environment called {{autodiff}} with all the dependencies installed. You can activate the development environment with
~~~
conda activate autodiff
~~~
and follow the instructions below to build {{autodiff}} from source.

## Installation using CMake

If you have `cmake` installed in your system, you can then not only install
{{autodiff}} but also build its examples and tests. First, you'll need to
download it by either git cloning its [GitHub repository][github]:

~~~
git clone https://github.com/autodiff/autodiff
~~~

or by [clicking here][zip] to start the download of a zip file, which you
should extract to a directory of your choice.

Then, execute the following steps (assuming you are in the root of the source code directory of {{autodiff}}!):

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

## Build Using Bazel

[bazel](https://bazel.build/) can be used as build system.

!!! attention

    [bazel](https://bazel.build/) support is part of the community effort. Therefore, it
    is not officially supported.

    Currently, running the unit tests and installing the library using
    [bazel](https://bazel.build/) is not supported.

### Build and Run Examples

Build all examples using [bazel](https://bazel.build/):

~~~
bazel build //examples/forward:all
bazel build //examples/reverse:all
~~~

Run all examples using [bazel](https://bazel.build/) and display their output:

~~~
bazel test //examples/forward:all --test_output=all
bazel test //examples/reverse:all --test_output=all
~~~

## Installation by copying

Assuming the git cloned repository or the extracted source code resides in a
directory named `autodiff`, you can now copy the sub-directory
`autodiff/autodiff` to somewhere in your project directory and directly use
{{autodiff}}.


## Installation failed. What do I do?

Discuss with us first your question on our [GitHub Discussion
Channel][discussion]. We may be able to respond more quickly there on how to
sort out your issue. If bug fixes are indeed required, we'll kindly ask you to
create a [GitHub Issue][issues], in which you can provide more details about the
issue and keep track of our progress on fixing it. You are also welcome to
recommend us installation improvements in the Gitter channel.


[discussion]: https://github.com/autodiff/autodiff/discussions/new?category=q-a
[Anaconda]: https://www.anaconda.com/distribution/
[Miniconda]: https://docs.conda.io/en/latest/miniconda.html
[github]: https://github.com/autodiff/autodiff
[zip]: https://github.com/autodiff/autodiff/archive/master.zip
[issues]: https://github.com/autodiff/autodiff/issues/new
[conda-devenv]: https://conda-devenv.readthedocs.io/en/latest/
