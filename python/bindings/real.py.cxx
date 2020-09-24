//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2020 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// autodiff includes
#include <autodiff/forward/real/real.hpp>
using namespace autodiff;

template<size_t N, typename T>
void exportReal(py::module& m, std::string typestr)
{
    auto __getitem__ = [](const Real<N, T>& self, size_t i)
    {
        return self[i];
    };

    auto __setitem__ = [](Real<N, T>& self, size_t i, const T& value)
    {
        self[i] = value;
    };

    auto __str__ = [](Real<N, T>& self)
    {
        std::stringstream ss;
        ss << self;
        return ss.str();
    };

    py::class_<Real<N, T>>(m, typestr.c_str())
        .def(py::init<>())
        .def(py::init<const T&>())
        .def(py::init<const std::array<T, N + 1>&>())
        .def(py::init<const Real<N, T>&>())
        .def("__getitem__", __getitem__)
        .def("__setitem__", __setitem__)
        .def("__str__", __str__)

        .def(-py::self)

        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)

        .def(py::self + T())
        .def(py::self - T())
        .def(py::self * T())
        .def(py::self / T())

        .def(T() + py::self)
        .def(T() - py::self)
        .def(T() * py::self)
        .def(T() / py::self)

        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= py::self)
        .def(py::self /= py::self)

        .def(py::self += T())
        .def(py::self -= T())
        .def(py::self *= T())
        .def(py::self /= T())
        ;

    py::implicitly_convertible<T, Real<N, T>>();
}

void export_real1st(py::module& m) { exportReal<1, double>(m, "real1st"); }
void export_real2nd(py::module& m) { exportReal<2, double>(m, "real2nd"); }
void export_real3rd(py::module& m) { exportReal<3, double>(m, "real3rd"); }
void export_real4th(py::module& m) { exportReal<4, double>(m, "real4th"); }
