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

// C++ includes
#include <sstream>

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
namespace py = pybind11;

// autodiff includes
#include <autodiff/forward/dual/dual.hpp>
using namespace autodiff;

template<typename T, typename G>
void exportDual(py::module& m, const char* typestr)
{
    auto __str__ = [](const Dual<T, G>& self)
    {
        std::stringstream ss;
        ss << self;
        return ss.str();
    };

    auto __repr__ = [](const Dual<T, G>& self)
    {
        return repr(self);
    };

    using U = autodiff::detail::DualValueType<T>;

    py::class_<Dual<T, G>>(m, typestr)
        .def(py::init<>())
        .def(py::init<const U&>())
        .def(py::init<const Dual<T, G>&>())
        .def("__str__", __str__)
        .def("__repr__", __repr__)

        .def(-py::self)

        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)

        .def(py::self + U())
        .def(py::self - U())
        .def(py::self * U())
        .def(py::self / U())

        .def(U() + py::self)
        .def(U() - py::self)
        .def(U() * py::self)
        .def(U() / py::self)

        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= py::self)
        .def(py::self /= py::self)

        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)

        .def(py::self += U())
        .def(py::self -= U())
        .def(py::self *= U())
        .def(py::self /= U())

        .def(py::self == U())
        .def(py::self != U())
        .def(py::self < U())
        .def(py::self > U())
        .def(py::self <= U())
        .def(py::self >= U())

        .def(U() == py::self)
        .def(U() != py::self)
        .def(U() < py::self)
        .def(U() > py::self)
        .def(U() <= py::self)
        .def(U() >= py::self)
        ;

    py::implicitly_convertible<T, Dual<T, G>>();
}

void export_dual1st(py::module& m) { exportDual<dual0th, dual0th>(m, "dual1st"); }
void export_dual2nd(py::module& m) { exportDual<dual1st, dual1st>(m, "dual2nd"); }
void export_dual3rd(py::module& m) { exportDual<dual2nd, dual2nd>(m, "dual3rd"); }
void export_dual4th(py::module& m) { exportDual<dual3rd, dual3rd>(m, "dual4th"); }
