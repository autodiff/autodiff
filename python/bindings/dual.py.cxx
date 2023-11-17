//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright © 2018–2024 Allan Leal
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
#include "pybind11.hxx"

// C++ includes
#include <sstream>

// autodiff includes
#include <autodiff/common/meta.hpp>
#include <autodiff/forward/dual/dual.hpp>
using namespace autodiff;
using autodiff::detail::isSame;

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

    auto __float__ = [](const Dual<T, G>& self)
    {
        return autodiff::val(self);
    };

    using U = autodiff::detail::NumericType<T>;

    auto cls = py::class_<Dual<T, G>>(m, typestr)
        .def(py::init<>())
        .def(py::init<const U&>())
        .def(py::init<const Dual<T, G>&>())
        .def("__str__", __str__)
        .def("__repr__", __repr__)
        .def("__float__", __float__)

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

    if constexpr (!isSame<U, int>) cls.def(py::init<int>());
    if constexpr (!isSame<U, long>) cls.def(py::init<long>());
    if constexpr (!isSame<U, float>) cls.def(py::init<float>());
    if constexpr (!isSame<U, double>) cls.def(py::init<double>());

    py::implicitly_convertible<T, Dual<T, G>>();

    if constexpr (!isSame<T, int>) py::implicitly_convertible<int, Dual<T, G>>();
    if constexpr (!isSame<T, long>) py::implicitly_convertible<long, Dual<T, G>>();
    if constexpr (!isSame<T, float>) py::implicitly_convertible<float, Dual<T, G>>();
    if constexpr (!isSame<T, double>) py::implicitly_convertible<double, Dual<T, G>>();

    m.def("abs"  , [](const Dual<T, G>& x) { return abs(x); });

    m.def("sin"  , [](const Dual<T, G>& x) { return sin(x); });
    m.def("cos"  , [](const Dual<T, G>& x) { return cos(x); });
    m.def("tan"  , [](const Dual<T, G>& x) { return tan(x); });

    m.def("asin" , [](const Dual<T, G>& x) { return asin(x); });
    m.def("acos" , [](const Dual<T, G>& x) { return acos(x); });
    m.def("atan" , [](const Dual<T, G>& x) { return atan(x); });

    // m.def("asinh", [](const Dual<T, G>& x) { return asinh(x); });
    // m.def("acosh", [](const Dual<T, G>& x) { return acosh(x); });
    // m.def("atanh", [](const Dual<T, G>& x) { return atanh(x); });

    m.def("sinh" , [](const Dual<T, G>& x) { return sinh(x); });
    m.def("cosh" , [](const Dual<T, G>& x) { return cosh(x); });
    m.def("tanh" , [](const Dual<T, G>& x) { return tanh(x); });

    m.def("arcsin" , [](const Dual<T, G>& x) { return asin(x); });
    m.def("arccos" , [](const Dual<T, G>& x) { return acos(x); });
    m.def("arctan" , [](const Dual<T, G>& x) { return atan(x); });

    // m.def("arcsinh", [](const Dual<T, G>& x) { return asinh(x); });
    // m.def("arccosh", [](const Dual<T, G>& x) { return acosh(x); });
    // m.def("arctanh", [](const Dual<T, G>& x) { return atanh(x); });

    m.def("sqrt" , [](const Dual<T, G>& x) { return sqrt(x); });
    // m.def("cbrt" , [](const Dual<T, G>& x) { return cbrt(x); });

    m.def("exp"  , [](const Dual<T, G>& x) { return exp(x);   });
    m.def("log"  , [](const Dual<T, G>& x) { return log(x);   });
    m.def("log10", [](const Dual<T, G>& x) { return log10(x); });

    m.def("pow"  , [](const Dual<T, G>& x, const Dual<T, G>& y) { return pow(x, y); });
    m.def("pow"  , [](const Dual<T, G>& x, const U& y)          { return pow(x, y); });
    m.def("pow"  , [](const U& x, const Dual<T, G>& y)          { return pow(x, y); });

    m.def("max"  , [](const Dual<T, G>& x, const Dual<T, G>& y) { return max(x, y); });
    m.def("max"  , [](const Dual<T, G>& x, const U& y)          { return max(x, y); });
    m.def("max"  , [](const U& x, const Dual<T, G>& y)          { return max(x, y); });

    m.def("min"  , [](const Dual<T, G>& x, const Dual<T, G>& y) { return min(x, y); });
    m.def("min"  , [](const Dual<T, G>& x, const U& y)          { return min(x, y); });
    m.def("min"  , [](const U& x, const Dual<T, G>& y)          { return min(x, y); });

    m.def("hypot"  , [](const Dual<T, G>& x, const Dual<T, G>& y) { return hypot(x, y); });
    m.def("hypot"  , [](const Dual<T, G>& x, const U& y)          { return hypot(x, y); });
    m.def("hypot"  , [](const U& x, const Dual<T, G>& y)          { return hypot(x, y); });

    m.def("erf"  , [](const Dual<T, G>& x) { return erf(x); });
}

void export_dual1st(py::module& m) { exportDual<dual0th, dual0th>(m, "dual1st"); }
void export_dual2nd(py::module& m) { exportDual<dual1st, dual1st>(m, "dual2nd"); }
void export_dual3rd(py::module& m) { exportDual<dual2nd, dual2nd>(m, "dual3rd"); }
void export_dual4th(py::module& m) { exportDual<dual3rd, dual3rd>(m, "dual4th"); }
