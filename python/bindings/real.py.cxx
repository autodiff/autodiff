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
#include <autodiff/forward/real/real.hpp>
#include <autodiff/forward/utils/derivative.hpp>
using namespace autodiff;
using autodiff::detail::isSame;

template<size_t N, typename T>
void exportReal(py::module& m, const char* typestr)
{
    auto __getitem__ = [](const Real<N, T>& self, size_t i)
    {
        return self[i];
    };

    auto __setitem__ = [](Real<N, T>& self, size_t i, const T& value)
    {
        self[i] = value;
    };

    auto __str__ = [](const Real<N, T>& self)
    {
        std::stringstream ss;
        ss << self;
        return ss.str();
    };

    auto __repr__ = [](const Real<N, T>& self)
    {
        return repr(self);
    };

    auto __float__ = [](const Real<N, T>& self)
    {
        return self[0];
    };

    auto __deepcopy__ = [](const Real<N, T>& self, py::object memo) -> Real<N, T>
    {
        return self;
    };

    auto cls = py::class_<Real<N, T>>(m, typestr)
        .def(py::init<>())
        .def(py::init<const T&>())
        .def(py::init<const std::array<T, N + 1>&>())
        .def(py::init<const Real<N, T>&>())
        .def("__getitem__", __getitem__)
        .def("__setitem__", __setitem__)
        .def("__str__", __str__)
        .def("__repr__", __repr__)
        .def("__float__", __float__)
        .def("__deepcopy__", __deepcopy__)  // needed when using autodiff::real types with plotly

        .def("exp", &autodiff::detail::exp<N, T>)
        .def("log", &autodiff::detail::log<N, T>)
        .def("log10", &autodiff::detail::log10<N, T>)
        .def("sqrt", &autodiff::detail::sqrt<N, T>)
        .def("cbrt", &autodiff::detail::cbrt<N, T>)
        .def("__pow__", &autodiff::detail::pow<N, T>)
        .def("__pow__", [](const Real<N, T>& self, const T& x) {return pow(self, x); })
        .def("__rpow__", [](const Real<N, T>& self, const T& x) {return pow(x, self); })
        .def("sin", &autodiff::detail::sin<N, T>)
        .def("cos", &autodiff::detail::cos<N, T>)
        .def("tan", &autodiff::detail::tan<N, T>)
        .def("arcsin", &autodiff::detail::asin<N, T>)
        .def("arccos", &autodiff::detail::acos<N, T>)
        .def("arctan", &autodiff::detail::atan<N, T>)
        .def("arctan2", &autodiff::detail::atan2<N, T>)
        .def("arctan2", [](const Real<N, T>& x, const T& y) {return atan2(x, y); })
        .def("sinh", &autodiff::detail::sinh<N, T>)
        .def("cosh", &autodiff::detail::cosh<N, T>)
        .def("tanh", &autodiff::detail::tanh<N, T>)
        .def("arcsinh", &autodiff::detail::asinh<N, T>)
        .def("arccosh", &autodiff::detail::acosh<N, T>)
        .def("__abs__", &autodiff::detail::abs<N, T>)
        .def("minimum", &autodiff::detail::min<N, T>)
        .def("minimum", [](const Real<N, T>& x, const T& y) {return min(x, y); })
        .def("minimum", [](const T& x, const Real<N, T>& y) {return min(x, y); })
        .def("maximum", &autodiff::detail::max<N, T>)
        .def("maximum", [](const Real<N, T>& x, const T& y) {return max(x, y); })
        .def("maximum", [](const T& x, const Real<N, T>& y) {return max(x, y); })

        .def("val", [](const Real<N, T>& self) { return self.val(); })
        .def("val", [](Real<N, T>& self) { return self.val(); })

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

        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)

        .def(py::self += T())
        .def(py::self -= T())
        .def(py::self *= T())
        .def(py::self /= T())

        .def(py::self == T())
        .def(py::self != T())
        .def(py::self < T())
        .def(py::self > T())
        .def(py::self <= T())
        .def(py::self >= T())

        .def(T() == py::self)
        .def(T() != py::self)
        .def(T() < py::self)
        .def(T() > py::self)
        .def(T() <= py::self)
        .def(T() >= py::self)
        ;

    if constexpr (!isSame<T, int>) cls.def(py::init<int>());
    if constexpr (!isSame<T, long>) cls.def(py::init<long>());
    if constexpr (!isSame<T, float>) cls.def(py::init<float>());
    if constexpr (!isSame<T, double>) cls.def(py::init<double>());

    py::implicitly_convertible<T, Real<N, T>>();

    if constexpr (!isSame<T, int>) py::implicitly_convertible<int, Real<N, T>>();
    if constexpr (!isSame<T, long>) py::implicitly_convertible<long, Real<N, T>>();
    if constexpr (!isSame<T, float>) py::implicitly_convertible<float, Real<N, T>>();
    if constexpr (!isSame<T, double>) py::implicitly_convertible<double, Real<N, T>>();

    m.def("seed", [](Real<N, T>& x) { x[1] = 1.0; });
    m.def("unseed", [](Real<N, T>& x) { x[1] = 1.0; });

    m.def("abs"  , [](const Real<N, T>& x) { return abs(x); });

    m.def("sin"  , [](const Real<N, T>& x) { return sin(x); });
    m.def("cos"  , [](const Real<N, T>& x) { return cos(x); });
    m.def("tan"  , [](const Real<N, T>& x) { return tan(x); });

    m.def("asin" , [](const Real<N, T>& x) { return asin(x); });
    m.def("acos" , [](const Real<N, T>& x) { return acos(x); });
    m.def("atan" , [](const Real<N, T>& x) { return atan(x); });

    m.def("asinh", [](const Real<N, T>& x) { return asinh(x); });
    m.def("acosh", [](const Real<N, T>& x) { return acosh(x); });
    m.def("atanh", [](const Real<N, T>& x) { return atanh(x); });

    m.def("sinh" , [](const Real<N, T>& x) { return sinh(x); });
    m.def("cosh" , [](const Real<N, T>& x) { return cosh(x); });
    m.def("tanh" , [](const Real<N, T>& x) { return tanh(x); });

    m.def("arcsin" , [](const Real<N, T>& x) { return asin(x); });
    m.def("arccos" , [](const Real<N, T>& x) { return acos(x); });
    m.def("arctan" , [](const Real<N, T>& x) { return atan(x); });

    m.def("arcsinh", [](const Real<N, T>& x) { return asinh(x); });
    m.def("arccosh", [](const Real<N, T>& x) { return acosh(x); });
    m.def("arctanh", [](const Real<N, T>& x) { return atanh(x); });

    m.def("sqrt" , [](const Real<N, T>& x) { return sqrt(x); });
    m.def("cbrt" , [](const Real<N, T>& x) { return cbrt(x); });

    m.def("exp"  , [](const Real<N, T>& x) { return exp(x);   });
    m.def("log"  , [](const Real<N, T>& x) { return log(x);   });
    m.def("log10", [](const Real<N, T>& x) { return log10(x); });

    m.def("pow"  , [](const Real<N, T>& x, const Real<N, T>& y) { return pow(x, y); });
    m.def("pow"  , [](const Real<N, T>& x, const T& y)          { return pow(x, y); });
    m.def("pow"  , [](const T& x, const Real<N, T>& y)          { return pow(x, y); });

    m.def("max"  , [](const Real<N, T>& x, const Real<N, T>& y) { return max(x, y); });
    m.def("max"  , [](const Real<N, T>& x, const T& y)          { return max(x, y); });
    m.def("max"  , [](const T& x, const Real<N, T>& y)          { return max(x, y); });

    m.def("min"  , [](const Real<N, T>& x, const Real<N, T>& y) { return min(x, y); });
    m.def("min"  , [](const Real<N, T>& x, const T& y)          { return min(x, y); });
    m.def("min"  , [](const T& x, const Real<N, T>& y)          { return min(x, y); });
}

void export_real1st(py::module& m) { exportReal<1, double>(m, "real1st"); }
void export_real2nd(py::module& m) { exportReal<2, double>(m, "real2nd"); }
void export_real3rd(py::module& m) { exportReal<3, double>(m, "real3rd"); }
void export_real4th(py::module& m) { exportReal<4, double>(m, "real4th"); }
