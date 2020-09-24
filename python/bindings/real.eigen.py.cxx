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
#include <pybind11/operators.h>
namespace py = pybind11;

// autodiff includes
#include <autodiff/forward/dual/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

template<typename Vec, typename T, bool isconst, bool isview>
void exportVector(py::module& m, std::string typestr)
{
    auto cls = py::class_<Vec>(m, typestr.c_str());

    if constexpr (!isview) {
        using VecRef = Eigen::Ref<Vec>;
        using VecConstRef = Eigen::Ref<const Vec>;

        cls.def(py::init<>());
        cls.def(py::init<long>());
        cls.def(py::init<const VecRef&>());
        cls.def(py::init<const VecConstRef&>());
    }

    // Define constructors if not a view (e.g., wrapped into an Eigen::Ref)
    if constexpr (!isview) {
        cls.def(py::init<>());
        cls.def(py::init<long>());
    }

    cls.def("__len__", [](const Vec& s) { return s.size(); });

    cls.def("__getitem__", [](const Vec& s, size_t i) {
        if(i >= s.size()) throw py::index_error();
        return s[i];
    });

    if constexpr (!isconst) {
        cls.def("__setitem__", [](Vec& s, size_t i, const T& val) {
            if(i >= s.size()) throw py::index_error();
            s[i] = val;
        });
    }

#if EIGEN_VERSION_AT_LEAST(3, 3, 90)
    cls.def("__iter__", [](const Vec& s) {
        return py::make_iterator(s.begin(), s.end()); // begin/end iterators have not always being available!
    }, py::keep_alive<0, 1>()); // keep object alive while iterator exists;
#endif

    cls.def("__str__", [](const Vec& s) { std::stringstream stream; stream << s; return stream.str(); });
}

constexpr auto isview(bool val) { return val; }
constexpr auto isconst(bool val) { return val; }

void exportVectorXreal(py::module& m)
{
    exportVector<VectorXreal0th, real0th, isconst(false), isview(false)>(m, "VectorXreal0th");
    exportVector<VectorXreal1st, real1st, isconst(false), isview(false)>(m, "VectorXreal1st");
    exportVector<VectorXreal2nd, real2nd, isconst(false), isview(false)>(m, "VectorXreal2nd");
    exportVector<VectorXreal3rd, real3rd, isconst(false), isview(false)>(m, "VectorXreal3rd");
    exportVector<VectorXreal4th, real4th, isconst(false), isview(false)>(m, "VectorXreal4th");

    exportVector<Eigen::Ref<VectorXreal0th>, real0th, isconst(false), isview(true)>(m, "VectorXreal0thRef");
    exportVector<Eigen::Ref<VectorXreal1st>, real1st, isconst(false), isview(true)>(m, "VectorXreal1stRef");
    exportVector<Eigen::Ref<VectorXreal2nd>, real2nd, isconst(false), isview(true)>(m, "VectorXreal2ndRef");
    exportVector<Eigen::Ref<VectorXreal3rd>, real3rd, isconst(false), isview(true)>(m, "VectorXreal3rdRef");
    exportVector<Eigen::Ref<VectorXreal4th>, real4th, isconst(false), isview(true)>(m, "VectorXreal4thRef");

    exportVector<Eigen::Ref<const VectorXreal0th>, real0th, isconst(true), isview(true)>(m, "VectorXreal0thConstRef");
    exportVector<Eigen::Ref<const VectorXreal1st>, real1st, isconst(true), isview(true)>(m, "VectorXreal1stConstRef");
    exportVector<Eigen::Ref<const VectorXreal2nd>, real2nd, isconst(true), isview(true)>(m, "VectorXreal2ndConstRef");
    exportVector<Eigen::Ref<const VectorXreal3rd>, real3rd, isconst(true), isview(true)>(m, "VectorXreal3rdConstRef");
    exportVector<Eigen::Ref<const VectorXreal4th>, real4th, isconst(true), isview(true)>(m, "VectorXreal4thConstRef");
}

void exportArrayXreal(py::module& m)
{
    exportVector<ArrayXreal0th, real0th, isconst(false), isview(false)>(m, "ArrayXreal0th");
    exportVector<ArrayXreal1st, real1st, isconst(false), isview(false)>(m, "ArrayXreal1st");
    exportVector<ArrayXreal2nd, real2nd, isconst(false), isview(false)>(m, "ArrayXreal2nd");
    exportVector<ArrayXreal3rd, real3rd, isconst(false), isview(false)>(m, "ArrayXreal3rd");
    exportVector<ArrayXreal4th, real4th, isconst(false), isview(false)>(m, "ArrayXreal4th");

    exportVector<Eigen::Ref<ArrayXreal0th>, real0th, isconst(false), isview(true)>(m, "ArrayXreal0thRef");
    exportVector<Eigen::Ref<ArrayXreal1st>, real1st, isconst(false), isview(true)>(m, "ArrayXreal1stRef");
    exportVector<Eigen::Ref<ArrayXreal2nd>, real2nd, isconst(false), isview(true)>(m, "ArrayXreal2ndRef");
    exportVector<Eigen::Ref<ArrayXreal3rd>, real3rd, isconst(false), isview(true)>(m, "ArrayXreal3rdRef");
    exportVector<Eigen::Ref<ArrayXreal4th>, real4th, isconst(false), isview(true)>(m, "ArrayXreal4thRef");

    exportVector<Eigen::Ref<const ArrayXreal0th>, real0th, isconst(true), isview(true)>(m, "ArrayXreal0thConstRef");
    exportVector<Eigen::Ref<const ArrayXreal1st>, real1st, isconst(true), isview(true)>(m, "ArrayXreal1stConstRef");
    exportVector<Eigen::Ref<const ArrayXreal2nd>, real2nd, isconst(true), isview(true)>(m, "ArrayXreal2ndConstRef");
    exportVector<Eigen::Ref<const ArrayXreal3rd>, real3rd, isconst(true), isview(true)>(m, "ArrayXreal3rdConstRef");
    exportVector<Eigen::Ref<const ArrayXreal4th>, real4th, isconst(true), isview(true)>(m, "ArrayXreal4thConstRef");
}

void exportVectorXdual(py::module& m)
{
    exportVector<VectorXdual0th, dual0th, isconst(false), isview(false)>(m, "VectorXdual0th");
    exportVector<VectorXdual1st, dual1st, isconst(false), isview(false)>(m, "VectorXdual1st");
    exportVector<VectorXdual2nd, dual2nd, isconst(false), isview(false)>(m, "VectorXdual2nd");
    exportVector<VectorXdual3rd, dual3rd, isconst(false), isview(false)>(m, "VectorXdual3rd");
    exportVector<VectorXdual4th, dual4th, isconst(false), isview(false)>(m, "VectorXdual4th");

    exportVector<Eigen::Ref<VectorXdual0th>, dual0th, isconst(false), isview(true)>(m, "VectorXdual0thRef");
    exportVector<Eigen::Ref<VectorXdual1st>, dual1st, isconst(false), isview(true)>(m, "VectorXdual1stRef");
    exportVector<Eigen::Ref<VectorXdual2nd>, dual2nd, isconst(false), isview(true)>(m, "VectorXdual2ndRef");
    exportVector<Eigen::Ref<VectorXdual3rd>, dual3rd, isconst(false), isview(true)>(m, "VectorXdual3rdRef");
    exportVector<Eigen::Ref<VectorXdual4th>, dual4th, isconst(false), isview(true)>(m, "VectorXdual4thRef");

    exportVector<Eigen::Ref<const VectorXdual0th>, dual0th, isconst(true), isview(true)>(m, "VectorXdual0thConstRef");
    exportVector<Eigen::Ref<const VectorXdual1st>, dual1st, isconst(true), isview(true)>(m, "VectorXdual1stConstRef");
    exportVector<Eigen::Ref<const VectorXdual2nd>, dual2nd, isconst(true), isview(true)>(m, "VectorXdual2ndConstRef");
    exportVector<Eigen::Ref<const VectorXdual3rd>, dual3rd, isconst(true), isview(true)>(m, "VectorXdual3rdConstRef");
    exportVector<Eigen::Ref<const VectorXdual4th>, dual4th, isconst(true), isview(true)>(m, "VectorXdual4thConstRef");
}

void exportArrayXdual(py::module& m)
{
    exportVector<ArrayXdual0th, dual0th, isconst(false), isview(false)>(m, "ArrayXdual0th");
    exportVector<ArrayXdual1st, dual1st, isconst(false), isview(false)>(m, "ArrayXdual1st");
    exportVector<ArrayXdual2nd, dual2nd, isconst(false), isview(false)>(m, "ArrayXdual2nd");
    exportVector<ArrayXdual3rd, dual3rd, isconst(false), isview(false)>(m, "ArrayXdual3rd");
    exportVector<ArrayXdual4th, dual4th, isconst(false), isview(false)>(m, "ArrayXdual4th");

    exportVector<Eigen::Ref<ArrayXdual0th>, dual0th, isconst(false), isview(true)>(m, "ArrayXdual0thRef");
    exportVector<Eigen::Ref<ArrayXdual1st>, dual1st, isconst(false), isview(true)>(m, "ArrayXdual1stRef");
    exportVector<Eigen::Ref<ArrayXdual2nd>, dual2nd, isconst(false), isview(true)>(m, "ArrayXdual2ndRef");
    exportVector<Eigen::Ref<ArrayXdual3rd>, dual3rd, isconst(false), isview(true)>(m, "ArrayXdual3rdRef");
    exportVector<Eigen::Ref<ArrayXdual4th>, dual4th, isconst(false), isview(true)>(m, "ArrayXdual4thRef");

    exportVector<Eigen::Ref<const ArrayXdual0th>, dual0th, isconst(true), isview(true)>(m, "ArrayXdual0thConstRef");
    exportVector<Eigen::Ref<const ArrayXdual1st>, dual1st, isconst(true), isview(true)>(m, "ArrayXdual1stConstRef");
    exportVector<Eigen::Ref<const ArrayXdual2nd>, dual2nd, isconst(true), isview(true)>(m, "ArrayXdual2ndConstRef");
    exportVector<Eigen::Ref<const ArrayXdual3rd>, dual3rd, isconst(true), isview(true)>(m, "ArrayXdual3rdConstRef");
    exportVector<Eigen::Ref<const ArrayXdual4th>, dual4th, isconst(true), isview(true)>(m, "ArrayXdual4thConstRef");
}