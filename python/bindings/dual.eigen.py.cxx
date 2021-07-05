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
namespace py = pybind11;

// autodiff includes
#include <autodiff/forward/dual/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include "common.eigen.py.hxx"
using namespace autodiff;

void exportVectorXdual(py::module& m)
{
    exportVector<VectorXdual0th, dual0th, isarray(false), isconst(false), isview(false)>(m, "VectorXdual0th");
    exportVector<VectorXdual1st, dual1st, isarray(false), isconst(false), isview(false)>(m, "VectorXdual1st");
    exportVector<VectorXdual2nd, dual2nd, isarray(false), isconst(false), isview(false)>(m, "VectorXdual2nd");
    exportVector<VectorXdual3rd, dual3rd, isarray(false), isconst(false), isview(false)>(m, "VectorXdual3rd");
    exportVector<VectorXdual4th, dual4th, isarray(false), isconst(false), isview(false)>(m, "VectorXdual4th");

    exportVector<Eigen::Ref<VectorXdual0th>, dual0th, isarray(false), isconst(false), isview(true)>(m, "VectorXdual0thRef");
    exportVector<Eigen::Ref<VectorXdual1st>, dual1st, isarray(false), isconst(false), isview(true)>(m, "VectorXdual1stRef");
    exportVector<Eigen::Ref<VectorXdual2nd>, dual2nd, isarray(false), isconst(false), isview(true)>(m, "VectorXdual2ndRef");
    exportVector<Eigen::Ref<VectorXdual3rd>, dual3rd, isarray(false), isconst(false), isview(true)>(m, "VectorXdual3rdRef");
    exportVector<Eigen::Ref<VectorXdual4th>, dual4th, isarray(false), isconst(false), isview(true)>(m, "VectorXdual4thRef");

    exportVector<Eigen::Ref<const VectorXdual0th>, dual0th, isarray(false), isconst(true), isview(true)>(m, "VectorXdual0thConstRef");
    exportVector<Eigen::Ref<const VectorXdual1st>, dual1st, isarray(false), isconst(true), isview(true)>(m, "VectorXdual1stConstRef");
    exportVector<Eigen::Ref<const VectorXdual2nd>, dual2nd, isarray(false), isconst(true), isview(true)>(m, "VectorXdual2ndConstRef");
    exportVector<Eigen::Ref<const VectorXdual3rd>, dual3rd, isarray(false), isconst(true), isview(true)>(m, "VectorXdual3rdConstRef");
    exportVector<Eigen::Ref<const VectorXdual4th>, dual4th, isarray(false), isconst(true), isview(true)>(m, "VectorXdual4thConstRef");
}

void exportArrayXdual(py::module& m)
{
    exportVector<ArrayXdual0th, dual0th, isarray(true), isconst(false), isview(false)>(m, "ArrayXdual0th");
    exportVector<ArrayXdual1st, dual1st, isarray(true), isconst(false), isview(false)>(m, "ArrayXdual1st");
    exportVector<ArrayXdual2nd, dual2nd, isarray(true), isconst(false), isview(false)>(m, "ArrayXdual2nd");
    exportVector<ArrayXdual3rd, dual3rd, isarray(true), isconst(false), isview(false)>(m, "ArrayXdual3rd");
    exportVector<ArrayXdual4th, dual4th, isarray(true), isconst(false), isview(false)>(m, "ArrayXdual4th");

    exportVector<Eigen::Ref<ArrayXdual0th>, dual0th, isarray(true), isconst(false), isview(true)>(m, "ArrayXdual0thRef");
    exportVector<Eigen::Ref<ArrayXdual1st>, dual1st, isarray(true), isconst(false), isview(true)>(m, "ArrayXdual1stRef");
    exportVector<Eigen::Ref<ArrayXdual2nd>, dual2nd, isarray(true), isconst(false), isview(true)>(m, "ArrayXdual2ndRef");
    exportVector<Eigen::Ref<ArrayXdual3rd>, dual3rd, isarray(true), isconst(false), isview(true)>(m, "ArrayXdual3rdRef");
    exportVector<Eigen::Ref<ArrayXdual4th>, dual4th, isarray(true), isconst(false), isview(true)>(m, "ArrayXdual4thRef");

    exportVector<Eigen::Ref<const ArrayXdual0th>, dual0th, isarray(true), isconst(true), isview(true)>(m, "ArrayXdual0thConstRef");
    exportVector<Eigen::Ref<const ArrayXdual1st>, dual1st, isarray(true), isconst(true), isview(true)>(m, "ArrayXdual1stConstRef");
    exportVector<Eigen::Ref<const ArrayXdual2nd>, dual2nd, isarray(true), isconst(true), isview(true)>(m, "ArrayXdual2ndConstRef");
    exportVector<Eigen::Ref<const ArrayXdual3rd>, dual3rd, isarray(true), isconst(true), isview(true)>(m, "ArrayXdual3rdConstRef");
    exportVector<Eigen::Ref<const ArrayXdual4th>, dual4th, isarray(true), isconst(true), isview(true)>(m, "ArrayXdual4thConstRef");
}
