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
#include <autodiff/forward/real/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include "common.eigen.py.hxx"
using namespace autodiff;

void exportVectorXreal(py::module& m)
{
    exportVector<VectorXreal0th, real0th, isarray(false), isconst(false), isview(false)>(m, "VectorXreal0th");
    exportVector<VectorXreal1st, real1st, isarray(false), isconst(false), isview(false)>(m, "VectorXreal1st");
    exportVector<VectorXreal2nd, real2nd, isarray(false), isconst(false), isview(false)>(m, "VectorXreal2nd");
    exportVector<VectorXreal3rd, real3rd, isarray(false), isconst(false), isview(false)>(m, "VectorXreal3rd");
    exportVector<VectorXreal4th, real4th, isarray(false), isconst(false), isview(false)>(m, "VectorXreal4th");

    exportVector<Eigen::Ref<VectorXreal0th>, real0th, isarray(false), isconst(false), isview(true)>(m, "VectorXreal0thRef");
    exportVector<Eigen::Ref<VectorXreal1st>, real1st, isarray(false), isconst(false), isview(true)>(m, "VectorXreal1stRef");
    exportVector<Eigen::Ref<VectorXreal2nd>, real2nd, isarray(false), isconst(false), isview(true)>(m, "VectorXreal2ndRef");
    exportVector<Eigen::Ref<VectorXreal3rd>, real3rd, isarray(false), isconst(false), isview(true)>(m, "VectorXreal3rdRef");
    exportVector<Eigen::Ref<VectorXreal4th>, real4th, isarray(false), isconst(false), isview(true)>(m, "VectorXreal4thRef");

    exportVector<Eigen::Ref<const VectorXreal0th>, real0th, isarray(false), isconst(true), isview(true)>(m, "VectorXreal0thConstRef");
    exportVector<Eigen::Ref<const VectorXreal1st>, real1st, isarray(false), isconst(true), isview(true)>(m, "VectorXreal1stConstRef");
    exportVector<Eigen::Ref<const VectorXreal2nd>, real2nd, isarray(false), isconst(true), isview(true)>(m, "VectorXreal2ndConstRef");
    exportVector<Eigen::Ref<const VectorXreal3rd>, real3rd, isarray(false), isconst(true), isview(true)>(m, "VectorXreal3rdConstRef");
    exportVector<Eigen::Ref<const VectorXreal4th>, real4th, isarray(false), isconst(true), isview(true)>(m, "VectorXreal4thConstRef");
}

void exportArrayXreal(py::module& m)
{
    exportVector<ArrayXreal0th, real0th, isarray(true), isconst(false), isview(false)>(m, "ArrayXreal0th");
    exportVector<ArrayXreal1st, real1st, isarray(true), isconst(false), isview(false)>(m, "ArrayXreal1st");
    exportVector<ArrayXreal2nd, real2nd, isarray(true), isconst(false), isview(false)>(m, "ArrayXreal2nd");
    exportVector<ArrayXreal3rd, real3rd, isarray(true), isconst(false), isview(false)>(m, "ArrayXreal3rd");
    exportVector<ArrayXreal4th, real4th, isarray(true), isconst(false), isview(false)>(m, "ArrayXreal4th");

    exportVector<Eigen::Ref<ArrayXreal0th>, real0th, isarray(true), isconst(false), isview(true)>(m, "ArrayXreal0thRef");
    exportVector<Eigen::Ref<ArrayXreal1st>, real1st, isarray(true), isconst(false), isview(true)>(m, "ArrayXreal1stRef");
    exportVector<Eigen::Ref<ArrayXreal2nd>, real2nd, isarray(true), isconst(false), isview(true)>(m, "ArrayXreal2ndRef");
    exportVector<Eigen::Ref<ArrayXreal3rd>, real3rd, isarray(true), isconst(false), isview(true)>(m, "ArrayXreal3rdRef");
    exportVector<Eigen::Ref<ArrayXreal4th>, real4th, isarray(true), isconst(false), isview(true)>(m, "ArrayXreal4thRef");

    exportVector<Eigen::Ref<const ArrayXreal0th>, real0th, isarray(true), isconst(true), isview(true)>(m, "ArrayXreal0thConstRef");
    exportVector<Eigen::Ref<const ArrayXreal1st>, real1st, isarray(true), isconst(true), isview(true)>(m, "ArrayXreal1stConstRef");
    exportVector<Eigen::Ref<const ArrayXreal2nd>, real2nd, isarray(true), isconst(true), isview(true)>(m, "ArrayXreal2ndConstRef");
    exportVector<Eigen::Ref<const ArrayXreal3rd>, real3rd, isarray(true), isconst(true), isview(true)>(m, "ArrayXreal3rdConstRef");
    exportVector<Eigen::Ref<const ArrayXreal4th>, real4th, isarray(true), isconst(true), isview(true)>(m, "ArrayXreal4thConstRef");
}
