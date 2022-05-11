//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2022 Allan Leal
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

// autodiff includes
#include <autodiff/forward/real/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include "eigen.hxx"
using namespace autodiff;

void exportVectorXreal3rd(py::module& m)
{
    exportVector<VectorXreal3rd, real3rd, isarray(false), isconst(false), isview(false)>(m, "VectorXreal3rd");
    exportVector<Eigen::Ref<VectorXreal3rd>, real3rd, isarray(false), isconst(false), isview(true)>(m, "VectorXreal3rdRef");
    exportVector<Eigen::Ref<const VectorXreal3rd>, real3rd, isarray(false), isconst(true), isview(true)>(m, "VectorXreal3rdConstRef");
}
