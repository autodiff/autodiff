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

#pragma once

// Eigen includes
#include <Eigen/Core>

//=====================================================================================================================
//
// EIGEN MACROS FOR CREATING NEW TYPE ALIASES
//
//=====================================================================================================================

#define AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)                        \
using Array##SizeSuffix##SizeSuffix##TypeSuffix = Eigen::Array<Type, Size, Size>;                 \
using Array##SizeSuffix##TypeSuffix             = Eigen::Array<Type, Size, 1>;                    \
using Matrix##SizeSuffix##TypeSuffix            = Eigen::Matrix<Type, Size, Size, 0, Size, Size>; \
using Vector##SizeSuffix##TypeSuffix            = Eigen::Matrix<Type, Size, 1, 0, Size, 1>;       \
using RowVector##SizeSuffix##TypeSuffix         = Eigen::Matrix<Type, 1, Size, 1, 1, Size>;

#define AUTODIFF_DEFINE_EIGEN_FIXED_TYPEDEFS(Type, TypeSuffix, Size)            \
using Array##Size##X##TypeSuffix  = Eigen::Array<Type, Size, -1>;               \
using Array##X##Size##TypeSuffix  = Eigen::Array<Type, -1, Size>;               \
using Matrix##Size##X##TypeSuffix = Eigen::Matrix<Type, Size, -1, 0, Size, -1>; \
using Matrix##X##Size##TypeSuffix = Eigen::Matrix<Type, -1, Size, 0, -1, Size>;

#define AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, 2, 2)             \
AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, 3, 3)             \
AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, 4, 4)             \
AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, -1, X)            \
AUTODIFF_DEFINE_EIGEN_FIXED_TYPEDEFS(Type, TypeSuffix, 2)          \
AUTODIFF_DEFINE_EIGEN_FIXED_TYPEDEFS(Type, TypeSuffix, 3)          \
AUTODIFF_DEFINE_EIGEN_FIXED_TYPEDEFS(Type, TypeSuffix, 4)
