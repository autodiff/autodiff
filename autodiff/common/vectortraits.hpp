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

#pragma once

// C++ includes
#include <vector>

// autodiff includes
#include <autodiff/common/meta.hpp>

namespace autodiff {
namespace detail {

/// An auxiliary template type to indicate VectorTraits has not been defined for a type.
template<typename V>
struct VectorTraitsNotDefinedFor {};

/// An auxiliary template type to indicate VectorTraits::ReplaceValueType is not supported for a type.
template<typename V>
struct VectorReplaceValueTypeNotSupportedFor {};

/// A vector traits to be defined for each autodiff number.
template<typename V, class Enable = void>
struct VectorTraits
{
    /// The value type of each entry in the vector.
    using ValueType = VectorTraitsNotDefinedFor<V>;

    /// The template alias to replace the value type of a vector type with another value type.
    using ReplaceValueType = VectorReplaceValueTypeNotSupportedFor<V>;
};

/// A template alias used to get the type of the values in a vector type.
template<typename V>
using VectorValueType = typename VectorTraits<PlainType<V>>::ValueType;

/// A template alias used to get the type of a vector that is equivalent to another but with a different value type.
template<typename V, typename NewValueType>
using VectorReplaceValueType = typename VectorTraits<PlainType<V>>::template ReplaceValueType<NewValueType>;

/// A compile-time constant that indicates with a type is a vector type.
template<typename V>
constexpr bool isVector = !std::is_same_v<VectorValueType<PlainType<V>>, VectorTraitsNotDefinedFor<PlainType<V>>>;


/// Implementation of VectorTraits for std::vector.
template<typename T, template<class> typename Allocator>
struct VectorTraits<std::vector<T, Allocator<T>>>
{
    using ValueType = T;

    template<typename NewValueType>
    using ReplaceValueType = std::vector<NewValueType, Allocator<NewValueType>>;
};

} // namespace detail
} // namespace autodiff
