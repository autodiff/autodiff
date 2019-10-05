//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2019 Allan Leal
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
#include <memory_resource>
#include <array>

namespace autodiff::reverse {
/// @brief memory managment for <code> autodiff::reverse </code> algorthms
namespace memory {

/// @brief stack_resource to allocate memory on stack
/// @details
/// stack_resource conatin <code> std::array </code> where we take memory
/// memory will free when dtor called
/// and cursor which points on free memory in <code> std::array </code> 
/// <br>
/// @todo do we always need to check capacity ? 
/// @tparam Size count of bytes which we will allocate
template<std::size_t Size>
class stack_resource final : public std::pmr::memory_resource 
{
    static_assert(Size != 0 && "We can't create resource with Size = 0");
    using void_t = void;
public:
    /// do allocation of memory
    /// @details
    /// @param bytes bytes count to allocate
    /// @param alignment unused
    /// @return <code> void_t* </code> pointer to memory block
    void_t* do_allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t) [[maybe_unused]]) override
    {
        check_capacity(bytes);
        void_t* resource = static_cast<void_t*>(storage.data() + cursor);  
        cursor += bytes;
        return resource;
    }
    /// do deallocation of memory
    /// @details
    /// just override base class method
    /// @param pointer unused
    /// @param bytes unused
    /// @param alignment unused
    void do_deallocate(void_t* pointer [[maybe_unused]], size_t bytes [[maybe_unused]], size_t alignment [[maybe_unused]]) noexcept override { }
    /// do check of equality
    /// @details
    /// @param other other memory_resource
    /// @return <code> true </code> if <code> other </code> and 
    /// <code> *this </code> have same adress
    bool do_is_equal(const memory_resource& other) const noexcept override
    {
        return std::addressof(*this) == std::addressof(other);
    }
    /// position of memory block
    /// @details
    /// @return cursor on current memory block
    std::size_t position() const noexcept 
    {
        return cursor;
    }
private:
    /// check storage capacity
    /// @details
    /// throws exception when lack of memory
    /// @param bytes bytes count to allocate
    void check_capacity(std::size_t bytes) const noexcept(false)
    {
        if(bytes + cursor > Size)
            throw std::runtime_error("Stack storage too small for your necessities");
    }

    std::size_t cursor = { 0 };     ///< cursor to free memory
    std::array<char, Size> storage; ///< memory storage
};
}
}
