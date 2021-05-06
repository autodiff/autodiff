// Catch includes
#include "catch.hpp"

// STL include
#include <memory>

// Autodiff include
#include <autodiff/reverse/memory.hpp>

using namespace autodiff::reverse::memory;

TEST_CASE("Test stack_resource")
{
    SECTION("position is zero after creation of resource")
    {
        constexpr auto size = 10u * sizeof(int);
        stack_resource<size> resource;
        REQUIRE(resource.position() == 0);
    }

    SECTION("position changed after allocation on correct size")
    {
        constexpr auto size = 10u * sizeof(int);
        stack_resource<size> resource;
        auto* p = resource.allocate(sizeof(int));
        (void)p;
        REQUIRE(resource.position() == sizeof(int));
    }

    SECTION("pointer is not equal nullptr after do_allocate")
    {
        constexpr auto size = 10u * sizeof(int);
        stack_resource<size> resource;
        auto* p = resource.allocate(sizeof(int));
        REQUIRE(p != nullptr);
    }

    SECTION("pointers are not equal after next allocation")
    {
        constexpr auto size = 10u * sizeof(int);
        stack_resource<size> resource;
        auto* p_first = resource.allocate(sizeof(int));
        auto* p_second = resource.allocate(sizeof(int));
        REQUIRE(p_first != p_second);
    }

    SECTION("allocte_shared via stack_resource don't return nullptr")
    {
        constexpr auto size = 10u * sizeof(int);
        stack_resource<size> resource;

        std::pmr::polymorphic_allocator<int> allocator { &resource };

        auto int_ptr = std::allocate_shared<int>(allocator, 42);

        REQUIRE(int_ptr != nullptr);
    }

    SECTION("allocte_shared via stack_resource contain correct value")
    {
        constexpr auto size = 10u * sizeof(int);
        stack_resource<size> resource;

        std::pmr::polymorphic_allocator<int> allocator{ &resource };

        auto expected_value = 42;
        auto int_ptr = std::allocate_shared<int>(allocator, expected_value);

        REQUIRE(*int_ptr == expected_value);
    }

    SECTION("allocte_shared via stack_resource throws exceprion when lack of memory")
    {
        // To allocate int as shared we need sizeof(int) + sizeof(control_block) 
        constexpr auto size = 1u * sizeof(int);
        stack_resource<size> resource;

        std::pmr::polymorphic_allocator<int> allocator{ &resource };

        REQUIRE_THROWS_WITH(std::allocate_shared<int>(allocator, 42), "Stack storage too small for your necessities");
    }

}
