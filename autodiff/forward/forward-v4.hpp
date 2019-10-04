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
// of this software and `associated documentation files (the "Software"), to deal
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
#include <cmath>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

// autodiff includes
#include <autodiff/forward/binomialcoefficient.hpp>

namespace autodiff::forward {

template<size_t I>
struct Index
{
    constexpr static size_t index = I;
    constexpr operator const size_t() const { return index; }
    constexpr operator size_t() { return index; }
};

namespace impl {

template<size_t I, size_t Imin, size_t Imax, typename Function>
constexpr auto For(Function&& f)
{
    if constexpr (I < Imax)
    {
        f(Index<I>{});
        For<I + 1, Imin, Imax>(std::forward<Function>(f));
    }
}

template<size_t I, size_t Imin, size_t Imax, typename Function>
constexpr auto ReverseFor(Function&& f)
{
    if constexpr (I < Imax)
    {
        ReverseFor<I + 1, Imin, Imax>(std::forward<Function>(f));
        f(Index<I>{});
    }
}

} // namespace impl

template<size_t Imin, size_t Imax, typename Function>
constexpr auto For(Function&& f)
{
    impl::For<Imin, Imin, Imax>(std::forward<Function>(f));
}

template<size_t Imax, typename Function>
constexpr auto For(Function&& f)
{
    For<0, Imax>(std::forward<Function>(f));
}

template<size_t Imin, size_t Imax, typename Function>
constexpr auto ReverseFor(Function&& f)
{
    impl::ReverseFor<Imin, Imin, Imax>(std::forward<Function>(f));
}

template<size_t Imax, typename Function>
constexpr auto ReverseFor(Function&& f)
{
    ReverseFor<0, Imax>(std::forward<Function>(f));
}

template<size_t Imin, size_t Imax, typename Function>
constexpr auto Sum(Function&& f)
{
    using T = std::invoke_result_t<Function, Index<Imin>>;
    auto aux = T{};
    For<Imin, Imax>([&](auto i) constexpr {
        aux += f(i);
    });
    return aux;
}

template<size_t i, size_t j>
constexpr double binomialCoefficient()
{
    static_assert(i <= detail::binomialcoeffs_nmax, "Violation of maximum order for binomial coefficient retrieval.");
    static_assert(j <= i, "Violation of j <= i condition for retrieving binomial coefficient C(i,j).");
    return detail::binomialcoeffs_data[detail::binomialcoeffs_offsets[i] + j];
}

template<size_t M, typename T>
class numarray
{
public:
    constexpr numarray() = delete;
    constexpr numarray(const numarray&) = delete;

    constexpr explicit numarray(T* data)
    : m_data(data)
    {}

    constexpr auto operator[](size_t i) -> T&
    {
        return m_data[i];
    }

    constexpr auto operator[](size_t i) const -> const T&
    {
        return m_data[i];
    }

    constexpr auto operator=(const numarray& other) -> numarray&
    {
        For<M>([&](auto i) constexpr { m_data[i] = other[i]; });
        return *this;
    }

    template<size_t N, typename U>
    constexpr auto operator=(const numarray<N, U>& other) -> numarray&
    {
        static_assert(M <= N);
        For<M>([&](auto i) constexpr { m_data[i] = other[i]; });
        return *this;
    }

    constexpr auto operator=(const T& scalar) -> numarray&
    {
        For<M>([&](auto i) constexpr { m_data[i] = scalar; });
        return *this;
    }

    constexpr auto operator+=(const T& scalar) -> numarray&
    {
        For<M>([&](auto i) constexpr { m_data[i] += scalar; });
        return *this;
    }

    constexpr auto operator-=(const T& scalar) -> numarray&
    {
        For<M>([&](auto i) constexpr { m_data[i] -= scalar; });
        return *this;
    }

    constexpr auto operator*=(const T& scalar) -> numarray&
    {
        For<M>([&](auto i) constexpr { m_data[i] *= scalar; });
        return *this;
    }

    constexpr auto operator/=(const T& scalar) -> numarray&
    {
        For<M>([&](auto i) constexpr { m_data[i] /= scalar; });
        return *this;
    }

    template<size_t N, typename U>
    constexpr auto operator+=(const numarray<N, U>& other) -> numarray&
    {
        static_assert(M <= N);
        For<M>([&](auto i) constexpr { m_data[i] += other[i]; });
        return *this;
    }

    template<size_t N, typename U>
    constexpr auto operator-=(const numarray<N, U>& other) -> numarray&
    {
        static_assert(M <= N);
        For<M>([&](auto i) constexpr { m_data[i] -= other[i]; });
        return *this;
    }

    template<size_t N, typename U>
    constexpr auto operator*=(const numarray<N, U>& other) -> numarray&
    {
        static_assert(M <= N);
        For<M>([&](auto i) constexpr { m_data[i] *= other[i]; });
        return *this;
    }

    template<size_t N, typename U>
    constexpr auto operator/=(const numarray<N, U>& other) -> numarray&
    {
        static_assert(M <= N);
        For<M>([&](auto i) constexpr { m_data[i] /= other[i]; });
        return *this;
    }

    template<size_t N, typename U>
    auto assignNegative(const numarray<N, U>& other) -> numarray&
    {
        static_assert(M <= N);
        For<M>([&](auto i) constexpr { m_data[i] = -other[i]; });
        return *this;
    }

    template<size_t N, typename U>
    auto assignScaled(const T& scalar, const numarray<N, U>& other) -> numarray&
    {
        static_assert(M <= N);
        For<M>([&](auto i) constexpr { m_data[i] += scalar * other[i]; });
        return *this;
    }

    auto fill(const T& value) -> numarray&
    {
        For<M>([&](auto i) constexpr { m_data[i] = value; });
        return *this;
    }

    auto negate() -> numarray&
    {
        For<M>([&](auto i) constexpr { m_data[i] = -m_data[i]; });
        return *this;
    }

private:
    T* m_data;
};

// template<size_t M, typename T>
// constexpr auto numwrap(std::array<T, M>& a) -> numarray<M, T>
// {
//     return { a.data() };
// }

// template<size_t M, typename T>
// constexpr auto numwrap(const std::array<T, M>& a) -> numarray<M, const T>
// {
//     return { a.data() };
// }

//=====================================================================================================================
//
// STANDARD TEMPLATE LIBRARY MATH FUNCTIONS
//
//=====================================================================================================================

using std::abs;
using std::acos;
using std::acosh;
using std::asin;
using std::asinh;
using std::atan;
using std::atanh;
using std::cos;
using std::cosh;
using std::exp;
using std::log;
using std::log10;
using std::pow;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;

//=====================================================================================================================
//
// TYPE TRAITS UTILITIES
//
//=====================================================================================================================

//-----------------------------------------------------------------------------
// ENABLE-IF FOR SFINAE USE
//-----------------------------------------------------------------------------
template<bool value>
using enableif = typename std::enable_if<value>::type;

//-----------------------------------------------------------------------------
// CONVENIENT TYPE TRAIT UTILITIES
//-----------------------------------------------------------------------------
template<typename T>
using plain = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template<typename A, typename B>
using common = typename std::common_type<A, B>::type;






template<typename T>
constexpr bool isNumber = std::is_arithmetic<plain<T>>::value;




//-----------------------------------------------------------------------------
// AUXILIARY CONSTEXPR CONSTANTS
//-----------------------------------------------------------------------------
template<typename T>
constexpr T Zero = static_cast<T>(0);

template<typename T>
constexpr T One = static_cast<T>(1);









//=====================================================================================================================
//
// EXPRESSION TYPES DEFINITION
//
//=====================================================================================================================

namespace detail {

template<size_t M, typename T>
auto begin(const std::array<T, M>& a)
{
    return a.begin();
}

template<size_t M, typename T>
auto begin(std::array<T, M>& a)
{
    return a.begin();
}

template<typename T>
auto begin(T* a) -> T*
{
    return a;
}

} // namespace detail

/// The type used to represent a *N*-th order dual number.
template<size_t N, typename T>
class Dual
{
private:
    /// The value and derivatives of the number up to order *N*.
    std::array<T, N + 1> m_data;

public:
    /// Construct a default Dual number of order *N* and type *T*.
    constexpr Dual()
    : m_data()
    {}

    /// Construct a Dual number with given data.
    constexpr Dual(const T& value)
    {
        m_data[0] = value;
        For<1, N+1>([&](auto i) constexpr { m_data[i] = Zero<T>; });
    }

    /// Construct a Dual number with given data.
    constexpr Dual(const std::array<T, N + 1>& data)
    : m_data(data)
    {}

    /// Construct a Dual number with given data.
    template<size_t M, typename U, enableif<isNumber<U>>...>
    constexpr Dual(const Dual<M, U>& other)
    {
        static_assert(N <= M);
        For<0, N + 1>([&](auto i) constexpr {
            m_data[i] = static_cast<T>(other[i]);
        });
    }

    constexpr auto begin() -> T*
    {
        return detail::begin(m_data);
    }

    constexpr auto begin() const -> const T*
    {
        return detail::begin(m_data);
    }

    constexpr auto data() -> numarray<N + 1, T>
    {
        return numarray<N + 1, T>{ begin() };
    }

    constexpr auto data() const -> numarray<N + 1, const T>
    {
        return numarray<N + 1, const T>{ begin() };
    }

    constexpr auto operator[](size_t i) -> T&
    {
        return begin()[i];
    }

    constexpr auto operator[](size_t i) const -> const T&
    {
        return begin()[i];
    }

    template<typename U, enableif<isNumber<U>>...>
    constexpr auto operator=(const U& value) -> Dual&
    {
        m_data[0] = value;
        For<1, N+1>([&](auto i) constexpr { m_data[i] = Zero<T>; });
        return *this;
    }

    constexpr auto operator=(const std::array<T, N + 1>& data)
    {
        m_data = data;
    }

    template<typename U, enableif<isNumber<U>>...>
    constexpr auto operator+=(const U& value) -> Dual&
    {
        m_data[0] += static_cast<T>(value);
        return *this;
    }

    template<typename U, enableif<isNumber<U>>...>
    constexpr auto operator-=(const U& value) -> Dual&
    {
        m_data[0] -= static_cast<T>(value);
        return *this;
    }

    template<typename U, enableif<isNumber<U>>...>
    constexpr auto operator*=(const U& value) -> Dual&
    {
        For<0, N+1>([&](auto i) constexpr { m_data[i] *= static_cast<T>(value); });
        return *this;
    }

    template<typename U, enableif<isNumber<U>>...>
    constexpr auto operator/=(const U& value) -> Dual&
    {
        For<0, N+1>([&](auto i) constexpr { m_data[i] /= static_cast<T>(value); });
        return *this;
    }

    constexpr auto operator+=(const Dual& y)
    {
        auto& x = *this;
        For<0, N+1>([&](auto i) constexpr { x[i] += y[i]; });
        return *this;
    }

    constexpr auto operator-=(const Dual& y)
    {
        auto& x = *this;
        For<0, N+1>([&](auto i) constexpr { x[i] -= y[i]; });
        return *this;
    }

    constexpr auto operator*=(const Dual& y)
    {
        auto& x = *this;
        ReverseFor<N + 1>([&](auto i) constexpr {
            x[i] = Sum<0, i + 1>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<i.index, j.index>;
                return c * x[i - j] * y[j];
            });
        });
        return *this;
    }

    constexpr auto operator/=(const Dual& y)
    {
        auto& x = *this;
        For<N + 1>([&](auto i) constexpr {
            x[i] -= Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<i.index, j.index>;
                return c * x[j] * y[i - j];
            });
            x[i] /= y[0];
        });
        return *this;
    }
};

/// Return the inverse of a Dual number.
template<size_t N, typename T>
constexpr auto inverse(const Dual<N, T>& x)
{
    Dual<N, T> y = One<T>;
    return y /= x;
}

//=====================================================================================================================
//
// UNARY OPERATORS +(Dual) AND -(Dual)
//
//=====================================================================================================================

template<size_t N, typename T>
auto operator+(const Dual<N, T>& x)
{
    return x;
}

template<size_t N, typename T>
auto operator-(Dual<N, T> x)
{
    For<0, N+1>([&](auto i) constexpr { x[i] = -x[i]; });
    return x;
}

//=====================================================================================================================
//
// BINARY OPERATOR +(Dual, Dual), +(Dual, Number), +(Number, Dual)
//
//=====================================================================================================================

template<size_t N, typename T>
auto operator+(Dual<N, T> x, const Dual<N, T>& y)
{
    return x += y;
}

template<size_t N, typename T, typename U, enableif<isNumber<U>>...>
auto operator+(Dual<N, T> x, const U& y)
{
    return x += y;
}

template<size_t N, typename T, typename U, enableif<isNumber<U>>...>
auto operator+(const U& x, Dual<N, T> y)
{
    return y += x;
}

//=====================================================================================================================
//
// BINARY OPERATOR -(Dual, Dual), -(Dual, Number), -(Number, Dual)
//
//=====================================================================================================================
template<size_t N, typename T>
auto operator-(Dual<N, T> x, const Dual<N, T>& y)
{
    return x -= y;
}

template<size_t N, typename T, typename U, enableif<isNumber<U>>...>
auto operator-(Dual<N, T> x, const U& y)
{
    return x -= y;
}

template<size_t N, typename T, typename U, enableif<isNumber<U>>...>
auto operator-(const U& x, Dual<N, T> y)
{
    For<0, N+1>([&](auto i) constexpr { y[i] = static_cast<T>(x) - y[i]; });
    return y;
}

//=====================================================================================================================
//
// BINARY OPERATOR *(Dual, Dual), *(Dual, Number), *(Number, Dual)
//
//=====================================================================================================================

template<size_t N, typename T>
auto operator*(Dual<N, T> x, const Dual<N, T>& y)
{
    return x *= y;
}

template<size_t N, typename T, typename U, enableif<isNumber<U>>...>
auto operator*(Dual<N, T> x, const U& y)
{
    return x *= y;
}

template<size_t N, typename T, typename U, enableif<isNumber<U>>...>
auto operator*(const U& x, Dual<N, T> y)
{
    return y *= x;
}

//=====================================================================================================================
//
// BINARY OPERATOR /(Dual, Dual), /(Dual, Number), /(Number, Dual)
//
//=====================================================================================================================

template<size_t N, typename T>
auto operator/(Dual<N, T> x, const Dual<N, T>& y)
{
    return x /= y;
}

template<size_t N, typename T, typename U, enableif<isNumber<U>>...>
auto operator/(Dual<N, T> x, const U& y)
{
    return x /= y;
}

template<size_t N, typename T, typename U, enableif<isNumber<U>>...>
auto operator/(const U& x, Dual<N, T> y)
{
    Dual<N, T> z = x;
    return z /= y;
}

//=====================================================================================================================
//
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T>
constexpr auto exp(const Dual<N, T>& x)
{
    Dual<N, T> expx;
    expx[0] = exp(x[0]);
    For<1, N + 1>([&](auto i) constexpr {
        expx[i] = Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<i.index - 1, j.index>;
            return c * x[i - j] * expx[j];
        });
    });
    return expx;
}

template<size_t N, typename T>
constexpr auto log(const Dual<N, T>& x)
{
    assert(x[0] != 0 && "autodiff: log has undefined value at zero");
    Dual<N, T> logx;
    logx[0] = log(x[0]);
    For<1, N + 1>([&](auto i) constexpr {
        logx[i] = x[i] - Sum<1, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<i.index - 1, j.index - 1>;
            return c * x[i - j] * logx[j];
        });
        logx[i] /= x[0];
    });
    return logx;
}

template<size_t N, typename T>
constexpr auto log10(const Dual<N, T>& x)
{
    assert(x[0] != 0 && "autodiff: log10 has undefined value at zero");
    const auto ln10 = 2.302585092994046;
    Dual<N, T> res = log(x);
    return res /= ln10;
}

template<size_t N, typename T>
constexpr auto sqrt(const Dual<N, T>& x)
{
    assert(x[0] != 0 && "autodiff: sqrt has undefined derivatives at zero");
    Dual<N, T> a;
    Dual<N, T> res;
    res[0] = sqrt(x[0]);
    For<1, N + 1>([&](auto i) constexpr {
        a[i] = x[i] - Sum<1, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<i.index - 1, j.index - 1>;
            return c * x[i - j] * a[j];
        });
        a[i] /= x[0];

        res[i] = 0.5 * Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<i.index - 1, j.index>;
            return c * a[i - j] * res[j];
        });
    });
    return res;
}

template<size_t N, typename T>
constexpr auto pow(const Dual<N, T>& x, const Dual<N, T>& y)
{
    Dual<N, T> res;
    res[0] = pow(x[0], y[0]);
    if constexpr (N > 0)
    {
        assert(x[0] != 0 && "autodiff: pow has undefined derivatives at zero");
        Dual<N, T> lnx = log(x);
        Dual<N, T> a;
        For<1, N + 1>([&](auto i) constexpr {
            a[i] = Sum<0, i + 1>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<i.index, j.index>;
                return c * y[i - j] * lnx[j];
            });

            res[i] = Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<i.index - 1, j.index>;
                return c * a[i - j] * res[j];
            });
        });
    }
    return res;
}

template<size_t N, typename T, typename U, enableif<isNumber<U>>...>
constexpr auto pow(const Dual<N, T>& x, const U& c)
{
    Dual<N, T> res;
    res[0] = pow(x[0], static_cast<T>(c));
    if constexpr (N > 0)
    {
        assert(x[0] != 0 && "autodiff: pow has undefined derivatives at zero");
        Dual<N, T> a = c * log(x);
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<i.index - 1, j.index>;
                return c * a[i - j] * res[j];
            });
        });
    }
    return res;
}

template<size_t N, typename T, typename U, enableif<isNumber<U>>...>
constexpr auto pow(const U& c, const Dual<N, T>& y)
{
    Dual<N, T> res;
    res[0] = pow(static_cast<T>(c), y[0]);
    if constexpr (N > 0)
    {
        assert(c != 0 && "autodiff: pow has undefined derivatives at zero");
        Dual<N, T> a = y * log(c);
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<i.index - 1, j.index>;
                return c * a[i - j] * res[j];
            });
        });
    }
    return res;
}

//=====================================================================================================================
//
// TRIGONOMETRIC FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T>
auto sincos(const Dual<N, T>& x) -> std::tuple<Dual<N, T>, Dual<N, T>>
{
    Dual<N, T> sinx;
    Dual<N, T> cosx;

    cosx[0] = cos(x[0]);
    sinx[0] = sin(x[0]);

    For<1, N+1>([&](auto i) constexpr {
        cosx[i] = -Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<i.index - 1, j.index>;
            return c * x[i - j] * sinx[j];
        });

        sinx[i] = Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<i.index - 1, j.index>;
            return c * x[i - j] * cosx[j];
        });
    });

    return {sinx, cosx};
}

template<size_t N, typename T>
auto sin(const Dual<N, T>& x)
{
    return std::get<0>(sincos(x));
}

template<size_t N, typename T>
auto cos(const Dual<N, T>& x)
{
    return std::get<1>(sincos(x));
}

template<size_t N, typename T>
auto tan(const Dual<N, T>& x)
{
    Dual<N, T> tanx;
    Dual<N, T> aux;

    tanx[0] = tan(x[0]);
    aux[0] = 1 + tanx[0]*tanx[0];

    For<1, N+1>([&](auto i) constexpr {
        tanx[i] = Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<i.index - 1, j.index>;
            return c * x[i - j] * aux[j];
        });

        aux[i] = 2*Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<i.index - 1, j.index>;
            return c * tanx[i - j] * tanx[j];
        });
    });

    return tanx;
}

template<size_t N, typename T>
constexpr auto asin(Dual<N, T> x)
{
    Dual<N - 1, T> aux(x);
    aux = 1/sqrt(1 - aux*aux);
    x[0] = asin(x[0]);
    For<1, N + 1>([&](auto i) constexpr {
        x[i] = aux[i - 1];
    });
    return x;
}

template<size_t N, typename T>
constexpr auto acos(Dual<N, T> x)
{
    Dual<N - 1, T> aux(x);
    aux = -1/sqrt(1 - aux*aux);
    x[0] = acos(x[0]);
    For<1, N + 1>([&](auto i) constexpr {
        x[i] = aux[i - 1];
    });
    return x;
}

template<size_t N, typename T>
constexpr auto atan(Dual<N, T> x)
{
    Dual<N - 1, T> aux(x);
    aux = 1/(1 + aux*aux);
    x[0] = atan(x[0]);
    For<1, N + 1>([&](auto i) constexpr {
        x[i] = aux[i - 1];
    });
    return x;
}

//=====================================================================================================================
//
// HYPERBOLIC FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T>
auto sinhcosh(const Dual<N, T>& x) -> std::tuple<Dual<N, T>, Dual<N, T>>
{
    Dual<N, T> sinhx;
    Dual<N, T> coshx;

    coshx[0] = cosh(x[0]);
    sinhx[0] = sinh(x[0]);

    For<1, N+1>([&](auto i) constexpr {
        coshx[i] = Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<i.index - 1, j.index>;
            return c * x[i - j] * sinhx[j];
        });

        sinhx[i] = Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<i.index - 1, j.index>;
            return c * x[i - j] * coshx[j];
        });
    });

    return {sinhx, coshx};
}

template<size_t N, typename T>
auto sinh(const Dual<N, T>& x)
{
    return std::get<0>(sinhcosh(x));
}

template<size_t N, typename T>
auto cosh(const Dual<N, T>& x)
{
    return std::get<1>(sinhcosh(x));
}


template<size_t N, typename T>
auto tanh(const Dual<N, T>& x)
{
    Dual<N, T> tanhx;
    Dual<N, T> aux;

    tanhx[0] = tanh(x[0]);
    aux[0] = 1 - tanhx[0]*tanhx[0];

    For<1, N+1>([&](auto i) constexpr {
        tanhx[i] = Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<i.index - 1, j.index>;
            return c * x[i - j] * aux[j];
        });

        aux[i] = -2*Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<i.index - 1, j.index>;
            return c * tanhx[i - j] * tanhx[j];
        });
    });

    return tanhx;
}

template<size_t N, typename T>
constexpr auto asinh(Dual<N, T> x)
{
    Dual<N - 1, T> aux(x);
    aux = 1/sqrt(aux*aux + 1);
    x[0] = asinh(x[0]);
    For<1, N + 1>([&](auto i) constexpr {
        x[i] = aux[i - 1];
    });
    return x;
}

template<size_t N, typename T>
constexpr auto acosh(Dual<N, T> x)
{
    Dual<N - 1, T> aux(x);
    aux = 1/sqrt(aux*aux - 1);
    x[0] = acosh(x[0]);
    For<1, N + 1>([&](auto i) constexpr {
        x[i] = aux[i - 1];
    });
    return x;
}

template<size_t N, typename T>
constexpr auto atanh(Dual<N, T> x)
{
    Dual<N - 1, T> aux(x);
    aux = 1/(1 - aux*aux);
    x[0] = atanh(x[0]);
    For<1, N + 1>([&](auto i) constexpr {
        x[i] = aux[i - 1];
    });
    return x;
}

//=====================================================================================================================
//
// OTHER FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T>
constexpr auto abs(const Dual<N, T>& x)
{
    Dual<N, T> res;
    res[0] = std::abs(x[0]);
    if constexpr (N > 0)
    {
        assert(x[0] != 0 && "autodiff: abs has undefined derivative at zero");
        const T s = std::copysign(1.0, x[0]);
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = s * x[i];
        });
    }
    return res;
}


template<size_t N, typename T>
std::ostream& operator<<(std::ostream& out, const Dual<N, T>& x)
{
    out << x.data[0];
    return out;
}

using dual1st = Dual<1, double>;
using dual2nd = Dual<2, double>;
using dual3rd = Dual<3, double>;
using dual4th = Dual<4, double>;
using dual5th = Dual<5, double>;
using dual6th = Dual<6, double>;
using dual7th = Dual<7, double>;
using dual8th = Dual<8, double>;
using dual9th = Dual<9, double>;

using dual = dual1st;

} // namespace autodiff::forward

namespace autodiff {

using forward::dual1st;
using forward::dual2nd;
using forward::dual3rd;
using forward::dual4th;
using forward::dual5th;
using forward::dual6th;
using forward::dual7th;
using forward::dual8th;
using forward::dual9th;
using forward::dual;

} // namespace autodiff
