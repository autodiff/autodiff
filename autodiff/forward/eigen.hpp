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

//------------------------------------------------------------------------------
// SUPPORT FOR EIGEN MATRICES AND VECTORS OF DUAL
//------------------------------------------------------------------------------
namespace Eigen {

template<typename T>
struct NumTraits;

template<typename T, typename G>
struct NumTraits<autodiff::forward::Dual<T, G>> : NumTraits<autodiff::forward::ValueType<T>> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
    typedef autodiff::forward::Dual<T, G> Real;
    typedef autodiff::forward::Dual<T, G> NonInteger;
    typedef autodiff::forward::Dual<T, G> Nested;
    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};

template<typename T, typename G, typename BinOp>
struct ScalarBinaryOpTraits<autodiff::forward::Dual<T, G>, autodiff::forward::ValueType<T>, BinOp>
{
    typedef autodiff::forward::Dual<T, G> ReturnType;
};

template<typename T, typename G, typename BinOp>
struct ScalarBinaryOpTraits<autodiff::forward::ValueType<T>, autodiff::forward::Dual<T, G>, BinOp>
{
    typedef autodiff::forward::Dual<T, G> ReturnType;
};

} // namespace Eigen


//------------------------------------------------------------------------------
// TYPEDEFS FOR EIGEN MATRICES, ARRAYS AND VECTORS OF DUAL
//------------------------------------------------------------------------------

namespace autodiff {

#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)   \
typedef Eigen::Matrix<Type, Size, Size, 0, Size, Size> Matrix##SizeSuffix##TypeSuffix;  \
typedef Eigen::Matrix<Type, Size, 1, 0, Size, 1>       Vector##SizeSuffix##TypeSuffix;  \
typedef Eigen::Array<Type, Size, 1, 0, Size, 1>        Array##SizeSuffix##TypeSuffix;  \
typedef Eigen::Matrix<Type, 1, Size, 1, 1, Size>       RowVector##SizeSuffix##TypeSuffix;

#define EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)         \
typedef Eigen::Array<Type, Size, -1, 0, Size, -1> Array##Size##X##TypeSuffix;  \
typedef Eigen::Matrix<Type, Size, -1, 0, Size, -1> Matrix##Size##X##TypeSuffix;  \
typedef Eigen::Matrix<Type, -1, Size, 0, -1, Size> Matrix##X##Size##TypeSuffix;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, -1, X) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

EIGEN_MAKE_TYPEDEFS_ALL_SIZES(autodiff::dual, dual)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(autodiff::HigherOrderDual<2>, dual2nd)

#undef EIGEN_MAKE_TYPEDEFS_ALL_SIZES
#undef EIGEN_MAKE_TYPEDEFS
#undef EIGEN_MAKE_FIXED_TYPEDEFS

} /* namespace autodiff */

namespace autodiff::forward {

namespace detail {
/// Compile time foreach for tuples
template <typename Tuple, typename Callable>
void forEach(Tuple&& tuple, Callable&& callable)
{
    std::apply(
        [&callable](auto&&... args) { (callable(std::forward<decltype(args)>(args)), ...); },
        std::forward<Tuple>(tuple)
    );
}

/// Create index sequence at interval [Start, End)
template <std::size_t Start, std::size_t End, std::size_t... Is>
auto makeIndexSequenceImpl() {
    if constexpr (End == Start) return std::index_sequence<Is...>();
    else return makeIndexSequenceImpl<Start, End - 1, End - 1, Is...>();
}

/// Create tuple view using index sequence
template <class Tuple, size_t... Is>
constexpr auto viewImpl(Tuple&& t,
    std::index_sequence<Is...>) {
    return std::forward_as_tuple(std::get<Is>(t)...);
}

/// Create index sequence on [Start, End)
template <std::size_t Start, std::size_t End>
using makeIndexSequence = std::decay_t<decltype(makeIndexSequenceImpl<Start, End>())>;

/// Create view on tuple tail size N
template <size_t N, class Tuple>
constexpr auto tailView(Tuple&& t) {
    constexpr auto size = std::tuple_size<std::remove_reference_t<Tuple>>::value;
    static_assert(N <= size, "N must be smaller or equal than size of tuple");
    return viewImpl(t, makeIndexSequence<size - N, size>{});
}

/// Return count of 'joined' tuple elements
template<typename Tuple>
auto count(Tuple&& t) -> std::size_t
{
    std::size_t n = 0;
    forEach(t, [&n](auto&& element) {
        n += element.size();
    });
    return n;
}

/// Wrap T to compatible array interface
template<typename T>
struct EigenVectorAdaptor {
    /// implicit construct from value
    constexpr EigenVectorAdaptor(T v) : val(v) { }
    /// operator [] to add array like access
    T operator[](Eigen::Index) const {
        return val;
    }
    /// size for compatibility
    Eigen::Index size() const {
        return 1;
    }
private:
    T val;
};

/// Meta function to endow floating point array compatible
template<typename T>
using makeTypeCompatible = std::conditional<isDual<std::decay_t<T>>, EigenVectorAdaptor<T>, T>;

/// Transform pack of types
template <typename Pack, template <typename> class Operation>
struct transformImpl;

/// Partial specialization
template <template <typename...> class Pack, typename... Types, template <typename> class Operation>
struct transformImpl<Pack<Types...>, Operation> { using type = Pack<typename Operation<Types>::type...>; };

/// Transformed Pack alias
template <typename Pack, template <typename> class Operation>
using transform = typename transformImpl<Pack, Operation>::type;
}

/// Create tuple with eigen compatible types from args
template<typename... Args>
constexpr auto wrtpack(Args&&... args)
{
    using comatible_tuple = detail::transform<std::tuple<Args...>, detail::makeTypeCompatible>;
    return (comatible_tuple(std::forward<typename detail::makeTypeCompatible<Args>::type>(args)...));
}

/// Return the gradient vector of scalar function *f* with respect to some or all variables *x*.
template<typename Function, typename Wrt, typename Args, typename Result>
auto gradient(const Function& f, Wrt&& wrt, Args&& args, Result& u) -> Eigen::VectorXd
{
    std::size_t n = detail::count(wrt);

    Eigen::VectorXd g(n);

    Eigen::Index current_index_pos = 0;
    detail::forEach(wrt, [&](auto&& w) {
        for(auto j = 0; j < w.size(); ++j)
        {
            w[j].grad = 1.0;
            u = std::apply(f, args);
            w[j].grad = 0.0;
            g[j + current_index_pos] = u.grad;
        }

        current_index_pos += w.size();
    });

    return g;
}

/// Return the gradient vector of scalar function *f* with respect to some or all variables *x*.
template<typename Function, typename Wrt, typename Args>
auto gradient(const Function& f, Wrt&& wrt, Args&& args) -> Eigen::VectorXd
{
    using Result = decltype(std::apply(f, args));
    Result u;
    return gradient(f, std::forward<Wrt>(wrt), std::forward<Args>(args), u);
}

/// Return the Jacobian matrix of a function *f* with respect to some or all variables.
template<typename Function, typename Wrt, typename Args, typename Result>
auto jacobian(const Function& f, Wrt&& wrt, Args&& args, Result& F) -> Eigen::MatrixXd
{
    std::size_t n = detail::count(wrt);

    if(n == 0) return {};

    std::get<0>(wrt)[0].grad = 1.0;
    F = std::apply(f, args);
    std::get<0>(wrt)[0].grad = 0.0;

    const auto m = F.size();

    Eigen::MatrixXd J(m, n);

    for(auto i = 0; i < m; ++i)
        J(i, 0) = F[i].grad;

    Eigen::Index current_index_pos = std::get<0>(wrt).size();
    constexpr auto wrt_count = std::tuple_size_v<std::remove_reference_t<Wrt>>;
    detail::forEach(detail::tailView<wrt_count - 1>(wrt), [&] (auto&& w) {
        w[0].grad = 1.0;
        F = std::apply(f, args);
        w[0].grad = 0.0;

        for(auto i = 0; i < m; ++i)
            J(i, current_index_pos) = F[i].grad;

        current_index_pos += w.size();
    });

    current_index_pos = 0;
    detail::forEach(wrt, [&] (auto&& w) {
        for(auto j = 1; j < w.size(); ++j)
        {
            w[j].grad = 1.0;
            F = std::apply(f, args);
            w[j].grad = 0.0;

            for(auto i = 0; i < m; ++i)
                J(i, j + current_index_pos) = F[i].grad;
        }

        current_index_pos += w.size();
    });

    return J;
}

/// Return the Jacobian matrix of a function *f* with respect to some or all variables.
template<typename Function, typename Wrt, typename Args>
auto jacobian(const Function& f, Wrt&& wrt, Args&& args) -> Eigen::MatrixXd
{
    using Result = decltype(std::apply(f, args));
    Result F;
    return jacobian(f, std::forward<Wrt>(wrt), std::forward<Args>(args), F);
}

/// Return the hessian matrix of scalar function *f* with respect to some or all variables *x*.
template<typename Function, typename Wrt, typename Args, typename Result, typename Gradient>
auto hessian(const Function& f, Wrt&& wrt, Args&& args, Result& u, Gradient& g) -> Eigen::MatrixXd
{
    std::size_t n = detail::count(wrt);

    Eigen::MatrixXd H(n, n);
    g.resize(n);

    // TODO: take symmetry into account (for tuple forEach)
    Eigen::Index current_index_pos_outer = 0;
    detail::forEach(wrt, [&](auto&& outer) {
        Eigen::Index current_index_pos_inner = 0;
        detail::forEach(wrt, [&](auto&& inner) {
            for (auto i = 0; i < outer.size(); i++)
            {
                outer[i].grad = 1.0;
                for (auto j = i; j < inner.size(); ++j)
                {
                    const auto ii = i + current_index_pos_outer;
                    const auto jj = j + current_index_pos_inner;

                    inner[j].val.grad = 1.0;
                    u = std::apply(f, args);
                    inner[j].val.grad = 0.0;

                    H(jj, ii) = H(ii, jj) = u.grad.grad;
                    g(ii) = static_cast<double>(u.grad);
                }
                outer[i].grad = 0.0;
            }
            current_index_pos_inner += inner.size();
        });
        current_index_pos_outer += outer.size();
    });

    return H;
}

/// Return the hessian matrix of scalar function *f* with respect to some or all variables *x*.
template<typename Function, typename Wrt, typename Args>
auto hessian(const Function& f, Wrt&& wrt, Args&& args) -> Eigen::MatrixXd
{
    using Result = decltype(std::apply(f, args));
    Result u;
    Eigen::VectorXd g;
    return hessian(f, std::forward<Wrt>(wrt), std::forward<Args>(args), u, g);
}
} // namespace autodiff::forward

namespace autodiff {
using forward::wrtpack;
}
