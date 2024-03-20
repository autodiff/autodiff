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

// autodiff includes
#include <autodiff/forward/real.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;

namespace {
template<typename To, typename Callable>
__global__ void assignKernel(std::size_t n, To* to, Callable callee)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) {
        return;
    }
    to[tid] = callee();
}

template<typename To, typename Callable>
void assign(To& to, Callable&& callee)
{
    To* toDevicePtr;
    CHECK(cudaMalloc(&toDevicePtr, sizeof(To)) == cudaSuccess);
    assignKernel<To, Callable><<<1, 1>>>(1, toDevicePtr, callee);
    CHECK(cudaMemcpy(&to, toDevicePtr, sizeof(To), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaFree(toDevicePtr) == cudaSuccess);
}

template<typename From, typename To, typename Callable>
__global__ void unaryKernel(const From* from, std::size_t n, To* to, Callable callee)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) {
        return;
    }
    to[tid] = callee(from[tid]);
}

template<typename From, typename To, typename Callable>
void unary(const From& from, To& to, Callable&& callee)
{
    From* fromDevicePtr;
    To* toDevicePtr;
    CHECK(cudaMalloc(&fromDevicePtr, sizeof(From)) == cudaSuccess);
    CHECK(cudaMalloc(&toDevicePtr, sizeof(To)) == cudaSuccess);
    CHECK(cudaMemcpy(fromDevicePtr, &from, sizeof(From), cudaMemcpyHostToDevice) == cudaSuccess);
    unaryKernel<From, To, Callable><<<1, 1>>>(fromDevicePtr, 1, toDevicePtr, callee);
    CHECK(cudaMemcpy(&to, toDevicePtr, sizeof(To), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaFree(toDevicePtr) == cudaSuccess);
    CHECK(cudaFree(fromDevicePtr) == cudaSuccess);
}

template<typename FromA, typename FromB, typename To, typename Callable>
__global__ void binaryKernel(const FromA* fromA, std::size_t n, const FromB* fromB, To* to, Callable callee)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) {
        return;
    }
    to[tid] = callee(fromA[tid], fromB[tid]);
}

template<typename FromA, typename FromB, typename To, typename Callable>
void binary(const FromA& fromA, const FromB& fromB, To& to, Callable&& callee)
{
    FromA* fromADevicePtr;
    FromB* fromBDevicePtr;
    To* toDevicePtr;
    CHECK(cudaMalloc(&fromADevicePtr, sizeof(FromA)) == cudaSuccess);
    CHECK(cudaMalloc(&fromBDevicePtr, sizeof(FromB)) == cudaSuccess);
    CHECK(cudaMalloc(&toDevicePtr, sizeof(To)) == cudaSuccess);
    CHECK(cudaMemcpy(fromADevicePtr, &fromA, sizeof(FromA), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(fromBDevicePtr, &fromB, sizeof(FromB), cudaMemcpyHostToDevice) == cudaSuccess);
    binaryKernel<FromA, FromB, To, Callable><<<1, 1>>>(fromADevicePtr, 1, fromBDevicePtr, toDevicePtr, callee);
    CHECK(cudaMemcpy(&to, toDevicePtr, sizeof(To), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaFree(toDevicePtr) == cudaSuccess);
    CHECK(cudaFree(fromBDevicePtr) == cudaSuccess);
    CHECK(cudaFree(fromADevicePtr) == cudaSuccess);
}

template<typename FromA, typename FromB, typename FromC, typename To, typename Callable>
__global__ void ternaryKernel(const FromA* fromA, std::size_t n, const FromB* fromB, const FromC* fromC, To* to, Callable callee)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) {
        return;
    }
    to[tid] = callee(fromA[tid], fromB[tid], fromC[tid]);
}

template<typename FromA, typename FromB, typename FromC, typename To, typename Callable>
void ternary(const FromA& fromA, const FromB& fromB, const FromC& fromC, To& to, Callable&& callee)
{
    FromA* fromADevicePtr;
    FromB* fromBDevicePtr;
    FromC* fromCDevicePtr;
    To* toDevicePtr;
    CHECK(cudaMalloc(&fromADevicePtr, sizeof(FromA)) == cudaSuccess);
    CHECK(cudaMalloc(&fromBDevicePtr, sizeof(FromB)) == cudaSuccess);
    CHECK(cudaMalloc(&fromCDevicePtr, sizeof(FromC)) == cudaSuccess);
    CHECK(cudaMalloc(&toDevicePtr, sizeof(To)) == cudaSuccess);
    CHECK(cudaMemcpy(fromADevicePtr, &fromA, sizeof(FromA), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(fromBDevicePtr, &fromB, sizeof(FromB), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(fromCDevicePtr, &fromC, sizeof(FromC), cudaMemcpyHostToDevice) == cudaSuccess);
    ternaryKernel<FromA, FromB, FromC, To, Callable><<<1, 1>>>(fromADevicePtr, 1, fromBDevicePtr, fromCDevicePtr, toDevicePtr, callee);
    CHECK(cudaMemcpy(&to, toDevicePtr, sizeof(To), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaFree(toDevicePtr) == cudaSuccess);
    CHECK(cudaFree(fromCDevicePtr) == cudaSuccess);
    CHECK(cudaFree(fromBDevicePtr) == cudaSuccess);
    CHECK(cudaFree(fromADevicePtr) == cudaSuccess);
}
} // namespace

#define CHECK_4TH_ORDER_REAL_NUMBERS(a, b) \
    CHECK_APPROX(a[0], b[0]);              \
    CHECK_APPROX(a[1], b[1]);              \
    CHECK_APPROX(a[2], b[2]);              \
    CHECK_APPROX(a[3], b[3]);              \
    CHECK_APPROX(a[4], b[4]);

#define CHECK_DERIVATIVES_REAL4TH_WRT(expr)                                                          \
{                                                                                                    \
    real4th x = 5, y = 7;                                                                            \
    auto f = [] __host__ __device__(const real4th& x, const real4th& y) -> real4th { return expr; }; \
    /* Check directional derivatives of f(x,y) along direction (3, 5) */                             \
    decltype(derivatives(f, along(3, 5), at(x, y))) dfdv;                                            \
                                                                                                     \
    binary(x, y, dfdv, [=] __device__(real4th x, real4th y) {                                        \
        auto dfdv = derivatives(f, along(3, 5), at(x, y));                                           \
        return dfdv;                                                                                 \
    });                                                                                              \
    x[1] = 3.0;                                                                                      \
    y[1] = 5.0;                                                                                      \
    u = expr;                                                                                        \
    x[1] = 0.0;                                                                                      \
    y[1] = 0.0;                                                                                      \
    CHECK_APPROX(dfdv[0], u[0]);                                                                     \
    CHECK_APPROX(dfdv[1], u[1]);                                                                     \
    CHECK_APPROX(dfdv[2], u[2]);                                                                     \
    CHECK_APPROX(dfdv[3], u[3]);                                                                     \
    CHECK_APPROX(dfdv[4], u[4]);                                                                     \
}

// Ensure Real can be used with CUDA variable memory space specifiers (must be
// default-constructible at compile time).
__device__ real4th x_device;
__shared__ real4th x_shared;
__constant__ real4th x_const;
__managed__ real4th x_managed;

// Auxiliary constants
#define ln10 (2.302585092994046)
#define pi (3.14159265359)

TEST_CASE("testing autodiff::real", "[forward][real]")
{
    real4th x, y, z, u, v, w;

    unary(x, x, [] __host__ __device__(real4th x) -> real4th { x = 1.0; return x; });

    CHECK_APPROX(x[0], 1.0);
    CHECK_APPROX(x[1], 0.0);
    CHECK_APPROX(x[2], 0.0);
    CHECK_APPROX(x[3], 0.0);
    CHECK_APPROX(x[4], 0.0);

    unary(x, x, [] __host__ __device__(real4th x) -> real4th { x = {0.5, 3.0, -5.0, -15.0, 11.0}; return x; });

    CHECK_APPROX(x[0], 0.5);
    CHECK_APPROX(x[1], 3.0);
    CHECK_APPROX(x[2], -5.0);
    CHECK_APPROX(x[3], -15.0);
    CHECK_APPROX(x[4], 11.0);

    y = +x;
    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return +x; });

    CHECK_APPROX(y[0], x[0]);
    CHECK_APPROX(y[1], x[1]);
    CHECK_APPROX(y[2], x[2]);
    CHECK_APPROX(y[3], x[3]);
    CHECK_APPROX(y[4], x[4]);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return -x; });

    CHECK_APPROX(y[0], -x[0]);
    CHECK_APPROX(y[1], -x[1]);
    CHECK_APPROX(y[2], -x[2]);
    CHECK_APPROX(y[3], -x[3]);
    CHECK_APPROX(y[4], -x[4]);

    z = x + y;
    binary(x, y, z, [] __host__ __device__(real4th x, real4th y) -> real4th { return x + y; });

    CHECK_APPROX(z[0], x[0] + y[0]);
    CHECK_APPROX(z[1], x[1] + y[1]);
    CHECK_APPROX(z[2], x[2] + y[2]);
    CHECK_APPROX(z[3], x[3] + y[3]);
    CHECK_APPROX(z[4], x[4] + y[4]);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return x + 1.0; });

    CHECK_APPROX(z[0], x[0] + 1.0);
    CHECK_APPROX(z[1], x[1]);
    CHECK_APPROX(z[2], x[2]);
    CHECK_APPROX(z[3], x[3]);
    CHECK_APPROX(z[4], x[4]);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return 1.0 + x; });

    CHECK_APPROX(z[0], x[0] + 1.0);
    CHECK_APPROX(z[1], x[1]);
    CHECK_APPROX(z[2], x[2]);
    CHECK_APPROX(z[3], x[3]);
    CHECK_APPROX(z[4], x[4]);

    binary(x, y, z, [] __host__ __device__(real4th x, real4th y) -> real4th { return x - y; });

    CHECK_APPROX(z[0], x[0] - y[0]);
    CHECK_APPROX(z[1], x[1] - y[1]);
    CHECK_APPROX(z[2], x[2] - y[2]);
    CHECK_APPROX(z[3], x[3] - y[3]);
    CHECK_APPROX(z[4], x[4] - y[4]);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return x - 1.0; });

    CHECK_APPROX(z[0], x[0] - 1.0);
    CHECK_APPROX(z[1], x[1]);
    CHECK_APPROX(z[2], x[2]);
    CHECK_APPROX(z[3], x[3]);
    CHECK_APPROX(z[4], x[4]);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return 1.0 - x; });

    CHECK_APPROX(z[0], 1.0 - x[0]);
    CHECK_APPROX(z[1], -x[1]);
    CHECK_APPROX(z[2], -x[2]);
    CHECK_APPROX(z[3], -x[3]);
    CHECK_APPROX(z[4], -x[4]);

    binary(x, y, z, [] __host__ __device__(real4th x, real4th y) -> real4th { return x * y; });

    CHECK_APPROX(z[0], x[0] * y[0]);
    CHECK_APPROX(z[1], x[1] * y[0] + x[0] * y[1]);
    CHECK_APPROX(z[2], x[2] * y[0] + 2 * x[1] * y[1] + x[0] * y[2]);
    CHECK_APPROX(z[3], x[3] * y[0] + 3 * x[2] * y[1] + 3 * x[1] * y[2] + x[0] * y[3]);
    CHECK_APPROX(z[4], x[4] * y[0] + 4 * x[3] * y[1] + 6 * x[2] * y[2] + 4 * x[1] * y[3] + x[0] * y[4]);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return x * 3.0; });

    CHECK_APPROX(z[0], x[0] * 3.0);
    CHECK_APPROX(z[1], x[1] * 3.0);
    CHECK_APPROX(z[2], x[2] * 3.0);
    CHECK_APPROX(z[3], x[3] * 3.0);
    CHECK_APPROX(z[4], x[4] * 3.0);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return 5.0 * x; });

    CHECK_APPROX(z[0], 5.0 * x[0]);
    CHECK_APPROX(z[1], 5.0 * x[1]);
    CHECK_APPROX(z[2], 5.0 * x[2]);
    CHECK_APPROX(z[3], 5.0 * x[3]);
    CHECK_APPROX(z[4], 5.0 * x[4]);

    binary(x, y, z, [] __host__ __device__(real4th x, real4th y) -> real4th { return x / y; });

    CHECK_APPROX(z[0], (x[0]) / y[0]);
    CHECK_APPROX(z[1], (x[1] - y[1] * z[0]) / y[0]);
    CHECK_APPROX(z[2], (x[2] - y[2] * z[0] - 2 * y[1] * z[1]) / y[0]);
    CHECK_APPROX(z[3], (x[3] - y[3] * z[0] - 3 * y[2] * z[1] - 3 * y[1] * z[2]) / y[0]);
    CHECK_APPROX(z[4], (x[4] - y[4] * z[0] - 4 * y[3] * z[1] - 6 * y[2] * z[2] - 4 * y[1] * z[3]) / y[0]);

    unary(y, z, [] __host__ __device__(real4th y) -> real4th { return 3.0 / y; });

    CHECK_APPROX(z[0], 3.0 / y[0]);
    CHECK_APPROX(z[1], -(y[1] * z[0]) / y[0]);
    CHECK_APPROX(z[2], -(y[2] * z[0] + 2 * y[1] * z[1]) / y[0]);
    CHECK_APPROX(z[3], -(y[3] * z[0] + 3 * y[2] * z[1] + 3 * y[1] * z[2]) / y[0]);
    CHECK_APPROX(z[4], -(y[4] * z[0] + 4 * y[3] * z[1] + 6 * y[2] * z[2] + 4 * y[1] * z[3]) / y[0]);

    unary(y, z, [] __host__ __device__(real4th y) -> real4th { return y / 5.0; });

    CHECK_APPROX(z[0], y[0] / 5.0);
    CHECK_APPROX(z[1], y[1] / 5.0);
    CHECK_APPROX(z[2], y[2] / 5.0);
    CHECK_APPROX(z[3], y[3] / 5.0);
    CHECK_APPROX(z[4], y[4] / 5.0);

    //=====================================================================================================================
    //
    // TESTING EXPONENTIAL AND LOGARITHMIC FUNCTIONS
    //
    //=====================================================================================================================

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return exp(x); });

    CHECK_APPROX(y[0], exp(x[0]));
    CHECK_APPROX(y[1], x[1] * y[0]);
    CHECK_APPROX(y[2], x[2] * y[0] + x[1] * y[1]);
    CHECK_APPROX(y[3], x[3] * y[0] + 2 * x[2] * y[1] + x[1] * y[2]);
    CHECK_APPROX(y[4], x[4] * y[0] + 3 * x[3] * y[1] + 3 * x[2] * y[2] + x[1] * y[3]);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return log(x); });

    CHECK_APPROX(y[0], log(x[0]));
    CHECK_APPROX(y[1], (x[1]) / x[0]);
    CHECK_APPROX(y[2], (x[2] - x[1] * y[1]) / x[0]);
    CHECK_APPROX(y[3], (x[3] - x[2] * y[1] - 2 * x[1] * y[2]) / x[0]);
    CHECK_APPROX(y[4], (x[4] - x[3] * y[1] - 3 * x[2] * y[2] - 3 * x[1] * y[3]) / x[0]);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return log10(x); });
    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return log(x) / ln10; });

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return sqrt(x); });
    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return exp(0.5 * log(x)); });

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return cbrt(x); });
    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return exp(1.0 / 3.0 * log(x)); });

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return pow(x, x); });
    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return exp(x * log(x)); });

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return pow(x, pi); });
    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return exp(pi * log(x)); });

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return pow(pi, x); });
    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return exp(x * log(pi)); });

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    //=====================================================================================================================
    //
    // TESTING TRIGONOMETRIC FUNCTIONS
    //
    //=====================================================================================================================

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return sin(x); });
    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return cos(x); });

    CHECK_APPROX(y[0], sin(x[0]));
    CHECK_APPROX(z[0], cos(x[0]));
    CHECK_APPROX(y[1], x[1] * z[0]);
    CHECK_APPROX(z[1], -x[1] * y[0]);
    CHECK_APPROX(y[2], x[2] * z[0] + x[1] * z[1]);
    CHECK_APPROX(z[2], -x[2] * y[0] - x[1] * y[1]);
    CHECK_APPROX(y[3], x[3] * z[0] + 2 * x[2] * z[1] + x[1] * z[2]);
    CHECK_APPROX(z[3], -x[3] * y[0] - 2 * x[2] * y[1] - x[1] * y[2]);
    CHECK_APPROX(y[4], x[4] * z[0] + 3 * x[3] * z[1] + 3 * x[2] * z[2] + x[1] * z[3]);
    CHECK_APPROX(z[4], -x[4] * y[0] - 3 * x[3] * y[1] - 3 * x[2] * y[2] - x[1] * y[3]);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return tan(x); });
    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return sin(x) / cos(x); });

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    //=====================================================================================================================
    //
    // TESTING INVERSE TRIGONOMETRIC FUNCTIONS
    //
    //=====================================================================================================================

    real4th xprime = {{x[1], x[2], x[3], x[4]}};

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return asin(x); });
    binary(x, xprime, z, [] __host__ __device__(real4th x, real4th xprime) -> real4th { return xprime / sqrt(1 - x * x); });

    CHECK_APPROX(y[0], asin(x[0]));
    CHECK_APPROX(y[1], z[0]);
    CHECK_APPROX(y[2], z[1]);
    CHECK_APPROX(y[3], z[2]);
    CHECK_APPROX(y[4], z[3]);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return acos(x); });
    binary(x, xprime, z, [] __host__ __device__(real4th x, real4th xprime) -> real4th { return -xprime / sqrt(1 - x * x); });

    CHECK_APPROX(y[0], acos(x[0]));
    CHECK_APPROX(y[1], z[0]);
    CHECK_APPROX(y[2], z[1]);
    CHECK_APPROX(y[3], z[2]);
    CHECK_APPROX(y[4], z[3]);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return atan(x); });
    binary(x, xprime, z, [] __host__ __device__(real4th x, real4th xprime) -> real4th { return xprime / (1 + x * x); });

    CHECK_APPROX(y[0], atan(x[0]));
    CHECK_APPROX(y[1], z[0]);
    CHECK_APPROX(y[2], z[1]);
    CHECK_APPROX(y[3], z[2]);
    CHECK_APPROX(y[4], z[3]);

    // atan2(double, real4th)
    constexpr double c = 2.0;
    unary(x, y, [=] __host__ __device__(real4th x) -> real4th { return atan2(c, x); });
    binary(x, xprime, z, [=] __host__ __device__(real4th x, real4th xprime) -> real4th { return xprime * (-c / (c * c + x * x)); });

    CHECK_APPROX(y[0], atan2(c, x[0]));
    CHECK_APPROX(y[1], z[0]);
    CHECK_APPROX(y[2], z[1]);
    CHECK_APPROX(y[3], z[2]);
    CHECK_APPROX(y[4], z[3]);

    // atan2(real4th, double)
    unary(x, y, [=] __host__ __device__(real4th x) -> real4th { return atan2(x, c); });
    binary(x, xprime, z, [=] __host__ __device__(real4th x, real4th xprime) -> real4th { return xprime * (c / (c * c + x * x)); });

    CHECK_APPROX(y[0], atan2(x[0], c));
    CHECK_APPROX(y[1], z[0]);
    CHECK_APPROX(y[2], z[1]);
    CHECK_APPROX(y[3], z[2]);
    CHECK_APPROX(y[4], z[3]);

    // atan2(real4th, real4th)
    real4th yprime = {{y[1], y[2], y[3], y[4]}};

    real4th s;
    binary(y, x, s, [] __host__ __device__(real4th y, real4th x) -> real4th { real4th s = atan2(y, x); return s; });
    unary(x, z, [=] __host__ __device__(real4th x) -> real4th { return (x[0] * yprime - y[0] * xprime) / (x[0] * x[0] + y[0] * y[0]); });

    CHECK_APPROX(s[0], atan2(y[0], x[0]));
    CHECK_APPROX(s[1], z[0]);
    CHECK_APPROX(s[2], z[1]);
    CHECK_APPROX(s[3], z[2]);
    CHECK_APPROX(s[4], z[3]);

    //=====================================================================================================================
    //
    // TESTING HYPERBOLIC FUNCTIONS
    //
    //=====================================================================================================================
    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return sinh(x); });
    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return cosh(x); });

    CHECK_APPROX(y[0], sinh(x[0]));
    CHECK_APPROX(z[0], cosh(x[0]));
    CHECK_APPROX(y[1], x[1] * z[0]);
    CHECK_APPROX(z[1], x[1] * y[0]);
    CHECK_APPROX(y[2], x[2] * z[0] + x[1] * z[1]);
    CHECK_APPROX(z[2], x[2] * y[0] + x[1] * y[1]);
    CHECK_APPROX(y[3], x[3] * z[0] + 2 * x[2] * z[1] + x[1] * z[2]);
    CHECK_APPROX(z[3], x[3] * y[0] + 2 * x[2] * y[1] + x[1] * y[2]);
    CHECK_APPROX(y[4], x[4] * z[0] + 3 * x[3] * z[1] + 3 * x[2] * z[2] + x[1] * z[3]);
    CHECK_APPROX(z[4], x[4] * y[0] + 3 * x[3] * y[1] + 3 * x[2] * y[2] + x[1] * y[3]);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return tanh(x); });
    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return sinh(x) / cosh(x); });

    CHECK_4TH_ORDER_REAL_NUMBERS(y, z);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return asinh(x); });
    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return 1 / sqrt(x * x + 1); });

    CHECK_APPROX(y[0], asinh(x[0]));
    CHECK_APPROX(y[1], z[0]);
    CHECK_APPROX(y[2], z[1]);
    CHECK_APPROX(y[3], z[2]);
    CHECK_APPROX(y[4], z[3]);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return acosh(10*x); /* acosh requires x > 1 */ });
    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return 1 / sqrt(100 * x * x - 1); });

    CHECK_APPROX(y[0], acosh(10 * x[0]));
    CHECK_APPROX(y[1], z[0]);
    CHECK_APPROX(y[2], z[1]);
    CHECK_APPROX(y[3], z[2]);
    CHECK_APPROX(y[4], z[3]);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return atanh(x); });
    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return 1 / (1 - x * x); });

    CHECK_APPROX(y[0], atanh(x[0]));
    CHECK_APPROX(y[1], z[0]);
    CHECK_APPROX(y[2], z[1]);
    CHECK_APPROX(y[3], z[2]);
    CHECK_APPROX(y[4], z[3]);

    //=====================================================================================================================
    //
    // TESTING OTHER FUNCTIONS
    //
    //=====================================================================================================================

    y = abs(x);
    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return abs(x); });

    CHECK_APPROX(y[0], std::abs(x[0]));
    CHECK_APPROX(y[1], std::abs(x[0]) / x[0] * x[1]);
    CHECK_APPROX(y[2], std::abs(x[0]) / x[0] * x[2]);
    CHECK_APPROX(y[3], std::abs(x[0]) / x[0] * x[3]);
    CHECK_APPROX(y[4], std::abs(x[0]) / x[0] * x[4]);

    unary(x, y, [] __host__ __device__(real4th x) -> real4th { return -x; });
    unary(y, z, [] __host__ __device__(real4th y) -> real4th { return abs(y); });

    CHECK_APPROX(z[0], std::abs(y[0]));
    CHECK_APPROX(z[1], std::abs(y[0]) / (y[0]) * y[1]);
    CHECK_APPROX(z[2], std::abs(y[0]) / (y[0]) * y[2]);
    CHECK_APPROX(z[3], std::abs(y[0]) / (y[0]) * y[3]);
    CHECK_APPROX(z[4], std::abs(y[0]) / (y[0]) * y[4]);

    //=====================================================================================================================
    //
    // TESTING MIN/MAX FUNCTIONS
    //
    //=====================================================================================================================

    x = {0.5, 3.0, -5.0, -15.0, 11.0};
    y = {4.5, 3.0, -5.0, -15.0, 11.0};

    binary(x, y, z, [] __host__ __device__(real4th x, real4th y) -> real4th { return min(x, y); });
    CHECK(z == x);

    binary(x, y, z, [] __host__ __device__(real4th x, real4th y) -> real4th { return min(y, x); });
    CHECK(z == x);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return min(x, 0.1); });
    CHECK(z == real4th(0.1));

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return min(0.2, x); });
    CHECK(z == real4th(0.2));

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return min(0.5, x); });
    CHECK(z == x);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return min(x, 0.5); });
    CHECK(z == x);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return min(3.5, x); });
    CHECK(z == x);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return min(x, 3.5); });
    CHECK(z == x);

    binary(x, y, z, [] __host__ __device__(real4th x, real4th y) -> real4th { return max(x, y); });
    CHECK(z == y);

    binary(x, y, z, [] __host__ __device__(real4th x, real4th y) -> real4th { return max(y, x); });
    CHECK(z == y);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return max(x, 0.1); });
    CHECK(z == x);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return max(0.2, x); });
    CHECK(z == x);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return max(0.5, x); });
    CHECK(z == x);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return max(x, 0.5); });
    CHECK(z == x);

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return max(8.5, x); });
    CHECK(z == real4th(8.5));

    unary(x, z, [] __host__ __device__(real4th x) -> real4th { return max(x, 8.5); });
    CHECK(z == real4th(8.5));

    //=====================================================================================================================
    //
    // TESTING COMPARISON OPERATORS
    //
    //=====================================================================================================================

    x = {0.5, 3.0, -5.0, -15.0, 11.0};

    // Check equality not only on value but also on the derivatives
    bool cond;
    unary(x, cond, [] __host__ __device__(real4th x) -> bool { return x == real4th({0.5, 3.0, -5.0, -15.0, 11.0}); });
    CHECK(cond);

    // Check equality against plain numeric types (double) do not require check against derivatives
    unary(x, cond, [] __host__ __device__(real4th x) -> bool { return x == 0.6; });
    CHECK_FALSE(cond);
    unary(x, cond, [] __host__ __device__(real4th x) -> bool { return x == 0.5; });
    CHECK(cond);

    // Check inequalities
    unary(x, cond, [] __host__ __device__(real4th x) -> bool { return x == real4th({0.5, 3.1, -5.0, -15.0, 11.0}); });
    CHECK_FALSE(cond);
    unary(x, cond, [] __host__ __device__(real4th x) -> bool { return x != real4th({0.5, 3.1, -5.0, -15.0, 11.0}); });
    CHECK(cond);
    unary(x, cond, [] __host__ __device__(real4th x) -> bool { return x != 1.0; });
    CHECK(cond);
    unary(x, cond, [] __host__ __device__(real4th x) -> bool { return x < 1.0; });
    CHECK(cond);
    unary(x, cond, [] __host__ __device__(real4th x) -> bool { return x > 0.1; });
    CHECK(cond);
    unary(x, cond, [] __host__ __device__(real4th x) -> bool { return x <= 1.0; });
    CHECK(cond);
    unary(x, cond, [] __host__ __device__(real4th x) -> bool { return x >= 0.1; });
    CHECK(cond);
    unary(x, cond, [] __host__ __device__(real4th x) -> bool { return x <= 0.5; });
    CHECK(cond);
    unary(x, cond, [] __host__ __device__(real4th x) -> bool { return x >= 0.5; });
    CHECK(cond);

    //=====================================================================================================================
    //
    // TESTING DERIVATIVE CALCULATIONS
    //
    //=====================================================================================================================

    CHECK_DERIVATIVES_REAL4TH_WRT(exp(log(2 * x + 3 * y)));
    CHECK_DERIVATIVES_REAL4TH_WRT(sin(2 * x + 3 * y));
    CHECK_DERIVATIVES_REAL4TH_WRT(exp(2 * x + 3 * y) * log(x / y));

    // Testing array-unpacking of derivatives for real number
    {
        real4th x = {{2.0, 3.0, 4.0, 5.0, 6.0}};

        decltype(derivatives(x)) xDerivatives;
        unary(x, xDerivatives, [] __host__ __device__(real4th x) { return derivatives(x); });
        auto [x0, x1, x2, x3, x4] = xDerivatives;

        CHECK_APPROX(x0, x[0]);
        CHECK_APPROX(x1, x[1]);
        CHECK_APPROX(x2, x[2]);
        CHECK_APPROX(x3, x[3]);
        CHECK_APPROX(x4, x[4]);
    }

//    // Testing array-unpacking of derivatives for vector of real numbers
//    {
//        real4th x = {{2.0, 3.0, 4.0, 5.0, 6.0}};
//        real4th y = {{3.0, 4.0, 5.0, 6.0, 7.0}};
//        real4th z = {{4.0, 5.0, 6.0, 7.0, 8.0}};
//
//        std::vector<real4th> u = { x, y, z };
//
//        auto [u0, u1, u2, u3, u4] = derivatives(u);
//
//        CHECK_APPROX( u0[0], x[0] ); CHECK_APPROX( u1[0], x[1] ); CHECK_APPROX( u2[0], x[2] ); CHECK_APPROX( u3[0], x[3] ); CHECK_APPROX( u4[0], x[4] );
//        CHECK_APPROX( u0[1], y[0] ); CHECK_APPROX( u1[1], y[1] ); CHECK_APPROX( u2[1], y[2] ); CHECK_APPROX( u3[1], y[3] ); CHECK_APPROX( u4[1], y[4] );
//        CHECK_APPROX( u0[2], z[0] ); CHECK_APPROX( u1[2], z[1] ); CHECK_APPROX( u2[2], z[2] ); CHECK_APPROX( u3[2], z[3] ); CHECK_APPROX( u4[2], z[4] );
//    }
}
