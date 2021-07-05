// C++ includes
#include <chrono>
#include <vector>
#include <iostream>
#include <functional>

// Eigen includes
#include <eigen3/Eigen/Core>
using namespace Eigen;

// autodiff includes
#define AUTODIFF_ENABLE_EIGEN_SUPPORT
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

template<typename T>
using MatrixX = Eigen::Matrix<T, -1, -1, 0, -1, -1>;

template<typename T>
using VectorX = Eigen::Matrix<T, -1, 1, 0, -1, 1>;

template<typename T>
using RowVectorX = Matrix<T, 1, -1, 1, 1, -1>;

template<typename T> T f1(const VectorX<T>& x);
template<typename T> T f2(const VectorX<T>& x);
template<typename T> T f3(const VectorX<T>& x);
template<typename T> T f4(const VectorX<T>& x);
template<typename T> T f5(const VectorX<T>& x);
template<typename T> T f6(const VectorX<T>& x);
template<typename T> T f7(const VectorX<T>& x);
template<typename T> T f8(const VectorX<T>& x);
template<typename T> T f9(const VectorX<T>& x);
template<typename T> T f10(const VectorX<T>& x);

VectorXd g1(const VectorXd& x);
VectorXd g2(const VectorXd& x);
VectorXd g3(const VectorXd& x);
VectorXd g4(const VectorXd& x);
VectorXd g5(const VectorXd& x);
VectorXd g6(const VectorXd& x);
VectorXd g7(const VectorXd& x);
VectorXd g8(const VectorXd& x);
VectorXd g9(const VectorXd& x);
VectorXd g10(const VectorXd& x);

/// Return the gradient of a scalar function using finite differences.
VectorXd findiff(const std::function<double(const VectorXd&)>&, const VectorXd&);

/// Return the average time taken to evaluate a function.
template<typename F, typename Vec>
double timeit(const F& f, const Vec& x);

const std::vector<std::function<double(const VectorXd&)>> f = {
    f1<double>,
    f2<double>,
    f3<double>,
    f4<double>,
    f5<double>,
    f6<double>,
    f7<double>,
    f8<double>,
    f9<double>,
    f10<double>
};

const std::vector<std::function<VectorXd(const VectorXd&)>> g = {
    g1,
    g2,
    g3,
    g4,
    g5,
    g6,
    g7,
    g8,
    g9,
    g10
};

const std::vector<std::function<var(const VectorXv&)>> f_autodiff = {
    f1<var>,
    f2<var>,
    f3<var>,
    f4<var>,
    f5<var>,
    f6<var>,
    f7<var>,
    f8<var>,
    f9<var>,
    f10<var>
};


int main(int argc, char **argv)
{
    const auto N = 50;
    const auto M = f.size();

    MatrixXd timing_func_evals(N, M);
    MatrixXd timing_func_evals_autodiff(N, M);

    MatrixXd timing_grad_evals_analytical(N, M);
    MatrixXd timing_grad_evals_findiff(N, M);
    MatrixXd timing_grad_evals_autodiff(N, M);

    for(auto nvar = 1; nvar < N; ++nvar)
    {
        std::cout << "Current variable number: " << nvar << std::endl;
        VectorXd x = VectorXd::Random(nvar);
        VectorXv xv = VectorXv::Random(nvar);
        for(auto ifunc = 0; ifunc < M; ++ifunc)
        {
            timing_func_evals(nvar, ifunc) = timeit(f[ifunc], x);
            timing_func_evals_autodiff(nvar, ifunc) = timeit(f_autodiff[ifunc], xv);

            var y = f_autodiff[ifunc](xv);

            auto g_findiff = [&](const VectorXd& x) { return findiff(f[ifunc], x); };
            auto g_autodiff = [&](const VectorXv& x) { return grad(y, x); };

            timing_grad_evals_analytical(nvar, ifunc) = timeit(g[ifunc], x);
            timing_grad_evals_findiff(nvar, ifunc) = timeit(g_findiff, x);
            timing_grad_evals_autodiff(nvar, ifunc) = timeit(g_autodiff, xv);
        }
    }

    std::cout << "timing_func_evals \n" << timing_func_evals << std::endl;
    std::cout << "timing_func_evals_autodiff \n" << timing_func_evals_autodiff << std::endl;
    std::cout << "timing_grad_evals_analytical \n" << timing_grad_evals_analytical << std::endl;
    std::cout << "timing_grad_evals_findiff \n" << timing_grad_evals_findiff << std::endl;
    std::cout << "timing_grad_evals_autodiff \n" << timing_grad_evals_autodiff << std::endl;
}

VectorXd indices(int n)
{
    VectorXd res(n);
    for(int i = 0; i < n; ++i)
        res[i] = i;
    return res;
}

VectorXd findiff(const std::function<double(const VectorXd&)>& f, const VectorXd& x)
{
    const auto n = x.size();
    const auto fval = f(x);
    const auto eps = 1e-8;
    VectorXd res(n);
    VectorXd xmod(x);
    for(auto i = 0; i < n; ++i)
    {
        const double h = std::abs(x[i]) * eps;
        xmod[i] += h;
        res[i] = (f(xmod) - fval)/h;
        xmod[i] = x[i];
    }
    return res;
}

template<typename F, typename Vec>
double timeit(const F& f, const Vec& x)
{
    const auto samples = 100;
    const auto begin = std::chrono::high_resolution_clock::now();
    for(auto i = 0; i < samples; ++i) f(x);
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - begin;
    return duration.count() / samples;
}

template<typename T>
T f1(const VectorX<T>& x)
{
    return x.sum();
}

template<typename T>
T f2(const VectorX<T>& x)
{
    T res = 0.0;
    const auto n = x.size();
    for(int i = 0; i < n; ++i)
        res += i * x[i];
    return res;
}

template<typename T>
T f3(const VectorX<T>& x)
{
    T res = 0.0;
    const auto n = x.size();
    for(int i = 0; i < n; ++i)
        res += i / x[i];
    return res;
}

template<typename T>
T f4(const VectorX<T>& x)
{
    T res = 0.0;
    T aux = 1.0;
    const auto n = x.size();
    for(int i = 0; i < n; ++i)
    {
        aux *= x[i];
        res += i * aux;
    }
    return res;
}

template<typename T>
T f5(const VectorX<T>& x)
{
    using std::sqrt;
    return sqrt(x.cwiseAbs2().sum());
}

template<typename T>
T f6(const VectorX<T>& x)
{
    return (x / x.sum()).sum();
}

template<typename T>
T f7(const VectorX<T>& x)
{
    return x.cwiseProduct((x / x.sum()).array().log().matrix()).sum();
}

template<typename T>
T f8(const VectorX<T>& x)
{
    auto sinx = x.array().sin().matrix();
    auto cosx = x.array().cos().matrix();
    return sinx.cwiseProduct(cosx).sum();
}

template<typename T>
T f9(const VectorX<T>& x)
{
    return x.array().exp().matrix().sum();
}

template<typename T>
T f10(const VectorX<T>& x)
{
    using std::log;
    T res = 0.0;
    T aux = 0.0;
    const auto n = x.size();
    for(int i = 1; i < n; ++i)
    {
        res += 1 + x[i] + x[i]*x[i] + x[i]*x[i]*x[i] + 1.0/x[i] + 1.0/(x[i]*x[i]) + 1.0/(x[i]*x[i]*x[i]) + x[i] * log(x[i]);
    }
    return res;
}


VectorXd g1(const VectorXd& x)
{
    return VectorXd::Ones(x.size());
}

VectorXd g2(const VectorXd& x)
{
    return indices(x.size());
}

VectorXd g3(const VectorXd& x)
{
    const auto n = x.size();
    VectorXd res(n);
    for(int i = 0; i < n; ++i)
        res[i] = -i/(x[i] * x[i]);
    return res;
}

VectorXd g4(const VectorXd& x)
{
    using std::sqrt;
    const auto n = x.size();
    VectorXd res(n);
    for(int i = 0; i < n; ++i)
        res[i] = 1 + i/sqrt(x[i]);
    return res;
}

VectorXd g5(const VectorXd& x)
{
    const auto n = x.size();
    const auto fval = f5(x);
    VectorXd res(n);
    for(int i = 0; i < n; ++i)
        res[i] = x[i] / fval;
    return res;
}

VectorXd g6(const VectorXd& x)
{
    const auto n = x.size();
    VectorXd res(n);
    for(int i = 0; i < n; ++i)
        res[i] = 1.0 / x[i];
    return res;
}

VectorXd g7(const VectorXd& x)
{
    using std::log;
    const auto n = x.size();
    const auto xsum = x.sum();
    VectorXd res(n);
    for(int i = 0; i < n; ++i)
        res[i] = log(x[i] / xsum);
    return res;
}

VectorXd g8(const VectorXd& x)
{
    using std::sin;
    using std::cos;
    const auto n = x.size();
    VectorXd res(n);
    for(int i = 0; i < n; ++i)
    {
        const auto sin_xi = sin(x[i]);
        const auto cos_xi = cos(x[i]);
        res[i] = cos_xi * cos_xi - sin_xi * sin_xi;
    }
    return res;
}

VectorXd g9(const VectorXd& x)
{
    using std::exp;
    const auto n = x.size();
    const auto fval = f9(x);
    VectorXd res(n);
    for(int i = 0; i < n; ++i)
        res[i] = i/x[i] * (1 - 1/x[i]) * exp(x[i] - fval);
    return res;
}

VectorXd g10(const VectorXd& x)
{
    using std::log;
    const auto n = x.size();
    VectorXd res(n);
    for(int i = 0; i < n; ++i)
        res[i] = 2 + 2*x[i] + 3*x[i]*x[i] - 1/(x[i]*x[i]) - 2/(x[i]*x[i]*x[i]) - 3/(x[i]*x[i]*x[i]*x[i]) + log(x[i]);
    return res;
}
