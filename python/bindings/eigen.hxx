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

// Eigen includes
#include <Eigen/Core>

// pybind11 includes
#include "pybind11.hxx"

template<typename Vec, typename T, bool isarray, bool isconst, bool isview>
void exportVector(py::module& m, const char* typestr)
{
    auto cls = py::class_<Vec>(m, typestr);

    if constexpr (!isview) {
        using VecRef = Eigen::Ref<Vec>;
        using VecConstRef = Eigen::Ref<const Vec>;

        cls.def(py::init<>());
        cls.def(py::init<long>());
        cls.def(py::init<const VecRef&>());
        cls.def(py::init<const VecConstRef&>());
    }

    // Define constructors if not a view (e.g., wrapped into an Eigen::Ref)
    if constexpr (!isview) {
        cls.def(py::init<>());
        cls.def(py::init<long>());
    }

    cls.def("__len__", [](const Vec& s) { return s.size(); });

    cls.def("__getitem__", [](const Vec& s, size_t i) {
        const size_t size = s.size();
        if(i >= size) throw py::index_error();
        return s[i];
    });

    if constexpr (!isconst) {
        cls.def("__setitem__", [](Vec& s, size_t i, const T& val) {
            const size_t size = s.size();
            if(i >= size) throw py::index_error();
            s[i] = val;
        });
    }

#if EIGEN_VERSION_AT_LEAST(3, 3, 90)
    cls.def("__iter__", [](const Vec& s) {
        return py::make_iterator(s.begin(), s.end()); // begin/end iterators have not always being available!
    }, py::keep_alive<0, 1>()); // keep object alive while iterator exists;
#endif

    cls.def("__str__", [](const Vec& s) {
        std::stringstream stream;
        stream << s;
        return stream.str();
    });

    cls.def("__repr__", [=](const Vec& s) {
        std::stringstream stream;
        stream << "autodiff." << typestr << "([";
        for(auto i = 0; i < s.size(); ++i)
            stream << (i == 0 ? "" : ", ") << s[i];
        stream << "])";
        return stream.str();
    });

    cls.def(-py::self);

    cls.def(py::self + py::self);
    cls.def(py::self - py::self);

    if constexpr(!isconst) {
        cls.def(py::self += py::self);
        cls.def(py::self -= py::self);
        cls.def(py::self *= T());
        cls.def(py::self /= T());
    }

    if constexpr (!isarray) {
        cls.def(py::self == py::self);
        cls.def(py::self != py::self);
    } else {
        cls.def("__eq__", [](const Vec& l, const Vec& r) { return (l == r).all(); } );
        cls.def("__eq__", [](const Vec& l, const py::array_t<double>& r) { return (l == Eigen::ArrayXd::Map(r.data(), r.size())).all(); } );
        cls.def("__ne__", [](const Vec& l, const Vec& r) { return (l != r).any(); } );
        cls.def("__eq__", [](const Vec& l, const py::array_t<double>& r) { return (l != Eigen::ArrayXd::Map(r.data(), r.size())).any(); } );
    }

    if constexpr (isarray) {
        cls.def(py::self * py::self);
        cls.def(py::self / py::self);

        cls.def(py::self + T());
        cls.def(py::self - T());
        cls.def(py::self * T());
        cls.def(py::self / T());

        cls.def(T() + py::self);
        cls.def(T() - py::self);
        cls.def(T() * py::self);
        cls.def(T() / py::self);

        if constexpr (!isconst) {
            cls.def(py::self *= py::self);
            cls.def(py::self /= py::self);
            cls.def(py::self += T());
            cls.def(py::self -= T());
        }
    }

    cls.def("asarray", [](const Vec& s) {
        py::array_t<double> res(s.size());
        auto data = res.mutable_data();
        for(auto i = 0; i < s.size(); ++i)
            data[i] = autodiff::val(s[i]);
        return res;
    });
}


template<typename Mat, typename T, bool isarray, bool isconst, bool isview>
void exportMatrix(py::module& m, const char* typestr)
{
    auto cls = py::class_<Mat>(m, typestr);

    if constexpr (!isview) {
        using MatRef = Eigen::Ref<Mat>;
        using MatConstRef = Eigen::Ref<const Mat>;

        cls.def(py::init<>());
        cls.def(py::init<long, long>());
        cls.def(py::init<const MatRef&>());
        cls.def(py::init<const MatConstRef&>());
    }

    // Define constructors if not a view (e.g., wrapped into an Eigen::Ref)
    if constexpr (!isview) {
        cls.def(py::init<>());
        cls.def(py::init<long, long>());
    }

    cls.def("__len__", [](const Mat& s) { return s.size(); });

    cls.def("rows", [](const Mat& s) { return s.rows(); });
    cls.def("cols", [](const Mat& s) { return s.cols(); });

    cls.def("__getitem__", [](const Mat& s, py::tuple pos) {
        const size_t rows = s.rows();
        const size_t cols = s.cols();
        const size_t i = pos[0].cast<size_t>();
        const size_t j = pos[1].cast<size_t>();
        if(i >= rows) throw py::index_error();
        if(j >= cols) throw py::index_error();
        return s(i, j);
    });

    if constexpr (!isconst) {
        cls.def("__setitem__", [](Mat& s, py::tuple pos, const T& val) {
            const size_t rows = s.rows();
            const size_t cols = s.cols();
            const size_t i = pos[0].cast<size_t>();
            const size_t j = pos[1].cast<size_t>();
            if(i >= rows) throw py::index_error();
            if(j >= cols) throw py::index_error();
            s(i, j) = val;
        });
    }

// #if EIGEN_VERSION_AT_LEAST(3, 3, 90)
//     cls.def("__iter__", [](const Mat& s) {
//         return py::make_iterator(s.begin(), s.end()); // begin/end iterators have not always being available!
//     }, py::keep_alive<0, 1>()); // keep object alive while iterator exists;
// #endif

    cls.def("__str__", [](const Mat& s) {
        std::stringstream stream;
        stream << s;
        return stream.str();
    });

    cls.def("__repr__", [=](const Mat& s) {
        std::stringstream stream;
        stream << "autodiff." << typestr << "([\n";
        for(auto i = 0; i < s.rows(); ++i) {
            stream << (i == 0 ? "" : "\n");
            for(auto j = 0; j < s.cols(); ++j)
                stream << (j == 0 ? "  [" : ", ") << s(i, j);
            stream << "],";
        }
        stream << "\n])";
        return stream.str();
    });

    cls.def(-py::self);

    cls.def(py::self + py::self);
    cls.def(py::self - py::self);

    if constexpr(!isconst) {
        cls.def(py::self += py::self);
        cls.def(py::self -= py::self);
        cls.def(py::self *= T());
        cls.def(py::self /= T());
    }

    cls.def(py::self == py::self);
    cls.def(py::self != py::self);

    cls.def("asarray", [](const Mat& s) {
        py::tuple shape = py::make_tuple(s.rows(), s.cols());
        py::array_t<double> res(shape);
        auto data = res.mutable_data();
        auto k = 0;
        for(auto j = 0; j < s.cols(); ++j)
            for(auto i = 0; i < s.rows(); ++i)
                data[k++] = autodiff::val(s(i, j));
        return res;
    });
}

constexpr auto isview(bool val) { return val; }
constexpr auto isconst(bool val) { return val; }
constexpr auto isarray(bool val) { return val; }
