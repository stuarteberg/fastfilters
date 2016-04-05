// fastfilters
// Copyright (c) 2016 Sven Peter
// sven.peter@iwr.uni-heidelberg.de or mail@svenpeter.me
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "fastfilters.hxx"

namespace py = pybind11;

namespace
{

template <class KernelType> class ConvolveFunctor
{
};

template <> class ConvolveFunctor<fastfilters::iir::Coefficients>
{
  public:
    void operator()(const float *input, const unsigned int pixel_stride, const unsigned int pixel_n,
                    const unsigned int dim_stride, const unsigned int n_dim, float *output,
                    const fastfilters::iir::Coefficients &coefs)
    {
        fastfilters::iir::convolve_iir(input, pixel_stride, pixel_n, dim_stride, n_dim, output, coefs);
    }
};

template <> class ConvolveFunctor<fastfilters::fir::Kernel>
{
  public:
    void operator()(const float *input, const unsigned int pixel_stride, const unsigned int pixel_n,
                    const unsigned int dim_stride, const unsigned int n_dim, float *output,
                    fastfilters::fir::Kernel &kernel)
    {
        fastfilters::fir::convolve_fir(input, pixel_stride, pixel_n, dim_stride, n_dim, output, kernel);
    }
};

template <class KernelType> py::array_t<float> convolve(KernelType &k, py::array_t<float> input)
{
    ConvolveFunctor<KernelType> conv;

    py::buffer_info info_in = input.request();
    const unsigned int ndim = (unsigned int)info_in.ndim;

    if (info_in.ndim <= 0)
        throw std::runtime_error("Number of dimensions must be > 0");

    auto result = py::array(py::buffer_info(nullptr, sizeof(float), py::format_descriptor<float>::value(), ndim,
                                            info_in.shape, info_in.strides));
    py::buffer_info info_out = result.request();

    const float *inptr = (float *)info_in.ptr;
    float *outptr = (float *)info_out.ptr;

    if (info_in.ndim == 1) {
        conv(inptr, info_in.shape[0], 1, 1, 1, outptr, k);
    } else if (info_in.ndim == 2) {
        conv(inptr, info_in.shape[0], info_in.shape[1], info_in.shape[1], 1, outptr, k);
        conv(outptr, info_in.shape[1], 1, info_in.shape[0], info_in.shape[1], outptr, k);
    } else {
        unsigned int n_dim = 1;
        for (unsigned int i = 1; i < ndim; ++i)
            n_dim *= info_in.shape[i];
        conv(inptr, info_in.shape[0], n_dim, n_dim, 1, outptr, k);

        n_dim = 1;
        for (unsigned int i = 0; i < ndim - 1; ++i)
            n_dim *= info_in.shape[i];
        conv(outptr, info_in.shape[ndim - 1], 1, n_dim, info_in.shape[ndim - 1], outptr, k);

        for (unsigned int i = ndim - 2; i > 0; --i) {
            unsigned int n_dim_inner = 1;
            unsigned int n_dim_outer = 1;
            for (unsigned int j = 0; j < i; ++j)
                n_dim_outer *= info_in.shape[j];
            for (unsigned int j = i + 1; j < ndim; ++j)
                n_dim_inner *= info_in.shape[j];

            for (unsigned int j = 0; j < n_dim_outer; ++j) {
                conv(outptr + n_dim_inner * info_in.shape[i - 1] * j, info_in.shape[i], n_dim_inner, n_dim_inner, 1,
                     outptr + n_dim_inner * info_in.shape[i - 1] * j, k);
            }
        }
    }

    return result;
}

} // anonymous namespace

PYBIND11_PLUGIN(fastfilters)
{
    py::module m_fastfilters("fastfilters", "fast gaussian kernel and derivative filters");

    py::class_<fastfilters::iir::Coefficients>(m_fastfilters, "IIRKernel")
        .def(py::init<const double, const unsigned int>())
        .def("__repr__", [](const fastfilters::iir::Coefficients &a) {
            std::ostringstream oss;
            oss << "<fastfilters.IIRKernel with sigma = " << a.sigma << " and order = " << a.order << ">";

            return oss.str();
        })
        .def_readonly("sigma", &fastfilters::iir::Coefficients::sigma)
        .def_readonly("order", &fastfilters::iir::Coefficients::order);

    py::class_<fastfilters::fir::Kernel>(m_fastfilters, "FIRKernel")
        .def(py::init<const bool, const std::vector<float>>())
        .def("__repr__", [](const fastfilters::fir::Kernel &a) {
            std::ostringstream oss;
            oss << "<fastfilters.FIRKernel with symmetric = " << a.is_symmetric << " and length = " << a.len() << ">";

            return oss.str();
        })
        .def("__getitem__", [](const fastfilters::fir::Kernel &s, size_t i) {
            if (i >= s.len())
                throw py::index_error();
            return s[i];
        });

    m_fastfilters.def("convolve", &convolve<fastfilters::iir::Coefficients>,
                      "apply IIR filter to all dimensions of array and return result.");
    m_fastfilters.def("convolve", &convolve<fastfilters::fir::Kernel>,
                      "apply FIR filter to all dimensions of array and return result.");

    m_fastfilters.def("cpu_has_avx_fma", &fastfilters::detail::cpu_has_avx_fma);
    m_fastfilters.def("cpu_has_avx", &fastfilters::detail::cpu_has_avx);
    m_fastfilters.def("cpu_has_avx2", &fastfilters::detail::cpu_has_avx2);

    return m_fastfilters.ptr();
}
