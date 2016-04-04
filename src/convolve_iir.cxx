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
#include "fastfilters.hxx"
#include "convolve_iir.hxx"

namespace fastfilters
{

namespace iir
{

void convolve_iir(const float *input, const unsigned int pixel_stride, const unsigned int pixel_n,
                  const unsigned int dim_stride, const unsigned int n_dim, float *output, const Coefficients &coefs)
{
    if (pixel_stride == 1) {
        if (detail::cpu_has_avx_fma() && dim_stride == 1)
            convolve_iir_inner_single<true, true, true, true>(input, pixel_stride, pixel_n, dim_stride, n_dim, output,
                                                              coefs);
        else if (detail::cpu_has_avx() && dim_stride == 1)
            convolve_iir_inner_single<true, false, true, false>(input, pixel_stride, pixel_n, dim_stride, n_dim, output,
                                                                coefs);
        else
            convolve_iir_inner_single<false, false, false, false>(input, pixel_stride, pixel_n, dim_stride, n_dim,
                                                                  output, coefs);
    } else {
        if (detail::cpu_has_avx_fma() && dim_stride == 1)
            convolve_iir_outer_single<true, true, true, true>(input, pixel_stride, pixel_n, dim_stride, n_dim, output,
                                                              coefs);
        else if (detail::cpu_has_avx() && dim_stride == 1)
            convolve_iir_outer_single<true, false, true, false>(input, pixel_stride, pixel_n, dim_stride, n_dim, output,
                                                                coefs);
        else
            convolve_iir_outer_single<false, false, false, false>(input, pixel_stride, pixel_n, dim_stride, n_dim,
                                                                  output, coefs);
    }
}

} // namepsace iir

} // namespace fastfilters