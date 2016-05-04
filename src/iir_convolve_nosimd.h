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

#ifndef FASTFILTERS_IIR_CONVOLVE_NOSIMD_H
#define FASTFILTERS_IIR_CONVOLVE_NOSIMD_H

DLL_LOCAL bool iir_convolve_inner(const float *inptr, size_t n_pixels, size_t pixel_stride, size_t n_outer,
                                  size_t outer_stride, float *outptr, size_t outptr_stride,
                                  fastfilters_kernel_t kernel_, fastfilters_border_treatment_t left_border,
                                  fastfilters_border_treatment_t right_border, const float *borderptr_left,
                                  const float *borderptr_right, size_t border_outer_stride);

DLL_LOCAL bool iir_convolve_outer(const float *inptr, size_t n_pixels, size_t pixel_stride, size_t n_outer,
                                  size_t outer_stride, float *outptr, size_t outptr_stride,
                                  fastfilters_kernel_t kernel_, fastfilters_border_treatment_t left_border,
                                  fastfilters_border_treatment_t right_border, const float *borderptr_left,
                                  const float *borderptr_right, size_t border_outer_stride);

#endif